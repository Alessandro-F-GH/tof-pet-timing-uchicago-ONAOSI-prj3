"""Group saved ML predictions by per-event spectral peak frequency AND phase.

Original prepared waveform only; saved development predictions only.
No filtering, signal modifications, feature transforms, inference or refitting.
Peak search is restricted to a user-specified frequency band.
Phase is defined with respect to common LED-relative t=0, at EACH event's
own fitted peak frequency (not the phase of a common frequency bin).
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from waveform_analysis.data.prepared_cache import PreparedCacheLookup
from waveform_analysis.data.results import read_run, resolve_run_config
from waveform_analysis.data.storage import RunStore
from waveform_analysis.data.view import mode_family
from waveform_analysis.engine import tshap_explainer as tshap
from waveform_analysis.scripts.analyze_waveform_fft import (
    remove_linear_trend, validate_grid)
from waveform_analysis.scripts.evaluate_conditional_knn import _prediction_payload


def event_peak_frequency_phase(x, t_ns, *, low_GHz=1.5,
                               high_GHz=3.5, fft_padding=4):
    """Detrended Hann FFT peak, then complex projection at event-specific peak.

    Zero padding interpolates frequency-bin positions; it does NOT increase
    true resolution of the measurement window. Phase is referenced to t=0.
    """
    x = np.asarray(x, np.float64)
    t = np.asarray(t_ns, np.float64)
    if x.ndim != 2 or t.shape != (x.shape[1],) or len(t) < 16:
        raise ValueError("Expected [events,samples] and a matching time grid")
    dt = float(np.median(np.diff(t)))
    if (fft_padding < 1 or not (0 < low_GHz < high_GHz < 0.5/dt)
            or not np.allclose(np.diff(t), dt, rtol=1e-4, atol=1e-6)):
        raise ValueError("Invalid frequency band, FFT padding or time grid")
    taper = np.hanning(len(t))
    detrended = remove_linear_trend(x)
    n_fft = len(t) * int(fft_padding)
    freq = np.fft.rfftfreq(n_fft, d=dt)
    band = np.flatnonzero((freq >= low_GHz) & (freq <= high_GHz))
    if len(band) < 2:
        raise ValueError("Peak-search band contains fewer than two FFT bins")
    z = np.fft.rfft(detrended*taper[None, :], n=n_fft, axis=1)
    power = np.abs(z[:, band])**2
    peak_col = np.argmax(power, axis=1)
    peak_frequency = freq[band[peak_col]]
    # Reproject with absolute (LED-relative) timestamps to compare phases
    # even when eventwise peak frequencies differ.
    kernel = np.exp(-2j*np.pi*peak_frequency[:, None]*t[None, :])
    coefficient = (2.0/taper.sum()) * np.sum(
        detrended*taper[None, :]*kernel, axis=1)
    amplitude = np.abs(coefficient)
    phase = np.angle(coefficient)
    band_power = np.sum(power, axis=1)
    peak_fraction = power[np.arange(len(x)), peak_col] / np.maximum(band_power, 1e-30)
    return peak_frequency, phase, amplitude, peak_fraction



def event_joint_peak_frequency_phases(pairs, t_ns, *, low_GHz=1.5,
                                      high_GHz=3.5, fft_padding=4,
                                      joint_mode="product"):
    """Find one peak frequency per paired event, then both LED-relative phases.

    Product/geometric_mean selects the same maximum and favors a frequency
    energetic in both channels. Sum is provided as a diagnostic alternative.
    No waveform modifications, and no model inference.
    """
    x = np.asarray(pairs, np.float64)
    t = np.asarray(t_ns, np.float64)
    if x.ndim != 3 or x.shape[1:] != (2, len(t)) or len(t) < 16:
        raise ValueError("Expected [events,2,time] and matching time axis")
    dt = float(np.median(np.diff(t)))
    if (fft_padding < 1 or not 0 < low_GHz < high_GHz < .5/dt
            or not np.allclose(np.diff(t), dt, rtol=1e-4, atol=1e-6)):
        raise ValueError("Invalid frequency band or grid")
    if joint_mode not in ("product", "geometric_mean", "sum"):
        raise ValueError("Invalid joint scoring mode")
    taper = np.hanning(len(t))
    residual = remove_linear_trend(x)
    n_fft = len(t)*int(fft_padding)
    f_grid = np.fft.rfftfreq(n_fft, d=dt)
    band = np.flatnonzero((f_grid >= low_GHz) & (f_grid <= high_GHz))
    if len(band) < 2:
        raise ValueError("Peak-search band too narrow")
    spectra = np.fft.rfft(residual*taper[None,None,:],
                          n=n_fft, axis=-1)
    powers = np.abs(spectra[:,:,band])**2
    # Normalize powers before multiplication to avoid overflow and arbitrary
    # detector gain favoring the higher-gain channel.
    if joint_mode == "sum":
        score = powers[:,0] + powers[:,1]
    else:
        norm = np.maximum(np.max(powers, axis=2, keepdims=True), 1e-30)
        scaled = powers / norm
        score = scaled[:,0]*scaled[:,1]
    peak_bin = np.argmax(score, axis=1)
    frequency = f_grid[band[peak_bin]]
    kernel = np.exp(-2j*np.pi*frequency[:,None]*t[None,:])
    coeff = 2.0/taper.sum()*np.sum(
        residual*taper[None,None,:]*kernel[:,None,:], axis=-1)
    phase = np.angle(coeff)
    amplitude = np.abs(coeff)
    delta_phase = np.angle(coeff[:,1]*np.conjugate(coeff[:,0]))
    return frequency, phase, amplitude, delta_phase


def group_grid(frequency, phase, prediction, *, frequency_bins=8,
               phase_bins=12, low=1.5, high=3.5,
               frequency_binning="quantile", phase_binning="uniform"):
    """Balanced rank-frequency bins, with optional balanced phase bins.

    Identical FFT peak frequencies remain in the same bin to prevent
    arbitrary event-order effects; bin populations are approximately balanced.
    """
    f = np.asarray(frequency, np.float64)
    phi = np.asarray(phase, np.float64)
    y = np.asarray(prediction, np.float64)
    if (f.ndim != 1 or f.shape != phi.shape or f.shape != y.shape
            or len(f) < frequency_bins
            or (phase_binning == "quantile" and len(phi) < phase_bins)
            or not all(np.isfinite(v).all() for v in (f, phi, y))):
        raise ValueError("Frequency, phase and prediction must align")
    if frequency_binning not in ("quantile", "uniform"):
        raise ValueError("Invalid frequency binning")
    if phase_binning not in ("quantile", "uniform"):
        raise ValueError("Invalid phase binning")
    if frequency_binning == "quantile":
        # Keep identical FFT peak frequencies together: splitting ties by
        # event order can create spurious phase/prediction differences.
        unique, inverse, counts = np.unique(
            f, return_inverse=True, return_counts=True)
        if len(unique) < frequency_bins:
            raise ValueError(
                f"Only {len(unique)} distinct peak frequencies; cannot form "
                f"{frequency_bins} nonempty tie-preserving frequency bins")
        cumulative = np.cumsum(counts)
        cuts = []
        previous = 0
        for j in range(1, frequency_bins):
            target = j * len(f) / frequency_bins
            # Candidate boundary after unique[k], while retaining enough
            # distinct frequencies for all remaining nonempty bins.
            eligible = np.arange(previous+1, len(unique)-(frequency_bins-j)+1)
            # eligible represents number of distinct frequencies consumed
            best = int(eligible[np.argmin(np.abs(cumulative[eligible-1]-target))])
            cuts.append(best)
            previous = best
        f_bin = np.searchsorted(cuts, inverse, side="right")
        f_ranges = [(float(f[f_bin == j].min()), float(f[f_bin == j].max()))
                    for j in range(frequency_bins)]
    else:
        f_edges = np.linspace(low, high, frequency_bins + 1)
        f_bin = np.clip(np.searchsorted(f_edges, f, side="right")-1,
                        0, frequency_bins-1)
        f_ranges = [(float(f_edges[j]), float(f_edges[j+1]))
                    for j in range(frequency_bins)]
    if phase_binning == "quantile":
        p_bin = np.empty(len(phi), dtype=int)
        for j, members in enumerate(np.array_split(np.argsort(phi, kind="stable"),
                                                    phase_bins)):
            p_bin[members] = j
        p_ranges = [(float(np.degrees(phi[p_bin == j].min())),
                     float(np.degrees(phi[p_bin == j].max())))
                    for j in range(phase_bins)]
    else:
        p_edges = np.linspace(-np.pi, np.pi, phase_bins + 1)
        p_bin = np.clip(np.searchsorted(p_edges, phi, side="right")-1,
                        0, phase_bins-1)
        p_ranges = [(float(np.degrees(p_edges[j])),
                     float(np.degrees(p_edges[j+1])))
                    for j in range(phase_bins)]
    global_mean = float(np.mean(y))
    records = []
    for i in range(frequency_bins):
        for j in range(phase_bins):
            values = y[(f_bin == i) & (p_bin == j)]
            records.append({
                "frequency_bin": i, "phase_bin": j,
                "frequency_min_GHz": f_ranges[i][0],
                "frequency_max_GHz": f_ranges[i][1],
                "frequency_center_GHz": float(np.mean(f_ranges[i])),
                "phase_min_deg": p_ranges[j][0],
                "phase_max_deg": p_ranges[j][1],
                "phase_center_deg": float(np.mean(p_ranges[j])),
                "n": int(len(values)),
                "mean_prediction_ps": float(values.mean()) if len(values) else None,
                "mean_deviation_ps": (
                    float(values.mean()-global_mean) if len(values) else None),
                "std_prediction_ps": (
                    float(values.std(ddof=1)) if len(values)>1 else None),
                "sem_prediction_ps": (
                    float(values.std(ddof=1)/np.sqrt(len(values)))
                    if len(values)>1 else None),
            })
    return records, f_bin, p_bin


def make_plots(out, tag, records, frequency, phase, n_f, n_p,
               min_cell_events, phase_binning="uniform"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    counts = np.zeros((n_f, n_p), int)
    means = np.full((n_f, n_p), np.nan)
    for rec in records:
        i, j = rec["frequency_bin"], rec["phase_bin"]
        counts[i, j] = rec["n"]
        if rec["n"] >= min_cell_events:
            means[i, j] = rec["mean_deviation_ps"]
    fig, ax = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)
    vmax = float(np.nanmax(np.abs(means))) if np.isfinite(means).any() else 1.
    vmax = max(vmax, .01)
    # Equally tall rows represent equally populated frequency groups, not
    # equally wide GHz intervals. Label with actual observed ranges.
    im = ax[0].imshow(means, origin="lower", aspect="auto",
                      extent=(0, n_p, 0, n_f), vmin=-vmax, vmax=vmax,
                      cmap="RdBu_r")
    fig.colorbar(im, ax=ax[0], label="Mean prediction minus global mean (ps)")
    ax[0].set_title(f"{tag} | min {min_cell_events} events/cell")
    im2 = ax[1].imshow(counts, origin="lower", aspect="auto",
                       extent=(0, n_p, 0, n_f), cmap="viridis")
    fig.colorbar(im2, ax=ax[1], label="Events")
    ax[1].set_title("Counts (all cells)")
    freq_labels = [
        f"{records[i*n_p]['frequency_min_GHz']:.3f}–"
        f"{records[i*n_p]['frequency_max_GHz']:.3f}"
        for i in range(n_f)]
    phase_labels = [f"{records[j]['phase_center_deg']:.0f}°"
                    for j in range(n_p)]
    for a in ax:
        a.set_yticks(np.arange(n_f)+.5, freq_labels)
        a.set_xticks(np.arange(n_p)+.5, phase_labels, rotation=45)
        a.set_xlabel("Phase bin" if phase_binning=="quantile" else "Phase (degrees)")
        a.set_ylabel("Peak frequency bin (GHz range)")
    fig.savefig(out/f"{tag}_frequency_phase.png", dpi=170)
    fig.savefig(out/f"{tag}_frequency_phase.pdf")
    plt.close(fig)
    fig, ax = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    ax[0].hist(frequency, bins=45)
    ax[0].set(xlabel="Event peak frequency (GHz)", ylabel="Events")
    ax[1].hist(np.degrees(phase), bins=36, range=(-180,180))
    ax[1].set(xlabel="Phase at event peak (degrees)", ylabel="Events")
    fig.savefig(out/f"{tag}_distributions.png", dpi=170)
    plt.close(fig)


def run(args):
    run_dir = tshap._resolve_model_run_directory(args.model_dir)
    record = read_run(run_dir)
    config = resolve_run_config(run_dir)
    manifest = record["manifest"]
    tshap._apply_tshap_cache(config, cache_root=args.cache_root,
                            artifact_root=args.artifact_root)
    dataset = PreparedCacheLookup().find("development", config, manifest)
    family = mode_family(config["mode"])
    waveforms = getattr(dataset, f"{family}_windows")
    time_ps, dt_ps = validate_grid(getattr(dataset, f"{family}_time_ps"))
    store = RunStore(run_dir, artifact_root=config["preprocessing"].get("artifact_root"))
    saved_path = store.development_predictions_path
    if not saved_path.is_file():
        saved_path = run_dir/"artifacts"/"development_pred.npz"
    predictions = _prediction_payload(saved_path, dataset, manifest)
    event_ids = np.asarray(dataset.event_index, np.int64)
    if args.max_events > 0 and args.max_events < dataset.n_events:
        rows = np.sort(np.random.default_rng(args.seed).choice(
            dataset.n_events, args.max_events, replace=False))
    else:
        rows = np.arange(dataset.n_events)
    time_ns = time_ps/1000.
    transform = getattr(dataset, f"{family}_transform")
    minimum = np.asarray(transform.minimum, np.float64).reshape(2, 1)
    span = np.asarray(transform.maximum, np.float64).reshape(2, 1)-minimum
    out = args.output_root.expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    summary = {
        "source_saved_predictions": str(saved_path),
        "model": manifest["model"], "events": int(len(rows)),
        "band_GHz": [args.band_GHz[0], args.band_GHz[1]],
        "frequency_bins": args.frequency_bins, "phase_bins": args.phase_bins,
        "frequency_binning": args.frequency_binning,
        "phase_binning": args.phase_binning,
        "amplitude_percentile": args.min_amplitude_percentile,
        "fft_padding": args.fft_padding,
        "joint_frequency_estimation": True,
        "joint_mode": args.joint_mode,
        "shared_frequency_bins": True,
        "phase_outputs": ["detector1", "detector2", "delta_phase"],
        "time_reference": "t=0 at prepared LED crossing",
        "note": ("Zero padding is interpolation, not extra spectral resolution. "
                 "Phase at variable frequency is relative to the same LED time zero. "
                 "Cells with small sample counts are hidden; association is descriptive. "
                 "No waveform modification or model inference."),
        "results": {},
    }
    all_records = []
    windows = {"tail": args.tail_window_ns}
    if args.include_pre_LED:
        windows["pre_LED"] = args.pre_window_ns
    for name, limits in windows.items():
        indices = np.flatnonzero((time_ns >= limits[0]-1e-9)
                                 & (time_ns <= limits[1]+1e-9))
        if len(indices) < 16:
            raise ValueError(f"Not enough samples for {name}")
        true_df_GHz = 1./(len(indices)*dt_ps/1000.)
        frequency = np.empty(len(rows), np.float64)
        phases = np.empty((len(rows), 2), np.float64)
        amps = np.empty((len(rows), 2), np.float64)
        differences = np.empty(len(rows), np.float64)
        for start in range(0, len(rows), args.chunk_size):
            stop = min(start+args.chunk_size, len(rows))
            pair = np.asarray(waveforms[rows[start:stop], :, :], np.float64)
            pair = pair[:, :, indices] * span[:,0][None,:,None] + minimum[:,0][None,:,None]
            f, p, a, delta = event_joint_peak_frequency_phases(
                pair, time_ns[indices], low_GHz=args.band_GHz[0],
                high_GHz=args.band_GHz[1], fft_padding=args.fft_padding,
                joint_mode=args.joint_mode)
            frequency[start:stop] = f
            phases[start:stop] = p
            amps[start:stop] = a
            differences[start:stop] = delta

        joint_amp = np.sqrt(amps[:,0]*amps[:,1])
        floor = float(np.percentile(joint_amp, args.min_amplitude_percentile))
        selected = (np.isfinite(frequency) & np.isfinite(differences)
                    & np.isfinite(phases).all(axis=1) & np.isfinite(joint_amp)
                    & (joint_amp >= floor))
        if selected.sum() < args.frequency_bins:
            raise ValueError("Too few common paired events after amplitude filter")

        # CRUCIAL: exactly the same selected events and frequency vector
        # enter all three groupings, guaranteeing common percentile bands.
        series = (("detector1", phases[:,0]),
                  ("detector2", phases[:,1]),
                  ("delta_phase", differences))
        shared_fbin = None
        shared_ranges = None
        for suffix, phase_values in series:
            records, fbin, pbin = group_grid(
                frequency[selected], phase_values[selected],
                predictions[rows][selected],
                frequency_bins=args.frequency_bins,
                phase_bins=args.phase_bins, low=args.band_GHz[0],
                high=args.band_GHz[1],
                frequency_binning=args.frequency_binning,
                phase_binning=args.phase_binning)
            ranges = [(records[k*args.phase_bins]["frequency_min_GHz"],
                       records[k*args.phase_bins]["frequency_max_GHz"])
                      for k in range(args.frequency_bins)]
            if shared_fbin is None:
                shared_fbin, shared_ranges = fbin.copy(), ranges
            elif not np.array_equal(fbin, shared_fbin) or ranges != shared_ranges:
                raise AssertionError("Frequency bins unexpectedly differ between phase views")
            tag = f"{name}_{suffix}"
            make_plots(out, tag, records, frequency[selected],
                       phase_values[selected], args.frequency_bins,
                       args.phase_bins, args.min_cell_events,
                       phase_binning=args.phase_binning)
            for rec in records:
                all_records.append({"window":name, "detector":suffix, **rec})
            occupied = sum(r["n"]>=args.min_cell_events for r in records)
            summary["results"][tag] = {
                "frequency_resolution_GHz": true_df_GHz,
                "n_selected": int(selected.sum()),
                "joint_amplitude_floor_mV": floor,
                "peak_frequency_mean_GHz": float(np.mean(frequency[selected])),
                "peak_frequency_std_GHz": float(np.std(frequency[selected])),
                "occupied_cells_above_threshold": occupied,
                "total_cells": len(records),
                "frequency_bin_counts": [
                    int(np.sum(fbin == k)) for k in range(args.frequency_bins)],
                "frequency_bin_ranges_GHz": [list(r) for r in ranges],
            }
            print(f"JOINT PEAK GROUP | {tag} | used={selected.sum()} "
                  f"| frequency={np.mean(frequency[selected]):.3f}±"
                  f"{np.std(frequency[selected]):.3f} GHz "
                  f"| df={true_df_GHz:.3f} GHz "
                  f"| populated_cells={occupied}/{len(records)}",
                  flush=True)
        np.savez_compressed(
            out/f"{name}_joint_events.npz",
            event_id=event_ids[rows],
            saved_prediction_ps=predictions[rows],
            joint_peak_frequency_GHz=frequency,
            detector1_phase_rad=phases[:,0],
            detector2_phase_rad=phases[:,1],
            delta_phase_rad=differences,
            detector1_amplitude_mV=amps[:,0],
            detector2_amplitude_mV=amps[:,1],
            joint_amplitude_mV=joint_amp,
            selected=selected,
            frequency_bin=np.where(selected, np.searchsorted(
                np.asarray([r[1] for r in shared_ranges[:-1]]),
                frequency, side="right"), -1),
        )
    with (out/"frequency_phase_bins.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(all_records[0]))
        writer.writeheader()
        writer.writerows(all_records)
    (out/"summary.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False), encoding="utf-8")
    return summary


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model-dir", type=Path, required=True)
    p.add_argument("--cache-root", type=Path)
    p.add_argument("--artifact-root", type=Path)
    p.add_argument("--output-root", type=Path, required=True)
    p.add_argument("--max-events", type=int, default=10000,
                   help="Use 0 for all events")
    p.add_argument("--band-GHz", nargs=2, type=float, default=[1.5,3.5])
    p.add_argument("--fft-padding", type=int, default=4)
    p.add_argument("--joint-mode", choices=("product", "geometric_mean", "sum"),
                   default="product")
    p.add_argument("--frequency-bins", type=int, default=8)
    p.add_argument("--phase-bins", type=int, default=12)
    p.add_argument("--frequency-binning", choices=("quantile", "uniform"),
                   default="quantile")
    p.add_argument("--phase-binning", choices=("uniform", "quantile"),
                   default="uniform")
    p.add_argument("--min-cell-events", type=int, default=20)
    p.add_argument("--min-amplitude-percentile", type=float, default=20.)
    p.add_argument("--tail-window-ns", nargs=2, type=float, default=[12.,45.])
    p.add_argument("--include-pre-LED", action="store_true")
    p.add_argument("--pre-window-ns", nargs=2, type=float, default=[-2.,0.])
    p.add_argument("--chunk-size", type=int, default=128)
    p.add_argument("--seed", type=int, default=42)
    return p


if __name__ == "__main__":
    run(parser().parse_args())
