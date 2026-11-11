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


def group_grid(frequency, phase, prediction, *, frequency_bins=8,
               phase_bins=12, low=1.5, high=3.5):
    f = np.asarray(frequency)
    phi = np.asarray(phase)
    y = np.asarray(prediction)
    if not (f.shape == phi.shape == y.shape):
        raise ValueError("Frequency, phase and prediction must align")
    f_edges = np.linspace(low, high, frequency_bins+1)
    p_edges = np.linspace(-np.pi, np.pi, phase_bins+1)
    f_bin = np.clip(np.searchsorted(f_edges, f, side="right")-1,
                    0, frequency_bins-1)
    p_bin = np.clip(np.searchsorted(p_edges, phi, side="right")-1,
                    0, phase_bins-1)
    global_mean = float(np.mean(y))
    records = []
    for i in range(frequency_bins):
        for j in range(phase_bins):
            values = y[(f_bin == i) & (p_bin == j)]
            records.append({
                "frequency_bin": i, "phase_bin": j,
                "frequency_center_GHz": float((f_edges[i]+f_edges[i+1])/2),
                "phase_center_deg": float(np.degrees((p_edges[j]+p_edges[j+1])/2)),
                "n": int(len(values)),
                "mean_prediction_ps": (float(values.mean()) if len(values) else None),
                "mean_deviation_ps": (
                    float(values.mean()-global_mean) if len(values) else None),
                "std_prediction_ps": (
                    float(values.std(ddof=1)) if len(values)>1 else None),
                "sem_prediction_ps": (
                    float(values.std(ddof=1)/np.sqrt(len(values)))
                    if len(values)>1 else None),
            })
    return records, f_bin, p_bin


def make_plots(out, tag, records, frequency, phase, n_f, n_p, min_cell_events):
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
    extent = [-180, 180, np.min(frequency), np.max(frequency)]
    # Use actual declared band centers, not observed extrema.
    centers = np.array([r["frequency_center_GHz"] for r in records[::n_p]])
    band_step = centers[1]-centers[0] if len(centers)>1 else 1.
    extent[2:] = [centers[0]-band_step/2, centers[-1]+band_step/2]
    vmax = float(np.nanmax(np.abs(means))) if np.isfinite(means).any() else 1.
    vmax = max(vmax, .01)
    im = ax[0].imshow(means, origin="lower", aspect="auto", extent=extent,
                      vmin=-vmax, vmax=vmax, cmap="RdBu_r")
    fig.colorbar(im, ax=ax[0], label="Mean prediction minus global mean (ps)")
    ax[0].set_title(f"{tag} | min {min_cell_events} events/cell")
    ax[0].set_xlabel("Phase at event-specific peak (degrees)")
    ax[0].set_ylabel("Peak frequency (GHz)")
    im2 = ax[1].imshow(counts, origin="lower", aspect="auto", extent=extent,
                       cmap="viridis")
    fig.colorbar(im2, ax=ax[1], label="Events")
    ax[1].set_title("Counts (all cells)")
    ax[1].set_xlabel("Phase at event-specific peak (degrees)")
    ax[1].set_ylabel("Peak frequency (GHz)")
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
        "amplitude_percentile": args.min_amplitude_percentile,
        "fft_padding": args.fft_padding,
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
        for d in (0, 1):
            freq = np.empty(len(rows))
            phase = np.empty(len(rows))
            amplitude = np.empty(len(rows))
            prominence = np.empty(len(rows))
            for start in range(0, len(rows), args.chunk_size):
                end = min(start+args.chunk_size, len(rows))
                signal = np.asarray(waveforms[rows[start:end], d, :], np.float64)
                signal = signal[:, indices]*span[d,0]+minimum[d,0]
                f, p, a, concentration = event_peak_frequency_phase(
                    signal, time_ns[indices], low_GHz=args.band_GHz[0],
                    high_GHz=args.band_GHz[1], fft_padding=args.fft_padding)
                freq[start:end], phase[start:end] = f, p
                amplitude[start:end], prominence[start:end] = a, concentration
            floor = float(np.percentile(amplitude, args.min_amplitude_percentile))
            selected = np.isfinite(freq)&np.isfinite(phase)&(amplitude>=floor)
            records, f_bin, p_bin = group_grid(
                freq[selected], phase[selected], predictions[rows][selected],
                frequency_bins=args.frequency_bins,
                phase_bins=args.phase_bins, low=args.band_GHz[0],
                high=args.band_GHz[1])
            tag = f"{name}_detector{d+1}"
            make_plots(out, tag, records, freq[selected], phase[selected],
                       args.frequency_bins, args.phase_bins, args.min_cell_events)
            for rec in records:
                all_records.append({"window":name, "detector":d+1, **rec})
            np.savez_compressed(
                out/f"{tag}_events.npz", event_id=event_ids[rows],
                saved_prediction_ps=predictions[rows], frequency_GHz=freq,
                phase_rad=phase, amplitude_mV=amplitude,
                spectral_peak_fraction=prominence, selected=selected)
            occupied = sum(r["n"]>=args.min_cell_events for r in records)
            summary["results"][tag] = {
                "frequency_resolution_GHz": true_df_GHz,
                "n_selected": int(selected.sum()), "amplitude_floor_mV": floor,
                "peak_frequency_mean_GHz": float(np.mean(freq[selected])),
                "peak_frequency_std_GHz": float(np.std(freq[selected])),
                "occupied_cells_above_threshold": occupied,
                "total_cells": len(records),
            }
            print(f"PEAK GROUP | {tag} | used={selected.sum()} "
                  f"| frequency={np.mean(freq[selected]):.3f}±"
                  f"{np.std(freq[selected]):.3f} GHz "
                  f"| df={true_df_GHz:.3f} GHz "
                  f"| populated_cells={occupied}/{len(records)}",
                  flush=True)
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
    p.add_argument("--frequency-bins", type=int, default=8)
    p.add_argument("--phase-bins", type=int, default=12)
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
