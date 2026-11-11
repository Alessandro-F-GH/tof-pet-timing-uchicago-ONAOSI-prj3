"""FFT of individual and LED-aligned aggregate prepared PET waveforms.

Single detector per invocation (never subtract or mix detector channels).
Compare mean individual power spectra, spectrum of the averaged trace, and
frequency-wise phase coherence. Analyze pre-LED, tail and full waveform in
separate windows to avoid mistaking the scintillation envelope for ringing.
No ROOT input, ML inference, refitting or stochastic imputation.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from waveform_analysis.data.prepared_cache import PreparedCacheLookup
from waveform_analysis.data.results import read_run, resolve_run_config
from waveform_analysis.data.view import mode_family
from waveform_analysis.engine import tshap_explainer as tshap


def validate_grid(t_ps):
    grid = np.asarray(t_ps, np.float64)
    if grid.ndim != 1 or grid.size < 4 or not np.isfinite(grid).all():
        raise ValueError("Invalid time grid")
    diffs = np.diff(grid)
    dt = float(np.median(diffs))
    if dt <= 0 or not np.allclose(diffs, dt, rtol=1.e-4, atol=.001):
        raise ValueError("FFT requires uniformly spaced prepared samples")
    return grid, dt


def remove_linear_trend(x):
    """Remove each event's least-squares linear trend (not waveform envelope)."""
    n = x.shape[-1]
    xcoord = np.linspace(-1., 1., n, dtype=np.float64)
    center = x.mean(axis=-1, keepdims=True)
    slope = (np.sum((x - center) * xcoord, axis=-1, keepdims=True)
             / np.sum(xcoord * xcoord))
    return x - center - slope * xcoord


def spectral_accumulators(waves, t_ps, *, selected_rows, detector,
                          window, max_individual, chunk_size,
                          physical_min, physical_span):
    """Welch-like single-segment averaged periodograms and coherent mean FFT.

    Coherent power = abs(mean(complex FFT))^2; incoherent power =
    mean(abs(complex FFT)^2). Their ratio is in [0,1] and measures
    phase-consistency in LED-relative coordinates, not physiological causality.
    """
    start_ns, end_ns = map(float, window)
    all_idx = np.flatnonzero((t_ps >= 1000 * start_ns - 1e-6)
                             & (t_ps <= 1000 * end_ns + 1e-6))
    if all_idx.size < 16:
        raise ValueError(f"Window {window} ns has <16 samples")
    if max_individual < 1:
        raise ValueError("Need at least one individual trace")
    dt_ns = float(np.median(np.diff(t_ps))) / 1000.
    freqs = np.fft.rfftfreq(all_idx.size, d=dt_ns)  # GHz (1/ns)
    taper = np.hanning(all_idx.size)
    taper_norm = np.sqrt(np.mean(taper * taper))
    if taper_norm <= 0:
        raise ValueError("Invalid Hann taper")
    sum_fft = np.zeros(freqs.shape, dtype=np.complex128)
    sum_power = np.zeros(freqs.shape, dtype=np.float64)
    sum_trace = np.zeros(all_idx.size, dtype=np.float64)
    snapshots = []
    count = 0
    for start in range(0, len(selected_rows), chunk_size):
        ids = selected_rows[start:start + chunk_size]
        # Always select EXACTLY ONE detector; no cross-channel aggregation.
        data = np.asarray(waves[ids, detector, :], dtype=np.float64)
        data = data[:, all_idx] * physical_span + physical_min
        if not np.isfinite(data).all():
            raise ValueError("Nonfinite waveform samples")
        # Save the physical mean trace before detrending, for visual inspection.
        sum_trace += np.sum(data, axis=0)
        residual = remove_linear_trend(data)
        fft = np.fft.rfft(residual * taper[None, :] / taper_norm, axis=-1)
        sum_fft += fft.sum(axis=0)
        sum_power += np.sum(np.abs(fft) ** 2, axis=0)
        if len(snapshots) < max_individual:
            snapshots.extend(data[:max_individual-len(snapshots)])
        count += len(ids)
    if count < 2:
        raise ValueError("At least two events required")
    mean_complex = sum_fft / count
    mean_individual_power = sum_power / count
    coherent_power = np.abs(mean_complex) ** 2
    coherence = np.clip(coherent_power / np.maximum(mean_individual_power, 1e-30),
                        0, 1)
    return {
        "frequency_GHz": freqs,
        "mean_individual_power": mean_individual_power,
        "coherent_power": coherent_power,
        "phase_consistency": coherence,
        "mean_waveform_mV": sum_trace / count,
        "time_ns": t_ps[all_idx] / 1000.,
        "individual_waveforms_mV": np.asarray(snapshots),
        "n_events": count,
        "window_ns": (start_ns, end_ns),
        "df_GHz": float(freqs[1] - freqs[0]),
        "nyquist_GHz": float(freqs[-1]),
        "normalization": ("Hann taper, eventwise linear detrend, FFT squared "
                          "magnitudes in common arbitrary scale; compare "
                          "coherent/incoherent spectra within the same window."),
    }


def write_plots(output, label, result, *, max_frequency_GHz=8.):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    f = result["frequency_GHz"]
    freq_mask = (f > 0) & (f <= max_frequency_GHz)
    t = result["time_ns"]
    mean = result["mean_waveform_mV"]
    individual = result["individual_waveforms_mV"]
    fig, axs = plt.subplots(3, 1, figsize=(11, 10), constrained_layout=True)
    for row in individual[:min(10, len(individual))]:
        axs[0].plot(t, row, linewidth=.6, alpha=.25)
    axs[0].plot(t, mean, linewidth=2, label=f"mean ({result['n_events']} events)")
    axs[0].set_ylabel("Amplitude (mV)")
    axs[0].set_title(f"{label} | single-event traces and LED-aligned average")
    axs[0].legend()
    axs[0].grid(alpha=.2)

    incoherent = result["mean_individual_power"]
    coherent = result["coherent_power"]
    axs[1].plot(f[freq_mask], incoherent[freq_mask],
                label="mean individual FFT power")
    axs[1].plot(f[freq_mask], coherent[freq_mask],
                label="FFT power of mean waveform")
    axs[1].set_yscale("log")
    axs[1].set_ylabel("Power (a.u., log)")
    axs[1].legend()
    axs[1].grid(alpha=.2)
    axs[2].plot(f[freq_mask], result["phase_consistency"][freq_mask])
    axs[2].set_ylim(0, 1.02)
    axs[2].set_xlabel("Frequency (GHz)")
    axs[2].set_ylabel("Phase consistency")
    axs[2].grid(alpha=.2)
    fig.savefig(output / f"{label}_fft.png", dpi=175)
    fig.savefig(output / f"{label}_fft.pdf")
    plt.close(fig)


def run(args):
    if args.cache_root and args.artifact_root:
        raise ValueError("Use --cache-root or --artifact-root, not both")
    if args.chunk_size < 1 or args.max_events < 2 or args.max_individual < 1:
        raise ValueError("Invalid event/chunk counts")
    run_dir = tshap._resolve_model_run_directory(args.model_dir)
    config = resolve_run_config(run_dir)
    manifest = read_run(run_dir)["manifest"]
    tshap._apply_tshap_cache(config, cache_root=args.cache_root,
                            artifact_root=args.artifact_root)
    dataset = PreparedCacheLookup().find(args.role, config, manifest)
    family = mode_family(config["mode"])
    waves = getattr(dataset, f"{family}_windows")
    t_ps, dt_ps = validate_grid(getattr(dataset, f"{family}_time_ps"))
    transform = getattr(dataset, f"{family}_transform")
    if waves is None or transform is None or waves.shape != (
            dataset.n_events, 2, len(t_ps)):
        raise ValueError("Paired prepared waveforms or saved transform missing")
    physical_minimum = np.asarray(transform.minimum, np.float64).reshape(2, -1)
    physical_maximum = np.asarray(transform.maximum, np.float64).reshape(2, -1)
    if physical_minimum.shape != (2, 1) or physical_maximum.shape != (2, 1):
        raise ValueError("Unsupported physical amplitude transform")
    detector = args.detector - 1
    minval = float(physical_minimum[detector, 0])
    span = float(physical_maximum[detector, 0] - minval)
    if span <= 0:
        raise ValueError("Invalid physical scale")

    total = dataset.n_events
    rng = np.random.default_rng(args.seed)
    rows = np.sort(rng.choice(total, min(total, args.max_events), replace=False))
    event_ids = np.asarray(dataset.event_index, np.int64)[rows]
    intervals = {"pre_LED": args.pre_window_ns,
                 "tail": args.tail_window_ns}
    if args.include_full:
        intervals["full"] = (float(t_ps[0] / 1000.),
                             float(t_ps[-1] / 1000.))
    out = args.output_root.expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    overview = {
        "run": str(run_dir), "dataset": str(dataset.directory),
        "mode": config["mode"], "role": args.role,
        "detector": args.detector, "detector_mixing": False,
        "event_count": int(len(rows)), "sampling_interval_ps": dt_ps,
        "sample_rate_GHz": 1000. / dt_ps,
        "note": ("Frequencies use LED-aligned prepared signals. Phase "
                 "consistency after LED alignment does not by itself establish "
                 "an anticipatory physical response. Full-window FFT may be "
                 "dominated by the main pulse envelope."),
        "windows": {},
    }
    np.save(out / "selected_event_ids.npy", event_ids)
    print(f"FFT | detector={args.detector} | role={args.role} "
          f"| events={len(rows)} | dt={dt_ps:.3f} ps", flush=True)
    for name, interval in intervals.items():
        result = spectral_accumulators(
            waves, t_ps, selected_rows=rows, detector=detector,
            window=interval, max_individual=args.max_individual,
            chunk_size=args.chunk_size, physical_min=minval,
            physical_span=span)
        np.savez_compressed(out / f"{name}_fft.npz", **{
            key: val for key, val in result.items()
            if key not in ("normalization", "window_ns")})
        write_plots(out, name, result, max_frequency_GHz=args.max_frequency_GHz)
        m = result["frequency_GHz"]
        incoherent = result["mean_individual_power"]
        if not np.any((m > 0) & (m <= args.max_frequency_GHz)):
            raise ValueError("No positive frequencies in requested graph range")
        pick = np.flatnonzero((m > 0) & (m <= args.max_frequency_GHz))
        peak = pick[int(np.argmax(incoherent[pick]))]
        overview["windows"][name] = {
            "bounds_ns": [float(v) for v in interval],
            "samples": len(result["time_ns"]),
            "frequency_resolution_GHz": result["df_GHz"],
            "nyquist_GHz": result["nyquist_GHz"],
            "largest_individual_power_peak_GHz": float(m[peak]),
            "phase_consistency_at_peak": float(result["phase_consistency"][peak]),
            "warning": ("A spectral peak is descriptive; harmonic content, "
                        "filtering, and window leakage need independent verification."),
        }
        print(f"WINDOW | {name} | df={result['df_GHz']:.4f} GHz "
              f"| peak={m[peak]:.4f} GHz "
              f"| coherence={result['phase_consistency'][peak]:.4f}", flush=True)
    (out / "summary.json").write_text(
        json.dumps(overview, indent=2, allow_nan=False), encoding="utf-8")
    return overview


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model-dir", type=Path, required=True,
                   help="Wide ML run providing prepared waveform cache and time grid")
    p.add_argument("--cache-root", type=Path)
    p.add_argument("--artifact-root", type=Path)
    p.add_argument("--output-root", type=Path, required=True)
    p.add_argument("--role", choices=("development", "blind"), default="development")
    p.add_argument("--detector", type=int, choices=(1, 2), default=1,
                   help="Analyze only this detector, never combine detector spectra")
    p.add_argument("--max-events", type=int, default=10000)
    p.add_argument("--max-individual", type=int, default=15,
                   help="Number of individual traces shown/saved")
    p.add_argument("--pre-window-ns", nargs=2, type=float, default=(-2., 0.))
    p.add_argument("--tail-window-ns", nargs=2, type=float, default=(12., 45.))
    p.add_argument("--include-full", action="store_true",
                   help="Also analyze whole waveform; can be dominated by pulse envelope")
    p.add_argument("--max-frequency-GHz", type=float, default=8.)
    p.add_argument("--chunk-size", type=int, default=128)
    p.add_argument("--seed", type=int, default=42)
    return p


if __name__ == "__main__":
    run(parser().parse_args())
