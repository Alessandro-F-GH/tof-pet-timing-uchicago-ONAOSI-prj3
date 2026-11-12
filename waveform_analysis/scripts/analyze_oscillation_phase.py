"""Measure eventwise oscillation amplitude/phase and association with MT correction.

Fixed-frequency, absolute LED-relative time reference; analyzes ONE detector
per invocation, without fitting models or changing prepared waveform cache.
A circular-linear diagnostic is a descriptive association, not evidence of
physical timing causality. Uses the same events in pre-LED and tail windows.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from waveform_analysis.data.prepared_cache import PreparedCacheLookup
from waveform_analysis.data.results import read_run, resolve_run_config
from waveform_analysis.data.view import mode_family, model_target
from waveform_analysis.engine import tshap_explainer as tshap
from waveform_analysis.scripts.analyze_waveform_fft import (
    validate_grid, remove_linear_trend)


def project_fixed_frequency(trace, time_ns, freq_GHz):
    """Windowed complex projection; phase referenced to t=0 (LED)."""
    signal = np.asarray(trace, np.float64)
    time = np.asarray(time_ns, np.float64)
    if signal.ndim != 2 or time.shape != (signal.shape[1],) or time.size < 16:
        raise ValueError("Expected [events, samples] and time grid")
    if freq_GHz <= 0 or not np.isfinite(freq_GHz):
        raise ValueError("Frequency must be positive")
    residual = remove_linear_trend(signal)
    taper = np.hanning(time.size)
    kernel = taper * np.exp(-2j * np.pi * freq_GHz * time)
    # 2/sum(Hann) converts complex projection of a real sinusoid to approximate
    # peak-to-zero physical amplitude, for windows spanning several periods.
    z = (2. / taper.sum()) * (residual @ kernel)
    amplitude = np.abs(z)
    phase = np.angle(z)
    return z, amplitude, phase


def circular_relation(phi, target):
    """OLS target ~ intercept + sin(phi) + cos(phi), no timing model training."""
    angle = np.asarray(phi, np.float64)
    y = np.asarray(target, np.float64)
    if angle.ndim != 1 or y.shape != angle.shape or angle.size < 4:
        raise ValueError("Need paired phase/target vectors")
    X = np.column_stack([np.ones(len(angle)), np.sin(angle), np.cos(angle)])
    beta, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
    fitted = X @ beta
    total = np.sum((y - y.mean()) ** 2)
    r2 = float(1 - np.sum((y - fitted) ** 2) / total) if total > 0 else None
    # The fitted sinusoid is descriptive; do not interpret R² as held-out skill.
    return {"r2_in_sample": r2, "intercept_ps": float(beta[0]),
            "sin_coefficient_ps": float(beta[1]),
            "cos_coefficient_ps": float(beta[2]),
            "sinusoid_magnitude_ps": float(np.hypot(beta[1], beta[2]))}


def phase_lock(phi_a, phi_b):
    """Unweighted within-event consistency across LED-referenced windows."""
    d = np.asarray(phi_a) - np.asarray(phi_b)
    resultant = np.mean(np.exp(1j * d))
    return {"resultant_length": float(np.abs(resultant)),
            "mean_phase_difference_rad": float(np.angle(resultant)),
            "n": int(len(d))}


def read_mt(path, dataset, mode, role):
    with np.load(path, allow_pickle=False) as f:
        required = {"event_id", "prediction_ps", "valid",
                    "led_residual_ps", "dataset_role", "method"}
        if absent := required - set(f.files):
            raise ValueError(f"Saved MT missing {sorted(absent)}")
        if str(f["dataset_role"].item()) != role:
            raise ValueError("MT dataset_role is incompatible with waveform role")
        if str(f["method"].item()) != "prepared_multithreshold_led":
            raise ValueError("Expected saved multi-threshold predictions")
        ids = np.asarray(f["event_id"], np.int64)
        pred = np.asarray(f["prediction_ps"], np.float64)
        valid = np.asarray(f["valid"], bool)
        residual = np.asarray(f["led_residual_ps"], np.float64)
    reference = np.asarray(dataset.event_index, np.int64)
    if (ids.ndim != 1 or ids.shape != pred.shape or valid.shape != ids.shape
            or residual.shape != ids.shape or np.unique(ids).size != ids.size
            or np.unique(reference).size != reference.size):
        raise ValueError("Invalid or duplicate event IDs / MT arrays")
    common, left, right = np.intersect1d(reference, ids, assume_unique=True,
                                        return_indices=True)
    target = np.asarray(model_target(dataset, mode), np.float64)
    if not np.allclose(target[left], residual[right], atol=1e-3, rtol=0):
        raise ValueError("Saved MT and prepared LED targets disagree for common IDs")
    valid_rows = valid[right] & np.isfinite(pred[right])
    if valid_rows.sum() < 10:
        raise ValueError("Insufficient valid MT events")
    return left[valid_rows], pred[right[valid_rows]], common[valid_rows]


def write_graphs(out, name, phi_pre, phi_tail, amp_pre, amp_tail, target, usable):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    ax[0, 0].hist(phi_pre, bins=48)
    ax[0, 0].set(xlabel="Pre-LED phase (rad)", ylabel="Events",
                 title="Pre-LED phase distribution")
    ax[0, 1].hist(phi_tail, bins=48)
    ax[0, 1].set(xlabel="Tail phase (rad)", ylabel="Events",
                 title="Tail phase distribution")
    ax[1, 0].scatter(phi_pre[usable], target[usable], alpha=.07, s=3, rasterized=True)
    ax[1, 0].set(xlabel="Pre-LED phase (rad)", ylabel="MT correction (ps)",
                 title="Eventwise phase vs MT correction")
    ax[1, 1].scatter(phi_tail[usable], target[usable], alpha=.07, s=3,
                     rasterized=True)
    ax[1, 1].set(xlabel="Tail phase (rad)", ylabel="MT correction (ps)",
                 title="Eventwise tail phase vs MT correction")
    for a in ax.flat:
        a.grid(alpha=.15)
    fig.savefig(out / f"{name}_phase.png", dpi=160)
    fig.savefig(out / f"{name}_phase.pdf")
    plt.close(fig)


def run(args):
    if args.cache_root is not None and args.artifact_root is not None:
        raise ValueError("Use --cache-root or --artifact-root, not both")
    if args.chunk_size < 1 or args.max_events < 10:
        raise ValueError("chunk-size must be positive, max-events >=10")
    if not 0 <= args.min_amplitude_percentile < 100:
        raise ValueError("min-amplitude-percentile must be in [0,100)")
    run_dir = tshap._resolve_model_run_directory(args.model_dir)
    config = resolve_run_config(run_dir)
    manifest = read_run(run_dir)["manifest"]
    tshap._apply_tshap_cache(config, cache_root=args.cache_root,
                            artifact_root=args.artifact_root)
    dataset = PreparedCacheLookup().find(args.role, config, manifest)
    family = mode_family(config["mode"])
    t_ps, dt_ps = validate_grid(getattr(dataset, f"{family}_time_ps"))
    waves = getattr(dataset, f"{family}_windows")
    transform = getattr(dataset, f"{family}_transform")
    if waves is None or transform is None or waves.shape != (
            dataset.n_events, 2, len(t_ps)):
        raise ValueError("Waveform grid or transform missing")
    if not 0 < args.frequency_GHz < 500. / dt_ps:
        raise ValueError("Frequency must be in (0, Nyquist)")
    mn = np.asarray(transform.minimum, np.float64).reshape(2, -1)
    mx = np.asarray(transform.maximum, np.float64).reshape(2, -1)
    if mn.shape != (2, 1) or mx.shape != (2, 1):
        raise ValueError("Unexpected prepared affine transform")
    d = args.detector - 1
    physical_min = float(mn[d, 0])
    scale = float(mx[d, 0] - mn[d, 0])
    if scale <= 0:
        raise ValueError("Invalid transform scale")
    selected, mt, event_ids = read_mt(
        args.mt_predictions_npz, dataset, config["mode"], args.role)
    if selected.size > args.max_events:
        rng = np.random.default_rng(args.seed)
        keep = np.sort(rng.choice(selected.size, args.max_events, replace=False))
        selected, mt, event_ids = selected[keep], mt[keep], event_ids[keep]
    full_time = t_ps / 1000.
    windows = {"pre": tuple(args.pre_window_ns),
               "tail": tuple(args.tail_window_ns)}
    indices = {}
    for name, (lo, hi) in windows.items():
        if lo >= hi or lo < full_time[0] or hi > full_time[-1]:
            raise ValueError(f"Window {name} not fully covered by prepared Wide grid")
        idx = np.flatnonzero((full_time >= lo - 1e-9)
                             & (full_time <= hi + 1e-9))
        if idx.size < 16 or args.frequency_GHz * (hi-lo) < 3:
            raise ValueError(f"Window {name} too short for reliable phase at "
                             f"{args.frequency_GHz} GHz")
        indices[name] = idx
    if args.highpass_GHz is not None:
        from scipy.signal import butter
        if not 0 < args.highpass_GHz < 500. / dt_ps:
            raise ValueError("High-pass cutoff must be in (0, Nyquist)")
        sos = butter(args.highpass_order, args.highpass_GHz, fs=1000. / dt_ps,
                     btype="highpass", output="sos")
    else:
        sos = None
    out = args.output_root.expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    data = {key: np.empty(len(selected), np.float64)
            for key in ("pre_amplitude_mV", "pre_phase_rad",
                        "tail_amplitude_mV", "tail_phase_rad")}
    for start in range(0, len(selected), args.chunk_size):
        end = min(len(selected), start + args.chunk_size)
        batch = np.asarray(waves[selected[start:end], d, :], np.float64)
        batch = batch * scale + physical_min
        if sos is not None:
            from scipy.signal import sosfiltfilt
            # Noncausal; pre-LED amplitude/phase may include impulse ringing.
            batch = sosfiltfilt(sos, batch, axis=-1)
        for name, idx in indices.items():
            _, amp, phase = project_fixed_frequency(
                batch[:, idx], full_time[idx], args.frequency_GHz)
            data[f"{name}_amplitude_mV"][start:end] = amp
            data[f"{name}_phase_rad"][start:end] = phase
    # Reject small-amplitude phase estimates using pre-registered percentile,
    # independent of the timing target. Report sensitivity to this filter.
    amp_pre, amp_tail = data["pre_amplitude_mV"], data["tail_amplitude_mV"]
    cut_pre = float(np.percentile(amp_pre, args.min_amplitude_percentile))
    cut_tail = float(np.percentile(amp_tail, args.min_amplitude_percentile))
    usable = (np.isfinite(amp_pre) & np.isfinite(amp_tail)
              & np.isfinite(data["pre_phase_rad"])
              & np.isfinite(data["tail_phase_rad"])
              & (amp_pre >= cut_pre) & (amp_tail >= cut_tail))
    if usable.sum() < 10:
        raise ValueError("Too few events with usable phase in both windows")
    phase_pre = data["pre_phase_rad"][usable]
    phase_tail = data["tail_phase_rad"][usable]
    mt_valid = mt[usable]
    lock = phase_lock(phase_pre, phase_tail)
    summary = {
        "run": str(run_dir), "mode": config["mode"], "role": args.role,
        "detector": args.detector, "frequency_GHz": args.frequency_GHz,
        "pre_window_ns": list(map(float, args.pre_window_ns)),
        "tail_window_ns": list(map(float, args.tail_window_ns)),
        "highpass_GHz": args.highpass_GHz,
        "phase_reference": "absolute LED-relative time, t=0 at LED",
        "detrend": "per-event linear detrend inside each window + Hann taper",
        "selection": {"matched_mt_events": int(len(selected)),
                      "usable_phase_events": int(usable.sum()),
                      "minimum_amplitude_percentile": args.min_amplitude_percentile,
                      "pre_amplitude_threshold_mV": cut_pre,
                      "tail_amplitude_threshold_mV": cut_tail},
        "pre_MT_association": circular_relation(phase_pre, mt_valid),
        "tail_MT_association": circular_relation(phase_tail, mt_valid),
        "pre_tail_phase_difference": lock,
        "warning": ("In-sample sinusoid R² and phase locking describe associations "
                    "only; they do not establish causality or predictive value. "
                    "Zero-phase high-pass filtering may induce pre-LED ringing."),
    }
    np.savez_compressed(out / "eventwise_phase.npz",
                        event_id=event_ids, mt_prediction_ps=mt, valid_phase=usable,
                        detector=np.asarray(args.detector),
                        frequency_GHz=np.asarray(args.frequency_GHz), **data)
    (out / "summary.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False), encoding="utf-8")
    with (out / "eventwise_phase.csv").open("w", newline="", encoding="utf-8") as f:
        fields = ["event_id", "mt_prediction_ps", "valid_phase"] + list(data)
        writer = csv.writer(f)
        writer.writerow(fields)
        for i, event in enumerate(event_ids):
            writer.writerow([int(event), float(mt[i]), bool(usable[i])] +
                            [float(data[k][i]) for k in data])
    write_graphs(out, "pre_tail", data["pre_phase_rad"],
                 data["tail_phase_rad"], amp_pre, amp_tail, mt, usable)
    print(f"PHASE | detector={args.detector} | f={args.frequency_GHz:.4g} GHz "
          f"| matched={len(selected)} usable={usable.sum()} "
          f"| phase locking={lock['resultant_length']:.4f} "
          f"| pre/MT R2={summary['pre_MT_association']['r2_in_sample']:.4f} "
          f"| tail/MT R2={summary['tail_MT_association']['r2_in_sample']:.4f}",
          flush=True)
    return summary


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model-dir", type=Path, required=True)
    p.add_argument("--mt-predictions-npz", type=Path, required=True)
    p.add_argument("--cache-root", type=Path)
    p.add_argument("--artifact-root", type=Path)
    p.add_argument("--output-root", type=Path, required=True)
    p.add_argument("--role", choices=("development", "blind"), default="development")
    p.add_argument("--detector", type=int, choices=(1, 2), default=1)
    p.add_argument("--frequency-GHz", type=float, default=2.5)
    p.add_argument("--pre-window-ns", nargs=2, type=float, default=(-2., 0.))
    p.add_argument("--tail-window-ns", nargs=2, type=float, default=(12., 45.))
    p.add_argument("--highpass-GHz", type=float, default=None,
                   help="Optional noncausal Butterworth filter, comparison only")
    p.add_argument("--highpass-order", type=int, default=4)
    p.add_argument("--min-amplitude-percentile", type=float, default=20.)
    p.add_argument("--max-events", type=int, default=10000)
    p.add_argument("--chunk-size", type=int, default=128)
    p.add_argument("--seed", type=int, default=42)
    return p


if __name__ == "__main__":
    run(parser().parse_args())
