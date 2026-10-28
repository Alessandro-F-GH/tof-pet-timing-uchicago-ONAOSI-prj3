#!/usr/bin/env python3
"""Iterative Singular Spectrum Analysis (SSA) imputation for wide waveforms.

For each held-out waveform and mask position, this performs iterative SSA gap
filling: embed the current complete series into a Hankel trajectory matrix,
compute a truncated SVD, reconstruct by diagonal averaging, then update only
the missing interval while keeping observed samples fixed.

This is a time-series-specific, per-waveform method: lagged windows preserve
temporal order and encode local serial dependence. It does not train on the
hidden targets of other events. An optional PPCA baseline can be loaded from a
previous run's model.npz. No TSHAP integration.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if __package__ in (None, "") and str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from sklearn.utils.extmath import randomized_svd

from waveform_analysis.scripts.train_wide_unet_imputer_detector_specific import (
    load_wide_development, root_relative,
)


def make_positions(length, width, stride):
    if not 0 < width < length or stride < 1:
        raise ValueError("Require 0 < mask width < waveform length and positive stride")
    starts = list(range(0, length - width + 1, stride))
    if starts[-1] != length - width:
        starts.append(length - width)
    return starts


def linear_fill(signal, start, stop):
    """Initialize a gap using only observed boundary samples."""
    length = len(signal)
    if start == 0:
        return np.full(stop - start, signal[stop], dtype=np.float64)
    if stop == length:
        return np.full(stop - start, signal[start - 1], dtype=np.float64)
    return np.interp(np.arange(start, stop),
                     np.array([start - 1, stop]),
                     np.array([signal[start - 1], signal[stop]]))


def hankelize(trajectory):
    """Diagonal-average a trajectory matrix back to a 1-D series."""
    rows, cols = trajectory.shape
    indices = np.add.outer(np.arange(rows), np.arange(cols)).ravel()
    sums = np.bincount(indices, weights=trajectory.ravel(), minlength=rows + cols - 1)
    counts = np.bincount(indices, minlength=rows + cols - 1)
    return sums / np.maximum(counts, 1)


def ssa_impute(signal, start, width, window_length, rank, max_iter, tolerance,
               damping, seed):
    """Iterative SSA imputation; returns reconstructed segment and diagnostics."""
    original = np.asarray(signal, dtype=np.float64)
    length = len(original)
    stop = start + width
    if not 0 <= start < stop <= length:
        raise ValueError("Invalid masked interval")
    if not 2 <= window_length <= length // 2:
        raise ValueError("window length must be in [2, waveform_length // 2]")
    if not 1 <= rank < window_length:
        raise ValueError("SSA rank must be >= 1 and smaller than window length")

    current = original.copy()
    current[start:stop] = linear_fill(original, start, stop)
    missing = np.arange(start, stop)
    relative_change = float("inf")
    iterations = 0

    for iteration in range(max_iter):
        trajectory = sliding_window_view(current, window_length).T
        u, singular_values, vt = randomized_svd(
            trajectory, n_components=rank, n_iter=2, random_state=seed + iteration
        )
        low_rank = (u * singular_values[None, :]) @ vt
        reconstructed = hankelize(low_rank)
        candidate = reconstructed[missing]
        updated = current[missing] + damping * (candidate - current[missing])
        relative_change = float(
            np.linalg.norm(updated - current[missing]) /
            (np.linalg.norm(current[missing]) + 1e-12)
        )
        current[missing] = updated
        iterations = iteration + 1
        if relative_change < tolerance:
            break

    return current[start:stop].copy(), iterations, relative_change


def interpolate_batch(signals, start, stop):
    width = stop - start
    length = signals.shape[1]
    if start == 0:
        return np.repeat(signals[:, stop:stop + 1], width, axis=1)
    if stop == length:
        return np.repeat(signals[:, start - 1:start], width, axis=1)
    alpha = (np.arange(start, stop) - (start - 1)) / (width + 1.0)
    return signals[:, start - 1:start] * (1.0 - alpha[None, :]) + signals[:, stop:stop + 1] * alpha[None, :]


class PPCABaseline:
    """Conditional mean from a saved PPCA model, with per-mask factors cached."""
    def __init__(self, path, length):
        with np.load(path) as saved:
            self.mean = np.asarray(saved["mean"], dtype=np.float64)
            self.loading = np.asarray(saved["loading"], dtype=np.float64)
            self.noise_variance = np.asarray(saved["noise_variance"], dtype=np.float64)
        if self.mean.shape != (2, length) or self.loading.shape[:2] != (2, length):
            raise ValueError("PPCA model dimensions do not match the waveform data")
        self.cache = {}

    def predict(self, detector, signals, start, width):
        key = (detector, start, width)
        if key not in self.cache:
            mean = self.mean[detector]
            loading = self.loading[detector]
            sigma2 = max(float(self.noise_variance[detector]), 1e-12)
            stop = start + width
            observed = np.ones(len(mean), dtype=bool)
            observed[start:stop] = False
            wo, wm = loading[observed], loading[start:stop]
            precision = np.eye(loading.shape[1]) + wo.T @ wo / sigma2
            factor = np.linalg.cholesky(precision)
            coefficient_map = np.linalg.solve(
                factor.T, np.linalg.solve(factor, wo.T / sigma2)
            ).T @ wm.T
            self.cache[key] = (observed, mean, coefficient_map)
        observed, mean, coefficient_map = self.cache[key]
        return mean[start:start + width] + (
            np.asarray(signals[:, observed], dtype=np.float64) - mean[observed]
        ) @ coefficient_map


def plot_metrics(rows, destination):
    fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
    for detector in (1, 2):
        ax = axes[detector - 1]
        subset = [r for r in rows if r["detector"] == detector]
        x = [r["start_ns"] for r in subset]
        series = [
            ("ssa_rmse", "Iterative SSA"),
            ("interpolation_rmse", "Linear interpolation"),
            ("ppca_rmse", "PPCA conditional mean"),
        ]
        for key, label in series:
            if subset and subset[0].get(key) is not None:
                ax.plot(x, [r[key] for r in subset], label=label)
        ax.set_ylabel(f"Detector {detector} RMSE")
        ax.grid(alpha=0.25)
        ax.legend()
    axes[-1].set_xlabel("Masked region start (ns)")
    fig.tight_layout()
    fig.savefig(destination / "rmse_by_position.png", dpi=180)
    fig.savefig(destination / "rmse_by_position.pdf")
    plt.close(fig)


def plot_examples(waves, eval_ids, starts, width, time_ps, destination, args,
                  ppca, seed):
    if args.plot_examples <= 0:
        return
    rng = np.random.default_rng(seed + 1009)
    del rng
    times = np.asarray(time_ps, dtype=np.float64) / 1000.0
    targets = list(dict.fromkeys(
        min(range(len(starts)), key=lambda i: abs(times[starts[i]] - target))
        for target in (0.0, 20.0, 40.0)
    ))
    folder = destination / "imputation_examples"
    folder.mkdir(parents=True, exist_ok=True)
    for detector in range(2):
        for position_idx in targets:
            start = starts[position_idx]
            stop = start + width
            for event_id in eval_ids[:args.plot_examples]:
                signal = np.asarray(waves[event_id, detector], dtype=np.float64)
                prediction, iterations, change = ssa_impute(
                    signal, start, width, args.window_length, args.rank,
                    args.max_iter, args.tolerance, args.damping, args.seed + int(event_id)
                )
                interp = linear_fill(signal, start, stop)
                fig, axes = plt.subplots(2, 1, figsize=(11, 7))
                masked = signal.copy()
                masked[start:stop] = np.nan
                for ax, zoom in zip(axes, (False, True)):
                    left = max(0, start - 2 * width) if zoom else 0
                    right = min(len(signal), stop + 2 * width) if zoom else len(signal)
                    sl = slice(left, right)
                    ax.plot(times[sl], signal[sl], color="black", lw=1.2, label="Ground truth")
                    ax.plot(times[sl], masked[sl], color="gray", ls="--", lw=1, label="Observed")
                    rec = signal.copy()
                    rec[start:stop] = prediction
                    lin = signal.copy()
                    lin[start:stop] = interp
                    ax.plot(times[sl], rec[sl], lw=1.3, label="SSA imputation")
                    ax.plot(times[sl], lin[sl], lw=1.0, alpha=.8, label="Interpolation")
                    if ppca is not None:
                        pp = signal.copy()
                        pp[start:stop] = ppca.predict(
                            detector, signal[None, :], start, width
                        )[0]
                        ax.plot(times[sl], pp[sl], lw=1.1, label="PPCA conditional mean")
                    ax.axvspan(times[start], times[stop - 1], color="tab:red", alpha=.08)
                    ax.set_ylabel("Normalized signal")
                    ax.grid(alpha=.2)
                axes[0].set_title(
                    f"Detector {detector + 1}, event {event_id}, mask {times[start]:.2f} ns; "
                    f"SSA iterations={iterations}, final change={change:.2g}"
                )
                axes[0].legend(ncol=3, fontsize=8)
                axes[1].set_xlabel("Time (ns)")
                fig.tight_layout()
                fig.savefig(
                    folder / f"detector_{detector+1}_event_{event_id}_position_{position_idx:02d}.png",
                    dpi=160
                )
                plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--channel", choices=("energy", "timing"), required=True)
    parser.add_argument("--cache-root", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--max-events", type=int, default=1000)
    parser.add_argument("--eval-events", type=int, default=150)
    parser.add_argument("--mask-ns", type=float, default=1.0)
    parser.add_argument("--mask-stride-samples", type=int)
    parser.add_argument("--window-length", type=int, default=80)
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--max-iter", type=int, default=10)
    parser.add_argument("--tolerance", type=float, default=1e-3)
    parser.add_argument("--damping", type=float, default=0.8)
    parser.add_argument("--ppca-model", type=Path,
                        help="Optional saved PPCA model.npz for comparison")
    parser.add_argument("--plot-examples", type=int, default=3)
    parser.add_argument("--eval-batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val-fraction", type=float, default=0.15)
    args = parser.parse_args()
    if (args.max_events < 10 or args.eval_events < 1 or args.mask_ns <= 0
        or args.window_length < 2 or args.rank < 1 or args.rank >= args.window_length
        or args.max_iter < 1 or args.tolerance <= 0 or not 0 < args.damping <= 1
        or args.plot_examples < 0 or args.eval_batch_size < 1
        or not 0 < args.val_fraction < 1
        or (args.mask_stride_samples is not None and args.mask_stride_samples < 1)):
        parser.error("Invalid parameter values")

    results = root_relative(args.results)
    waves, time_ps, source, manifest = load_wide_development(
        results, args.channel, root_relative(args.cache_root)
    )
    length = waves.shape[-1]
    if args.window_length > length // 2:
        parser.error("--window-length must not exceed half the waveform length")
    dt_ps = float(np.median(np.diff(time_ps)))
    if dt_ps <= 0 or not np.allclose(np.diff(time_ps), dt_ps, rtol=1e-4, atol=1e-3):
        raise ValueError("Expected uniformly sampled waveform")
    width = max(1, round(args.mask_ns * 1000.0 / dt_ps))
    starts = make_positions(length, width, args.mask_stride_samples or width)

    selected = np.random.default_rng(args.seed).permutation(len(waves))[:args.max_events]
    n_val = max(1, round(len(selected) * args.val_fraction))
    eval_ids = selected[:min(n_val, args.eval_events)]
    if len(eval_ids) == 0:
        parser.error("No evaluation events selected")

    destination = (root_relative(args.output) if args.output else
                   ROOT / "waveform_analysis/imputation_experiments" /
                   results.name / args.channel / "ssa_imputer")
    destination.mkdir(parents=True, exist_ok=True)

    ppca = None
    if args.ppca_model is not None:
        ppca_path = root_relative(args.ppca_model)
        if not ppca_path.is_file():
            parser.error(f"PPCA model not found: {ppca_path}")
        ppca = PPCABaseline(ppca_path, length)
        print(f"Loaded PPCA baseline: {ppca_path}", flush=True)

    print(f"Iterative SSA: selected={len(selected)}, reserved validation={n_val}, "
          f"evaluated={len(eval_ids)}, waveform={length}, mask={width}, "
          f"positions={len(starts)}, window={args.window_length}, rank={args.rank}, "
          f"max_iter={args.max_iter}", flush=True)

    rows = []
    for detector in range(2):
        print(f"Detector {detector + 1}: evaluating", flush=True)
        for position_idx, start in enumerate(starts):
            stop = start + width
            sse_ssa = 0.0
            sse_interp = 0.0
            sse_ppca = 0.0
            slope_sse = 0.0
            slope_count = 0
            total_samples = 0
            iteration_sum = 0
            convergence_sum = 0.0
            started = time.perf_counter()
            for offset in range(0, len(eval_ids), args.eval_batch_size):
                batch_ids = eval_ids[offset:offset + args.eval_batch_size]
                batch = np.asarray(waves[batch_ids, detector, :], dtype=np.float64)
                truth = batch[:, start:stop]
                interp = interpolate_batch(batch, start, stop)
                for j, signal in enumerate(batch):
                    pred, iterations, change = ssa_impute(
                        signal, start, width, args.window_length, args.rank,
                        args.max_iter, args.tolerance, args.damping,
                        args.seed + int(batch_ids[j])
                    )
                    sse_ssa += float(np.square(pred - truth[j]).sum())
                    iteration_sum += iterations
                    convergence_sum += change
                    if start > 0:
                        true_left = signal[start] - signal[start - 1]
                        pred_left = pred[0] - signal[start - 1]
                        slope_sse += float((pred_left - true_left) ** 2)
                        slope_count += 1
                    if stop < length:
                        true_right = signal[stop] - signal[stop - 1]
                        pred_right = signal[stop] - pred[-1]
                        slope_sse += float((pred_right - true_right) ** 2)
                        slope_count += 1
                sse_interp += float(np.square(interp - truth).sum())
                if ppca is not None:
                    pp = ppca.predict(detector, batch, start, width)
                    sse_ppca += float(np.square(pp - truth).sum())
                total_samples += truth.size
            elapsed = time.perf_counter() - started
            row = dict(
                detector=detector + 1, mask_position=position_idx,
                start_sample=start, start_ns=float(time_ps[start] / 1000.0),
                evaluation_events=len(eval_ids), window_length=args.window_length,
                rank=args.rank, max_iter=args.max_iter,
                ssa_rmse=float(np.sqrt(sse_ssa / total_samples)),
                interpolation_rmse=float(np.sqrt(sse_interp / total_samples)),
                ppca_rmse=float(np.sqrt(sse_ppca / total_samples)) if ppca is not None else None,
                boundary_slope_rmse=float(np.sqrt(slope_sse / slope_count)) if slope_count else None,
                mean_iterations=float(iteration_sum / len(eval_ids)),
                mean_final_relative_change=float(convergence_sum / len(eval_ids)),
                position_seconds=float(elapsed),
            )
            rows.append(row)
            if position_idx % 5 == 0 or position_idx == len(starts) - 1:
                print(
                    f"Detector {detector + 1}: position {position_idx+1}/{len(starts)} "
                    f"at {row['start_ns']:.2f} ns; SSA RMSE={row['ssa_rmse']:.6g}; "
                    f"time={elapsed:.1f}s", flush=True
                )

    with (destination / "position_metrics.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    plot_metrics(rows, destination)
    plot_examples(waves, eval_ids, starts, width, time_ps, destination,
                  args, ppca, args.seed)

    summary = dict(
        method="iterative_singular_spectrum_analysis",
        source_run=str(source), channel=args.channel,
        dataset_identity=manifest.get("analysis_protocol_identity"),
        selected_events=len(selected), evaluation_events=len(eval_ids),
        waveform_samples=length, mask_samples=width, mask_positions=len(starts),
        window_length=args.window_length, rank=args.rank, max_iter=args.max_iter,
        tolerance=args.tolerance, damping=args.damping, seed=args.seed,
        ppca_model=str(root_relative(args.ppca_model)) if args.ppca_model else None,
        output=str(destination),
        note="Per-waveform iterative SSA; observed samples are clamped at every iteration. No model fitting on masked targets.",
    )
    (destination / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Finished. Results: {destination}", flush=True)


if __name__ == "__main__":
    main()
