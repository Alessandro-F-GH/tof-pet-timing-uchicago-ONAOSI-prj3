#!/usr/bin/env python3
"""Estimate a local conditional-variance floor for masked waveform segments.

Nearest neighbors are selected using only observed waveform samples. Temporal
bin averages compress the observed signal and suppress high-frequency noise.
For each K, the sample variance of the neighbors' true masked segments estimates
local conditional variability; the neighbor-mean RMSE is reported as a sanity
check. This is an empirical local estimate, not a guaranteed theoretical bound.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if __package__ in (None, "") and str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.neighbors import NearestNeighbors

from waveform_analysis.scripts.train_probabilistic_pca_imputer import positions
from waveform_analysis.scripts.train_wide_unet_imputer_detector_specific import (
    load_wide_development,
    root_relative,
)


def parse_k_values(value):
    try:
        values = sorted(set(int(part.strip()) for part in value.split(",")))
    except ValueError as exc:
        raise argparse.ArgumentTypeError("K values must be comma-separated integers") from exc
    if not values or any(k < 2 for k in values):
        raise argparse.ArgumentTypeError("Every K must be at least 2")
    return values


def observed_bin_features(signals, start, width, n_bins):
    """Mean-pool temporal bins, excluding every sample inside the masked gap."""
    signals = np.asarray(signals)
    length = signals.shape[1]
    stop = start + width
    edges = np.linspace(0, length, n_bins + 1, dtype=int)
    observed = np.ones(length, dtype=bool)
    observed[start:stop] = False
    columns = []
    for left, right in zip(edges[:-1], edges[1:]):
        idx = np.arange(left, right)
        idx = idx[observed[idx]]
        if idx.size:
            columns.append(signals[:, idx].mean(axis=1))
    if not columns:
        raise ValueError("No observed temporal features remain")
    return np.stack(columns, axis=1)


def read_comparison_metrics(path):
    if path is None:
        return {}
    mapping = {}
    with Path(path).open("r", newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            value = row.get("blend_rmse")
            if value not in (None, ""):
                mapping[(int(row["detector"]), int(row["mask_position"]))] = float(value)
    return mapping


def bootstrap_rmse_ci(per_event_mse, rng, replicates):
    if replicates <= 0:
        return float("nan"), float("nan")
    n = len(per_event_mse)
    indices = rng.integers(0, n, size=(replicates, n))
    values = np.sqrt(np.maximum(per_event_mse[indices].mean(axis=1), 0.0))
    low, high = np.quantile(values, [0.025, 0.975])
    return float(low), float(high)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--channel", choices=("energy", "timing"), required=True)
    parser.add_argument("--cache-root", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--max-events", type=int, default=5000)
    parser.add_argument("--max-reference-events", type=int, default=3000,
                        help="Cap training reference events; 0 uses all training events")
    parser.add_argument("--mask-ns", type=float, default=1.0)
    parser.add_argument("--mask-stride-samples", type=int)
    parser.add_argument("--feature-bins", type=int, default=48)
    parser.add_argument("--k-values", type=parse_k_values, default=[5, 10, 20, 50])
    parser.add_argument("--eval-events", type=int, default=500)
    parser.add_argument("--tuning-events", type=int, default=500,
                        help="Reserve this many validation events before evaluation, matching the blend experiment split")
    parser.add_argument("--val-fraction", type=float, default=0.15)
    parser.add_argument("--query-batch-size", type=int, default=128)
    parser.add_argument("--n-jobs", type=int, default=-1)
    parser.add_argument("--bootstrap-replicates", type=int, default=200)
    parser.add_argument("--comparison-metrics", type=Path,
                        help="Optional position_metrics.csv from the PPCA/interpolation blend experiment")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if (
        args.max_events < 10 or args.max_reference_events < 0
        or args.mask_ns <= 0 or args.feature_bins < 8
        or args.eval_events < 1 or args.tuning_events < 1
        or args.query_batch_size < 1 or args.bootstrap_replicates < 0
        or not 0 < args.val_fraction < 1
        or (args.mask_stride_samples is not None and args.mask_stride_samples < 1)
    ):
        parser.error("Invalid parameter")

    results = root_relative(args.results)
    waves, time_ps, source, manifest = load_wide_development(
        results, args.channel, root_relative(args.cache_root)
    )
    length = waves.shape[-1]
    dt_ps = float(np.median(np.diff(time_ps)))
    if dt_ps <= 0 or not np.allclose(np.diff(time_ps), dt_ps, rtol=1e-4, atol=1e-3):
        raise ValueError("Expected a uniformly sampled time grid")
    width = max(1, round(args.mask_ns * 1000.0 / dt_ps))
    starts = positions(length, width, args.mask_stride_samples or width)

    selected = np.random.default_rng(args.seed).permutation(len(waves))[:args.max_events]
    if len(selected) < 10:
        parser.error("Not enough events were loaded")
    n_val = max(2, round(len(selected) * args.val_fraction))
    if n_val >= len(selected):
        parser.error("Not enough training events")
    validation = selected[:n_val]
    training = selected[n_val:]
    n_reserved = min(args.tuning_events, max(1, len(validation) // 2))
    evaluation = validation[n_reserved:n_reserved + args.eval_events]
    if len(evaluation) == 0:
        parser.error("No evaluation events remain; reduce --tuning-events or --eval-events")

    rng = np.random.default_rng(args.seed + 901)
    if args.max_reference_events and len(training) > args.max_reference_events:
        reference_ids = rng.choice(training, size=args.max_reference_events, replace=False)
    else:
        reference_ids = training.copy()

    destination = (
        root_relative(args.output) if args.output else
        ROOT / "waveform_analysis" / "imputation_experiments"
        / results.name / args.channel / "knn_conditional_variance_floor"
    )
    destination.mkdir(parents=True, exist_ok=True)
    comparison = read_comparison_metrics(root_relative(args.comparison_metrics))
    k_values = args.k_values
    if max(k_values) > len(reference_ids):
        parser.error(f"Maximum K={max(k_values)} exceeds reference pool size {len(reference_ids)}")
    max_k = max(k_values)

    print(
        f"Events: training={len(training)}, references={len(reference_ids)}, "
        f"evaluation={len(evaluation)}; waveform={length}; mask={width}; "
        f"positions={len(starts)}; bins={args.feature_bins}; K={k_values}",
        flush=True,
    )
    rows = []
    rng_bootstrap = np.random.default_rng(args.seed + 902)
    for det in range(2):
        print(f"Detector {det + 1}: preparing reference waveforms", flush=True)
        reference_signals = np.asarray(waves[reference_ids, det, :], dtype=np.float32)
        evaluation_signals = np.asarray(waves[evaluation, det, :], dtype=np.float32)
        for pos_idx, start in enumerate(starts):
            stop = start + width
            ref_features = observed_bin_features(
                reference_signals, start, width, args.feature_bins
            ).astype(np.float32, copy=False)
            query_features = observed_bin_features(
                evaluation_signals, start, width, args.feature_bins
            ).astype(np.float32, copy=False)
            model = NearestNeighbors(
                n_neighbors=max_k, algorithm="brute", metric="euclidean",
                n_jobs=args.n_jobs,
            )
            model.fit(ref_features)
            all_distances = []
            all_indices = []
            for offset in range(0, len(evaluation_signals), args.query_batch_size):
                distances, indices = model.kneighbors(
                    query_features[offset:offset + args.query_batch_size],
                    n_neighbors=max_k, return_distance=True,
                )
                all_distances.append(distances)
                all_indices.append(indices)
            distances = np.concatenate(all_distances, axis=0)
            indices = np.concatenate(all_indices, axis=0)
            neighbor_segments = reference_signals[indices, start:stop]
            truth = evaluation_signals[:, start:stop]

            for k in k_values:
                segments = neighbor_segments[:, :k, :].astype(np.float64, copy=False)
                conditional_var = np.var(segments, axis=1, ddof=1)
                per_event_floor_mse = conditional_var.mean(axis=1)
                floor_rmse = float(np.sqrt(max(float(per_event_floor_mse.mean()), 0.0)))
                prediction = segments.mean(axis=1)
                knn_mse_by_event = np.square(prediction - truth).mean(axis=1)
                knn_rmse = float(np.sqrt(max(float(knn_mse_by_event.mean()), 0.0)))
                ci_low, ci_high = bootstrap_rmse_ci(
                    per_event_floor_mse, rng_bootstrap, args.bootstrap_replicates
                )
                blend_rmse = comparison.get((det + 1, pos_idx))
                rows.append(dict(
                    detector=det + 1,
                    mask_position=pos_idx,
                    start_sample=start,
                    start_ns=float(time_ps[start] / 1000.0),
                    k_neighbors=k,
                    evaluation_events=len(evaluation),
                    reference_events=len(reference_ids),
                    feature_dimensions=int(ref_features.shape[1]),
                    mean_neighbor_distance=float(distances[:, :k].mean()),
                    estimated_conditional_floor_rmse=floor_rmse,
                    floor_rmse_ci95_low=ci_low,
                    floor_rmse_ci95_high=ci_high,
                    knn_conditional_mean_rmse=knn_rmse,
                    comparison_blend_rmse=blend_rmse,
                    blend_minus_floor_rmse=(
                        float(blend_rmse - floor_rmse) if blend_rmse is not None else None
                    ),
                ))
            if pos_idx % 10 == 0 or pos_idx == len(starts) - 1:
                print(
                    f"Detector {det + 1}: position {pos_idx + 1}/{len(starts)} "
                    f"at {time_ps[start] / 1000.0:.2f} ns",
                    flush=True,
                )

    csv_path = destination / "conditional_variance_floor.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    fig, axes = plt.subplots(2, 1, figsize=(11, 8), sharex=True)
    for det in range(2):
        ax = axes[det]
        matching_det = [r for r in rows if r["detector"] == det + 1]
        for k in k_values:
            matching = [r for r in matching_det if r["k_neighbors"] == k]
            x = [r["start_ns"] for r in matching]
            y = [r["estimated_conditional_floor_rmse"] for r in matching]
            lo = [r["floor_rmse_ci95_low"] for r in matching]
            hi = [r["floor_rmse_ci95_high"] for r in matching]
            ax.plot(x, y, label=f"Local variance estimate, K={k}")
            if args.bootstrap_replicates > 0:
                ax.fill_between(x, lo, hi, alpha=0.12)
        if comparison:
            matching = [r for r in matching_det if r["k_neighbors"] == k_values[0]]
            if any(r["comparison_blend_rmse"] is not None for r in matching):
                ax.plot(
                    [r["start_ns"] for r in matching],
                    [r["comparison_blend_rmse"] for r in matching],
                    color="black", linestyle="--", label="Blend held-out RMSE",
                )
        ax.set_ylabel(f"Detector {det + 1} RMSE")
        ax.grid(alpha=0.25)
        ax.legend(fontsize="small")
    axes[-1].set_xlabel("Masked region start (ns)")
    fig.tight_layout()
    fig.savefig(destination / "conditional_variance_floor.png", dpi=180)
    fig.savefig(destination / "conditional_variance_floor.pdf")
    plt.close(fig)

    fig, axes = plt.subplots(2, 1, figsize=(11, 8), sharex=True)
    for det in range(2):
        ax = axes[det]
        matching_det = [r for r in rows if r["detector"] == det + 1]
        for k in k_values:
            matching = [r for r in matching_det if r["k_neighbors"] == k]
            ax.plot(
                [r["start_ns"] for r in matching],
                [r["knn_conditional_mean_rmse"] for r in matching],
                label=f"KNN mean, K={k}",
            )
        ax.set_ylabel(f"Detector {det + 1} RMSE")
        ax.grid(alpha=0.25)
        ax.legend(fontsize="small")
    axes[-1].set_xlabel("Masked region start (ns)")
    fig.tight_layout()
    fig.savefig(destination / "knn_mean_rmse.png", dpi=180)
    plt.close(fig)

    summary = dict(
        method="observed_waveform_knn_local_conditional_variance",
        dataset_identity=manifest.get("analysis_protocol_identity"),
        source_run=str(source),
        channel=args.channel,
        selected_events=len(selected),
        training_events=len(training),
        reference_events=len(reference_ids),
        validation_events=len(validation),
        reserved_validation_events=n_reserved,
        evaluation_events=len(evaluation),
        waveform_length=length,
        mask_samples=width,
        mask_positions=len(starts),
        temporal_feature_bins=args.feature_bins,
        feature_scaling="raw temporal-bin means; no per-bin standardization",
        k_values=k_values,
        bootstrap_replicates=args.bootstrap_replicates,
        comparison_metrics=str(root_relative(args.comparison_metrics)) if args.comparison_metrics else None,
        note=(
            "For each held-out event, neighbors are selected using only observed samples "
            "outside the mask. The sample variance of their true masked segments estimates "
            "local conditional variability. Finite-neighborhood mismatch can inflate this "
            "estimate; it is not a guaranteed theoretical lower bound. K sensitivity and "
            "the held-out RMSE of the neighbor mean are reported as diagnostics."
        ),
    )
    (destination / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(f"Finished: {destination}", flush=True)


if __name__ == "__main__":
    main()
