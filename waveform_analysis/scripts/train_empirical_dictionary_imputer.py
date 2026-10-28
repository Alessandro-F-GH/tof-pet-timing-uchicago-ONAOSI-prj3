#!/usr/bin/env python3
"""Interval-specific empirical-dictionary imputation for full wide waveforms.

For each mask position, training events provide a dictionary of actual missing
segments. Neighbors are selected using a low-dimensional PCA representation of
the OBSERVED context only. The missing segment is reconstructed by nonnegative
weighted combinations of dictionary patterns, with an optional soft boundary
value/slope penalty. Event splitting occurs before masks are generated.

Optional PPCA baseline accepts the model.npz saved by
train_probabilistic_pca_imputer.py. No TSHAP integration.
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
from sklearn.decomposition import PCA

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


def boundary_descriptors(patterns):
    """Map [events, width] patterns to endpoint values and inward slopes."""
    return np.stack((
        patterns[:, 0],
        patterns[:, 1] - patterns[:, 0],
        patterns[:, -1],
        patterns[:, -1] - patterns[:, -2],
    ), axis=-1)


def boundary_targets(waveforms, start, stop, length):
    """Boundary targets estimated exclusively from observed samples."""
    target = np.zeros((len(waveforms), 4), dtype=np.float32)
    active = []
    if start >= 2:
        left_slope = waveforms[:, start - 1] - waveforms[:, start - 2]
        target[:, 0] = waveforms[:, start - 1] + left_slope
        target[:, 1] = left_slope
        active.extend((0, 1))
    if stop + 1 < length:
        right_slope = waveforms[:, stop + 1] - waveforms[:, stop]
        target[:, 2] = waveforms[:, stop] - right_slope
        target[:, 3] = right_slope
        active.extend((2, 3))
    return target, active


def context_features(full_scores, waveforms, start, stop, mean, components, scale):
    """PCA scores after replacing the hidden interval by its training mean.

    The subtraction exactly removes the masked samples' contribution to the
    full-waveform PCA scores. The resulting feature depends only on observed
    samples and the fixed training mean; no target values leak into retrieval.
    """
    delta = np.asarray(waveforms[:, start:stop], dtype=np.float32) - mean[start:stop]
    masked_scores = full_scores - delta @ components[:, start:stop].T
    return np.asarray(masked_scores / scale[None, :], dtype=np.float32)


def interpolation_batch(signals, start, stop, length):
    width = stop - start
    if start == 0:
        return np.repeat(signals[:, stop:stop + 1], width, axis=1)
    if stop == length:
        return np.repeat(signals[:, start - 1:start], width, axis=1)
    alpha = (np.arange(start, stop, dtype=np.float32) - (start - 1)) / (width + 1.0)
    return (signals[:, start - 1:start] * (1.0 - alpha[None, :])
            + signals[:, stop:stop + 1] * alpha[None, :])


def retrieve_and_impute(train_features, query_features, train_waveforms,
                        query_waveforms, start, width, args):
    """Return inverse-distance KNN and boundary-aware dictionary predictions."""
    stop = start + width
    n_train = len(train_features)
    k = min(args.neighbors, n_train)
    # Euclidean search is batched so the query-by-dictionary distance matrix
    # never grows beyond eval_batch_size x n_train.
    train_norm = np.einsum("ij,ij->i", train_features, train_features)
    qnorm = np.einsum("ij,ij->i", query_features, query_features)
    dist2 = qnorm[:, None] + train_norm[None, :] - 2.0 * (query_features @ train_features.T)
    np.maximum(dist2, 0.0, out=dist2)
    indices = np.argpartition(dist2, kth=k - 1, axis=1)[:, :k]
    neighbor_d2 = np.take_along_axis(dist2, indices, axis=1)
    order = np.argsort(neighbor_d2, axis=1)
    indices = np.take_along_axis(indices, order, axis=1)
    neighbor_d2 = np.take_along_axis(neighbor_d2, order, axis=1)

    neighbor_patterns = train_waveforms[indices, start:stop]
    # Standard nonnegative inverse-distance weights, with an exact-match guard.
    distances = np.sqrt(neighbor_d2)
    exact = distances[:, 0] < 1e-8
    weights_knn = 1.0 / np.maximum(distances, 1e-8)
    weights_knn /= weights_knn.sum(axis=1, keepdims=True)
    if np.any(exact):
        weights_knn[exact] = 0.0
        weights_knn[exact, 0] = 1.0
    prediction_knn = np.einsum("bk,bkw->bw", weights_knn, neighbor_patterns)

    # Fit local convex-combination weights in the reduced observed-context space.
    # A sum-to-one constraint is imposed analytically; negative coefficients are
    # clipped and renormalized to prevent unstable cancellation of patterns.
    neighbor_features = train_features[indices]
    p = train_features.shape[1]
    gram = np.matmul(neighbor_features, np.transpose(neighbor_features, (0, 2, 1))) / p
    rhs = np.einsum("bkp,bp->bk", neighbor_features, query_features) / p

    if args.boundary_weight > 0 and width >= 2:
        all_descriptors = boundary_descriptors(train_waveforms[:, start:stop])
        boundary_scale = np.maximum(np.std(all_descriptors, axis=0), 1e-6)
        pattern_desc = boundary_descriptors(
            neighbor_patterns.reshape(-1, width)
        ).reshape(len(query_features), k, 4)
        pattern_desc = pattern_desc / boundary_scale[None, None, :]
        desired, active = boundary_targets(query_waveforms, start, stop, train_waveforms.shape[1])
        if active:
            target_desc = desired[:, active] / boundary_scale[None, active]
            candidate_desc = pattern_desc[:, :, active]
            n_active = len(active)
            gram += args.boundary_weight * (
                candidate_desc @ np.transpose(candidate_desc, (0, 2, 1))
            ) / n_active
            rhs += args.boundary_weight * np.einsum(
                "bka,ba->bk", candidate_desc, target_desc
            ) / n_active

    gram += args.ridge * np.eye(k, dtype=np.float32)[None, :, :]
    ones = np.ones((len(query_features), k, 1), dtype=np.float32)
    rhs_col = rhs[:, :, None]
    try:
        solved_rhs = np.linalg.solve(gram, rhs_col)
        solved_ones = np.linalg.solve(gram, ones)
        denominator = np.sum(solved_ones, axis=1, keepdims=True)
        weights = solved_rhs + solved_ones * (
            (1.0 - np.sum(solved_rhs, axis=1, keepdims=True))
            / np.maximum(denominator, 1e-12)
        )
        weights = weights[:, :, 0]
        weights = np.maximum(weights, 0.0)
        sums = weights.sum(axis=1, keepdims=True)
        bad = sums[:, 0] < 1e-12
        weights /= np.maximum(sums, 1e-12)
        if np.any(bad):
            weights[bad] = weights_knn[bad]
    except np.linalg.LinAlgError:
        weights = weights_knn
    prediction_boundary = np.einsum("bk,bkw->bw", weights, neighbor_patterns)
    interpolation = interpolation_batch(query_waveforms, start, stop, train_waveforms.shape[1])
    return prediction_knn, prediction_boundary, interpolation


def ppca_conditional_mean(model, detector, signals, start, width):
    mean = model["mean"][detector].astype(np.float64)
    loading = model["loading"][detector].astype(np.float64)
    sigma2 = float(model["noise_variance"][detector])
    stop = start + width
    observed = np.ones(len(mean), dtype=bool)
    observed[start:stop] = False
    wo, wm = loading[observed], loading[start:stop]
    precision = np.eye(loading.shape[1]) + wo.T @ wo / sigma2
    factor = np.linalg.cholesky(precision)
    coefficient_map = np.linalg.solve(
        factor.T, np.linalg.solve(factor, wo.T / sigma2)
    ).T @ wm.T
    return mean[start:stop] + (
        np.asarray(signals[:, observed], dtype=np.float64) - mean[observed]
    ) @ coefficient_map


def plot_metrics(rows, destination):
    fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
    methods = [
        ("knn_rmse", "Weighted empirical neighbors"),
        ("boundary_rmse", "Boundary-aware dictionary"),
        ("interpolation_rmse", "Linear interpolation"),
        ("ppca_rmse", "PPCA conditional mean"),
    ]
    for detector in (1, 2):
        ax = axes[detector - 1]
        subset = [row for row in rows if row["detector"] == detector]
        x = [row["start_ns"] for row in subset]
        for key, label in methods:
            if subset and subset[0].get(key) is not None:
                ax.plot(x, [row[key] for row in subset], label=label)
        ax.set_ylabel(f"Detector {detector} RMSE")
        ax.grid(alpha=0.25)
        ax.legend()
    axes[-1].set_xlabel("Masked region start (ns)")
    fig.tight_layout()
    fig.savefig(destination / "rmse_by_position.png", dpi=180)
    fig.savefig(destination / "rmse_by_position.pdf")
    plt.close(fig)


def plot_examples(models, waves, eval_ids, starts, width, time_ps, destination,
                  args, seed):
    rng = np.random.default_rng(seed + 918)
    times = np.asarray(time_ps) / 1000.0
    target_positions = list(dict.fromkeys(
        min(range(len(starts)), key=lambda i: abs(times[starts[i]] - target))
        for target in (0.0, 20.0, 40.0)
    ))
    for detector, state in enumerate(models):
        train_waves, train_scores, pca = state["train_waves"], state["train_scores"], state["pca"]
        eval_waves, eval_scores = state["eval_waves"], state["eval_scores"]
        scale = np.maximum(pca.explained_variance_, 1e-12) ** args.distance_whitening
        for pos_idx in target_positions:
            start = starts[pos_idx]
            stop = start + width
            train_feat = context_features(train_scores, train_waves, start, stop,
                                          pca.mean_, pca.components_, scale)
            for local_idx, event_id in enumerate(eval_ids[:args.plot_examples]):
                signal = eval_waves[local_idx:local_idx + 1]
                query_feat = context_features(eval_scores[local_idx:local_idx + 1],
                                              signal, start, stop, pca.mean_,
                                              pca.components_, scale)
                knn, boundary, interp = retrieve_and_impute(
                    train_feat, query_feat, train_waves, signal, start, width, args
                )
                ppca = None
                if state.get("ppca") is not None:
                    ppca = ppca_conditional_mean(state["ppca"], detector, signal, start, width)[0]
                original = signal[0]
                fig, axes = plt.subplots(2, 1, figsize=(11, 7))
                for ax, zoom in zip(axes, (False, True)):
                    left = max(0, start - 2 * width) if zoom else 0
                    right = min(len(original), stop + 2 * width) if zoom else len(original)
                    sl = slice(left, right)
                    masked = original.copy()
                    masked[start:stop] = np.nan
                    ax.plot(times[sl], original[sl], color="black", lw=1.2, label="Ground truth")
                    ax.plot(times[sl], masked[sl], color="gray", ls="--", lw=1, label="Observed")
                    full_knn, full_boundary, full_interp = (original.copy(), original.copy(), original.copy())
                    full_knn[start:stop] = knn[0]
                    full_boundary[start:stop] = boundary[0]
                    full_interp[start:stop] = interp[0]
                    ax.plot(times[sl], full_knn[sl], label="Weighted neighbors", lw=1.2)
                    ax.plot(times[sl], full_boundary[sl], label="Boundary-aware", lw=1.3)
                    ax.plot(times[sl], full_interp[sl], label="Interpolation", lw=1.0, alpha=.8)
                    if ppca is not None:
                        full_ppca = original.copy()
                        full_ppca[start:stop] = ppca
                        ax.plot(times[sl], full_ppca[sl], label="PPCA mean", lw=1.1)
                    ax.axvspan(times[start], times[stop - 1], color="tab:red", alpha=.08)
                    ax.set_ylabel("Normalized signal")
                    ax.grid(alpha=.2)
                axes[0].set_title(
                    f"Detector {detector + 1}, event {event_id}, mask at {times[start]:.2f} ns"
                )
                axes[0].legend(ncol=3, fontsize=8)
                axes[1].set_xlabel("Time (ns)")
                fig.tight_layout()
                folder = destination / "imputation_examples"
                folder.mkdir(parents=True, exist_ok=True)
                fig.savefig(folder / f"detector_{detector+1}_event_{event_id}_position_{pos_idx:02d}.png",
                            dpi=160)
                plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--channel", choices=("energy", "timing"), required=True)
    parser.add_argument("--cache-root", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--max-events", type=int, default=5000)
    parser.add_argument("--context-components", type=int, default=32)
    parser.add_argument("--neighbors", type=int, default=10)
    parser.add_argument("--mask-ns", type=float, default=1.0)
    parser.add_argument("--mask-stride-samples", type=int)
    parser.add_argument("--eval-events", type=int, default=1000)
    parser.add_argument("--eval-batch-size", type=int, default=128)
    parser.add_argument("--distance-whitening", type=float, default=0.25)
    parser.add_argument("--ridge", type=float, default=1e-2)
    parser.add_argument("--boundary-weight", type=float, default=0.1)
    parser.add_argument("--ppca-model", type=Path,
                        help="Optional PPCA model.npz for a same-split baseline")
    parser.add_argument("--plot-examples", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val-fraction", type=float, default=0.15)
    args = parser.parse_args()
    if (args.max_events < 10 or args.context_components < 1 or args.neighbors < 1
        or args.mask_ns <= 0 or args.eval_events < 1 or args.eval_batch_size < 1
        or args.distance_whitening < 0 or args.ridge <= 0 or args.boundary_weight < 0
        or args.plot_examples < 0 or not 0 < args.val_fraction < 1
        or (args.mask_stride_samples is not None and args.mask_stride_samples < 1)):
        parser.error("Invalid parameter values")

    results = root_relative(args.results)
    waves, time_ps, source, manifest = load_wide_development(
        results, args.channel, root_relative(args.cache_root)
    )
    length = waves.shape[-1]
    dt_ps = float(np.median(np.diff(time_ps)))
    if dt_ps <= 0 or not np.allclose(np.diff(time_ps), dt_ps, rtol=1e-4, atol=1e-3):
        raise ValueError("Expected uniformly sampled waveform")
    width = max(1, round(args.mask_ns * 1000.0 / dt_ps))
    if width < 2:
        raise ValueError("Mask must contain at least two samples for boundary descriptors")
    starts = make_positions(length, width, args.mask_stride_samples or width)
    selected = np.random.default_rng(args.seed).permutation(len(waves))[:args.max_events]
    n_val = max(1, round(len(selected) * args.val_fraction))
    eval_ids = selected[:n_val][:args.eval_events]
    train_ids = selected[n_val:]
    if len(train_ids) <= args.context_components:
        parser.error("Need more training events than context components")
    destination = (root_relative(args.output) if args.output else
                  ROOT / "waveform_analysis/imputation_experiments" /
                  results.name / args.channel / "empirical_dictionary")
    destination.mkdir(parents=True, exist_ok=True)

    ppca = None
    if args.ppca_model is not None:
        ppca_path = root_relative(args.ppca_model)
        if not ppca_path.is_file():
            parser.error(f"PPCA model not found: {ppca_path}")
        with np.load(ppca_path) as saved:
            ppca = {key: np.array(saved[key]) for key in
                    ("mean", "loading", "noise_variance")}
        if ppca["mean"].shape != (2, length) or ppca["loading"].shape[:2] != (2, length):
            parser.error("PPCA model waveform dimensions do not match this dataset")
        print(f"Loaded optional PPCA baseline: {ppca_path}", flush=True)

    all_rows = []
    plot_states = []
    print(f"Selected {len(selected)} events: train={len(train_ids)}, "
          f"validation={n_val}, evaluated={len(eval_ids)}; length={length}, "
          f"mask={width} samples, positions={len(starts)}", flush=True)

    for detector in range(2):
        print(f"Detector {detector + 1}: fitting context PCA", flush=True)
        train_waves = np.asarray(waves[train_ids, detector, :], dtype=np.float32)
        eval_waves = np.asarray(waves[eval_ids, detector, :], dtype=np.float32)
        pca = PCA(n_components=args.context_components, svd_solver="randomized",
                  random_state=args.seed)
        train_scores = pca.fit_transform(train_waves)
        eval_scores = pca.transform(eval_waves)
        scale = np.maximum(pca.explained_variance_, 1e-12) ** args.distance_whitening
        print(f"Detector {detector + 1}: PCA context variance explained="
              f"{pca.explained_variance_ratio_.sum():.4f}", flush=True)
        state = dict(train_waves=train_waves, eval_waves=eval_waves,
                     train_scores=train_scores, eval_scores=eval_scores,
                     pca=pca, ppca=ppca)
        for position_idx, start in enumerate(starts):
            stop = start + width
            train_feat = context_features(train_scores, train_waves, start, stop,
                                          pca.mean_, pca.components_, scale)
            sums = {"knn": 0.0, "boundary": 0.0, "interp": 0.0, "ppca": 0.0}
            count = 0
            for offset in range(0, len(eval_waves), args.eval_batch_size):
                batch = eval_waves[offset:offset + args.eval_batch_size]
                qfeat = context_features(eval_scores[offset:offset + args.eval_batch_size],
                                         batch, start, stop, pca.mean_, pca.components_, scale)
                knn, boundary, interp = retrieve_and_impute(
                    train_feat, qfeat, train_waves, batch, start, width, args
                )
                truth = batch[:, start:stop]
                sums["knn"] += float(np.square(knn - truth).sum())
                sums["boundary"] += float(np.square(boundary - truth).sum())
                sums["interp"] += float(np.square(interp - truth).sum())
                if ppca is not None:
                    pred = ppca_conditional_mean(ppca, detector, batch, start, width)
                    sums["ppca"] += float(np.square(pred - truth).sum())
                count += truth.size
            row = dict(
                detector=detector + 1, mask_position=position_idx,
                start_sample=start, start_ns=float(time_ps[start] / 1000.0),
                evaluation_events=len(eval_waves), neighbors=min(args.neighbors, len(train_waves)),
                context_components=args.context_components,
                knn_rmse=float(np.sqrt(sums["knn"] / count)),
                boundary_rmse=float(np.sqrt(sums["boundary"] / count)),
                interpolation_rmse=float(np.sqrt(sums["interp"] / count)),
                ppca_rmse=float(np.sqrt(sums["ppca"] / count)) if ppca is not None else None,
            )
            all_rows.append(row)
            if position_idx % 5 == 0 or position_idx == len(starts) - 1:
                print(f"Detector {detector + 1}: position {position_idx + 1}/{len(starts)} "
                      f"at {row['start_ns']:.2f} ns", flush=True)
        if args.plot_examples:
            plot_states.append(state)
        del train_waves, eval_waves, train_scores, eval_scores

    with (destination / "position_metrics.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(all_rows[0]))
        writer.writeheader()
        writer.writerows(all_rows)
    plot_metrics(all_rows, destination)
    if args.plot_examples and plot_states:
        plot_examples(plot_states, waves, eval_ids, starts, width, time_ps,
                      destination, args, args.seed)

    summary = dict(
        method="interval_specific_empirical_dictionary",
        source_run=str(source), channel=args.channel,
        dataset_identity=manifest.get("analysis_protocol_identity"),
        selected_events=len(selected), training_events=len(train_ids),
        validation_events=n_val, evaluation_events=len(eval_ids),
        waveform_samples=length, mask_samples=width, mask_positions=len(starts),
        context_components=args.context_components, neighbors=args.neighbors,
        distance_whitening=args.distance_whitening, ridge=args.ridge,
        boundary_weight=args.boundary_weight, seed=args.seed,
        ppca_model=str(root_relative(args.ppca_model)) if args.ppca_model else None,
        output=str(destination),
        note="Dictionary targets are interval-specific training segments; retrieval features exclude the masked interval.",
    )
    (destination / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Finished. Results: {destination}", flush=True)


if __name__ == "__main__":
    main()
