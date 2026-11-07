"""Heatmaps of saved teacher prediction conditional on waveform amplitude or first difference.

Six views: per-detector standardized waveform, paired z-score difference,
per-detector standardized first difference, and paired derivative z difference.
Each heatmap is accompanied by empirical support counts. All normalization
and plotted events are train-only under the full-signal seed-42 split.
Teacher inference and transformations are never performed.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from waveform_analysis.engine.sample_normalization import (
    fit_first_difference_stats, fit_time_sample_stats)
from waveform_analysis.scripts.evaluate_conditional_knn import _load_inputs, _logger

CHANNEL_NAMES = ("detector_1", "detector_2", "detector_2_minus_1")



def load_prediction_difference(args, reference_predictions, reference_provenance, log):
    """Only subtract predictions aligned to exactly the same prepared dataset."""
    from argparse import Namespace
    second_args = Namespace(**vars(args))
    second_args.model_dir = args.comparison_model_dir
    other_waves, other_predictions, _, second = _load_inputs(second_args, log)
    if reference_provenance["mode"] != second["mode"]:
        raise ValueError("Different mode between runs")
    if reference_provenance["dataset"] != second["dataset"]:
        raise ValueError(
            "Different prepared dataset artifacts: event row ordering and waveform "
            "identity are not guaranteed. Refusing unsafe subtraction.")
    with np.load(reference_provenance["prediction_artifact"], allow_pickle=False) as archive:
        a = np.asarray(archive["event_id"], dtype=np.int64)
    with np.load(second["prediction_artifact"], allow_pickle=False) as archive:
        b = np.asarray(archive["event_id"], dtype=np.int64)
    if a.size != b.size or not np.array_equal(np.sort(a), np.sort(b)):
        raise ValueError("Different event populations; cannot subtract predictions")
    if len(other_predictions) != len(reference_predictions):
        raise ValueError("Different prediction array lengths")
    log.info("DIFFERENCE | reference %s minus comparison %s",
             reference_provenance["model"], second["model"])
    return reference_predictions - other_predictions, second


def accumulate_heatmaps(waves, labels, rows, positions, mean, scale,
                        derivative_mean, derivative_scale, *,
                        bins=80, z_limit=3., chunk_size=128):
    """Return weighted target sums and counts, shape [2 signal types,3 channels,bins,time].

    Values outside [-z_limit,z_limit] are omitted (never clipped to edges).
    Difference is z(detector2)-z(detector1), not z-scored again.
    """
    if bins < 2 or z_limit <= 0 or chunk_size < 1:
        raise ValueError("Invalid binning")
    positions = np.asarray(positions, dtype=np.int64)
    if positions.ndim != 1 or not len(positions):
        raise ValueError("At least one time position required")
    ntime = len(positions)
    flat_size = bins * ntime
    sums = np.zeros((2, 3, bins, ntime), dtype=np.float64)
    counts = np.zeros((2, 3, bins, ntime), dtype=np.int64)
    t_offsets = np.arange(ntime, dtype=np.int64)[None, :]
    edges = np.linspace(-z_limit, z_limit, bins + 1)
    m = np.asarray(mean[:, positions], dtype=np.float64)
    s = np.asarray(scale[:, positions], dtype=np.float64)
    dm = np.asarray(derivative_mean[:, positions], dtype=np.float64)
    ds = np.asarray(derivative_scale[:, positions], dtype=np.float64)
    for left in range(0, len(rows), chunk_size):
        ids = rows[left:left + chunk_size]
        x = np.asarray(waves[ids], dtype=np.float64)
        signal = (x[:, :, positions] - m[None]) / s[None]
        # Compute first differences on raw amplitudes before z-normalization.
        difference = np.zeros((len(ids), 2, ntime), dtype=np.float64)
        valid_derivative = positions > 0
        if valid_derivative.any():
            p = positions[valid_derivative]
            difference[:, :, valid_derivative] = x[:, :, p] - x[:, :, p - 1]
        derivative = (difference - dm[None]) / ds[None]
        for kind, source in enumerate((signal, derivative)):
            for channel in range(3):
                values = (source[:, channel, :] if channel < 2 else
                          source[:, 1, :] - source[:, 0, :])
                which = np.floor((values + z_limit) * bins / (2 * z_limit)).astype(np.int64)
                # Include exact upper edge in the last bin.
                which = np.where(values == z_limit, bins - 1, which)
                valid = (np.isfinite(values) & (values >= -z_limit) &
                         (values <= z_limit) & (which >= 0) & (which < bins))
                flat_indices = which * ntime + t_offsets
                selected = flat_indices[valid]
                count = np.bincount(selected, minlength=flat_size)
                weighted = np.bincount(
                    selected,
                    weights=np.broadcast_to(np.asarray(labels[ids], dtype=np.float64)[:, None],
                                            valid.shape)[valid],
                    minlength=flat_size)
                counts[kind, channel] += count.reshape(bins, ntime)
                sums[kind, channel] += weighted.reshape(bins, ntime)
    return edges, sums, counts


def run(args):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm, TwoSlopeNorm

    root = args.output_root.expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    log = _logger(root)
    waves, predictions, retained, provenance = _load_inputs(args, log)
    comparison_provenance = None
    if args.comparison_model_dir is not None:
        predictions, comparison_provenance = load_prediction_difference(
            args, predictions, provenance, log)
    n = len(predictions)
    if args.tuning_queries < 1 or args.holdout_queries < 1:
        raise ValueError("Tuning/holdout must be nonempty")
    if args.tuning_queries + args.holdout_queries >= n:
        raise ValueError("Insufficient training population")
    if args.time_stride < 1 or args.z_bins < 2 or args.z_limit <= 0:
        raise ValueError("Invalid grid settings")
    if args.derivative_end_sample is not None and args.derivative_end_sample <= 0:
        raise ValueError("--derivative-end-sample must be positive")
    rng = np.random.default_rng(args.seed)
    excluded = rng.choice(n, args.tuning_queries + args.holdout_queries, replace=False)
    mask = np.ones(n, dtype=bool)
    mask[excluded] = False
    training = np.flatnonzero(mask)
    log.info("TRAIN ONLY | n=%d | tuning=%d holdout=%d seed=%d",
             len(training), args.tuning_queries, args.holdout_queries, args.seed)
    means, scales, signal_meta = fit_time_sample_stats(
        waves, training, chunk_size=args.chunk_size,
        scale_floor_ratio=args.normalization_scale_floor_ratio)
    dmeans, dscales, derivative_meta = fit_first_difference_stats(
        waves, training, chunk_size=args.chunk_size,
        scale_floor_ratio=args.normalization_scale_floor_ratio)
    positions = np.arange(0, waves.shape[-1], args.time_stride, dtype=np.int64)
    if args.retained_only:
        positions = positions[retained[positions]]
    edges, sums, counts = accumulate_heatmaps(
        waves, predictions, training, positions, means, scales, dmeans, dscales,
        bins=args.z_bins, z_limit=args.z_limit, chunk_size=args.chunk_size)
    mean_target = np.full_like(sums, np.nan)
    populated = counts >= args.min_count
    np.divide(sums, counts, out=mean_target, where=populated)
    # Center colors on the train-only mean prediction: color represents
    # an absolute mean in ps, but palette symmetry is around that mean.
    baseline = float(np.mean(predictions[training]))
    deviation = np.abs(mean_target[np.isfinite(mean_target)] - baseline)
    radius = float(np.percentile(deviation, 98)) if len(deviation) else 1.
    radius = max(radius, 1.)
    target_norm = TwoSlopeNorm(vmin=baseline - radius, vcenter=baseline,
                               vmax=baseline + radius)
    result_files = []
    for kind, title in enumerate(("waveform", "first_difference")):
        for channel, channel_name in enumerate(CHANNEL_NAMES):
            fig, (ax_mean, ax_count) = plt.subplots(
                2, 1, figsize=(13, 8), sharex=True, sharey=True,
                constrained_layout=True)
            extent = [float(positions[0]), float(positions[-1] + args.time_stride),
                      float(edges[0]), float(edges[-1])]
            im = ax_mean.imshow(
                np.ma.masked_invalid(mean_target[kind, channel]),
                origin="lower", aspect="auto", interpolation="nearest",
                extent=extent, cmap="coolwarm", norm=target_norm)
            fig.colorbar(im, ax=ax_mean, label=(
                "Mean prediction difference (reference - comparison, ps)"
                if comparison_provenance else "Mean saved teacher prediction (ps)"))
            count_array = counts[kind, channel]
            supported = np.ma.masked_where(count_array == 0, count_array)
            positive = count_array[count_array > 0]
            color_norm = LogNorm(vmin=1, vmax=max(int(positive.max()) if positive.size else 1, 2))
            im_support = ax_count.imshow(
                supported, origin="lower", aspect="auto", interpolation="nearest",
                extent=extent, cmap="viridis", norm=color_norm)
            fig.colorbar(im_support, ax=ax_count, label="Events per cell (log scale)")
            ax_mean.set_title(f"{title} | {channel_name} | mean prediction (min count {args.min_count})")
            ax_count.set_title("Empirical support (counts)")
            ylabel = ("Standardized derivative difference" if kind and channel == 2
                      else "Standardized waveform difference" if not kind and channel == 2
                      else "Standardized first difference (z)" if kind
                      else "Standardized waveform (z)")
            ax_mean.set_ylabel(ylabel)
            ax_count.set_ylabel(ylabel)
            ax_count.set_xlabel("Waveform sample position")
            if kind == 1 and args.derivative_end_sample is not None:
                # Zoom just the derivative plots: keep full-resolution analysis
                # and all data in the NPZ unchanged.
                ax_count.set_xlim(0, min(args.derivative_end_sample,
                                         waves.shape[-1] - 1))
            filename = f"{title}_{channel_name}_target_heatmap.png"
            fig.savefig(root / filename, dpi=180)
            plt.close(fig)
            result_files.append(filename)
            log.info("FIGURE | %s", filename)
    np.savez_compressed(
        root / "conditional_target_heatmaps.npz",
        positions=positions, z_edges=edges, target_mean_ps=mean_target,
        support_counts=counts, target_sums_ps=sums,
        training_prediction_mean_ps=baseline,
        prediction_type=np.array("reference_minus_comparison" if comparison_provenance
                                 else "saved_teacher"),
        channel_names=np.array(CHANNEL_NAMES),
        feature_names=np.array(["waveform", "first_difference"]))
    (root / "summary.json").write_text(json.dumps({
        "provenance": provenance,
        "comparison_provenance": comparison_provenance,
        "prediction_sign": "reference minus comparison" if comparison_provenance else None,
        "events": int(len(training)), "seed": args.seed,
        "target": ("SAVED reference minus comparison prediction, not calibrated ground truth"
                   if comparison_provenance else "SAVED full teacher prediction, not calibrated ground truth"),
        "signal_normalization": signal_meta,
        "first_difference_normalization": derivative_meta,
        "differential_definition": "z(detector2)-z(detector1); not separately standardized",
        "z_range": [-args.z_limit, args.z_limit],
        "z_bins": args.z_bins, "time_stride": args.time_stride,
        "min_count": args.min_count, "figures": result_files,
        "derivative_end_sample": args.derivative_end_sample,
        "target_mean_ps": baseline, "symmetric_color_radius_ps": radius,
        "note": "Descriptive 1D conditional mean per time coordinate, not model attribution.",
    }, indent=2), encoding="utf-8")
    log.info("COMPLETE | %d figures | %s", len(result_files), root)


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model-dir", type=Path, required=True)
    p.add_argument("--comparison-model-dir", type=Path,
                   help="Second saved model run; map reference minus comparison predictions")
    p.add_argument("--cache-root", type=Path)
    p.add_argument("--artifact-root", type=Path)
    p.add_argument("--output-root", type=Path, required=True)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--tuning-queries", type=int, default=500)
    p.add_argument("--holdout-queries", type=int, default=500)
    p.add_argument("--chunk-size", type=int, default=128)
    p.add_argument("--normalization-scale-floor-ratio", type=float, default=0.05)
    p.add_argument("--time-stride", type=int, default=8)
    p.add_argument("--z-bins", type=int, default=80)
    p.add_argument("--z-limit", type=float, default=3.)
    p.add_argument("--min-count", type=int, default=25)
    p.add_argument("--derivative-end-sample", type=int, default=1200,
                   help="Rightmost sample shown in derivative plots; 0-1200 by default. "
                        "The saved numerical heatmap remains full-length.")
    p.add_argument("--retained-only", action="store_true")
    return p


if __name__ == "__main__":
    run(parser().parse_args())
