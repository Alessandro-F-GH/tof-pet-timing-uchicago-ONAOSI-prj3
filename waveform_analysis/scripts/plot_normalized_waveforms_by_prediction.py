"""Plot train-standardized waveform deviations by saved teacher-prediction percentile.

The exact full-signal CNN input standardization is used (per detector and
time position, fitted on training events only). No teacher model inference.
Percentile groups are defined on the training predictions only.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from waveform_analysis.engine.sample_normalization import fit_time_sample_stats
from waveform_analysis.scripts.evaluate_conditional_knn import _load_inputs, _logger


BANDS = ((2., 12.), (45., 55.), (88., 98.))


def prediction_groups(predictions, training, bands=BANDS):
    values = np.asarray(predictions)[training]
    thresholds = np.percentile(values, sorted({v for pair in bands for v in pair}))
    lookup = dict(zip(sorted({v for pair in bands for v in pair}), thresholds))
    groups = []
    for low, high in bands:
        left, right = lookup[low], lookup[high]
        # Percentiles are disjoint here; upper bound inclusive only for last group.
        mask = (values >= left) & (values < right)
        groups.append((f"P{low:g}–P{high:g}", training[mask], float(left), float(right)))
    return groups


def group_moments(waveforms, indices, mean, scale, chunk_size=256):
    """Streaming population mean and standard deviation of standardized waves."""
    count = 0
    first = np.zeros_like(mean, dtype=np.float64)
    second = np.zeros_like(mean, dtype=np.float64)
    for start in range(0, len(indices), chunk_size):
        ids = indices[start:start + chunk_size]
        x = np.asarray(waveforms[ids], dtype=np.float64)
        z = (x - mean[None, :, :]) / scale[None, :, :]
        first += z.sum(axis=0)
        second += np.square(z).sum(axis=0)
        count += len(ids)
    if count == 0:
        raise ValueError("Empty percentile group")
    mu = first / count
    std = np.sqrt(np.maximum(second / count - np.square(mu), 0))
    return mu, std


def run(args):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output = args.output_root.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    log = _logger(output)
    waves, predictions, retained, provenance = _load_inputs(args, log)
    n = len(predictions)
    if args.tuning_queries + args.holdout_queries >= n:
        raise ValueError("Not enough training events")
    rng = np.random.default_rng(args.seed)
    excluded = rng.choice(n, args.tuning_queries + args.holdout_queries, replace=False)
    keep = np.ones(n, dtype=bool)
    keep[excluded] = False
    training = np.flatnonzero(keep)
    means, scales, norm_info = fit_time_sample_stats(
        waves, training, chunk_size=args.chunk_size,
        scale_floor_ratio=args.normalization_scale_floor_ratio)
    groups = prediction_groups(predictions, training)
    positions = np.arange(waves.shape[-1])
    if args.retained_only:
        positions = positions[retained]
    result = []
    summaries = []
    for name, ids, lower, upper in groups:
        avg, sd = group_moments(waves, ids, means, scales, args.chunk_size)
        result.append((name, avg[:, positions], sd[:, positions]))
        summaries.append({"group": name, "count": len(ids),
                          "prediction_lower_ps": lower, "prediction_upper_ps": upper})
        log.info("%s | n=%d | predicted ps range [%.4f, %.4f]", name, len(ids), lower, upper)

    # Two detectors, rows are group-conditional mean z and group-conditional std z.
    fig, axes = plt.subplots(2, 2, figsize=(15, 8), sharex=True)
    for detector in range(2):
        for name, avg, sd in result:
            axes[0, detector].plot(positions, avg[detector], linewidth=1.2, label=name)
            axes[1, detector].plot(positions, sd[detector], linewidth=1.2, label=name)
        axes[0, detector].set_title(f"Detector {detector + 1}")
        axes[0, detector].set_ylabel("Mean standardized waveform deviation (z)")
        axes[1, detector].set_ylabel("Within-group std of z")
        axes[1, detector].set_xlabel("Waveform sample position")
        for row in range(2):
            axes[row, detector].grid(alpha=0.2)
            axes[row, detector].legend()
    fig.suptitle("Waveform z-score by saved teacher prediction percentile (train only)")
    fig.tight_layout()
    figure = output / "normalized_waveforms_by_prediction_percentile.png"
    fig.savefig(figure, dpi=180)
    plt.close(fig)

    np.savez_compressed(
        output / "normalized_waveforms_by_prediction_percentile.npz",
        positions=positions,
        group_names=np.array([r[0] for r in result]),
        group_counts=np.array([v["count"] for v in summaries]),
        group_mean_z=np.stack([r[1] for r in result]),
        group_std_z=np.stack([r[2] for r in result]),
        normalized_reference_mean=means[:, positions],
        normalized_reference_scale=scales[:, positions])
    (output / "summary.json").write_text(json.dumps({
        "provenance": provenance,
        "training_events": int(len(training)),
        "seed": args.seed,
        "normalization": norm_info,
        "bands": summaries,
        "figure": str(figure),
    }, indent=2), encoding="utf-8")
    log.info("COMPLETE | %s", figure)


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model-dir", type=Path, required=True)
    p.add_argument("--cache-root", type=Path)
    p.add_argument("--artifact-root", type=Path)
    p.add_argument("--output-root", type=Path, required=True)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--tuning-queries", type=int, default=500)
    p.add_argument("--holdout-queries", type=int, default=500)
    p.add_argument("--chunk-size", type=int, default=256)
    p.add_argument("--normalization-scale-floor-ratio", type=float, default=0.05)
    p.add_argument("--retained-only", action="store_true",
                   help="Use retained sample positions only")
    return p


if __name__ == "__main__":
    run(parser().parse_args())
