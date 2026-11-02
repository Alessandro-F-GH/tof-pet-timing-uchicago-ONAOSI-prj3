#!/usr/bin/env python3
"""Evaluate joint-detector PPCA imputation in both complementary TSHAP coalitions.

The SAME jointly fitted PPCA distribution is conditioned on each observed set.
Development events only are used for fitting; held-out events for evaluation.
Reference substitution uses fixed control waveforms shared between both modes.
All reconstruction RMSE values are expressed in physical mV.
"""
from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from waveform_analysis.data.imputation import load_wide_development, root_relative
from waveform_analysis.models.ppca_imputer import ProbabilisticPCA, positions
from waveform_analysis.scripts.train_ppca_interpolation_blend import linear_interpolation_batch


METHODS = ("ppca_joint", "median", "original_tshap")
LABELS = {"ppca_joint": "Joint PPCA", "median": "Development median",
          "original_tshap": "Reference substitution", "linear": "Linear interpolation"}


def parse_dataset(value):
    if "=" not in value:
        raise argparse.ArgumentTypeError("Expected LABEL=RESULTS_PATH")
    label, path = value.split("=", 1)
    if not label or not path:
        raise argparse.ArgumentTypeError("Expected LABEL=RESULTS_PATH")
    return label, Path(path)


def write_csv(path, rows):
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def bootstrap_rmse(event_sse, count_per_event, seed, repetitions):
    rng = np.random.default_rng(seed)
    n = len(event_sse)
    if n < 2:
        value = float(np.sqrt(np.sum(event_sse) / (n * count_per_event)))
        return value, value
    values = [np.sqrt(np.sum(event_sse[rng.integers(n, size=n)]) /
                      (n * count_per_event)) for _ in range(repetitions)]
    return tuple(float(v) for v in np.quantile(values, [0.025, 0.975]))


def conditional_mean(model, batch, observed, missing):
    """Joint Gaussian PPCA conditional mean, with arbitrary observed indices.

    Input and output features are detector-major [det1 time, det2 time].
    The latent posterior is computed once per mask position, not per event.
    """
    w_observed = model.W[observed]
    w_missing = model.W[missing]
    precision = np.eye(model.components) + w_observed.T @ w_observed / model.sigma2
    right = w_observed.T / model.sigma2
    coefficients = np.linalg.solve(precision, right).T @ w_missing.T
    return model.mean[missing] + (batch[:, observed] - model.mean[observed]) @ coefficients


def plot_positions(rows, folder, label, channel, mode):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharex=True, sharey=True)
    for d, ax in enumerate(axes, start=1):
        subset = [r for r in rows if r["detector"] == d]
        for method in dict.fromkeys(r["method"] for r in subset):
            series = sorted((r for r in subset if r["method"] == method),
                            key=lambda r: r["start_ns"])
            ax.plot([r["start_ns"] for r in series],
                    [r["rmse"] for r in series], marker=".", markersize=3,
                    label=LABELS.get(method, method))
        ax.set_title(f"Detector {d}")
        ax.grid(alpha=.25)
    axes[0].set_ylabel("Reconstruction RMSE (mV)")
    axes[0].legend(fontsize=8)
    fig.supxlabel("Window start (ns)")
    fig.suptitle(f"{label} {channel} — {mode}")
    fig.tight_layout()
    safe = re.sub(r"[^A-Za-z0-9_.-]", "_", f"{label}_{channel}_{mode}")
    fig.savefig(folder / f"rmse_by_position_{safe}.pdf")
    plt.close(fig)


def plot_overall(rows, folder):
    groups = list(dict.fromkeys((r["dataset"], r["channel"], r["occlusion_mode"])
                                for r in rows))
    fig, ax = plt.subplots(figsize=(max(10, len(groups) * 1.3), 5))
    methods = list(dict.fromkeys(r["method"] for r in rows))
    x = np.arange(len(groups))
    width = .8 / len(methods)
    for j, method in enumerate(methods):
        values = []
        for group in groups:
            item = next((r for r in rows
                         if (r["dataset"], r["channel"], r["occlusion_mode"]) == group
                         and r["method"] == method), None)
            values.append(float(item["rmse"]) if item else np.nan)
        ax.bar(x - .4 + width * (j + .5), values, width,
               label=LABELS.get(method, method))
    ax.set_xticks(x)
    ax.set_xticklabels([f"{d} {c}\n{m}" for d, c, m in groups], fontsize=8)
    ax.set_ylabel("Pooled reconstruction RMSE (mV)")
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=.2)
    fig.tight_layout()
    fig.savefig(folder / "overall_rmse.pdf")
    plt.close(fig)


def benchmark(label, path, channel, args, folder, cache_root):
    waves, time_ps, source, manifest, transform, controls, mask = load_wide_development(
        root_relative(path), channel, root_relative(cache_root),
        return_transform=True, include_control=True, include_sample_mask=True)
    waves = np.asarray(transform.inverse(waves), np.float64)
    controls = np.asarray(transform.inverse(controls), np.float64)
    time_ps = np.asarray(time_ps, np.float64)
    valid = np.asarray(mask, bool)
    n_events, detectors, length = waves.shape
    if detectors != 2 or len(valid) != length:
        raise ValueError(f"{label}/{channel}: unexpected waveform/mask shape")
    dt = float(np.median(np.diff(time_ps)))
    if dt <= 0 or not np.allclose(np.diff(time_ps), dt):
        raise ValueError("Nonuniform waveform time grid")
    width = int(round(args.mask_ns * 1000 / dt))
    if not 0 < width < length:
        raise ValueError("Invalid window width")
    starts = [p for p in positions(length, width, args.mask_stride_samples or width)
              if np.all(valid[p:p+width])]
    if not starts:
        raise ValueError(f"{label}/{channel}: no unmasked windows of requested width")
    order = np.random.default_rng(args.seed).permutation(n_events)
    order = order[:min(args.max_events, n_events)]
    n_eval = min(args.eval_events, max(2, int(round(len(order) * args.eval_fraction))))
    if len(order) - n_eval <= args.components:
        raise ValueError("Insufficient training events for PPCA components")
    train_ids, test_ids = order[:-n_eval], order[-n_eval:]
    # Fit once across BOTH detectors using only the independent training cohort.
    # Scale is not standardized across detector channels: inputs are physical mV.
    train = waves[train_ids].reshape(len(train_ids), -1)
    test = waves[test_ids].reshape(len(test_ids), -1)
    model = ProbabilisticPCA(args.components).fit(train)
    median = np.median(train, axis=0)
    rng = np.random.default_rng(args.seed + 17)
    baseline_ids = rng.choice(len(controls), min(args.n_baselines, len(controls)),
                              replace=False)
    baselines = controls[baseline_ids].reshape(len(baseline_ids), -1)
    # Original TSHAP background is sampled once and shared across modes/positions.
    print(f"[{label}/{channel}] train={len(train)} eval={len(test)} "
          f"references={len(baselines)} positions={len(starts)}", flush=True)
    position_rows, overall_rows = [], []
    for mode in ("remove_window", "keep_only"):
        methods = METHODS + (("linear",) if mode == "remove_window" else ())
        total_sse = {m: np.zeros(len(test)) for m in methods}
        per_event_count = 0
        for pi, start in enumerate(starts):
            segment = np.zeros(length, bool)
            segment[start:start+width] = True
            observed_time = valid & (~segment if mode == "remove_window" else segment)
            missing_time = valid & (segment if mode == "remove_window" else ~segment)
            observed = np.tile(observed_time, 2)
            missing = np.tile(missing_time, 2)
            n_missing = int(np.sum(missing_time))
            if not observed.any() or n_missing == 0:
                continue
            per_event_count += int(missing.sum())
            predictions = {
                "ppca_joint": conditional_mean(model, test, observed, missing),
                "median": np.broadcast_to(median[missing], (len(test), int(missing.sum())))
            }
            if mode == "remove_window":
                # Linear interpolation is a local reconstruction reference.
                reconstructed = [
                    linear_interpolation_batch(waves[test_ids, d, :], start, width)
                    for d in range(2)
                ]
                predictions["linear"] = np.concatenate(reconstructed, axis=1)
            truth = test[:, missing]
            # Reference substitution: average squared error across full baseline
            # draws, NOT squared error of the averaged reference waveform.
            reference_error = np.mean(
                np.square(baselines[:, missing][None, :, :] - truth[:, None, :]),
                axis=1)
            errors = {m: np.square(pred - truth) for m, pred in predictions.items()}
            errors["original_tshap"] = reference_error
            for method, err in errors.items():
                total_sse[method] += np.sum(err, axis=1)
                for d in (0, 1):
                    start_d = d * length
                    mask_d = missing[start_d:start_d+length]
                    chunk = err[:, d*n_missing:(d+1)*n_missing]
                    position_rows.append(dict(dataset=label, channel=channel,
                        occlusion_mode=mode, detector=d+1, method=method,
                        mask_position=pi, start_sample=start,
                        start_ns=float(time_ps[start] / 1000.),
                        mask_width_samples=width, mask_width_ns=width*dt/1000.,
                        missing_valid_samples=int(mask_d.sum()),
                        evaluation_events=len(test),
                        rmse=float(np.sqrt(np.mean(chunk)))))
            if pi % max(1, len(starts) // 10) == 0 or pi == len(starts)-1:
                print(f"[{label}/{channel}/{mode}] {pi+1}/{len(starts)}", flush=True)
        for j, method in enumerate(methods):
            overall_rmse = float(np.sqrt(np.sum(total_sse[method]) /
                                         (len(test)*per_event_count)))
            lo, hi = bootstrap_rmse(total_sse[method], per_event_count,
                                    args.seed + 1000 + j, args.bootstrap)
            overall_rows.append(dict(dataset=label, channel=channel,
                occlusion_mode=mode, method=method, rmse=overall_rmse,
                ci95_low=lo, ci95_high=hi, training_events=len(train),
                evaluation_events=len(test), mask_positions=len(starts),
                components=args.components, mask_width_ns=width*dt/1000.))
        plot_positions([r for r in position_rows if r["occlusion_mode"] == mode],
                       folder, label, channel, mode)
    details = dict(dataset=label, channel=channel, source_run=str(source),
                   dataset_identity=manifest.get("analysis_protocol_identity"),
                   training_events=len(train), evaluation_events=len(test),
                   reference_events=len(baselines), sample_interval_ps=dt,
                   mask_width_samples=width, mask_positions=len(starts),
                   ppca="joint_two_detector_PPCA_same_model_for_both_masks",
                   modes=["remove_window", "keep_only"])
    return position_rows, overall_rows, details


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", action="append", type=parse_dataset, required=True,
                        help="Repeat: UC=waveform_analysis/results/UC")
    parser.add_argument("--channels", nargs="+", choices=["energy", "timing"],
                        default=["energy", "timing"])
    parser.add_argument("--cache-root", type=Path,
                        help="Single cache root (use per-dataset LABEL=PATH overrides if needed)")
    parser.add_argument("--dataset-cache-root", action="append", default=[],
                        type=parse_dataset, help="Repeat: UC=/home/user/UC_cache")
    parser.add_argument("--output", type=Path, default=Path("imputation_joint_ppca"))
    parser.add_argument("--max-events", type=int, default=5000)
    parser.add_argument("--components", type=int, default=32)
    parser.add_argument("--n-baselines", type=int, default=8)
    parser.add_argument("--mask-ns", type=float, default=.25)
    parser.add_argument("--mask-stride-samples", type=int)
    parser.add_argument("--eval-events", type=int, default=500)
    parser.add_argument("--eval-fraction", type=float, default=.2)
    parser.add_argument("--bootstrap", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if (args.mask_ns <= 0 or args.components < 1 or args.n_baselines < 1
            or args.bootstrap < 20 or args.max_events < 10 or args.eval_events < 1
            or not 0 < args.eval_fraction < 1
            or args.mask_stride_samples is not None and args.mask_stride_samples < 1):
        parser.error("Invalid mask width, PPCA components, baselines, or bootstrap count")
    roots = dict(args.dataset_cache_root)
    folder = root_relative(args.output)
    folder.mkdir(parents=True, exist_ok=True)
    positions_out, overall_out, settings = [], [], []
    for label, path in args.dataset:
        cache = roots.get(label, args.cache_root)
        for channel in args.channels:
            p, o, info = benchmark(label, path, channel, args, folder, cache)
            positions_out.extend(p)
            overall_out.extend(o)
            settings.append(info)
    write_csv(folder / "position_metrics.csv", positions_out)
    write_csv(folder / "overall_metrics.csv", overall_out)
    plot_overall(overall_out, folder)
    (folder / "summary.json").write_text(json.dumps({
        "description": "Joint PPCA benchmark, both complementary masks; independent held-out evaluation.",
        "settings": settings, "methods": LABELS, "mask_width_ns": args.mask_ns,
        "seed": args.seed, "bootstrap_replicates": args.bootstrap,
        "baseline_note": "Reference substitution reports expected MSE across draws, not error of mean waveform.",
        "ppca_note": "One joint Gaussian distribution fitted on both detectors and conditioned separately for each coalition."
    }, indent=2), encoding="utf-8")
    print(f"Completed: {folder}", flush=True)


if __name__ == "__main__":
    main()
