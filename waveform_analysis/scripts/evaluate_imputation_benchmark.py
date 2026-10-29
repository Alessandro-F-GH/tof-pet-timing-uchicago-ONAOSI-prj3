#!/usr/bin/env python3
"""Benchmark waveform imputation methods on held-out, artificially masked signals.

Methods: linear interpolation, cubic Hermite, standard PPCA conditional mean,
and the calibrated smooth-envelope Hermite-PPCA blend. Outputs pooled RMSE,
event-bootstrap confidence intervals, RMSE by temporal position, plots and LaTeX.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import re
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from waveform_analysis.data.imputation import load_wide_development, root_relative
from waveform_analysis.models.ppca_imputer import ProbabilisticPCA, positions
from waveform_analysis.scripts.train_ppca_interpolation_blend import (
    linear_interpolation_batch,
    hermite_interpolation_batch,
    smooth_blend_envelope,
)


METHODS = ("original_tshap", "linear", "hermite", "ppca", "hermite_ppca_blend")
LABELS = {
    "original_tshap": "Multi-baseline",
    "linear": "Linear interpolation",
    "hermite": "Cubic Hermite",
    "ppca": "Standard PPCA",
    "hermite_ppca_blend": "Hermite-PPCA blend",
}


def parse_dataset(value):
    if "=" not in value:
        raise argparse.ArgumentTypeError("Use DATASET_LABEL=RESULTS_PATH")
    label, path = value.split("=", 1)
    if not label.strip() or not path.strip():
        raise argparse.ArgumentTypeError("Use DATASET_LABEL=RESULTS_PATH")
    return label.strip(), Path(path.strip())


def rmse_from_sse(sse, n):
    return float(np.sqrt(max(float(sse), 0.0) / max(int(n), 1)))


def bootstrap_rmse(event_sse, values_per_event, seed, replicates=500):
    """Event-level bootstrap CI; each sampled event carries all its masks/detectors."""
    event_sse = np.asarray(event_sse, dtype=np.float64)
    n_events = len(event_sse)
    if n_events < 2:
        value = rmse_from_sse(event_sse.sum(), values_per_event * n_events)
        return value, value
    rng = np.random.default_rng(seed)
    estimates = np.empty(replicates, dtype=np.float64)
    for b in range(replicates):
        ids = rng.integers(0, n_events, size=n_events)
        estimates[b] = rmse_from_sse(event_sse[ids].sum(), values_per_event * n_events)
    return tuple(np.quantile(estimates, [0.025, 0.975]).tolist())


def physical_unit_scale(values):
    """RMSE values are stored in mV; use nV for any scale below 1 mV."""
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite) & (finite > 0)]
    largest = float(np.max(finite)) if finite.size else 1.0
    return (1.0, "mV") if largest >= 1.0 else (1e6, "nV")


def format_rmse_label(value_mv):
    """Use nV for individual bar labels below 1 mV."""
    value_mv = float(value_mv)
    if value_mv < 1.0:
        return f"{value_mv * 1e6:.3g} nV"
    return f"{value_mv:.3g} mV"


def save_position_plot(rows, output, label, channel):
    scale, unit = physical_unit_scale([r["rmse"] for r in rows])
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.7), sharex=True, sharey=True)
    handles = []
    for detector, ax in enumerate(axes, start=1):
        selected = [r for r in rows if r["detector"] == detector]
        for method in METHODS:
            series = sorted(
                (r for r in selected if r["method"] == method),
                key=lambda r: r["start_ns"],
            )
            x = np.asarray([r["start_ns"] for r in series])
            y = np.asarray([r["rmse"] for r in series]) * scale
            # NaNs break the line over excluded timing masks; log axes cannot
            # meaningfully display zero or negative reconstruction errors.
            y[~np.isfinite(y) | (y <= 0)] = np.nan
            (line,) = ax.plot(x, y, label=LABELS[method], lw=1.5)
            if detector == 1:
                handles.append(line)
        ax.set_title(f"Detector {detector}")
        ax.set_yscale("log")
        ax.grid(alpha=0.25, which="both")
    # Keep the one shared legend between the figure title and subplot titles.
    # A figure-level x label remains at the bottom with no competing legend.
    fig.suptitle(f"{label} - {channel.capitalize()} channel", y=0.99)
    fig.legend(handles, [LABELS[m] for m in METHODS],
               loc="upper center", bbox_to_anchor=(0.5, 0.935),
               ncol=len(METHODS), fontsize="small", frameon=False)
    axes[0].set_ylabel(f"Reconstruction RMSE ({unit})", labelpad=5)
    fig.supxlabel("Masked interval start (ns)", y=0.02)
    fig.tight_layout(rect=(0.01, 0.07, 0.99, 0.82))
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", f"{label}_{channel}")
    fig.savefig(output / f"rmse_by_position_{safe}.pdf", bbox_inches="tight")
    fig.savefig(output / f"rmse_by_position_{safe}.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def evaluate_configuration(label, results, channel, args, output):
    waves, time_ps, source, manifest, transform, control_waves, sample_mask = load_wide_development(
        root_relative(results), channel, root_relative(args.cache_root),
        return_transform=True, include_control=True, include_sample_mask=True
    )
    # The prepared arrays are normalized/clipped for model training. Convert them
    # back through the frozen per-detector transform before fitting imputers or
    # computing reconstruction errors, so all reported RMSE values are in mV.
    waves = np.asarray(transform.inverse(waves), dtype=np.float64)
    time_ps = np.asarray(time_ps, dtype=np.float64)
    n_total, n_detectors, length = waves.shape
    if n_detectors != 2:
        raise ValueError(f"{label}/{channel}: expected 2 detectors, got {n_detectors}")
    dt_ps = float(np.median(np.diff(time_ps)))
    if dt_ps <= 0 or not np.allclose(np.diff(time_ps), dt_ps, rtol=1e-4, atol=1e-3):
        raise ValueError(f"{label}/{channel}: expected uniform time grid")
    width = max(1, round(args.mask_ns * 1000.0 / dt_ps))
    if width >= length:
        raise ValueError(f"{label}/{channel}: mask width {width} >= waveform length {length}")
    starts = positions(length, width, args.mask_stride_samples or width)
    # Only fully retained timing intervals are eligible for plotted/pooled RMSE.
    # Excluded waveform regions otherwise produce artificial near-zero errors.
    if channel == "timing":
        starts = [start for start in starts if np.all(sample_mask[start:start + width])]
    if not starts:
        raise ValueError(f"{label}/{channel}: no valid mask positions remain")

    selected = np.random.default_rng(args.seed).permutation(n_total)[:min(args.max_events, n_total)]
    if len(selected) < 20:
        raise ValueError(f"{label}/{channel}: need at least 20 events, found {len(selected)}")
    # Fixed, disjoint event roles: fit PPCA, calibrate blend, evaluate final errors.
    n_eval = min(args.eval_events, max(2, int(round(len(selected) * args.eval_fraction))))
    n_tune = min(args.tuning_events, max(2, int(round(len(selected) * args.tuning_fraction))))
    n_train = len(selected) - n_eval - n_tune
    if n_train < max(10, args.components + 1):
        raise ValueError("Not enough training events after reserving tuning/evaluation cohorts")
    training, tuning, evaluation = (
        selected[:n_train], selected[n_train:n_train+n_tune], selected[n_train+n_tune:]
    )
    if len(evaluation) == 0 or len(tuning) == 0:
        raise ValueError("Empty tuning/evaluation split; adjust event/fraction settings")

    print(f"[{label}/{channel}] events={len(selected)} train={len(training)} "
          f"tune={len(tuning)} test={len(evaluation)} length={length} "
          f"mask={width} samples positions={len(starts)}", flush=True)
    n_reference = min(args.n_baselines, len(training))
    if n_reference < 1:
        raise ValueError("No training events available for original TSHAP references")
    # The original TSHAP occlusion method substitutes reference-waveform segments.
    # References come from the non-evaluation cohort and are fixed across positions.
    control_waves = np.asarray(transform.inverse(control_waves), dtype=np.float64)
    if len(control_waves) < 1:
        raise ValueError("No control waveforms available for original TSHAP reference substitution")
    n_reference = min(args.n_baselines, len(control_waves))
    reference_ids = np.random.default_rng(args.seed + 17).choice(
        len(control_waves), size=n_reference, replace=False
    )
    reference_waves = control_waves[reference_ids]

    models = []
    prepared_masks = []
    for det in range(2):
        model = ProbabilisticPCA(args.components).fit(
            np.asarray(waves[training, det, :], dtype=np.float64)
        )
        models.append(model)
        prepared_masks.append([model.prepare_mask(start, width) for start in starts])

    # Tune a separate smooth-envelope PPCA alpha for each detector and position.
    alphas = np.zeros((2, len(starts)), dtype=np.float64)
    for det, model in enumerate(models):
        for pi, start in enumerate(starts):
            stop = start + width
            env = smooth_blend_envelope(start, width, length)
            numerator = denominator = 0.0
            for offset in range(0, len(tuning), args.batch_size):
                ids = tuning[offset:offset+args.batch_size]
                batch = np.asarray(waves[ids, det, :], dtype=np.float64)
                truth = batch[:, start:stop]
                hermite = hermite_interpolation_batch(batch, start, width, args.slope_points)
                ppca = model.conditional_mean_batch(batch, start, width, prepared_masks[det][pi])
                delta = env[None, :] * (ppca - hermite)
                numerator += float(np.sum(delta * (truth - hermite)))
                denominator += float(np.sum(delta * delta))
            alphas[det, pi] = (
                float(np.clip(numerator / denominator, 0.0, 1.0))
                if denominator > 1e-18 else 0.0
            )

    # Per-position and event-wise squared errors are retained for pooled metrics/CI.
    position_rows = []
    event_sse = {method: np.zeros(len(evaluation), dtype=np.float64) for method in METHODS}
    values_per_event = 2 * len(starts) * width
    for pi, start in enumerate(starts):
        stop = start + width
        sse = {method: np.zeros(2, dtype=np.float64) for method in METHODS}
        for det, model in enumerate(models):
            env = smooth_blend_envelope(start, width, length)
            alpha = alphas[det, pi]
            for offset in range(0, len(evaluation), args.batch_size):
                local = slice(offset, min(offset + args.batch_size, len(evaluation)))
                ids = evaluation[local]
                batch = np.asarray(waves[ids, det, :], dtype=np.float64)
                truth = batch[:, start:stop]
                linear = linear_interpolation_batch(batch, start, width)
                hermite = hermite_interpolation_batch(batch, start, width, args.slope_points)
                ppca = model.conditional_mean_batch(batch, start, width, prepared_masks[det][pi])
                blend = hermite + alpha * env[None, :] * (ppca - hermite)
                predictions = {
                    "linear": linear, "hermite": hermite,
                    "ppca": ppca, "hermite_ppca_blend": blend,
                }
                # Original TSHAP replaces each masked segment with the corresponding
                # segment from every reference baseline. Score expected squared error
                # over references per event, rather than imputing their mean waveform.
                reference_segments = reference_waves[:, det, start:stop]
                reference_squared = np.square(
                    reference_segments[None, :, :] - truth[:, None, :]
                )
                original_sse_per_event = np.mean(
                    np.sum(reference_squared, axis=2), axis=1
                )
                sse["original_tshap"][det] += float(original_sse_per_event.sum())
                event_sse["original_tshap"][local] += original_sse_per_event
                for method, pred in predictions.items():
                    squared = np.square(pred - truth)
                    sse[method][det] += float(squared.sum())
                    event_sse[method][local] += squared.sum(axis=1)
        for det in range(2):
            for method in METHODS:
                position_rows.append({
                    "dataset": label, "channel": channel, "detector": det + 1,
                    "method": method, "mask_position": pi, "start_sample": int(start),
                    "start_ns": float(time_ps[start] / 1000.0),
                    "mask_width_samples": int(width), "mask_width_ns": float(width * dt_ps / 1000.0),
                    "evaluation_events": int(len(evaluation)),
                    "rmse": rmse_from_sse(sse[method][det], len(evaluation) * width),
                })
        if pi % max(1, len(starts)//10) == 0 or pi == len(starts)-1:
            print(f"[{label}/{channel}] position {pi+1}/{len(starts)}", flush=True)

    overall_rows = []
    for method in METHODS:
        total_sse = float(event_sse[method].sum())
        n_values = len(evaluation) * values_per_event
        lo, hi = bootstrap_rmse(event_sse[method], values_per_event,
                                args.seed + 1000 + METHODS.index(method), args.bootstrap)
        overall_rows.append({
            "dataset": label, "channel": channel, "method": method,
            "rmse": rmse_from_sse(total_sse, n_values),
            "ci95_low": lo, "ci95_high": hi,
            "evaluation_events": len(evaluation), "training_events": len(training),
            "tuning_events": len(tuning), "mask_positions": len(starts),
            "mask_width_samples": width, "mask_width_ns": width * dt_ps / 1000.0,
            "components": args.components,
        })
    return position_rows, overall_rows, {
        "dataset": label, "channel": channel, "source_run": str(source),
        "dataset_identity": manifest.get("analysis_protocol_identity"),
        "total_available_events": int(n_total), "selected_events": int(len(selected)),
        "training_events": int(len(training)), "tuning_events": int(len(tuning)),
        "evaluation_events": int(len(evaluation)), "waveform_samples": int(length),
        "sample_interval_ps": dt_ps, "mask_width_samples": width,
        "mask_width_ns": width * dt_ps / 1000.0, "mask_positions": len(starts),
        "components": args.components, "seed": args.seed,
        "methods": list(METHODS), "slope_points": args.slope_points,
        "n_original_tshap_reference_baselines": int(n_reference),
        "original_tshap_reference_source": "independent control dataset",
        "signal_units": "mV (inverse of frozen per-detector input transform)",
    }


def write_csv(path, rows):
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_latex(path, rows):
    configs = sorted({(r["dataset"], r["channel"]) for r in rows})
    lines = [
        r"\begin{table}[t]", r"\centering",
        r"\caption{Held-out waveform reconstruction RMSE in mV. Confidence intervals are event-bootstrap 95\% intervals.}",
        r"\label{tab:imputation-reconstruction-rmse}",
        r"\begin{tabular}{lllrrr}", r"\toprule",
        r"Dataset & Channel & Method & RMSE & 95\% CI & $\Delta$ vs.\ Hermite (\%) \\",
        r"\midrule",
    ]
    for dataset, channel in configs:
        subset = [r for r in rows if r["dataset"] == dataset and r["channel"] == channel]
        hermite = next(r["rmse"] for r in subset if r["method"] == "hermite")
        for r in subset:
            delta = 100.0 * (1.0 - r["rmse"] / hermite) if hermite else 0.0
            label = LABELS[r["method"]].replace("--", r"\textendash{}")
            ci = f'[{r["ci95_low"]:.3g}, {r["ci95_high"]:.3g}]'
            lines.append(
                f'{dataset} & {channel} & {label} & {r["rmse"]:.4g} & {ci} & {delta:+.1f} \\\\'
            )
        lines.append(r"\midrule")
    if lines[-1] == r"\midrule":
        lines.pop()
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    path.write_text("\n".join(lines), encoding="utf-8")


def save_overall_plot(rows, output):
    """One grouped pooled-RMSE plot with event-bootstrap confidence intervals."""
    configs = list(dict.fromkeys((r["dataset"], r["channel"]) for r in rows))
    scale, unit = physical_unit_scale([r["rmse"] for r in rows])
    x = np.arange(len(configs), dtype=np.float64)
    total_width = 0.82
    bar_width = total_width / len(METHODS)
    fig, ax = plt.subplots(figsize=(max(9.0, 1.7 * len(configs)), 5.2))
    for mi, method in enumerate(METHODS):
        values, low, high = [], [], []
        for dataset, channel in configs:
            row = next(r for r in rows if r["dataset"] == dataset and r["channel"] == channel
                       and r["method"] == method)
            values.append(row["rmse"] * scale)
            low.append(row["ci95_low"] * scale)
            high.append(row["ci95_high"] * scale)
        values = np.asarray(values)
        yerr = np.vstack((values - np.asarray(low), np.asarray(high) - values))
        xpos = x - total_width / 2 + (mi + 0.5) * bar_width
        bars = ax.bar(xpos, values, width=bar_width, yerr=yerr, capsize=2.5,
                      label=LABELS[method])
        for bar, value in zip(bars, values):
            ax.annotate(format_rmse_label(value / scale),
                        (bar.get_x() + bar.get_width()/2, bar.get_height()),
                        xytext=(0, 3), textcoords="offset points", ha="center",
                        va="bottom", fontsize=7, rotation=90 if len(configs) > 3 else 0)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{dataset}\n{channel.capitalize()}" for dataset, channel in configs])
    ax.set_ylabel(f"Pooled held-out RMSE ({unit})")
    ax.set_title("Overall imputation reconstruction error")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(fontsize="small", ncol=min(3, len(METHODS)))
    ax.margins(y=0.18)
    fig.tight_layout()
    fig.savefig(output / "overall_rmse.pdf", bbox_inches="tight")
    fig.savefig(output / "overall_rmse.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", action="append", type=parse_dataset, required=True,
                        help="Repeat as LABEL=RESULTS_PATH, e.g. UC=waveform_analysis/results/UC/RUN")
    parser.add_argument("--channels", nargs="+", choices=("energy", "timing"),
                        default=("energy", "timing"))
    parser.add_argument("--cache-root", type=Path)
    parser.add_argument("--output", type=Path, default=Path("waveform_analysis/imputation_benchmark"))
    parser.add_argument("--max-events", type=int, default=5000)
    parser.add_argument("--components", type=int, default=32)
    parser.add_argument("--n-baselines", type=int, default=8,
                        help="Number of non-evaluation reference waveforms for original TSHAP substitution")
    parser.add_argument("--mask-ns", type=float, default=0.25)
    parser.add_argument("--mask-stride-samples", type=int)
    parser.add_argument("--eval-events", type=int, default=500)
    parser.add_argument("--eval-fraction", type=float, default=0.20)
    parser.add_argument("--tuning-events", type=int, default=500)
    parser.add_argument("--tuning-fraction", type=float, default=0.20)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--slope-points", type=int, default=5)
    parser.add_argument("--bootstrap", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if (args.max_events < 20 or args.components < 1 or args.n_baselines < 1 or args.mask_ns <= 0
            or args.batch_size < 1 or args.eval_events < 1 or args.tuning_events < 1
            or not 0 < args.eval_fraction < 1 or not 0 < args.tuning_fraction < 1
            or args.slope_points < 2 or args.bootstrap < 20
            or args.eval_fraction + args.tuning_fraction >= 0.8):
        parser.error("Invalid settings: check event counts, fractions, mask width and bootstrap count")
    output = root_relative(args.output)
    output.mkdir(parents=True, exist_ok=True)
    all_positions, all_overall, configurations = [], [], []
    for label, result_path in args.dataset:
        for channel in args.channels:
            positions_rows, overall_rows, config = evaluate_configuration(
                label, result_path, channel, args, output
            )
            all_positions.extend(positions_rows)
            all_overall.extend(overall_rows)
            configurations.append(config)
            safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", f"{label}_{channel}")
            save_position_plot(positions_rows, output, label, channel)
            write_csv(output / f"position_metrics_{safe}.csv", positions_rows)
    write_csv(output / "position_metrics.csv", all_positions)
    write_csv(output / "overall_metrics.csv", all_overall)
    write_latex(output / "reconstruction_metrics.tex", all_overall)
    save_overall_plot(all_overall, output)
    summary = {
        "description": "Held-out reconstruction benchmark in physical mV units; original TSHAP multi-baseline substitution is evaluated as expected reconstruction error over reference segments; event-level bootstrap CIs.",
        "configurations": configurations,
        "methods": list(METHODS), "method_labels": LABELS,
        "mask_width_ns": args.mask_ns, "seed": args.seed,
        "n_baselines": args.n_baselines,
        "signal_units": "mV; plot values use nV if all RMSE values are below 1 mV",
        "bootstrap_replicates": args.bootstrap,
        "warning": "Results are valid only if each results path resolves to the intended development dataset and split.",
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Benchmark complete. Outputs: {output}", flush=True)
    print(f"  {output / 'overall_metrics.csv'}")
    print(f"  {output / 'position_metrics.csv'}")
    print(f"  {output / 'reconstruction_metrics.tex'}")


if __name__ == "__main__":
    main()
