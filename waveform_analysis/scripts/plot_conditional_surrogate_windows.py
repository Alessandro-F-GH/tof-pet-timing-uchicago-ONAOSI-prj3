"""Plot per-window conditional surrogate fidelity from saved diagnostics only.

Reads NN-Olsen / temporal CNN output directories (diagnostics.npz).
No waveform loading, fitting, inference, or original-model access.

Example:
python -m waveform_analysis.scripts.plot_conditional_surrogate_windows \
    --mlp-results ~/conditional_nn/UC_per_sample \
    --cnn-results ~/conditional_temporal_cnn/UC_per_sample \
    --output-dir ~/conditional_comparison/UC
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def _read_run(root: Path):
    root = root.expanduser().resolve()
    with np.load(root / "diagnostics.npz", allow_pickle=False) as source:
        expected = {
            "window_starts",
            "tuning_query_rows", "holdout_query_rows",
            "tuning_original_prediction_ps", "holdout_original_prediction_ps",
            "tuning_conditional_prediction_ps", "holdout_conditional_prediction_ps",
        }
        absent = expected.difference(source.files)
        if absent:
            raise ValueError(f"{root}: missing diagnostics keys {sorted(absent)}")
        raw = {key: np.array(source[key], copy=True) for key in expected}
    for split in ("tuning", "holdout"):
        original = raw[f"{split}_original_prediction_ps"]
        estimates = raw[f"{split}_conditional_prediction_ps"]
        if original.ndim != 1 or estimates.shape != (len(original), len(raw["window_starts"]), 2):
            raise ValueError(f"{root}: unexpected {split} estimate shape {estimates.shape}")
        if not np.isfinite(original).all() or not np.isfinite(estimates).all():
            raise ValueError(f"{root}: non-finite {split} predictions")
    if np.diff(raw["window_starts"]).min(initial=1) <= 0:
        raise ValueError(f"{root}: window starts must be strictly increasing")
    summary_path = root / "summary.json"
    raw["summary"] = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.exists() else {}
    raw["directory"] = str(root)
    return raw


def _check_same_queries(runs):
    reference = next(iter(runs.values()))
    for label, run in runs.items():
        np.testing.assert_array_equal(
            run["window_starts"], reference["window_starts"],
            err_msg=f"{label}: windows differ; cannot compare per-window performance")
        for split in ("tuning", "holdout"):
            np.testing.assert_array_equal(
                run[f"{split}_query_rows"], reference[f"{split}_query_rows"],
                err_msg=f"{label}: {split} events differ; cannot compare paired results")
            np.testing.assert_allclose(
                run[f"{split}_original_prediction_ps"],
                reference[f"{split}_original_prediction_ps"], rtol=0, atol=1e-5,
                err_msg=f"{label}: {split} original targets differ")


def _window_metrics(run, split):
    original = run[f"{split}_original_prediction_ps"]
    predicted = run[f"{split}_conditional_prediction_ps"]
    errors = predicted - original[:, None, None]
    return {
        "rmse_ps": np.sqrt(np.mean(errors ** 2, axis=0)),
        "mae_ps": np.mean(np.abs(errors), axis=0),
        "bias_ps": np.mean(errors, axis=0),
        "pred_std_ps": np.std(predicted, axis=0),
        "original_std_ps": float(np.std(original)),
        "original_mean_baseline_rmse_ps": float(np.std(original)),
    }


def generate(mlp_results: Path, cnn_results: Path, output_dir: Path,
             *, sample_period_ps: float | None = None):
    """Write per-window CSV and diagnostic plots; return generated file paths."""
    runs = {"MLP": _read_run(mlp_results), "CNN": _read_run(cnn_results)}
    _check_same_queries(runs)
    if sample_period_ps is not None and sample_period_ps <= 0:
        raise ValueError("Sample period must be positive")
    output_dir = output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    starts = runs["MLP"]["window_starts"]
    x = starts.astype(float) if sample_period_ps is None else starts * sample_period_ps / 1000
    xlabel = "Window start (sample index)" if sample_period_ps is None else "Window start (ns)"
    metrics = {split: {name: _window_metrics(run, split)
                       for name, run in runs.items()}
               for split in ("tuning", "holdout")}

    csv_path = output_dir / "per_window_metrics.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        columns = ["model", "split", "window_start_sample", "window_start_ns",
                   "coalition", "rmse_ps", "mae_ps", "bias_ps",
                   "pred_std_ps", "original_std_ps"]
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for split in ("tuning", "holdout"):
            for model_name in runs:
                stats = metrics[split][model_name]
                for j, sample in enumerate(starts):
                    for c, coalition in enumerate(("W", "R")):
                        writer.writerow({
                            "model": model_name, "split": split,
                            "window_start_sample": int(sample),
                            "window_start_ns": ("" if sample_period_ps is None
                                                else float(x[j])),
                            "coalition": coalition,
                            "rmse_ps": float(stats["rmse_ps"][j, c]),
                            "mae_ps": float(stats["mae_ps"][j, c]),
                            "bias_ps": float(stats["bias_ps"][j, c]),
                            "pred_std_ps": float(stats["pred_std_ps"][j, c]),
                            "original_std_ps": stats["original_std_ps"],
                        })

    outputs = [csv_path]
    for split in ("holdout", "tuning"):
        fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True,
                                 constrained_layout=True)
        for c, coalition in enumerate(("W", "R")):
            ax = axes[c]
            for name in runs:
                ax.plot(x, metrics[split][name]["rmse_ps"][:, c],
                        label=name, linewidth=1.7)
            ref = metrics[split]["MLP"]["original_mean_baseline_rmse_ps"]
            ax.axhline(ref, color="0.5", linestyle="--", linewidth=1,
                       label=f"Constant predictor (std={ref:.1f} ps)")
            ax.set_ylabel(f"RMSE {coalition} (ps)")
            ax.grid(alpha=0.25)
            ax.legend()
        axes[-1].set_xlabel(xlabel)
        fig.suptitle(f"Conditional surrogate fidelity by window — {split}")
        plot_path = output_dir / f"{split}_rmse_by_window.png"
        fig.savefig(plot_path, dpi=180)
        plt.close(fig)
        outputs.append(plot_path)

    fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True,
                             constrained_layout=True)
    for c, coalition in enumerate(("W", "R")):
        ax = axes[c]
        for name in runs:
            for split, style in (("holdout", "-"), ("tuning", "--")):
                ax.plot(x, metrics[split][name]["rmse_ps"][:, c],
                        linestyle=style, linewidth=1.5, label=f"{name} {split}")
        ax.set_ylabel(f"RMSE {coalition} (ps)")
        ax.grid(alpha=0.25)
        ax.legend(ncol=2)
    axes[-1].set_xlabel(xlabel)
    fig.suptitle("Per-window RMSE: holdout (solid) vs tuning (dashed)")
    plot_path = output_dir / "tuning_vs_holdout_rmse_by_window.png"
    fig.savefig(plot_path, dpi=180)
    plt.close(fig)
    outputs.append(plot_path)

    for metric, ylabel in (("mae_ps", "MAE (ps)"),
                          ("bias_ps", "Bias (ps)"),
                          ("pred_std_ps", "Estimated prediction std (ps)")):
        fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True,
                                 constrained_layout=True)
        for c, coalition in enumerate(("W", "R")):
            ax = axes[c]
            for name in runs:
                ax.plot(x, metrics["holdout"][name][metric][:, c],
                        linewidth=1.5, label=name)
            if metric == "bias_ps":
                ax.axhline(0, color="0.5", linestyle=":", linewidth=1)
            if metric == "pred_std_ps":
                std = metrics["holdout"]["MLP"]["original_std_ps"]
                ax.axhline(std, color="0.5", linestyle="--",
                           linewidth=1, label=f"Original std ({std:.1f} ps)")
            ax.set_ylabel(f"{coalition} — {ylabel}")
            ax.grid(alpha=0.25)
            ax.legend()
        axes[-1].set_xlabel(xlabel)
        fig.suptitle(f"Holdout per-window {metric.replace('_ps', '')}")
        plot_path = output_dir / f"holdout_{metric}_by_window.png"
        fig.savefig(plot_path, dpi=180)
        plt.close(fig)
        outputs.append(plot_path)
    return outputs


def _parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mlp-results", type=Path, required=True)
    parser.add_argument("--cnn-results", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--sample-period-ps", type=float, default=None,
                        help="Optional known sampling period in ps; otherwise x is sample index")
    return parser


if __name__ == "__main__":
    args = _parser().parse_args()
    for path in generate(args.mlp_results, args.cnn_results, args.output_dir,
                         sample_period_ps=args.sample_period_ps):
        print(path)
