#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import json
import zlib
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import linregress, pearsonr, spearmanr

from utils_fit import fit_ctr_ps
from waveform_analysis.ml_pipeline.common import voltage_from_name


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Analyse correlation between bootstrap CTR estimates for two timing methods. "
            "The same event indices are resampled for both methods on every bootstrap repeat, "
            "so covariance in their CTR uncertainties is measured directly."
        )
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        required=True,
        help="Completed waveform-analysis study directory containing manifest.json and artifacts/.",
    )
    parser.add_argument(
        "--reference",
        default="led",
        help="Reference method artifact name. Default: led.",
    )
    parser.add_argument(
        "--method",
        default="cnn",
        help="Method to compare against the reference. Default: cnn.",
    )
    parser.add_argument(
        "--stage",
        choices=("train", "test"),
        default="test",
        help="Residual population to analyse. Default: test.",
    )
    parser.add_argument(
        "--bootstrap-samples",
        type=int,
        default=None,
        help="Number of paired bootstrap repeats. Default: fit.bootstrap_samples from study manifest.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Base RNG seed. Default: validation.seed from study manifest.",
    )
    parser.add_argument(
        "--dataset",
        action="append",
        default=None,
        help="Optional dataset-name filter. May be repeated.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Default: <run-dir>/bootstrap_ctr_correlation/<reference>_vs_<method>/.",
    )
    return parser.parse_args()


def _read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as stream:
        value = json.load(stream)
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return value


def _fit_config(manifest: dict[str, Any]) -> dict[str, Any]:
    config = manifest.get("config") or {}
    fit = config.get("fit") or manifest.get("fit")
    if not isinstance(fit, dict):
        raise KeyError("Study manifest does not contain config.fit")
    return dict(fit)


def _default_seed(manifest: dict[str, Any]) -> int:
    config = manifest.get("config") or {}
    validation = config.get("validation") or {}
    return int(validation.get("seed", 0))


def _dataset_seed(base_seed: int, dataset: str, reference: str, method: str) -> int:
    token = f"{dataset}|{reference}|{method}".encode("utf-8")
    return (int(base_seed) + int(zlib.crc32(token))) & 0xFFFFFFFF


def _artifact(run: Path, dataset: str, method: str, stage: str) -> Path:
    return run / "artifacts" / dataset / f"{method}_{stage}_residuals_ps.npy"


def _discover_datasets(run: Path, reference: str, method: str, stage: str) -> list[str]:
    root = run / "artifacts"
    if not root.is_dir():
        raise FileNotFoundError(f"Missing artifacts directory: {root}")
    datasets = []
    for directory in root.iterdir():
        if not directory.is_dir():
            continue
        if _artifact(run, directory.name, reference, stage).is_file() and _artifact(run, directory.name, method, stage).is_file():
            datasets.append(directory.name)
    return sorted(datasets, key=voltage_from_name)


def _paired_values(
    reference_values: np.ndarray,
    method_values: np.ndarray,
    fit_config: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray, dict[str, int]]:
    reference_values = np.asarray(reference_values, dtype=np.float64).reshape(-1)
    method_values = np.asarray(method_values, dtype=np.float64).reshape(-1)
    if reference_values.shape != method_values.shape:
        raise ValueError(
            "Residual artifacts do not have the same length; paired bootstrap requires event-wise aligned arrays"
        )

    finite = np.isfinite(reference_values) & np.isfinite(method_values)
    mask = finite.copy()
    limit = fit_config.get("max_abs_ps")
    if limit is not None:
        limit = float(limit)
        if not np.isfinite(limit) or limit <= 0:
            raise ValueError("fit.max_abs_ps must be positive when configured")
        mask &= (np.abs(reference_values) <= limit) & (np.abs(method_values) <= limit)

    counts = {
        "n_total": int(reference_values.size),
        "n_nonfinite_pair": int(np.count_nonzero(~finite)),
        "n_outside_common_fit_range": int(np.count_nonzero(finite & ~mask)),
        "n_common": int(np.count_nonzero(mask)),
    }
    return reference_values[mask], method_values[mask], counts


def _paired_bootstrap(
    reference_values: np.ndarray,
    method_values: np.ndarray,
    fit_config: dict[str, Any],
    *,
    repeats: int,
    seed: int,
) -> list[dict[str, float | int]]:
    if repeats < 2:
        raise ValueError("bootstrap-samples must be >= 2")
    n = int(reference_values.size)
    if n < int(fit_config.get("min_events", 100)):
        raise ValueError(f"Only {n} common events; insufficient for CTR estimation")

    rng = np.random.default_rng(int(seed))
    rows: list[dict[str, float | int]] = []
    for repeat in range(int(repeats)):
        indices = rng.integers(0, n, size=n)
        try:
            reference_ctr = fit_ctr_ps(reference_values[indices], fit_config, bootstrap=False).ctr_ps
            method_ctr = fit_ctr_ps(method_values[indices], fit_config, bootstrap=False).ctr_ps
        except ValueError:
            continue
        if not np.isfinite(reference_ctr) or not np.isfinite(method_ctr):
            continue
        delta = float(method_ctr - reference_ctr)
        rows.append(
            {
                "repeat": repeat,
                "reference_ctr_ps": float(reference_ctr),
                "method_ctr_ps": float(method_ctr),
                "delta_ctr_ps": delta,
                "relative_improvement_percent": float(100.0 * (reference_ctr - method_ctr) / reference_ctr),
            }
        )
    return rows


def _correlation_summary(rows: list[dict[str, float | int]]) -> dict[str, float | int]:
    reference = np.asarray([float(row["reference_ctr_ps"]) for row in rows], dtype=np.float64)
    method = np.asarray([float(row["method_ctr_ps"]) for row in rows], dtype=np.float64)
    delta = method - reference

    result: dict[str, float | int] = {
        "bootstrap_successful": int(reference.size),
        "reference_bootstrap_mean_ps": float(np.mean(reference)) if reference.size else float("nan"),
        "reference_bootstrap_std_ps": float(np.std(reference, ddof=1)) if reference.size > 1 else float("nan"),
        "method_bootstrap_mean_ps": float(np.mean(method)) if method.size else float("nan"),
        "method_bootstrap_std_ps": float(np.std(method, ddof=1)) if method.size > 1 else float("nan"),
        "delta_bootstrap_mean_ps": float(np.mean(delta)) if delta.size else float("nan"),
        "delta_bootstrap_std_ps": float(np.std(delta, ddof=1)) if delta.size > 1 else float("nan"),
        "delta_q16_ps": float(np.quantile(delta, 0.16)) if delta.size else float("nan"),
        "delta_q50_ps": float(np.quantile(delta, 0.50)) if delta.size else float("nan"),
        "delta_q84_ps": float(np.quantile(delta, 0.84)) if delta.size else float("nan"),
        "method_better_fraction": float(np.mean(delta < 0.0)) if delta.size else float("nan"),
        "pearson_r": float("nan"),
        "pearson_p": float("nan"),
        "spearman_rho": float("nan"),
        "spearman_p": float("nan"),
        "covariance_ps2": float("nan"),
        "independent_delta_uncertainty_ps": float("nan"),
        "covariance_predicted_delta_uncertainty_ps": float("nan"),
        "paired_to_independent_uncertainty_ratio": float("nan"),
        "regression_slope": float("nan"),
        "regression_intercept_ps": float("nan"),
        "regression_r_squared": float("nan"),
    }
    if reference.size < 3 or np.std(reference) == 0.0 or np.std(method) == 0.0:
        return result

    pearson = pearsonr(reference, method)
    spearman = spearmanr(reference, method)
    covariance = float(np.cov(reference, method, ddof=1)[0, 1])
    sigma_reference = float(np.std(reference, ddof=1))
    sigma_method = float(np.std(method, ddof=1))
    independent = float(np.sqrt(sigma_reference**2 + sigma_method**2))
    covariance_predicted = float(
        np.sqrt(max(0.0, sigma_reference**2 + sigma_method**2 - 2.0 * covariance))
    )
    regression = linregress(reference, method)
    paired = float(np.std(delta, ddof=1))

    result.update(
        pearson_r=float(pearson.statistic),
        pearson_p=float(pearson.pvalue),
        spearman_rho=float(spearman.statistic),
        spearman_p=float(spearman.pvalue),
        covariance_ps2=covariance,
        independent_delta_uncertainty_ps=independent,
        covariance_predicted_delta_uncertainty_ps=covariance_predicted,
        paired_to_independent_uncertainty_ratio=(paired / independent if independent > 0 else float("nan")),
        regression_slope=float(regression.slope),
        regression_intercept_ps=float(regression.intercept),
        regression_r_squared=float(regression.rvalue**2),
    )
    return result


def _write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _plot_pairs(
    path: Path,
    dataset: str,
    reference_name: str,
    method_name: str,
    stage: str,
    rows: list[dict[str, Any]],
    summary: dict[str, Any],
) -> None:
    reference = np.asarray([float(row["reference_ctr_ps"]) for row in rows], dtype=np.float64)
    method = np.asarray([float(row["method_ctr_ps"]) for row in rows], dtype=np.float64)
    if not reference.size:
        return

    fig, ax = plt.subplots(figsize=(7.0, 6.2))
    ax.scatter(reference, method, s=28, alpha=0.55)

    low = float(min(np.min(reference), np.min(method)))
    high = float(max(np.max(reference), np.max(method)))
    span = max(high - low, 1.0)
    low -= 0.06 * span
    high += 0.06 * span
    ax.plot([low, high], [low, high], ls="--", lw=1.2, label="equal CTR")

    slope = float(summary["regression_slope"])
    intercept = float(summary["regression_intercept_ps"])
    if np.isfinite(slope) and np.isfinite(intercept):
        x_line = np.linspace(low, high, 200)
        ax.plot(x_line, intercept + slope * x_line, lw=1.5, label="bootstrap regression")

    ax.set_xlim(low, high)
    ax.set_ylim(low, high)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel(f"{reference_name} bootstrap CTR FWHM [ps]")
    ax.set_ylabel(f"{method_name} bootstrap CTR FWHM [ps]")
    ax.set_title(f"{dataset} · {stage} · paired CTR bootstrap")
    ax.grid(alpha=0.2)
    ax.legend(loc="best")
    ax.text(
        0.02,
        0.98,
        (
            f"paired resamples: {summary['bootstrap_successful']}\n"
            f"Pearson r = {summary['pearson_r']:+.3f}\n"
            f"Spearman ρ = {summary['spearman_rho']:+.3f}\n"
            f"cov = {summary['covariance_ps2']:+.2f} ps²\n"
            f"σ(ΔCTR) paired = {summary['delta_bootstrap_std_ps']:.2f} ps\n"
            f"σ(ΔCTR) independent = {summary['independent_delta_uncertainty_ps']:.2f} ps"
        ),
        transform=ax.transAxes,
        ha="left",
        va="top",
        bbox={"boxstyle": "round,pad=0.35", "facecolor": "white", "alpha": 0.9},
    )
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def _plot_delta(
    path: Path,
    dataset: str,
    reference_name: str,
    method_name: str,
    rows: list[dict[str, Any]],
    summary: dict[str, Any],
) -> None:
    delta = np.asarray([float(row["delta_ctr_ps"]) for row in rows], dtype=np.float64)
    if not delta.size:
        return
    fig, ax = plt.subplots(figsize=(7.4, 4.8))
    bins = max(10, min(25, int(round(np.sqrt(delta.size)))))
    ax.hist(delta, bins=bins, alpha=0.45, edgecolor="black")
    ax.axvline(0.0, ls="--", lw=1.2, label="no CTR change")
    ax.axvline(float(np.mean(delta)), ls="-", lw=1.6, label=f"mean ΔCTR {np.mean(delta):+.2f} ps")
    ax.set_xlabel(f"ΔCTR = {method_name} − {reference_name} [ps]")
    ax.set_ylabel("Bootstrap resamples")
    ax.set_title(f"{dataset} · paired bootstrap improvement")
    ax.grid(axis="y", alpha=0.2)
    ax.legend()
    ax.text(
        0.02,
        0.97,
        (
            f"σ paired = {summary['delta_bootstrap_std_ps']:.2f} ps\n"
            f"16–84% = [{summary['delta_q16_ps']:+.2f}, {summary['delta_q84_ps']:+.2f}] ps\n"
            f"P({method_name} better) = {100.0 * summary['method_better_fraction']:.1f}%"
        ),
        transform=ax.transAxes,
        ha="left",
        va="top",
        bbox={"boxstyle": "round,pad=0.35", "facecolor": "white", "alpha": 0.9},
    )
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def _plot_summary(path: Path, summaries: list[dict[str, Any]], method_name: str) -> None:
    rows = [row for row in summaries if np.isfinite(float(row["voltage_V"]))]
    if not rows:
        return
    rows = sorted(rows, key=lambda row: float(row["voltage_V"]))
    voltage = np.arange(len(rows), dtype=float)
    labels = [f"{float(row['voltage_V']):g} V" for row in rows]
    correlation = np.asarray([float(row["pearson_r"]) for row in rows], dtype=float)

    fig, ax = plt.subplots(figsize=(8.2, 4.8))
    ax.bar(voltage, correlation, alpha=0.75)
    ax.axhline(0.0, lw=1.0, ls="--")
    ax.set_xticks(voltage)
    ax.set_xticklabels(labels)
    ax.set_ylim(-1.0, 1.0)
    ax.set_xlabel("Bias voltage")
    ax.set_ylabel("Pearson correlation of paired bootstrap CTR")
    ax.set_title(f"Bootstrap CTR covariance with {method_name}")
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def analyse_dataset(
    run: Path,
    dataset: str,
    reference_name: str,
    method_name: str,
    stage: str,
    fit_config: dict[str, Any],
    repeats: int,
    seed: int,
    output_root: Path,
) -> dict[str, Any]:
    reference_path = _artifact(run, dataset, reference_name, stage)
    method_path = _artifact(run, dataset, method_name, stage)
    reference_raw = np.asarray(np.load(reference_path), dtype=np.float64)
    method_raw = np.asarray(np.load(method_path), dtype=np.float64)
    reference, method, counts = _paired_values(reference_raw, method_raw, fit_config)

    full_reference = fit_ctr_ps(reference, fit_config, bootstrap=False).ctr_ps
    full_method = fit_ctr_ps(method, fit_config, bootstrap=False).ctr_ps

    bootstrap_rows = _paired_bootstrap(
        reference,
        method,
        fit_config,
        repeats=repeats,
        seed=seed,
    )
    statistics = _correlation_summary(bootstrap_rows)
    if int(statistics["bootstrap_successful"]) < max(10, repeats // 2):
        raise RuntimeError(
            f"{dataset}: only {statistics['bootstrap_successful']}/{repeats} paired bootstrap resamples produced valid FWHM"
        )

    dataset_dir = output_root / dataset
    dataset_dir.mkdir(parents=True, exist_ok=True)
    _write_rows(dataset_dir / "bootstrap_pairs.csv", bootstrap_rows)
    _plot_pairs(
        dataset_dir / "ctr_bootstrap_correlation.pdf",
        dataset,
        reference_name.upper(),
        method_name.upper(),
        stage,
        bootstrap_rows,
        statistics,
    )
    _plot_delta(
        dataset_dir / "delta_ctr_bootstrap.pdf",
        dataset,
        reference_name.upper(),
        method_name.upper(),
        bootstrap_rows,
        statistics,
    )

    summary = {
        "dataset": dataset,
        "voltage_V": voltage_from_name(dataset),
        "stage": stage,
        "reference": reference_name,
        "method": method_name,
        "bootstrap_requested": repeats,
        **counts,
        "fit_bin_width_ps": float(fit_config.get("bin_width_ps", np.nan)),
        "full_reference_ctr_ps": float(full_reference),
        "full_method_ctr_ps": float(full_method),
        "full_delta_ctr_ps": float(full_method - full_reference),
        **statistics,
    }
    with (dataset_dir / "summary.json").open("w", encoding="utf-8") as stream:
        json.dump(summary, stream, indent=2, allow_nan=True)
        stream.write("\n")
    return summary


def main() -> None:
    args = parse_args()
    run = args.run_dir.resolve()
    manifest_path = run / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Missing study manifest: {manifest_path}")
    manifest = _read_json(manifest_path)
    fit_config = _fit_config(manifest)
    repeats = int(args.bootstrap_samples if args.bootstrap_samples is not None else fit_config.get("bootstrap_samples", 100))
    if repeats < 2:
        raise ValueError("bootstrap-samples must be >= 2")
    base_seed = int(args.seed if args.seed is not None else _default_seed(manifest))

    datasets = _discover_datasets(run, args.reference, args.method, args.stage)
    if args.dataset:
        wanted = set(args.dataset)
        datasets = [dataset for dataset in datasets if dataset in wanted]
        missing = wanted - set(datasets)
        if missing:
            raise FileNotFoundError(
                f"Requested dataset(s) missing paired {args.reference}/{args.method} {args.stage} residual artifacts: {sorted(missing)}"
            )
    if not datasets:
        raise FileNotFoundError(
            f"No datasets contain both {args.reference}_{args.stage}_residuals_ps.npy and "
            f"{args.method}_{args.stage}_residuals_ps.npy"
        )

    output_root = (
        args.output_dir
        or run / "bootstrap_ctr_correlation" / f"{args.reference}_vs_{args.method}"
    ).resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    summaries = []
    for dataset in datasets:
        print(f"Paired CTR bootstrap {dataset} | {args.reference} vs {args.method} | stage={args.stage} ...")
        summary = analyse_dataset(
            run,
            dataset,
            args.reference,
            args.method,
            args.stage,
            fit_config,
            repeats,
            _dataset_seed(base_seed, dataset, args.reference, args.method),
            output_root,
        )
        summaries.append(summary)
        print(
            f"  r={summary['pearson_r']:+.3f} | cov={summary['covariance_ps2']:+.2f} ps^2 | "
            f"sigma_delta paired={summary['delta_bootstrap_std_ps']:.2f} ps vs independent="
            f"{summary['independent_delta_uncertainty_ps']:.2f} ps | "
            f"P({args.method} better)={100.0 * summary['method_better_fraction']:.1f}%"
        )

    _write_rows(output_root / "summary.csv", summaries)
    _plot_summary(output_root / "correlation_vs_voltage.pdf", summaries, args.method.upper())
    print(f"Outputs: {output_root}")


if __name__ == "__main__":
    main()
