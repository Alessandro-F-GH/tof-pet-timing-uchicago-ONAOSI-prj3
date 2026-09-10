#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import linregress, pearsonr, spearmanr

from waveform_analysis.ml_pipeline.dataset import PreparedDataset, load_prepared_dataset
from waveform_analysis.ml_pipeline.view import calibrated_led, inverse_pair, waveform_view

WIDTH_FACTOR = 2.35


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Relate event-wise waveform-channel distance to absolute calibrated LED timing error. "
            "For this diagnostic only, timing width is defined directly as 2.35 * sample sigma."
        )
    )
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--dataset", action="append", default=None)
    parser.add_argument(
        "--stage",
        choices=("training", "validation", "development", "test"),
        default="development",
    )
    parser.add_argument("--window-ns", type=float, nargs=2, metavar=("START", "STOP"), default=None)
    parser.add_argument("--distance", choices=("rms", "mean_abs", "max_abs"), default="rms")
    parser.add_argument("--trend-bins", type=int, default=12)
    parser.add_argument(
        "--efficiency-step-percent",
        type=float,
        default=1.0,
        help="Accepted-event efficiency step. Default: 1%%.",
    )
    parser.add_argument(
        "--bootstrap-samples",
        type=int,
        default=None,
        help="Bootstrap repeats for the uncertainty of 2.35*sigma. Default: study fit.bootstrap_samples.",
    )
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser.parse_args()


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return value


def _stage_indices(dataset: PreparedDataset, stage: str) -> np.ndarray:
    if stage == "training":
        return np.asarray(dataset.training, dtype=np.int64)
    if stage == "validation":
        return np.asarray(dataset.validation, dtype=np.int64)
    if stage == "development":
        return np.asarray(dataset.development, dtype=np.int64)
    if stage == "test":
        return np.asarray(dataset.test, dtype=np.int64)
    raise ValueError(stage)


def _time_mask(time_ps: np.ndarray, window_ns: tuple[float, float] | None) -> np.ndarray:
    time_ns = np.asarray(time_ps, dtype=np.float64) / 1000.0
    if window_ns is None:
        return np.ones(time_ns.size, dtype=bool)
    start, stop = map(float, window_ns)
    if stop <= start:
        raise ValueError("--window-ns must satisfy START < STOP")
    mask = (time_ns >= start) & (time_ns <= stop)
    if np.count_nonzero(mask) < 2:
        raise ValueError(f"Window [{start:g}, {stop:g}] ns contains fewer than two prepared samples")
    return mask


def _distance(difference: np.ndarray, metric: str) -> np.ndarray:
    if metric == "rms":
        return np.sqrt(np.mean(difference**2, axis=1))
    if metric == "mean_abs":
        return np.mean(np.abs(difference), axis=1)
    if metric == "max_abs":
        return np.max(np.abs(difference), axis=1)
    raise ValueError(metric)


def _waveform_distance(
    dataset: PreparedDataset,
    mode: str,
    indices: np.ndarray,
    *,
    metric: str,
    window_ns: tuple[float, float] | None,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    first = waveform_view(dataset, mode, indices[:1])
    time_ps = np.asarray(first.time_ps, dtype=np.float64)
    mask = _time_mask(time_ps, window_ns)
    values = np.full(indices.size, np.nan, dtype=np.float64)

    for start in range(0, indices.size, int(batch_size)):
        stop = min(indices.size, start + int(batch_size))
        view = waveform_view(dataset, mode, indices[start:stop])
        physical = inverse_pair(dataset, mode, view.materialize())
        difference = np.asarray(physical[:, 0, mask] - physical[:, 1, mask], dtype=np.float64)
        finite = np.all(np.isfinite(difference), axis=1)
        if np.any(finite):
            values[start:stop][finite] = _distance(difference[finite], metric)
    return values, time_ps[mask]


def _source_metadata(dataset: PreparedDataset, indices: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    source_dataset_path = dataset.directory / "source_dataset.npy"
    source_event_path = dataset.directory / "source_event_index.npy"
    if source_dataset_path.is_file():
        source_dataset = np.asarray(np.load(source_dataset_path, mmap_mode="r")[indices]).astype(str)
    else:
        source = Path(str(dataset.manifest.get("source", dataset.directory.name))).stem
        source_dataset = np.full(indices.size, source, dtype="U128")
    if source_event_path.is_file():
        source_event = np.asarray(np.load(source_event_path, mmap_mode="r")[indices], dtype=np.int64)
    else:
        source_event = np.asarray(dataset.event_index[indices], dtype=np.int64)
    return source_dataset, source_event


def _correlation(distance: np.ndarray, absolute_error: np.ndarray) -> dict[str, Any]:
    finite = np.isfinite(distance) & np.isfinite(absolute_error)
    x = np.asarray(distance[finite], dtype=np.float64)
    y = np.asarray(absolute_error[finite], dtype=np.float64)
    result = {
        "n": int(x.size),
        "distance_mean_mV": float(np.mean(x)),
        "distance_std_mV": float(np.std(x)),
        "abs_led_error_mean_ps": float(np.mean(y)),
        "abs_led_error_median_ps": float(np.median(y)),
        "pearson_r": float("nan"),
        "pearson_p": float("nan"),
        "spearman_rho": float("nan"),
        "spearman_p": float("nan"),
        "linear_slope_ps_per_mV": float("nan"),
        "linear_intercept_ps": float("nan"),
        "linear_r_squared": float("nan"),
    }
    if x.size >= 3 and np.std(x) > 0.0 and np.std(y) > 0.0:
        p = pearsonr(x, y)
        s = spearmanr(x, y)
        r = linregress(x, y)
        result.update(
            pearson_r=float(p.statistic),
            pearson_p=float(p.pvalue),
            spearman_rho=float(s.statistic),
            spearman_p=float(s.pvalue),
            linear_slope_ps_per_mV=float(r.slope),
            linear_intercept_ps=float(r.intercept),
            linear_r_squared=float(r.rvalue**2),
        )
    return result


def _binned_trend(distance: np.ndarray, absolute_error: np.ndarray, bins: int) -> list[dict[str, Any]]:
    finite = np.isfinite(distance) & np.isfinite(absolute_error)
    x = np.asarray(distance[finite], dtype=np.float64)
    y = np.asarray(absolute_error[finite], dtype=np.float64)
    edges = np.unique(np.quantile(x, np.linspace(0.0, 1.0, max(2, int(bins)) + 1)))
    rows = []
    for i, (left, right) in enumerate(zip(edges[:-1], edges[1:])):
        mask = (x >= left) & (x <= right if i == len(edges) - 2 else x < right)
        if not np.any(mask):
            continue
        values = y[mask]
        rows.append(
            {
                "bin": i + 1,
                "n": int(np.count_nonzero(mask)),
                "distance_low_mV": float(left),
                "distance_high_mV": float(right),
                "distance_median_mV": float(np.median(x[mask])),
                "abs_led_error_mean_ps": float(np.mean(values)),
                "abs_led_error_median_ps": float(np.median(values)),
                "abs_led_error_q16_ps": float(np.quantile(values, 0.16)),
                "abs_led_error_q84_ps": float(np.quantile(values, 0.84)),
            }
        )
    return rows


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _plot_relation(path: Path, dataset_name: str, stage: str, metric: str, distance: np.ndarray,
                   absolute_error: np.ndarray, trend: list[dict[str, Any]], stats: dict[str, Any]) -> None:
    finite = np.isfinite(distance) & np.isfinite(absolute_error)
    x, y = distance[finite], absolute_error[finite]
    fig, ax = plt.subplots(figsize=(8.4, 5.6))
    ax.scatter(x, y, s=8, alpha=0.12, label="events")
    if trend:
        tx = np.asarray([r["distance_median_mV"] for r in trend])
        ty = np.asarray([r["abs_led_error_median_ps"] for r in trend])
        low = ty - np.asarray([r["abs_led_error_q16_ps"] for r in trend])
        high = np.asarray([r["abs_led_error_q84_ps"] for r in trend]) - ty
        ax.errorbar(tx, ty, yerr=np.vstack([low, high]), marker="o", capsize=3,
                    label="distance-bin median ± 16–84%")
    ax.set_xlabel(f"Waveform-channel {metric} distance [mV]")
    ax.set_ylabel("Absolute calibrated LED error [ps]")
    ax.set_title(f"{dataset_name} · {stage} · waveform distance vs |LED error|")
    ax.grid(alpha=0.2)
    ax.legend(loc="best")
    ax.text(
        0.02, 0.98,
        f"n={stats['n']}\nPearson r={stats['pearson_r']:+.3f}\n"
        f"Spearman ρ={stats['spearman_rho']:+.3f}\nlinear R²={stats['linear_r_squared']:.3f}",
        transform=ax.transAxes, ha="left", va="top",
        bbox={"boxstyle": "round,pad=0.3", "facecolor": "white", "alpha": 0.9},
    )
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def _width(values: np.ndarray) -> float:
    x = np.asarray(values, dtype=np.float64)
    x = x[np.isfinite(x)]
    if x.size < 2:
        raise ValueError("At least two finite values are required")
    return float(WIDTH_FACTOR * np.std(x, ddof=1))


def _width_with_bootstrap(values: np.ndarray, *, samples: int, seed: int) -> tuple[float, float, int]:
    x = np.asarray(values, dtype=np.float64)
    x = x[np.isfinite(x)]
    central = _width(x)
    if samples < 2:
        return central, float("nan"), 0
    rng = np.random.default_rng(seed)
    estimates = np.asarray(
        [_width(x[rng.integers(0, x.size, size=x.size)]) for _ in range(samples)],
        dtype=np.float64,
    )
    estimates = estimates[np.isfinite(estimates)]
    uncertainty = float(np.std(estimates, ddof=1)) if estimates.size > 1 else float("nan")
    return central, uncertainty, int(estimates.size)


def _efficiency_grid(step_percent: float) -> np.ndarray:
    step = float(step_percent)
    if not 0.0 < step <= 100.0:
        raise ValueError("--efficiency-step-percent must satisfy 0 < STEP <= 100")
    return np.append(np.arange(step, 100.0, step, dtype=np.float64), 100.0)


def _selection_scan(distance: np.ndarray, signed_error: np.ndarray, *, minimum: int,
                    efficiency_step_percent: float, bootstrap_samples: int, seed: int) -> list[dict[str, Any]]:
    finite = np.isfinite(distance) & np.isfinite(signed_error)
    x = np.asarray(distance[finite], dtype=np.float64)
    error = np.asarray(signed_error[finite], dtype=np.float64)
    rows = []
    for i, requested_percent in enumerate(_efficiency_grid(efficiency_step_percent)):
        threshold = float(np.quantile(x, requested_percent / 100.0))
        accepted = x <= threshold
        n = int(np.count_nonzero(accepted))
        if n < minimum:
            continue
        width, uncertainty, successful = _width_with_bootstrap(
            error[accepted], samples=bootstrap_samples, seed=seed + i
        )
        rows.append(
            {
                "requested_efficiency_percent": float(requested_percent),
                "efficiency_percent": float(100.0 * n / x.size),
                "distance_threshold_mV": threshold,
                "n_accepted": n,
                "n_total": int(x.size),
                "width_2p35sigma_ps": width,
                "width_uncertainty_ps": uncertainty,
                "bootstrap_successful": successful,
                "mean_abs_led_error_ps": float(np.mean(np.abs(error[accepted]))),
                "median_abs_led_error_ps": float(np.median(np.abs(error[accepted]))),
            }
        )
    return rows


def _plot_selection_scan(path: Path, dataset_name: str, stage: str, metric: str,
                         rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    efficiency = np.asarray([r["efficiency_percent"] for r in rows])
    width = np.asarray([r["width_2p35sigma_ps"] for r in rows])
    uncertainty = np.asarray([r["width_uncertainty_ps"] for r in rows])
    fig, ax = plt.subplots(figsize=(8.2, 5.0))
    ax.errorbar(efficiency, width, yerr=uncertainty, marker=".", ms=4, capsize=2, lw=1.0)
    ax.set_xlabel("Accepted events with lowest waveform distance [%]")
    ax.set_ylabel(r"Timing width $2.35\sigma$ [ps]")
    ax.set_title(f"{dataset_name} · {stage} · low-{metric}-distance selection")
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def _selected_values(distance: np.ndarray, signed_error: np.ndarray, threshold: float) -> np.ndarray:
    finite = np.isfinite(distance) & np.isfinite(signed_error)
    return np.asarray(signed_error[finite & (distance <= threshold)], dtype=np.float64)


def _plot_best_distribution(path: Path, dataset_name: str, stage: str, metric: str,
                            values: np.ndarray, best: dict[str, Any], *, bootstrap_samples: int,
                            seed: int) -> dict[str, Any]:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    width, uncertainty, successful = _width_with_bootstrap(values, samples=bootstrap_samples, seed=seed)
    mean = float(np.mean(values))
    sigma = float(np.std(values, ddof=1))

    # The histogram is visualization only. The metric is computed directly from the samples.
    low, high = np.quantile(values, [0.005, 0.995])
    if high <= low:
        low, high = float(np.min(values)), float(np.max(values))
    margin = 0.06 * max(float(high - low), 1.0)
    edges = np.linspace(float(low - margin), float(high + margin), 31)
    counts, edges = np.histogram(values, bins=edges)
    centers = 0.5 * (edges[:-1] + edges[1:])
    bin_width = float(edges[1] - edges[0])
    gaussian = (
        values.size * bin_width / (sigma * np.sqrt(2.0 * np.pi))
        * np.exp(-0.5 * ((centers - mean) / sigma) ** 2)
    ) if sigma > 0.0 else np.zeros_like(centers)

    fig, ax = plt.subplots(figsize=(8.4, 5.1))
    ax.hist(values, bins=edges, histtype="stepfilled", alpha=0.42, edgecolor="black", label="selected events")
    ax.plot(centers, gaussian, lw=1.8, label="Gaussian with sample mean and σ")
    ax.set_xlabel("Calibrated LED timing residual [ps]")
    ax.set_ylabel("Events / display bin")
    ax.set_title(f"{dataset_name} · best low-{metric}-distance selection ({stage})")
    ax.grid(axis="y", alpha=0.2)
    ax.legend(loc="best")
    ax.text(
        0.02, 0.97,
        f"efficiency = {float(best['efficiency_percent']):.1f}%\n"
        f"distance ≤ {float(best['distance_threshold_mV']):.4g} mV\n"
        f"n = {int(best['n_accepted'])}\nσ = {sigma:.2f} ps\n"
        f"2.35σ = {width:.2f} ± {uncertainty:.2f} ps",
        transform=ax.transAxes, ha="left", va="top",
        bbox={"boxstyle": "round,pad=0.3", "facecolor": "white", "alpha": 0.9},
    )
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)

    return {
        **best,
        "sigma_ps": sigma,
        "recomputed_width_2p35sigma_ps": width,
        "recomputed_width_uncertainty_ps": uncertainty,
        "bootstrap_successful": successful,
    }


def _remove_stale_outputs(directory: Path) -> None:
    # Earlier revisions wrote histogram-FWHM diagnostics under these names. Remove them so an
    # old PDF cannot be mistaken for output from the current sigma-only analysis.
    for name in (
        "led_ctr_vs_low_distance_efficiency.pdf",
        "best_low_distance_ctr_distribution.pdf",
    ):
        path = directory / name
        if path.is_file():
            path.unlink()


def _per_voltage_summary(distance: np.ndarray, absolute_error: np.ndarray,
                         voltage: np.ndarray) -> list[dict[str, Any]]:
    rows = []
    voltage = np.asarray(voltage, dtype=np.float64)
    for value in np.unique(voltage[np.isfinite(voltage)]):
        mask = np.isclose(voltage, value, rtol=0.0, atol=1e-9)
        rows.append({"voltage_V": float(value), **_correlation(distance[mask], absolute_error[mask])})
    return rows


def analyse_dataset(run: Path, manifest: dict[str, Any], dataset_name: str,
                    args: argparse.Namespace, output_root: Path) -> None:
    prepared_dir = Path(manifest["datasets"][dataset_name]["prepared_dir"])
    dataset = load_prepared_dataset(prepared_dir)
    mode = str(manifest.get("mode") or manifest["config"]["mode"])
    indices = _stage_indices(dataset, args.stage)
    if indices.size == 0:
        raise RuntimeError(f"{dataset_name}: {args.stage} split is empty")

    window_ns = None if args.window_ns is None else tuple(map(float, args.window_ns))
    distance, distance_time_ps = _waveform_distance(
        dataset, mode, indices,
        metric=args.distance, window_ns=window_ns, batch_size=int(args.batch_size),
    )
    signed_error = np.asarray(calibrated_led(dataset, mode)[indices], dtype=np.float64)
    absolute_error = np.abs(signed_error)
    finite = np.isfinite(distance) & np.isfinite(signed_error)
    if np.count_nonzero(finite) < 3:
        raise RuntimeError(f"{dataset_name}: fewer than three finite distance/error pairs")

    source_dataset, source_event = _source_metadata(dataset, indices)
    voltage = np.asarray(dataset.bias_voltage_V[indices], dtype=np.float64)
    stats = _correlation(distance, absolute_error)
    trend = _binned_trend(distance, absolute_error, int(args.trend_bins))

    config = manifest.get("config") or {}
    fit_config = dict(config.get("fit") or {})
    minimum = int(fit_config.get("min_events", 100))
    bootstrap_samples = int(
        args.bootstrap_samples if args.bootstrap_samples is not None
        else fit_config.get("bootstrap_samples", 100)
    )
    seed = int((config.get("validation") or {}).get("seed", 0))
    scan = _selection_scan(
        distance, signed_error,
        minimum=minimum,
        efficiency_step_percent=float(args.efficiency_step_percent),
        bootstrap_samples=bootstrap_samples,
        seed=seed,
    )

    dataset_dir = output_root / dataset_name
    dataset_dir.mkdir(parents=True, exist_ok=True)
    _remove_stale_outputs(dataset_dir)
    _plot_relation(
        dataset_dir / "waveform_distance_vs_abs_led_error.pdf",
        dataset_name, args.stage, args.distance, distance, absolute_error, trend, stats,
    )
    _plot_selection_scan(
        dataset_dir / "led_2p35sigma_vs_low_distance_efficiency.pdf",
        dataset_name, args.stage, args.distance, scan,
    )

    best = None
    eligible = [r for r in scan if float(r["efficiency_percent"]) > 10.0]
    if eligible:
        best_scan = min(eligible, key=lambda r: float(r["width_2p35sigma_ps"]))
        selected = _selected_values(distance, signed_error, float(best_scan["distance_threshold_mV"]))
        best = _plot_best_distribution(
            dataset_dir / "best_low_distance_2p35sigma_distribution.pdf",
            dataset_name, args.stage, args.distance, selected, best_scan,
            bootstrap_samples=bootstrap_samples, seed=seed + 100000,
        )
        (dataset_dir / "best_low_distance_selection.json").write_text(
            json.dumps(best, indent=2, allow_nan=True) + "\n", encoding="utf-8"
        )

    event_rows = [
        {
            "position": i,
            "prepared_index": int(indices[i]),
            "source_dataset": str(source_dataset[i]),
            "source_event_index": int(source_event[i]),
            "bias_voltage_V": float(voltage[i]),
            "waveform_distance_mV": float(distance[i]),
            "led_error_ps": float(signed_error[i]),
            "abs_led_error_ps": float(absolute_error[i]),
        }
        for i in range(indices.size)
    ]
    _write_csv(dataset_dir / "events.csv", event_rows)
    _write_csv(dataset_dir / "distance_binned_error.csv", trend)
    _write_csv(dataset_dir / "low_distance_selection_scan.csv", scan)
    per_voltage = _per_voltage_summary(distance, absolute_error, voltage)
    _write_csv(dataset_dir / "correlation_by_voltage.csv", per_voltage)

    actual_window = (
        [float(distance_time_ps[0] / 1000.0), float(distance_time_ps[-1] / 1000.0)]
        if distance_time_ps.size else [float("nan"), float("nan")]
    )
    summary = {
        "dataset": dataset_name,
        "stage": args.stage,
        "mode": mode,
        "prepared_dir": str(prepared_dir),
        "distance_metric": args.distance,
        "distance_definition": {
            "rms": "sqrt(mean_t((s1-s2)^2))",
            "mean_abs": "mean_t(abs(s1-s2))",
            "max_abs": "max_t(abs(s1-s2))",
        }[args.distance],
        "signal_units": "physical mV after inverse prepared-input normalization",
        "requested_window_ns": None if window_ns is None else list(window_ns),
        "actual_sample_window_ns": actual_window,
        "led_error_definition": "abs(calibrated_led_pair_residual_ps)",
        "resolution_estimator": "2.35 * sample standard deviation (ddof=1), using all finite selected residuals",
        "histogram_used_for_resolution": False,
        "efficiency_step_percent": float(args.efficiency_step_percent),
        "n_requested": int(indices.size),
        "n_finite_pairs": int(np.count_nonzero(finite)),
        "correlation": stats,
        "per_voltage": per_voltage,
        "best_selection_above_10_percent": best,
        "selection_scan_note": (
            "This standalone diagnostic uses 2.35*sigma only. The histogram in the best-selection "
            "figure is display-only. Choose any cut on development and freeze it before blind evaluation."
        ),
    }
    (dataset_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, allow_nan=True) + "\n", encoding="utf-8"
    )

    print(
        f"{dataset_name} | stage={args.stage} | n={stats['n']} | "
        f"Pearson r={stats['pearson_r']:+.4f} | Spearman rho={stats['spearman_rho']:+.4f}"
    )
    if best is not None:
        print(
            f"  best selection >10%: 2.35*sigma={best['recomputed_width_2p35sigma_ps']:.2f} ± "
            f"{best['recomputed_width_uncertainty_ps']:.2f} ps | "
            f"efficiency={best['efficiency_percent']:.1f}% | "
            f"distance <= {best['distance_threshold_mV']:.4g} mV"
        )
    print(f"  outputs: {dataset_dir}")


def main() -> None:
    args = parse_args()
    if int(args.batch_size) < 1:
        raise ValueError("--batch-size must be positive")
    if int(args.trend_bins) < 2:
        raise ValueError("--trend-bins must be >= 2")
    if not 0.0 < float(args.efficiency_step_percent) <= 100.0:
        raise ValueError("--efficiency-step-percent must satisfy 0 < STEP <= 100")
    if args.bootstrap_samples is not None and int(args.bootstrap_samples) < 0:
        raise ValueError("--bootstrap-samples must be non-negative")

    run = args.run_dir.resolve()
    manifest = _read_json(run / "manifest.json")
    available = list((manifest.get("datasets") or {}).keys())
    if args.dataset:
        wanted = set(args.dataset)
        datasets = [name for name in available if name in wanted]
        missing = wanted - set(datasets)
        if missing:
            raise FileNotFoundError(f"Requested study dataset(s) not found: {sorted(missing)}")
    else:
        datasets = available
    if not datasets:
        raise FileNotFoundError("Study manifest contains no datasets")

    if args.stage == "test":
        print(
            "WARNING: analysing the blind/test population. Treat this as confirmation only; "
            "do not tune a distance cut on these results."
        )

    output_root = (args.output_dir or run / "waveform_distance_led_error").resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    for dataset_name in datasets:
        analyse_dataset(run, manifest, dataset_name, args, output_root)


if __name__ == "__main__":
    main()
