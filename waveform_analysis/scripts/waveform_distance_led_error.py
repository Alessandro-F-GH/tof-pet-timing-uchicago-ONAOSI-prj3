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

from utils_fit import fit_ctr_ps
from waveform_analysis.ml_pipeline.dataset import PreparedDataset, load_prepared_dataset
from waveform_analysis.ml_pipeline.view import calibrated_led, inverse_pair, waveform_view


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Measure the event-wise distance between the two aligned waveform channels and test "
            "its relation to the absolute calibrated LED timing error. The default distance is "
            "the RMS of the physical-mV difference s1(t)-s2(t) over the full ML window."
        )
    )
    parser.add_argument("--run-dir", type=Path, required=True, help="Completed study directory.")
    parser.add_argument(
        "--dataset",
        action="append",
        default=None,
        help="Optional study dataset name. May be repeated. Default: all datasets in the study.",
    )
    parser.add_argument(
        "--stage",
        choices=("training", "validation", "development", "test"),
        default="development",
        help="Population to analyse. Default: development (training + validation).",
    )
    parser.add_argument(
        "--window-ns",
        type=float,
        nargs=2,
        metavar=("START", "STOP"),
        default=None,
        help="Optional time window for the distance. Default: the complete prepared ML window.",
    )
    parser.add_argument(
        "--distance",
        choices=("rms", "mean_abs", "max_abs"),
        default="rms",
        help="Waveform-channel distance metric. Default: rms.",
    )
    parser.add_argument(
        "--trend-bins",
        type=int,
        default=12,
        help="Equal-count distance bins for the median |LED error| trend. Default: 12.",
    )
    parser.add_argument(
        "--efficiency-points",
        type=int,
        default=10,
        help="Number of low-distance acceptance points from 10%% to 100%% for the CTR scan. Default: 10.",
    )
    parser.add_argument(
        "--bootstrap-samples",
        type=int,
        default=None,
        help="CTR bootstrap repeats for the efficiency scan. Default: fit.bootstrap_samples from the study.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=512,
        help="Waveform batch size. Default: 512.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Default: <run-dir>/waveform_distance_led_error/.",
    )
    return parser.parse_args()


def _read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as stream:
        value = json.load(stream)
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
        raise ValueError(
            f"Requested distance window [{start:g}, {stop:g}] ns contains fewer than two prepared samples"
        )
    return mask


def _distance_from_difference(difference: np.ndarray, metric: str) -> np.ndarray:
    difference = np.asarray(difference, dtype=np.float64)
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
    if indices.size == 0:
        return np.empty(0, dtype=np.float64), np.empty(0, dtype=np.float64)
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
            batch_values = np.full(difference.shape[0], np.nan, dtype=np.float64)
            batch_values[finite] = _distance_from_difference(difference[finite], metric)
            values[start:stop] = batch_values
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


def _correlation(distance: np.ndarray, absolute_error: np.ndarray) -> dict[str, float | int]:
    distance = np.asarray(distance, dtype=np.float64)
    absolute_error = np.asarray(absolute_error, dtype=np.float64)
    finite = np.isfinite(distance) & np.isfinite(absolute_error)
    x, y = distance[finite], absolute_error[finite]
    result: dict[str, float | int] = {
        "n": int(x.size),
        "distance_mean_mV": float(np.mean(x)) if x.size else float("nan"),
        "distance_std_mV": float(np.std(x)) if x.size else float("nan"),
        "abs_led_error_mean_ps": float(np.mean(y)) if y.size else float("nan"),
        "abs_led_error_median_ps": float(np.median(y)) if y.size else float("nan"),
        "pearson_r": float("nan"),
        "pearson_p": float("nan"),
        "spearman_rho": float("nan"),
        "spearman_p": float("nan"),
        "linear_slope_ps_per_mV": float("nan"),
        "linear_intercept_ps": float("nan"),
        "linear_r_squared": float("nan"),
    }
    if x.size < 3 or np.std(x) == 0.0 or np.std(y) == 0.0:
        return result
    pearson = pearsonr(x, y)
    spearman = spearmanr(x, y)
    regression = linregress(x, y)
    result.update(
        pearson_r=float(pearson.statistic),
        pearson_p=float(pearson.pvalue),
        spearman_rho=float(spearman.statistic),
        spearman_p=float(spearman.pvalue),
        linear_slope_ps_per_mV=float(regression.slope),
        linear_intercept_ps=float(regression.intercept),
        linear_r_squared=float(regression.rvalue**2),
    )
    return result


def _binned_trend(distance: np.ndarray, absolute_error: np.ndarray, bins: int) -> list[dict[str, float | int]]:
    finite = np.isfinite(distance) & np.isfinite(absolute_error)
    x = np.asarray(distance[finite], dtype=np.float64)
    y = np.asarray(absolute_error[finite], dtype=np.float64)
    if x.size == 0:
        return []
    bins = max(2, min(int(bins), int(x.size)))
    edges = np.unique(np.quantile(x, np.linspace(0.0, 1.0, bins + 1)))
    rows = []
    for index, (left, right) in enumerate(zip(edges[:-1], edges[1:])):
        mask = (x >= left) & (x <= right if index == len(edges) - 2 else x < right)
        if not np.any(mask):
            continue
        values = y[mask]
        rows.append(
            {
                "bin": index + 1,
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


def _plot_relation(
    path: Path,
    dataset_name: str,
    stage: str,
    metric: str,
    distance: np.ndarray,
    absolute_error: np.ndarray,
    trend: list[dict[str, Any]],
    statistics: dict[str, Any],
) -> None:
    finite = np.isfinite(distance) & np.isfinite(absolute_error)
    x, y = distance[finite], absolute_error[finite]
    fig, ax = plt.subplots(figsize=(8.4, 5.6))
    ax.scatter(x, y, s=8, alpha=0.12, label="events")
    if trend:
        tx = np.asarray([row["distance_median_mV"] for row in trend], dtype=float)
        ty = np.asarray([row["abs_led_error_median_ps"] for row in trend], dtype=float)
        low = ty - np.asarray([row["abs_led_error_q16_ps"] for row in trend], dtype=float)
        high = np.asarray([row["abs_led_error_q84_ps"] for row in trend], dtype=float) - ty
        ax.errorbar(tx, ty, yerr=np.vstack([low, high]), marker="o", capsize=3, lw=1.4, label="distance-bin median ± 16–84%")
    ax.set_xlabel(f"Waveform-channel {metric} distance [mV]")
    ax.set_ylabel("Absolute calibrated LED error [ps]")
    ax.set_title(f"{dataset_name} · {stage} · waveform distance vs |LED error|")
    ax.grid(alpha=0.2)
    ax.legend(loc="best")
    ax.text(
        0.02,
        0.98,
        (
            f"n={statistics['n']}\n"
            f"Pearson r={statistics['pearson_r']:+.3f}\n"
            f"Spearman ρ={statistics['spearman_rho']:+.3f}\n"
            f"linear R²={statistics['linear_r_squared']:.3f}"
        ),
        transform=ax.transAxes,
        ha="left",
        va="top",
        bbox={"boxstyle": "round,pad=0.3", "facecolor": "white", "alpha": 0.9},
    )
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def _selection_scan(
    distance: np.ndarray,
    signed_led_error: np.ndarray,
    fit_config: dict[str, Any],
    *,
    points: int,
    bootstrap_samples: int,
    seed: int,
) -> list[dict[str, Any]]:
    finite = np.isfinite(distance) & np.isfinite(signed_led_error)
    x = np.asarray(distance[finite], dtype=np.float64)
    error = np.asarray(signed_led_error[finite], dtype=np.float64)
    minimum = int(fit_config.get("min_events", 100))
    if x.size < minimum:
        return []
    fractions = np.linspace(0.1, 1.0, max(2, int(points)))
    rows = []
    for index, requested_fraction in enumerate(fractions):
        threshold = float(np.quantile(x, requested_fraction))
        accepted = x <= threshold
        n = int(np.count_nonzero(accepted))
        if n < minimum:
            continue
        local_fit = dict(fit_config)
        local_fit["bootstrap_samples"] = int(bootstrap_samples)
        try:
            result = fit_ctr_ps(error[accepted], local_fit, seed=int(seed + index), bootstrap=True)
        except ValueError:
            continue
        rows.append(
            {
                "requested_efficiency_percent": float(100.0 * requested_fraction),
                "efficiency_percent": float(100.0 * n / x.size),
                "distance_threshold_mV": threshold,
                "n_accepted": n,
                "n_total": int(x.size),
                "ctr_ps": float(result.ctr_ps),
                "ctr_uncertainty_ps": float(result.ctr_error_ps),
                "bootstrap_successful": int(result.bootstrap_successful),
                "mean_abs_led_error_ps": float(np.mean(np.abs(error[accepted]))),
                "median_abs_led_error_ps": float(np.median(np.abs(error[accepted]))),
            }
        )
    return rows


def _plot_selection_scan(path: Path, dataset_name: str, stage: str, metric: str, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    efficiency = np.asarray([row["efficiency_percent"] for row in rows], dtype=float)
    ctr = np.asarray([row["ctr_ps"] for row in rows], dtype=float)
    uncertainty = np.asarray([row["ctr_uncertainty_ps"] for row in rows], dtype=float)
    fig, ax = plt.subplots(figsize=(7.8, 5.0))
    ax.errorbar(efficiency, ctr, yerr=uncertainty, marker="o", capsize=3)
    ax.set_xlabel("Accepted events with lowest waveform distance [%]")
    ax.set_ylabel("LED CTR FWHM [ps]")
    ax.set_title(f"{dataset_name} · {stage} · low-{metric}-distance selection")
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def _per_voltage_summary(distance: np.ndarray, absolute_error: np.ndarray, voltage: np.ndarray) -> list[dict[str, Any]]:
    rows = []
    voltage = np.asarray(voltage, dtype=np.float64)
    for value in np.unique(voltage[np.isfinite(voltage)]):
        mask = np.isclose(voltage, value, rtol=0.0, atol=1e-9)
        stats = _correlation(distance[mask], absolute_error[mask])
        rows.append({"voltage_V": float(value), **stats})
    return rows


def analyse_dataset(
    run: Path,
    manifest: dict[str, Any],
    dataset_name: str,
    args: argparse.Namespace,
    output_root: Path,
) -> None:
    dataset_info = manifest.get("datasets", {}).get(dataset_name)
    if not isinstance(dataset_info, dict):
        raise KeyError(f"Study manifest has no dataset {dataset_name!r}")
    prepared_dir = Path(dataset_info["prepared_dir"])
    dataset = load_prepared_dataset(prepared_dir)
    mode = str(manifest.get("mode") or manifest["config"]["mode"])
    indices = _stage_indices(dataset, args.stage)
    if indices.size == 0:
        raise RuntimeError(f"{dataset_name}: {args.stage} split is empty")

    window_ns = None if args.window_ns is None else tuple(map(float, args.window_ns))
    distance, distance_time_ps = _waveform_distance(
        dataset,
        mode,
        indices,
        metric=args.distance,
        window_ns=window_ns,
        batch_size=int(args.batch_size),
    )
    signed_error = np.asarray(calibrated_led(dataset, mode)[indices], dtype=np.float64)
    absolute_error = np.abs(signed_error)
    finite = np.isfinite(distance) & np.isfinite(signed_error)
    if np.count_nonzero(finite) < 3:
        raise RuntimeError(f"{dataset_name}: fewer than three finite distance/LED-error pairs")

    source_dataset, source_event = _source_metadata(dataset, indices)
    voltage = np.asarray(dataset.bias_voltage_V[indices], dtype=np.float64)
    statistics = _correlation(distance, absolute_error)
    trend = _binned_trend(distance, absolute_error, int(args.trend_bins))

    config = manifest.get("config") or {}
    fit_config = dict(config.get("fit") or {})
    bootstrap_samples = int(
        args.bootstrap_samples
        if args.bootstrap_samples is not None
        else fit_config.get("bootstrap_samples", 100)
    )
    seed = int((config.get("validation") or {}).get("seed", 0))
    scan = _selection_scan(
        distance,
        signed_error,
        fit_config,
        points=int(args.efficiency_points),
        bootstrap_samples=bootstrap_samples,
        seed=seed,
    )

    dataset_dir = output_root / dataset_name
    dataset_dir.mkdir(parents=True, exist_ok=True)
    _plot_relation(
        dataset_dir / "waveform_distance_vs_abs_led_error.pdf",
        dataset_name,
        args.stage,
        args.distance,
        distance,
        absolute_error,
        trend,
        statistics,
    )
    _plot_selection_scan(
        dataset_dir / "led_ctr_vs_low_distance_efficiency.pdf",
        dataset_name,
        args.stage,
        args.distance,
        scan,
    )

    event_rows = []
    for position in range(indices.size):
        event_rows.append(
            {
                "position": position,
                "prepared_index": int(indices[position]),
                "source_dataset": str(source_dataset[position]),
                "source_event_index": int(source_event[position]),
                "bias_voltage_V": float(voltage[position]),
                "waveform_distance_mV": float(distance[position]),
                "led_error_ps": float(signed_error[position]),
                "abs_led_error_ps": float(absolute_error[position]),
            }
        )
    _write_csv(dataset_dir / "events.csv", event_rows)
    _write_csv(dataset_dir / "distance_binned_error.csv", trend)
    _write_csv(dataset_dir / "low_distance_selection_scan.csv", scan)

    per_voltage = _per_voltage_summary(distance, absolute_error, voltage)
    _write_csv(dataset_dir / "correlation_by_voltage.csv", per_voltage)

    if distance_time_ps.size:
        actual_window = [float(distance_time_ps[0] / 1000.0), float(distance_time_ps[-1] / 1000.0)]
    else:
        actual_window = [float("nan"), float("nan")]
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
        "waveforms_are_aligned": "each detector waveform is on the prepared LED/native-anchor-relative ML time grid",
        "requested_window_ns": None if window_ns is None else list(window_ns),
        "actual_sample_window_ns": actual_window,
        "led_error_definition": "abs(calibrated_led_pair_residual_ps)",
        "n_requested": int(indices.size),
        "n_finite_pairs": int(np.count_nonzero(finite)),
        "correlation": statistics,
        "per_voltage": per_voltage,
        "selection_scan_note": (
            "Descriptive CTR-versus-efficiency scan on the selected stage. Use development to choose a cut, "
            "then freeze it before evaluating test/blind data."
        ),
    }
    with (dataset_dir / "summary.json").open("w", encoding="utf-8") as stream:
        json.dump(summary, stream, indent=2, allow_nan=True)
        stream.write("\n")

    print(
        f"{dataset_name} | stage={args.stage} | n={statistics['n']} | "
        f"Pearson r={statistics['pearson_r']:+.4f} | Spearman rho={statistics['spearman_rho']:+.4f}"
    )
    if scan:
        best = min(scan, key=lambda row: float(row["ctr_ps"]))
        print(
            f"  lowest scanned LED CTR={best['ctr_ps']:.2f} ± {best['ctr_uncertainty_ps']:.2f} ps "
            f"at efficiency={best['efficiency_percent']:.1f}% "
            f"(distance <= {best['distance_threshold_mV']:.4g} mV)"
        )
    print(f"  outputs: {dataset_dir}")


def main() -> None:
    args = parse_args()
    if int(args.batch_size) < 1:
        raise ValueError("--batch-size must be positive")
    if int(args.trend_bins) < 2:
        raise ValueError("--trend-bins must be >= 2")
    if int(args.efficiency_points) < 2:
        raise ValueError("--efficiency-points must be >= 2")

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
