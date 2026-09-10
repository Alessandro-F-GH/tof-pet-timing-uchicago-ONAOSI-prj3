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

from waveform_analysis.ml_pipeline.config import discover_root_files, load_config, mode_family


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Standalone diagnostic of baseline-level difference versus LED timing error. "
            "It reads existing native-preprocessed caches and does not modify or run the main pipeline."
        )
    )
    parser.add_argument("--config", type=Path, required=True, help="Experiment JSON used to locate caches and timing settings.")
    parser.add_argument(
        "--split",
        choices=("development", "test", "all"),
        default="development",
        help="Population to analyse. Default: development, so the blind test is not inspected.",
    )
    parser.add_argument(
        "--baseline-window-ns",
        type=float,
        nargs=2,
        metavar=("START", "STOP"),
        default=None,
        help="Baseline window relative to the selected trigger. Default: preprocessing.selection.baseline_noise.window_ns.",
    )
    parser.add_argument(
        "--threshold-mv",
        type=float,
        default=None,
        help="Override the selected LED threshold. Otherwise read it from the prepared-dataset manifest.",
    )
    parser.add_argument(
        "--dataset",
        action="append",
        default=None,
        help="Optional dataset-name filter. May be repeated; exact ROOT stem is expected.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory. Default: <study_output>/baseline_led_correlation.",
    )
    return parser.parse_args()


def _read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as stream:
        value = json.load(stream)
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return value


def _load_required(directory: Path, name: str) -> np.ndarray:
    path = directory / f"{name}.npy"
    if not path.is_file():
        raise FileNotFoundError(f"Missing preprocessed array: {path}")
    return np.load(path, mmap_mode="r")


def _family_arrays(directory: Path, family: str) -> tuple[np.ndarray, ...]:
    return (
        _load_required(directory, f"{family}_windows_mV"),
        _load_required(directory, f"{family}_window_start_time_s"),
        _load_required(directory, f"{family}_sample_interval_s"),
        _load_required(directory, f"{family}_rising_start"),
        _load_required(directory, f"{family}_rising_stop"),
    )


def _crossing_ps(
    signal: np.ndarray,
    start_time_s: float,
    interval_s: float,
    rising_start: int,
    rising_stop: int,
    level_mV: float,
) -> float:
    """Same two-sample LED interpolation used by the waveform pipeline."""
    y = np.asarray(signal, dtype=np.float64)
    a, b = int(rising_start), int(rising_stop)
    if a < 0 or b >= y.size or b <= a or not np.isfinite(level_mV):
        return float("nan")
    y0, y1 = y[a:b], y[a + 1 : b + 1]
    crossings = np.flatnonzero(np.isfinite(y0) & np.isfinite(y1) & (y0 < level_mV) & (y1 >= level_mV))
    if crossings.size == 0:
        crossings = np.flatnonzero(np.isfinite(y0) & np.isfinite(y1) & (y0 == level_mV) & (y1 > level_mV))
    if crossings.size == 0:
        return float("nan")
    lower = a + int(crossings[-1])
    denominator = float(y[lower + 1] - y[lower])
    if denominator == 0.0 or not np.isfinite(denominator):
        return float("nan")
    fraction = (float(level_mV) - float(y[lower])) / denominator
    if not 0.0 <= fraction <= 1.0:
        return float("nan")
    return (float(start_time_s) + (float(lower) + fraction) * float(interval_s)) * 1.0e12


def _baseline_level(
    signal: np.ndarray,
    interval_s: float,
    materialized_before_ns: float,
    baseline_window_ns: tuple[float, float],
) -> float:
    """Mean baseline level in a window expressed relative to the selected trigger."""
    y = np.asarray(signal, dtype=np.float64)
    dt_ns = float(interval_s) * 1.0e9
    if not np.isfinite(dt_ns) or dt_ns <= 0.0:
        return float("nan")
    trigger_index = int(np.ceil(float(materialized_before_ns) / dt_ns))
    relative_ns = (np.arange(y.size, dtype=np.float64) - trigger_index) * dt_ns
    start_ns, stop_ns = map(float, baseline_window_ns)
    mask = (relative_ns >= start_ns) & (relative_ns <= stop_ns) & np.isfinite(y)
    if np.count_nonzero(mask) < 2:
        return float("nan")
    return float(np.mean(y[mask]))


def _selected_threshold(
    prepared_manifest: dict[str, Any] | None,
    family: str,
    override: float | None,
) -> tuple[float, str]:
    if override is not None:
        return float(override), "command_line_override"
    if prepared_manifest is None:
        raise FileNotFoundError(
            "Prepared manifest unavailable and --threshold-mv was not supplied. "
            "Run dataset preparation first or pass an explicit threshold."
        )
    thresholds = prepared_manifest.get("led_threshold_mV") or {}
    if family not in thresholds:
        raise KeyError(f"Prepared manifest has no selected LED threshold for {family}")
    return float(thresholds[family]), "prepared_manifest"


def _calibration_bias(
    delta_led_ps: np.ndarray,
    split: np.ndarray,
    true_tof_ps: float,
    family: str,
    threshold_mV: float,
    threshold_source: str,
    prepared_manifest: dict[str, Any] | None,
) -> tuple[float, str]:
    if threshold_source == "prepared_manifest" and prepared_manifest is not None:
        means = prepared_manifest.get("led_training_mean_ps") or {}
        if family in means:
            training_mean = float(means[family])
            if np.isfinite(training_mean):
                return training_mean - float(true_tof_ps), "prepared_training_mean"

    development = (np.asarray(split) == 0) & np.isfinite(delta_led_ps)
    if np.count_nonzero(development) < 2:
        raise RuntimeError("Not enough development LED pairs to estimate the fixed channel offset")
    # A constant centering term does not change correlation. Mean is used here to
    # match the pipeline calibration convention when an explicit threshold is tested.
    mean_delta = float(np.mean(np.asarray(delta_led_ps)[development]))
    return mean_delta - float(true_tof_ps), f"development_mean_at_{threshold_mV:g}mV"


def _stats(x: np.ndarray, y: np.ndarray) -> dict[str, float]:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    finite = np.isfinite(x) & np.isfinite(y)
    x, y = x[finite], y[finite]
    result = {
        "n": int(x.size),
        "pearson_r": float("nan"),
        "pearson_p": float("nan"),
        "spearman_rho": float("nan"),
        "spearman_p": float("nan"),
        "slope_ps_per_mV": float("nan"),
        "intercept_ps": float("nan"),
        "r_squared": float("nan"),
    }
    if x.size < 3 or np.all(x == x[0]) or np.all(y == y[0]):
        return result
    pearson = pearsonr(x, y)
    spearman = spearmanr(x, y)
    regression = linregress(x, y)
    result.update(
        pearson_r=float(pearson.statistic),
        pearson_p=float(pearson.pvalue),
        spearman_rho=float(spearman.statistic),
        spearman_p=float(spearman.pvalue),
        slope_ps_per_mV=float(regression.slope),
        intercept_ps=float(regression.intercept),
        r_squared=float(regression.rvalue**2),
    )
    return result


def _write_events(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = [
        "event_index",
        "split",
        "baseline_1_mV",
        "baseline_2_mV",
        "delta_baseline_mV",
        "led_1_ps",
        "led_2_ps",
        "raw_led_error_ps",
        "calibrated_led_error_ps",
        "inside_pipeline_coincidence",
    ]
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _plot(
    output: Path,
    dataset: str,
    family: str,
    split_name: str,
    threshold_mV: float,
    baseline_window_ns: tuple[float, float],
    delta_baseline: np.ndarray,
    led_error: np.ndarray,
    inside_coincidence: np.ndarray,
    statistics: dict[str, float],
) -> None:
    finite = np.isfinite(delta_baseline) & np.isfinite(led_error)
    x = np.asarray(delta_baseline)[finite]
    y = np.asarray(led_error)[finite]
    inside = np.asarray(inside_coincidence, dtype=bool)[finite]
    if x.size == 0:
        return

    fig, ax = plt.subplots(figsize=(8.8, 5.8))
    if np.any(inside):
        ax.scatter(x[inside], y[inside], s=18, alpha=0.25, label=f"inside coincidence ({np.count_nonzero(inside)})")
    if np.any(~inside):
        ax.scatter(x[~inside], y[~inside], s=24, alpha=0.55, marker="x", label=f"outside coincidence ({np.count_nonzero(~inside)})")

    slope = statistics["slope_ps_per_mV"]
    intercept = statistics["intercept_ps"]
    if np.isfinite(slope) and np.isfinite(intercept):
        x_line = np.linspace(float(np.quantile(x, 0.01)), float(np.quantile(x, 0.99)), 200)
        ax.plot(x_line, intercept + slope * x_line, lw=1.8, label=f"linear slope {slope:+.2f} ps/mV")

    ax.axhline(0.0, ls="--", lw=1.0, alpha=0.5)
    ax.axvline(0.0, ls="--", lw=1.0, alpha=0.5)
    ax.set_xlabel(r"Baseline-level difference $B_1-B_2$ [mV]")
    ax.set_ylabel("Calibrated LED timing error [ps]")
    ax.set_title(f"{dataset} · {family} · LED {threshold_mV:g} mV · {split_name}")
    ax.grid(alpha=0.2)
    ax.legend(loc="best")
    ax.text(
        0.02,
        0.98,
        (
            f"baseline window: [{baseline_window_ns[0]:g}, {baseline_window_ns[1]:g}] ns\n"
            f"Pearson r = {statistics['pearson_r']:+.4f} (p={statistics['pearson_p']:.3g})\n"
            f"Spearman ρ = {statistics['spearman_rho']:+.4f} (p={statistics['spearman_p']:.3g})\n"
            f"R² = {statistics['r_squared']:.4f}"
        ),
        transform=ax.transAxes,
        ha="left",
        va="top",
        bbox={"boxstyle": "round,pad=0.35", "facecolor": "white", "alpha": 0.9},
    )
    fig.tight_layout()
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)


def analyse_dataset(
    root: Path,
    config: dict[str, Any],
    args: argparse.Namespace,
    output_root: Path,
) -> dict[str, Any]:
    dataset = root.stem
    family = mode_family(str(config["mode"]))
    preprocessed = Path(config["preprocessing"]["preprocessed_dir"]) / dataset
    prepared = Path(config["preprocessing"]["prepared_dir"]) / dataset
    if not preprocessed.is_dir():
        raise FileNotFoundError(f"Missing preprocessed cache for {dataset}: {preprocessed}")

    prepared_manifest = _read_json(prepared / "manifest.json") if (prepared / "manifest.json").is_file() else None
    threshold_mV, threshold_source = _selected_threshold(prepared_manifest, family, args.threshold_mv)
    true_tof_ps = float(config["data"]["true_tof_ps"])
    coincidence_window_ps = 1000.0 * float(config["standard_methods"].get("led_coincidence_window_ns", 2.0))
    default_baseline = config["preprocessing"]["selection"]["baseline_noise"]["window_ns"]
    baseline_window_ns = tuple(map(float, args.baseline_window_ns or default_baseline))
    if baseline_window_ns[1] > 0.0 or baseline_window_ns[1] <= baseline_window_ns[0]:
        raise ValueError("Baseline window must satisfy START < STOP <= 0 ns relative to trigger")

    manifest = _read_json(preprocessed / "manifest.json")
    materialized_before_ns = float(manifest["materialized_window_ns"]["before"])
    waves, starts, intervals, rising_start, rising_stop = _family_arrays(preprocessed, family)
    event_index = np.asarray(_load_required(preprocessed, "event_index"), dtype=np.int64)
    split = np.asarray(_load_required(preprocessed, "split"), dtype=np.int8)
    n = event_index.size

    baseline = np.full((n, 2), np.nan, dtype=np.float64)
    led = np.full((n, 2), np.nan, dtype=np.float64)
    for event in range(n):
        for detector in range(2):
            baseline[event, detector] = _baseline_level(
                waves[event, detector],
                float(intervals[event, detector]),
                materialized_before_ns,
                baseline_window_ns,
            )
            led[event, detector] = _crossing_ps(
                waves[event, detector],
                float(starts[event, detector]),
                float(intervals[event, detector]),
                int(rising_start[event, detector]),
                int(rising_stop[event, detector]),
                threshold_mV,
            )

    delta_baseline = baseline[:, 0] - baseline[:, 1]
    delta_led = led[:, 0] - led[:, 1]
    raw_error = delta_led - true_tof_ps
    calibration_bias_ps, calibration_source = _calibration_bias(
        delta_led,
        split,
        true_tof_ps,
        family,
        threshold_mV,
        threshold_source,
        prepared_manifest,
    )
    calibrated_error = raw_error - calibration_bias_ps
    inside_coincidence = np.isfinite(raw_error) & (np.abs(raw_error) <= coincidence_window_ps)

    if args.split == "development":
        requested = split == 0
    elif args.split == "test":
        requested = split == 1
    else:
        requested = np.ones(n, dtype=bool)
    finite = requested & np.isfinite(delta_baseline) & np.isfinite(calibrated_error)
    statistics = _stats(delta_baseline[finite], calibrated_error[finite])
    inside_stats = _stats(delta_baseline[finite & inside_coincidence], calibrated_error[finite & inside_coincidence])

    dataset_dir = output_root / dataset
    dataset_dir.mkdir(parents=True, exist_ok=True)
    event_rows = []
    for i in np.flatnonzero(requested):
        event_rows.append(
            {
                "event_index": int(event_index[i]),
                "split": "development" if split[i] == 0 else "test",
                "baseline_1_mV": float(baseline[i, 0]),
                "baseline_2_mV": float(baseline[i, 1]),
                "delta_baseline_mV": float(delta_baseline[i]),
                "led_1_ps": float(led[i, 0]),
                "led_2_ps": float(led[i, 1]),
                "raw_led_error_ps": float(raw_error[i]),
                "calibrated_led_error_ps": float(calibrated_error[i]),
                "inside_pipeline_coincidence": bool(inside_coincidence[i]),
            }
        )
    _write_events(dataset_dir / "events.csv", event_rows)
    _plot(
        dataset_dir / "baseline_vs_led_error.pdf",
        dataset,
        family,
        args.split,
        threshold_mV,
        baseline_window_ns,
        delta_baseline[requested],
        calibrated_error[requested],
        inside_coincidence[requested],
        statistics,
    )

    summary = {
        "dataset": dataset,
        "mode": str(config["mode"]),
        "family": family,
        "split": args.split,
        "threshold_mV": threshold_mV,
        "threshold_source": threshold_source,
        "baseline_window_start_ns": baseline_window_ns[0],
        "baseline_window_stop_ns": baseline_window_ns[1],
        "true_tof_ps": true_tof_ps,
        "calibration_bias_ps": calibration_bias_ps,
        "calibration_source": calibration_source,
        "coincidence_window_ps": coincidence_window_ps,
        "baseline_noise_filter_enabled": bool(config["preprocessing"]["selection"]["baseline_noise"].get("enabled", False)),
        "n_requested": int(np.count_nonzero(requested)),
        "n_finite": int(np.count_nonzero(finite)),
        "n_inside_pipeline_coincidence": int(np.count_nonzero(finite & inside_coincidence)),
        "delta_baseline_mean_mV": float(np.mean(delta_baseline[finite])) if np.any(finite) else float("nan"),
        "delta_baseline_std_mV": float(np.std(delta_baseline[finite])) if np.any(finite) else float("nan"),
        "led_error_mean_ps": float(np.mean(calibrated_error[finite])) if np.any(finite) else float("nan"),
        "led_error_std_ps": float(np.std(calibrated_error[finite])) if np.any(finite) else float("nan"),
        **statistics,
        "inside_coincidence_pearson_r": inside_stats["pearson_r"],
        "inside_coincidence_pearson_p": inside_stats["pearson_p"],
        "inside_coincidence_spearman_rho": inside_stats["spearman_rho"],
        "inside_coincidence_spearman_p": inside_stats["spearman_p"],
        "inside_coincidence_slope_ps_per_mV": inside_stats["slope_ps_per_mV"],
        "inside_coincidence_r_squared": inside_stats["r_squared"],
    }
    with (dataset_dir / "summary.json").open("w", encoding="utf-8") as stream:
        json.dump(summary, stream, indent=2, allow_nan=True)
        stream.write("\n")
    return summary


def write_summary(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    fields = list(rows[0])
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    output_root = (args.output_dir or (Path(config["experiment"]["output_dir"]) / "baseline_led_correlation")).resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    roots = discover_root_files(config)
    if args.dataset:
        wanted = set(args.dataset)
        roots = [root for root in roots if root.stem in wanted]
        missing = wanted - {root.stem for root in roots}
        if missing:
            raise FileNotFoundError(f"Requested dataset(s) not found in config ROOT discovery: {sorted(missing)}")
    if not roots:
        raise FileNotFoundError("No ROOT datasets matched the configured source")

    if bool(config["preprocessing"]["selection"]["baseline_noise"].get("enabled", False)):
        print("WARNING: baseline_noise selection is enabled; the analysed preprocessed population is already baseline-noise selected.")

    summaries = []
    for root in roots:
        print(f"Analysing {root.stem} ...")
        summary = analyse_dataset(root, config, args, output_root)
        summaries.append(summary)
        print(
            f"  n={summary['n_finite']} | Pearson r={summary['pearson_r']:+.4f} | "
            f"Spearman rho={summary['spearman_rho']:+.4f} | slope={summary['slope_ps_per_mV']:+.3f} ps/mV"
        )

    write_summary(output_root / "summary.csv", summaries)
    print(f"Outputs: {output_root}")


if __name__ == "__main__":
    main()
