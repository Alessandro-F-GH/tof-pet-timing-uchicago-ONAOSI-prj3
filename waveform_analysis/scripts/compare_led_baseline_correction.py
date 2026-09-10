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

from utils_fit import fit_ctr_ps
from waveform_analysis.ml_pipeline.common import voltage_from_name
from waveform_analysis.ml_pipeline.config import discover_root_files, load_config, mode_family


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Standalone comparison of the current LED timing with an event-wise "
            "baseline-subtracted LED. Existing native-preprocessed caches are read; "
            "the main pipeline and its caches are never modified."
        )
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--split",
        choices=("development", "test", "all"),
        default="development",
        help="Population to compare after the original LED coincidence cut. Default: development.",
    )
    parser.add_argument(
        "--baseline-window-ns",
        type=float,
        nargs=2,
        metavar=("START", "STOP"),
        default=None,
        help="Baseline mean window relative to selected trigger. Default: baseline_noise.window_ns.",
    )
    parser.add_argument(
        "--threshold-mv",
        type=float,
        default=None,
        help="Override the selected LED threshold; otherwise read it from the prepared manifest.",
    )
    parser.add_argument(
        "--dataset",
        action="append",
        default=None,
        help="Optional exact ROOT stem filter. May be repeated.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Default: <study_output>/led_baseline_correction.",
    )
    return parser.parse_args()


def _read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as stream:
        value = json.load(stream)
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return value


def _load(directory: Path, name: str) -> np.ndarray:
    path = directory / f"{name}.npy"
    if not path.is_file():
        raise FileNotFoundError(f"Missing preprocessed array: {path}")
    return np.load(path, mmap_mode="r")


def _family_arrays(directory: Path, family: str) -> tuple[np.ndarray, ...]:
    return (
        _load(directory, f"{family}_windows_mV"),
        _load(directory, f"{family}_window_start_time_s"),
        _load(directory, f"{family}_sample_interval_s"),
        _load(directory, f"{family}_rising_start"),
        _load(directory, f"{family}_rising_stop"),
    )


def _crossing_ps(
    signal: np.ndarray,
    start_time_s: float,
    interval_s: float,
    rising_start: int,
    rising_stop: int,
    threshold_mV: float,
) -> float:
    """Two-sample LED interpolation, matching the main timing implementation."""
    y = np.asarray(signal, dtype=np.float64)
    a, b = int(rising_start), int(rising_stop)
    if a < 0 or b >= y.size or b <= a or not np.isfinite(threshold_mV):
        return float("nan")
    y0, y1 = y[a:b], y[a + 1 : b + 1]
    crossings = np.flatnonzero(
        np.isfinite(y0) & np.isfinite(y1) & (y0 < threshold_mV) & (y1 >= threshold_mV)
    )
    if crossings.size == 0:
        crossings = np.flatnonzero(
            np.isfinite(y0) & np.isfinite(y1) & (y0 == threshold_mV) & (y1 > threshold_mV)
        )
    if crossings.size == 0:
        return float("nan")
    lower = a + int(crossings[-1])
    denominator = float(y[lower + 1] - y[lower])
    if denominator == 0.0 or not np.isfinite(denominator):
        return float("nan")
    fraction = (float(threshold_mV) - float(y[lower])) / denominator
    if not 0.0 <= fraction <= 1.0:
        return float("nan")
    return (float(start_time_s) + (float(lower) + fraction) * float(interval_s)) * 1.0e12


def _baseline_level(
    signal: np.ndarray,
    interval_s: float,
    materialized_before_ns: float,
    window_ns: tuple[float, float],
) -> float:
    y = np.asarray(signal, dtype=np.float64)
    dt_ns = float(interval_s) * 1.0e9
    if not np.isfinite(dt_ns) or dt_ns <= 0.0:
        return float("nan")
    trigger_index = int(np.ceil(float(materialized_before_ns) / dt_ns))
    relative_ns = (np.arange(y.size, dtype=np.float64) - trigger_index) * dt_ns
    start_ns, stop_ns = map(float, window_ns)
    mask = (relative_ns >= start_ns) & (relative_ns <= stop_ns) & np.isfinite(y)
    if np.count_nonzero(mask) < 2:
        return float("nan")
    return float(np.mean(y[mask]))


def _selected_threshold(prepared_manifest: dict[str, Any] | None, family: str, override: float | None) -> tuple[float, str]:
    if override is not None:
        return float(override), "command_line_override"
    if prepared_manifest is None:
        raise FileNotFoundError(
            "Prepared manifest unavailable and --threshold-mv was not supplied. "
            "Run dataset preparation first or provide an explicit threshold."
        )
    thresholds = prepared_manifest.get("led_threshold_mV") or {}
    if family not in thresholds:
        raise KeyError(f"Prepared manifest has no selected LED threshold for {family}")
    return float(thresholds[family]), "prepared_manifest"


def _dataset_seed(base_seed: int, dataset: str) -> int:
    return (int(base_seed) + int(zlib.crc32(dataset.encode("utf-8")))) & 0xFFFFFFFF


def _summary(values: np.ndarray) -> dict[str, float]:
    x = np.asarray(values, dtype=np.float64)
    x = x[np.isfinite(x)]
    if not x.size:
        return {"mean_ps": float("nan"), "std_ps": float("nan"), "rmse_ps": float("nan")}
    return {
        "mean_ps": float(np.mean(x)),
        "std_ps": float(np.std(x)),
        "rmse_ps": float(np.sqrt(np.mean(x**2))),
    }


def _paired_bootstrap_delta(
    original_ps: np.ndarray,
    corrected_ps: np.ndarray,
    fit_config: dict[str, Any],
    *,
    seed: int,
    repeats: int,
) -> tuple[float, int]:
    original = np.asarray(original_ps, dtype=np.float64)
    corrected = np.asarray(corrected_ps, dtype=np.float64)
    if original.shape != corrected.shape or original.size < 3 or repeats < 2:
        return float("nan"), 0
    rng = np.random.default_rng(int(seed))
    deltas: list[float] = []
    for _ in range(int(repeats)):
        index = rng.integers(0, original.size, size=original.size)
        try:
            ctr_original = fit_ctr_ps(original[index], fit_config, bootstrap=False).ctr_ps
            ctr_corrected = fit_ctr_ps(corrected[index], fit_config, bootstrap=False).ctr_ps
        except ValueError:
            continue
        if np.isfinite(ctr_original) and np.isfinite(ctr_corrected):
            deltas.append(float(ctr_corrected - ctr_original))
    if len(deltas) < 2:
        return float("nan"), len(deltas)
    return float(np.std(deltas, ddof=1)), len(deltas)


def _display_edges(original: np.ndarray, corrected: np.ndarray, bins: int = 30) -> np.ndarray:
    pooled = np.concatenate([np.asarray(original, dtype=float), np.asarray(corrected, dtype=float)])
    pooled = pooled[np.isfinite(pooled)]
    lo, hi = np.quantile(pooled, [0.005, 0.995])
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        center = float(np.median(pooled)) if pooled.size else 0.0
        return np.linspace(center - 1.0, center + 1.0, bins + 1)
    margin = 0.06 * float(hi - lo)
    return np.linspace(float(lo - margin), float(hi + margin), bins + 1)


def _plot_dataset(
    path: Path,
    dataset: str,
    family: str,
    split_name: str,
    threshold_mV: float,
    original: np.ndarray,
    corrected: np.ndarray,
    original_fit,
    corrected_fit,
    delta_ctr_error_ps: float,
) -> None:
    edges = _display_edges(original, corrected)
    fig, ax = plt.subplots(figsize=(9.0, 5.8))
    colors = plt.rcParams["axes.prop_cycle"].by_key().get("color", [])
    color_original = colors[0] if colors else None
    color_corrected = colors[1] if len(colors) > 1 else None

    ax.hist(original, bins=edges, histtype="stepfilled", alpha=0.14, color=color_original, edgecolor=color_original,
            label=f"Original LED · FWHM {original_fit.ctr_ps:.1f} ± {original_fit.ctr_error_ps:.1f} ps")
    ax.hist(corrected, bins=edges, histtype="stepfilled", alpha=0.14, color=color_corrected, edgecolor=color_corrected,
            label=f"Baseline corrected · FWHM {corrected_fit.ctr_ps:.1f} ± {corrected_fit.ctr_error_ps:.1f} ps")

    for value in (original_fit.left_half_ps, original_fit.right_half_ps):
        ax.axvline(float(value), color=color_original, ls="--", lw=1.4, alpha=0.9)
    for value in (corrected_fit.left_half_ps, corrected_fit.right_half_ps):
        ax.axvline(float(value), color=color_corrected, ls="--", lw=1.4, alpha=0.9)

    delta = float(corrected_fit.ctr_ps - original_fit.ctr_ps)
    improvement = 100.0 * (original_fit.ctr_ps - corrected_fit.ctr_ps) / original_fit.ctr_ps
    ax.text(
        0.02,
        0.97,
        f"ΔCTR = {delta:+.1f} ± {delta_ctr_error_ps:.1f} ps\nRelative change = {improvement:+.1f}%",
        transform=ax.transAxes,
        ha="left",
        va="top",
        bbox={"boxstyle": "round,pad=0.35", "facecolor": "white", "alpha": 0.9},
    )
    ax.set_xlabel("Calibrated LED timing error [ps]")
    ax.set_ylabel("Events / display bin")
    ax.set_title(f"{dataset} · {family} · LED {threshold_mV:g} mV · {split_name}")
    ax.grid(alpha=0.2)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def _write_events(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = [
        "event_index",
        "split",
        "baseline_1_mV",
        "baseline_2_mV",
        "delta_baseline_mV",
        "original_led_1_ps",
        "original_led_2_ps",
        "corrected_led_1_ps",
        "corrected_led_2_ps",
        "original_error_ps",
        "corrected_error_ps",
        "error_change_ps",
    ]
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def analyse_dataset(root: Path, config: dict[str, Any], args: argparse.Namespace, output_root: Path) -> dict[str, Any]:
    dataset = root.stem
    family = mode_family(str(config["mode"]))
    preprocessed = Path(config["preprocessing"]["preprocessed_dir"]) / dataset
    prepared = Path(config["preprocessing"]["prepared_dir"]) / dataset
    if not preprocessed.is_dir():
        raise FileNotFoundError(f"Missing native-preprocessed cache for {dataset}: {preprocessed}")

    prepared_manifest = _read_json(prepared / "manifest.json") if (prepared / "manifest.json").is_file() else None
    threshold_mV, threshold_source = _selected_threshold(prepared_manifest, family, args.threshold_mv)
    baseline_default = config["preprocessing"]["selection"]["baseline_noise"]["window_ns"]
    baseline_window_ns = tuple(map(float, args.baseline_window_ns or baseline_default))
    if baseline_window_ns[1] > 0.0 or baseline_window_ns[1] <= baseline_window_ns[0]:
        raise ValueError("Baseline window must satisfy START < STOP <= 0 ns")

    pre_manifest = _read_json(preprocessed / "manifest.json")
    materialized_before_ns = float(pre_manifest["materialized_window_ns"]["before"])
    waves, starts, intervals, rising_start, rising_stop = _family_arrays(preprocessed, family)
    event_index = np.asarray(_load(preprocessed, "event_index"), dtype=np.int64)
    split = np.asarray(_load(preprocessed, "split"), dtype=np.int8)
    n = event_index.size

    baseline = np.full((n, 2), np.nan, dtype=np.float64)
    original_led = np.full((n, 2), np.nan, dtype=np.float64)
    corrected_led = np.full((n, 2), np.nan, dtype=np.float64)
    for event in range(n):
        for detector in range(2):
            signal = np.asarray(waves[event, detector], dtype=np.float64)
            baseline[event, detector] = _baseline_level(
                signal,
                float(intervals[event, detector]),
                materialized_before_ns,
                baseline_window_ns,
            )
            original_led[event, detector] = _crossing_ps(
                signal,
                float(starts[event, detector]),
                float(intervals[event, detector]),
                int(rising_start[event, detector]),
                int(rising_stop[event, detector]),
                threshold_mV,
            )
            if np.isfinite(baseline[event, detector]):
                corrected_led[event, detector] = _crossing_ps(
                    signal - baseline[event, detector],
                    float(starts[event, detector]),
                    float(intervals[event, detector]),
                    int(rising_start[event, detector]),
                    int(rising_stop[event, detector]),
                    threshold_mV,
                )

    true_tof_ps = float(config["data"]["true_tof_ps"])
    coincidence_window_ps = 1000.0 * float(config["standard_methods"].get("led_coincidence_window_ns", 2.0))
    delta_original = original_led[:, 0] - original_led[:, 1]
    delta_corrected = corrected_led[:, 0] - corrected_led[:, 1]
    raw_original_error = delta_original - true_tof_ps

    # Freeze the event selection to the current/original LED. The baseline-corrected
    # version is never allowed to choose its own population, avoiding an artificial
    # resolution improvement from outcome-dependent event rejection.
    original_coincidence = np.isfinite(raw_original_error) & (np.abs(raw_original_error) <= coincidence_window_ps)
    paired_valid = (
        original_coincidence
        & np.all(np.isfinite(baseline), axis=1)
        & np.all(np.isfinite(original_led), axis=1)
        & np.all(np.isfinite(corrected_led), axis=1)
    )

    development_calibration = (split == 0) & paired_valid
    if np.count_nonzero(development_calibration) < int(config["fit"]["min_events"]):
        raise RuntimeError(f"{dataset}: insufficient paired development events for calibration")
    original_bias_ps = float(np.mean(delta_original[development_calibration]) - true_tof_ps)
    corrected_bias_ps = float(np.mean(delta_corrected[development_calibration]) - true_tof_ps)
    original_error = delta_original - true_tof_ps - original_bias_ps
    corrected_error = delta_corrected - true_tof_ps - corrected_bias_ps

    if args.split == "development":
        requested = split == 0
    elif args.split == "test":
        requested = split == 1
    else:
        requested = np.ones(n, dtype=bool)
    analysed = requested & paired_valid
    original_values = np.asarray(original_error[analysed], dtype=np.float64)
    corrected_values = np.asarray(corrected_error[analysed], dtype=np.float64)
    if original_values.size < int(config["fit"]["min_events"]):
        raise RuntimeError(f"{dataset}: only {original_values.size} paired events remain")

    seed = _dataset_seed(int(config["validation"]["seed"]), dataset)
    original_fit = fit_ctr_ps(original_values, config["fit"], seed=seed, bootstrap=True)
    corrected_fit = fit_ctr_ps(corrected_values, config["fit"], seed=seed, bootstrap=True)
    delta_error, delta_bootstrap_success = _paired_bootstrap_delta(
        original_values,
        corrected_values,
        config["fit"],
        seed=seed,
        repeats=int(config["fit"].get("bootstrap_samples", 100)),
    )

    dataset_dir = output_root / dataset
    dataset_dir.mkdir(parents=True, exist_ok=True)
    _plot_dataset(
        dataset_dir / "led_baseline_correction_comparison.pdf",
        dataset,
        family,
        args.split,
        threshold_mV,
        original_values,
        corrected_values,
        original_fit,
        corrected_fit,
        delta_error,
    )

    rows = []
    analysed_indices = np.flatnonzero(analysed)
    for position, i in enumerate(analysed_indices):
        rows.append(
            {
                "event_index": int(event_index[i]),
                "split": "development" if split[i] == 0 else "test",
                "baseline_1_mV": float(baseline[i, 0]),
                "baseline_2_mV": float(baseline[i, 1]),
                "delta_baseline_mV": float(baseline[i, 0] - baseline[i, 1]),
                "original_led_1_ps": float(original_led[i, 0]),
                "original_led_2_ps": float(original_led[i, 1]),
                "corrected_led_1_ps": float(corrected_led[i, 0]),
                "corrected_led_2_ps": float(corrected_led[i, 1]),
                "original_error_ps": float(original_values[position]),
                "corrected_error_ps": float(corrected_values[position]),
                "error_change_ps": float(corrected_values[position] - original_values[position]),
            }
        )
    _write_events(dataset_dir / "events.csv", rows)

    original_summary = _summary(original_values)
    corrected_summary = _summary(corrected_values)
    delta_ctr_ps = float(corrected_fit.ctr_ps - original_fit.ctr_ps)
    relative_improvement = 100.0 * (original_fit.ctr_ps - corrected_fit.ctr_ps) / original_fit.ctr_ps
    summary = {
        "dataset": dataset,
        "voltage_V": voltage_from_name(dataset),
        "mode": str(config["mode"]),
        "family": family,
        "split": args.split,
        "threshold_mV": threshold_mV,
        "threshold_source": threshold_source,
        "baseline_window_start_ns": baseline_window_ns[0],
        "baseline_window_stop_ns": baseline_window_ns[1],
        "true_tof_ps": true_tof_ps,
        "coincidence_window_ps": coincidence_window_ps,
        "selection_definition": "original_LED_coincidence_then_common_finite_pair",
        "n_requested": int(np.count_nonzero(requested)),
        "n_original_coincidence": int(np.count_nonzero(requested & original_coincidence)),
        "n_corrected_missing_on_original_coincidence": int(np.count_nonzero(requested & original_coincidence & ~np.all(np.isfinite(corrected_led), axis=1))),
        "n_analysed_paired": int(original_values.size),
        "original_calibration_bias_ps": original_bias_ps,
        "corrected_calibration_bias_ps": corrected_bias_ps,
        "calibration_shift_ps": corrected_bias_ps - original_bias_ps,
        "original_ctr_ps": float(original_fit.ctr_ps),
        "original_ctr_uncertainty_ps": float(original_fit.ctr_error_ps),
        "corrected_ctr_ps": float(corrected_fit.ctr_ps),
        "corrected_ctr_uncertainty_ps": float(corrected_fit.ctr_error_ps),
        "delta_ctr_corrected_minus_original_ps": delta_ctr_ps,
        "delta_ctr_paired_bootstrap_uncertainty_ps": delta_error,
        "delta_ctr_paired_bootstrap_successful": int(delta_bootstrap_success),
        "relative_ctr_improvement_percent": relative_improvement,
        "original_mean_ps": original_summary["mean_ps"],
        "original_std_ps": original_summary["std_ps"],
        "original_rmse_ps": original_summary["rmse_ps"],
        "corrected_mean_ps": corrected_summary["mean_ps"],
        "corrected_std_ps": corrected_summary["std_ps"],
        "corrected_rmse_ps": corrected_summary["rmse_ps"],
        "fit_bin_width_ps": float(config["fit"]["bin_width_ps"]),
        "bootstrap_samples": int(config["fit"]["bootstrap_samples"]),
    }
    with (dataset_dir / "summary.json").open("w", encoding="utf-8") as stream:
        json.dump(summary, stream, indent=2, allow_nan=True)
        stream.write("\n")
    return summary


def _write_summary(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _plot_voltage_summary(path: Path, rows: list[dict[str, Any]]) -> None:
    finite_rows = [r for r in rows if np.isfinite(float(r["voltage_V"]))]
    if not finite_rows:
        return
    ordered = sorted(finite_rows, key=lambda r: float(r["voltage_V"]))
    voltage = np.asarray([float(r["voltage_V"]) for r in ordered])
    original = np.asarray([float(r["original_ctr_ps"]) for r in ordered])
    original_error = np.asarray([float(r["original_ctr_uncertainty_ps"]) for r in ordered])
    corrected = np.asarray([float(r["corrected_ctr_ps"]) for r in ordered])
    corrected_error = np.asarray([float(r["corrected_ctr_uncertainty_ps"]) for r in ordered])

    fig, ax = plt.subplots(figsize=(8.4, 4.8))
    ax.errorbar(voltage, original, yerr=original_error, marker="o", capsize=3, label="Original LED")
    ax.errorbar(voltage, corrected, yerr=corrected_error, marker="o", capsize=3, label="Baseline-corrected LED")
    ax.set_xlabel("Bias voltage [V]")
    ax.set_ylabel("CTR FWHM [ps]")
    ax.set_title("LED timing: original vs event-wise baseline subtraction")
    ax.grid(alpha=0.2)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    output_root = (args.output_dir or (Path(config["experiment"]["output_dir"]) / "led_baseline_correction")).resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    roots = discover_root_files(config)
    if args.dataset:
        wanted = set(args.dataset)
        roots = [root for root in roots if root.stem in wanted]
        missing = wanted - {root.stem for root in roots}
        if missing:
            raise FileNotFoundError(f"Requested dataset(s) not found: {sorted(missing)}")
    if not roots:
        raise FileNotFoundError("No ROOT datasets matched the configured source")

    if bool(config["preprocessing"]["selection"]["baseline_noise"].get("enabled", False)):
        print("WARNING: baseline_noise selection is enabled; native-preprocessed caches already contain that selected population.")

    summaries = []
    for root in roots:
        print(f"Comparing {root.stem} ...")
        result = analyse_dataset(root, config, args, output_root)
        summaries.append(result)
        print(
            f"  n={result['n_analysed_paired']} | original={result['original_ctr_ps']:.2f} ± {result['original_ctr_uncertainty_ps']:.2f} ps | "
            f"corrected={result['corrected_ctr_ps']:.2f} ± {result['corrected_ctr_uncertainty_ps']:.2f} ps | "
            f"Δ={result['delta_ctr_corrected_minus_original_ps']:+.2f} ± {result['delta_ctr_paired_bootstrap_uncertainty_ps']:.2f} ps | "
            f"improvement={result['relative_ctr_improvement_percent']:+.2f}%"
        )

    _write_summary(output_root / "summary.csv", summaries)
    _plot_voltage_summary(output_root / "ctr_vs_voltage.pdf", summaries)
    print(f"Outputs: {output_root}")


if __name__ == "__main__":
    main()
