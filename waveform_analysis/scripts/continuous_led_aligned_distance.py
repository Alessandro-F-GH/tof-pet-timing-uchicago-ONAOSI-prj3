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

from waveform_analysis.ml_pipeline.data import PreprocessedData
from waveform_analysis.ml_pipeline.timing import family_arrays, led_grid, pair_delta


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Continuously align the two detector waveforms to their interpolated LED crossings, "
            "so each waveform has its LED crossing exactly at t=0, then measure event-wise channel "
            "distance and correlate it with the absolute calibrated LED timing error. Waveforms are "
            "read directly from the selected-event preprocessing cache; no native-grid anchor or "
            "anchor correction is used."
        )
    )
    parser.add_argument("--run-dir", type=Path, required=True, help="Completed study directory.")
    parser.add_argument(
        "--dataset",
        default=None,
        help="Study dataset name. Default: the only dataset, or the concatenated dataset if unambiguous.",
    )
    parser.add_argument(
        "--stage",
        choices=("development", "test", "all"),
        default="development",
        help="Preprocessing population to analyse. Default: development.",
    )
    parser.add_argument(
        "--window-ns",
        type=float,
        nargs=2,
        metavar=("START", "STOP"),
        default=None,
        help="Relative-time window after continuous LED alignment. Default: study ml_input.window_ns.",
    )
    parser.add_argument(
        "--step-ps",
        type=float,
        default=None,
        help="Common interpolation-grid spacing. Default: median native sample interval.",
    )
    parser.add_argument(
        "--led-threshold-mV",
        type=float,
        default=None,
        help="Override the LED threshold. Default: selected/fixed threshold from prepared dataset manifest.",
    )
    parser.add_argument(
        "--distance",
        choices=("rms", "mean_abs", "max_abs"),
        default="rms",
        help="Distance between the two continuously aligned physical-mV waveforms. Default: rms.",
    )
    parser.add_argument("--trend-bins", type=int, default=16, help="Equal-count bins for trend plot. Default: 16.")
    parser.add_argument("--batch-size", type=int, default=512, help="Event batch size. Default: 512.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Default: <run-dir>/continuous_led_aligned_distance/.",
    )
    return parser.parse_args()


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return value


def _existing_path(value: str | Path, *fallbacks: Path) -> Path:
    value_path = Path(value)
    candidates = [value_path]
    if not value_path.is_absolute():
        candidates.extend(fallback / value_path for fallback in fallbacks)
    candidates.extend(fallbacks)
    for candidate in candidates:
        if candidate.is_dir():
            return candidate.resolve()
    raise FileNotFoundError(f"Cannot locate directory {value!s}; tried {[str(p) for p in candidates]}")


def _optional(directory: Path, name: str) -> np.ndarray | None:
    path = directory / f"{name}.npy"
    return np.load(path, mmap_mode="r") if path.is_file() else None


def _load_preprocessed(directory: Path) -> PreprocessedData:
    manifest = _read_json(directory / "manifest.json")
    return PreprocessedData(
        directory=directory,
        manifest=manifest,
        event_index=np.load(directory / "event_index.npy", mmap_mode="r"),
        split=np.load(directory / "split.npy", mmap_mode="r"),
        bias_voltage_V=np.load(directory / "bias_voltage_V.npy", mmap_mode="r"),
        energy_windows_mV=_optional(directory, "energy_windows_mV"),
        timing_windows_mV=_optional(directory, "timing_windows_mV"),
        energy_window_start_time_s=_optional(directory, "energy_window_start_time_s"),
        timing_window_start_time_s=_optional(directory, "timing_window_start_time_s"),
        energy_sample_interval_s=_optional(directory, "energy_sample_interval_s"),
        timing_sample_interval_s=_optional(directory, "timing_sample_interval_s"),
        energy_rising_start=_optional(directory, "energy_rising_start"),
        timing_rising_start=_optional(directory, "timing_rising_start"),
        energy_rising_stop=_optional(directory, "energy_rising_stop"),
        timing_rising_stop=_optional(directory, "timing_rising_stop"),
    )


def _mode_family(mode: str) -> str:
    if mode == "energy_to_energy":
        return "energy"
    if mode == "timing_to_timing":
        return "timing"
    raise ValueError(f"Unsupported mode: {mode}")


def _resolve_dataset(run: Path, run_manifest: dict[str, Any], requested: str | None) -> tuple[str, Path, dict[str, Any]]:
    datasets = run_manifest.get("datasets") or {}
    if requested is None:
        if len(datasets) != 1:
            raise ValueError(f"Study has {len(datasets)} datasets; specify --dataset from {sorted(datasets)}")
        requested = next(iter(datasets))
    if requested not in datasets:
        raise KeyError(f"Dataset {requested!r} not found; available: {sorted(datasets)}")
    prepared_dir = _existing_path(datasets[requested]["prepared_dir"], run, run.parent)
    return requested, prepared_dir, _read_json(prepared_dir / "manifest.json")


def _source_preprocessed_dirs(
    prepared_dir: Path,
    prepared_manifest: dict[str, Any],
    config: dict[str, Any],
) -> list[tuple[str, Path]]:
    preprocessed_root = Path(config["preprocessing"]["preprocessed_dir"]).resolve()
    if bool(prepared_manifest.get("concatenated", False)):
        rows = prepared_manifest.get("source_datasets") or []
        if not rows:
            raise ValueError("Concatenated prepared manifest has no source_datasets")
        sources = []
        for row in rows:
            source = str(row["source"])
            stem = Path(source).stem
            directory = preprocessed_root / stem
            if not directory.is_dir():
                source_prepared = Path(str(row.get("prepared_dir", "")))
                if source_prepared.is_dir():
                    source_manifest = _read_json(source_prepared / "manifest.json")
                    directory = Path(source_manifest["preprocessed_dir"])
            if not directory.is_dir():
                raise FileNotFoundError(f"Preprocessed cache not found for {stem}: {directory}")
            sources.append((stem, directory.resolve()))
        return sources

    source = str(prepared_manifest["source"])
    stem = Path(source).stem
    saved = prepared_manifest.get("preprocessed_dir")
    candidates = [Path(saved)] if saved else []
    candidates.append(preprocessed_root / stem)
    for directory in candidates:
        if directory.is_dir():
            return [(stem, directory.resolve())]
    raise FileNotFoundError(f"Preprocessed cache not found for {stem}: {[str(p) for p in candidates]}")


def _stage_mask(data: PreprocessedData, stage: str) -> np.ndarray:
    split = np.asarray(data.split, dtype=np.int8)
    if stage == "development":
        return split == 0
    if stage == "test":
        return split == 1
    if stage == "all":
        return np.ones(data.n_events, dtype=bool)
    raise ValueError(stage)


def _relative_grid(window_ns: tuple[float, float], step_ps: float) -> np.ndarray:
    start_ps, stop_ps = 1000.0 * float(window_ns[0]), 1000.0 * float(window_ns[1])
    if not start_ps < 0.0 < stop_ps:
        raise ValueError("Alignment window must include t=0 strictly inside the window")
    if not np.isfinite(step_ps) or step_ps <= 0.0:
        raise ValueError("Interpolation step must be positive")
    first = int(np.ceil(start_ps / step_ps - 1e-10))
    last = int(np.floor(stop_ps / step_ps + 1e-10))
    grid = np.arange(first, last + 1, dtype=np.float64) * float(step_ps)
    if not np.any(grid == 0.0):
        raise RuntimeError("Internal error: interpolation grid does not contain t=0")
    if grid.size < 3:
        raise ValueError("Interpolation grid contains fewer than three samples")
    return grid


def _linear_shift_to_led(
    signal: np.ndarray,
    start_time_s: float,
    interval_s: float,
    led_time_ps: float,
    relative_time_ps: np.ndarray,
) -> np.ndarray | None:
    """Evaluate one native waveform on a grid exactly relative to its interpolated LED crossing.

    This is a continuous translation implemented by linear interpolation. It never rounds the LED
    time to a native sample. The t=0 output therefore evaluates the waveform at the same continuous
    crossing time returned by the repository LED interpolation.
    """
    y = np.asarray(signal, dtype=np.float64)
    dt_ps = float(interval_s) * 1.0e12
    if y.ndim != 1 or y.size < 2 or not np.isfinite(dt_ps) or dt_ps <= 0.0 or not np.isfinite(led_time_ps):
        return None
    start_ps = float(start_time_s) * 1.0e12
    fractional_index = (float(led_time_ps) + relative_time_ps - start_ps) / dt_ps
    lower = np.floor(fractional_index).astype(np.int64)
    fraction = fractional_index - lower
    if np.any(lower < 0) or np.any(lower + 1 >= y.size):
        return None
    shifted = y[lower] + fraction * (y[lower + 1] - y[lower])
    return shifted if np.all(np.isfinite(shifted)) else None


def _distance(difference: np.ndarray, metric: str) -> float:
    if metric == "rms":
        return float(np.sqrt(np.mean(difference**2)))
    if metric == "mean_abs":
        return float(np.mean(np.abs(difference)))
    if metric == "max_abs":
        return float(np.max(np.abs(difference)))
    raise ValueError(metric)


def _source_records(
    source_name: str,
    data: PreprocessedData,
    family: str,
    threshold_mV: float,
    stage: str,
    relative_time_ps: np.ndarray,
    metric: str,
    batch_size: int,
) -> tuple[list[dict[str, Any]], np.ndarray, np.ndarray]:
    waves, starts, intervals, _rising_start, _rising_stop = family_arrays(data, family)
    indices = np.flatnonzero(_stage_mask(data, stage)).astype(np.int64)
    led_times = led_grid(data, family, indices, np.asarray([threshold_mV], dtype=np.float64))[:, :, 0]
    zero_index = int(np.flatnonzero(relative_time_ps == 0.0)[0])
    records: list[dict[str, Any]] = []
    sum_waveforms = np.zeros((2, relative_time_ps.size), dtype=np.float64)
    mean_count = np.zeros(2, dtype=np.int64)

    for batch_start in range(0, indices.size, int(batch_size)):
        batch_stop = min(indices.size, batch_start + int(batch_size))
        for local in range(batch_start, batch_stop):
            event = int(indices[local])
            led = np.asarray(led_times[local], dtype=np.float64)
            row: dict[str, Any] = {
                "source_dataset": source_name,
                "preprocessed_row": event,
                "event_index": int(data.event_index[event]),
                "bias_voltage_V": float(data.bias_voltage_V[event]),
                "led1_ps": float(led[0]),
                "led2_ps": float(led[1]),
                "delta_led_ps": float(led[0] - led[1]) if np.all(np.isfinite(led)) else float("nan"),
                "distance_mV": float("nan"),
                "aligned_ch1_at_t0_mV": float("nan"),
                "aligned_ch2_at_t0_mV": float("nan"),
                "ch1_t0_minus_threshold_mV": float("nan"),
                "ch2_t0_minus_threshold_mV": float("nan"),
                "continuous_window_valid": False,
            }
            if np.all(np.isfinite(led)):
                shifted = []
                for detector in range(2):
                    aligned = _linear_shift_to_led(
                        waves[event, detector],
                        float(starts[event, detector]),
                        float(intervals[event, detector]),
                        float(led[detector]),
                        relative_time_ps,
                    )
                    if aligned is None:
                        shifted = []
                        break
                    shifted.append(aligned)
                if len(shifted) == 2:
                    pair = np.stack(shifted)
                    difference = pair[0] - pair[1]
                    row["distance_mV"] = _distance(difference, metric)
                    row["aligned_ch1_at_t0_mV"] = float(pair[0, zero_index])
                    row["aligned_ch2_at_t0_mV"] = float(pair[1, zero_index])
                    row["ch1_t0_minus_threshold_mV"] = float(pair[0, zero_index] - threshold_mV)
                    row["ch2_t0_minus_threshold_mV"] = float(pair[1, zero_index] - threshold_mV)
                    row["continuous_window_valid"] = True
                    sum_waveforms += pair
                    mean_count += 1
            records.append(row)

    return records, sum_waveforms, mean_count


def _correlation(distance: np.ndarray, absolute_error: np.ndarray) -> dict[str, Any]:
    finite = np.isfinite(distance) & np.isfinite(absolute_error)
    x = np.asarray(distance[finite], dtype=np.float64)
    y = np.asarray(absolute_error[finite], dtype=np.float64)
    result: dict[str, Any] = {
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
    if x.size >= 3 and np.std(x) > 0.0 and np.std(y) > 0.0:
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


def _trend(distance: np.ndarray, absolute_error: np.ndarray, bins: int) -> list[dict[str, Any]]:
    finite = np.isfinite(distance) & np.isfinite(absolute_error)
    x = np.asarray(distance[finite], dtype=np.float64)
    y = np.asarray(absolute_error[finite], dtype=np.float64)
    if x.size == 0:
        return []
    edges = np.unique(np.quantile(x, np.linspace(0.0, 1.0, min(int(bins), x.size) + 1)))
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


def _plot_relation(
    path: Path,
    dataset_name: str,
    stage: str,
    metric: str,
    distance: np.ndarray,
    absolute_error: np.ndarray,
    trend: list[dict[str, Any]],
    stats: dict[str, Any],
) -> None:
    finite = np.isfinite(distance) & np.isfinite(absolute_error)
    x, y = distance[finite], absolute_error[finite]
    fig, ax = plt.subplots(figsize=(8.5, 5.6))
    ax.scatter(x, y, s=8, alpha=0.12, label="preprocessed selected events")
    if trend:
        tx = np.asarray([row["distance_median_mV"] for row in trend])
        ty = np.asarray([row["abs_led_error_median_ps"] for row in trend])
        low = ty - np.asarray([row["abs_led_error_q16_ps"] for row in trend])
        high = np.asarray([row["abs_led_error_q84_ps"] for row in trend]) - ty
        ax.errorbar(
            tx,
            ty,
            yerr=np.vstack([low, high]),
            marker="o",
            capsize=3,
            lw=1.3,
            label="distance-bin median ± 16–84%",
        )
    ax.set_xlabel(f"Continuously LED-aligned channel {metric} distance [mV]")
    ax.set_ylabel("Absolute LED error, no anchor correction [ps]")
    ax.set_title(f"{dataset_name} · {stage} · continuous LED alignment")
    ax.grid(alpha=0.2)
    ax.legend(loc="best")
    ax.text(
        0.02,
        0.98,
        f"n={stats['n']}\nPearson r={stats['pearson_r']:+.4f}\n"
        f"Spearman ρ={stats['spearman_rho']:+.4f}\nR²={stats['linear_r_squared']:.5f}",
        transform=ax.transAxes,
        ha="left",
        va="top",
        bbox={"boxstyle": "round,pad=0.3", "facecolor": "white", "alpha": 0.9},
    )
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def _plot_mean_waveforms(
    path: Path,
    dataset_name: str,
    relative_time_ps: np.ndarray,
    sum_waveforms: np.ndarray,
    count: int,
    threshold_mV: float,
) -> None:
    if count <= 0:
        return
    mean = sum_waveforms / float(count)
    time_ns = relative_time_ps / 1000.0
    fig, ax = plt.subplots(figsize=(8.5, 5.0))
    ax.plot(time_ns, mean[0], label="detector 1 mean")
    ax.plot(time_ns, mean[1], label="detector 2 mean")
    ax.axvline(0.0, ls="--", lw=1.2, label="exact interpolated LED time")
    ax.scatter([0.0], [threshold_mV], marker="o", s=35, zorder=5, label=f"LED threshold = {threshold_mV:g} mV")
    ax.set_xlabel("Time relative to interpolated LED crossing [ns]")
    ax.set_ylabel("Signal [mV]")
    ax.set_title(f"{dataset_name} · continuously LED-aligned mean waveforms")
    ax.grid(alpha=0.2)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    if args.batch_size < 1:
        raise ValueError("--batch-size must be positive")
    if args.trend_bins < 2:
        raise ValueError("--trend-bins must be >= 2")

    run = args.run_dir.resolve()
    run_manifest = _read_json(run / "manifest.json")
    dataset_name, prepared_dir, prepared_manifest = _resolve_dataset(run, run_manifest, args.dataset)
    config = run_manifest.get("config") or {}
    mode = str(prepared_manifest.get("mode") or run_manifest.get("mode") or config["mode"])
    family = _mode_family(mode)

    threshold_map = prepared_manifest.get("led_threshold_mV") or {}
    threshold_mV = float(
        args.led_threshold_mV if args.led_threshold_mV is not None else threshold_map[family]
    )
    true_tof_ps = float(prepared_manifest.get("true_tof_ps", (config.get("data") or {}).get("true_tof_ps", 0.0)))
    calibration_map = prepared_manifest.get("calibration_bias_ps") or {}
    calibration_bias_ps = float(calibration_map.get(family, float("nan")))

    window_cfg = (config.get("ml_input") or {}).get("window_ns") or {}
    window_ns = (
        tuple(map(float, args.window_ns))
        if args.window_ns is not None
        else (float(window_cfg.get("start", -2.0)), float(window_cfg.get("end", 30.0)))
    )

    source_dirs = _source_preprocessed_dirs(prepared_dir, prepared_manifest, config)
    sources = [(name, _load_preprocessed(directory)) for name, directory in source_dirs]
    interval_values = []
    for _name, data in sources:
        _waves, _starts, intervals, _rise_a, _rise_b = family_arrays(data, family)
        interval_values.append(np.asarray(intervals, dtype=np.float64).reshape(-1) * 1.0e12)
    native_intervals_ps = np.concatenate(interval_values)
    native_intervals_ps = native_intervals_ps[np.isfinite(native_intervals_ps) & (native_intervals_ps > 0.0)]
    if not native_intervals_ps.size:
        raise RuntimeError("No valid native sample intervals")
    step_ps = float(args.step_ps if args.step_ps is not None else np.median(native_intervals_ps))
    relative_time_ps = _relative_grid(window_ns, step_ps)

    records: list[dict[str, Any]] = []
    waveform_sum = np.zeros((2, relative_time_ps.size), dtype=np.float64)
    waveform_count = 0
    for source_name, data in sources:
        source_rows, source_sum, source_count = _source_records(
            source_name,
            data,
            family,
            threshold_mV,
            args.stage,
            relative_time_ps,
            args.distance,
            args.batch_size,
        )
        records.extend(source_rows)
        waveform_sum += source_sum
        waveform_count += int(source_count[0])

    if not records:
        raise RuntimeError("No preprocessing events found for requested stage")

    delta_led = np.asarray([row["delta_led_ps"] for row in records], dtype=np.float64)
    distance = np.asarray([row["distance_mV"] for row in records], dtype=np.float64)

    # Calibration is allowed; anchor correction is not. Prefer the frozen training calibration used
    # by the prepared dataset. If unavailable, estimate one offset from preprocessing development
    # events only, never from the blind/test population.
    calibration_source = "prepared_training_calibration_bias"
    if not np.isfinite(calibration_bias_ps):
        development_delta = []
        for _source_name, data in sources:
            indices = np.flatnonzero(_stage_mask(data, "development")).astype(np.int64)
            led = led_grid(data, family, indices, np.asarray([threshold_mV], dtype=np.float64))[:, :, 0]
            values = pair_delta(led)
            development_delta.append(values[np.isfinite(values)])
        pooled = np.concatenate(development_delta) if development_delta else np.empty(0)
        if not pooled.size:
            raise RuntimeError("Cannot determine LED calibration bias")
        calibration_bias_ps = float(np.mean(pooled) - true_tof_ps)
        calibration_source = "preprocessing_development_mean_fallback"

    led_error = delta_led - true_tof_ps - calibration_bias_ps
    absolute_error = np.abs(led_error)
    for row, error in zip(records, led_error):
        row["true_tof_ps"] = true_tof_ps
        row["calibration_bias_ps"] = calibration_bias_ps
        row["led_error_no_anchor_ps"] = float(error)
        row["abs_led_error_no_anchor_ps"] = float(abs(error))

    finite = np.isfinite(distance) & np.isfinite(absolute_error)
    if np.count_nonzero(finite) < 3:
        raise RuntimeError("Fewer than three valid continuously aligned distance/error pairs")

    stats = _correlation(distance, absolute_error)
    trend = _trend(distance, absolute_error, args.trend_bins)

    output = (args.output_dir or run / "continuous_led_aligned_distance").resolve() / dataset_name
    output.mkdir(parents=True, exist_ok=True)
    _write_csv(output / "events.csv", records)
    _write_csv(output / "distance_binned_error.csv", trend)
    _plot_relation(
        output / "continuous_distance_vs_abs_led_error.pdf",
        dataset_name,
        args.stage,
        args.distance,
        distance,
        absolute_error,
        trend,
        stats,
    )
    _plot_mean_waveforms(
        output / "continuous_led_aligned_mean_waveforms.pdf",
        dataset_name,
        relative_time_ps,
        waveform_sum,
        waveform_count,
        threshold_mV,
    )

    voltage = np.asarray([row["bias_voltage_V"] for row in records], dtype=np.float64)
    per_voltage = []
    for value in np.unique(voltage[np.isfinite(voltage)]):
        mask = np.isclose(voltage, value, rtol=0.0, atol=1e-9)
        per_voltage.append({"voltage_V": float(value), **_correlation(distance[mask], absolute_error[mask])})
    _write_csv(output / "correlation_by_voltage.csv", per_voltage)

    t0_error = np.asarray(
        [
            value
            for row in records
            for value in (row["ch1_t0_minus_threshold_mV"], row["ch2_t0_minus_threshold_mV"])
        ],
        dtype=np.float64,
    )
    t0_error = t0_error[np.isfinite(t0_error)]
    valid_led = int(np.count_nonzero(np.isfinite(delta_led)))
    valid_continuous = int(np.count_nonzero(np.isfinite(distance)))
    summary = {
        "dataset": dataset_name,
        "stage": args.stage,
        "mode": mode,
        "family": family,
        "sources": [{"dataset": name, "preprocessed_dir": str(data.directory), "n_events": data.n_events} for name, data in sources],
        "population": "events from preprocessing cache after event selection; no prepared native-anchor materialization is used",
        "led_threshold_mV": threshold_mV,
        "alignment": "linear interpolation to an exact per-detector LED-relative grid; t_LED(detector)=0 exactly",
        "relative_window_ns": [float(window_ns[0]), float(window_ns[1])],
        "interpolation_step_ps": step_ps,
        "native_interval_median_ps": float(np.median(native_intervals_ps)),
        "native_interval_min_ps": float(np.min(native_intervals_ps)),
        "native_interval_max_ps": float(np.max(native_intervals_ps)),
        "distance_metric": args.distance,
        "distance_definition": {
            "rms": "sqrt(mean_t((s1_continuous_LED_aligned - s2_continuous_LED_aligned)^2))",
            "mean_abs": "mean_t(abs(s1_continuous_LED_aligned - s2_continuous_LED_aligned))",
            "max_abs": "max_t(abs(s1_continuous_LED_aligned - s2_continuous_LED_aligned))",
        }[args.distance],
        "led_error_definition": "(t_LED1 - t_LED2) - true_tof - calibration_bias; NO anchor correction",
        "true_tof_ps": true_tof_ps,
        "calibration_bias_ps": calibration_bias_ps,
        "calibration_source": calibration_source,
        "anchor_correction_used": False,
        "n_preprocessed_stage_events": len(records),
        "n_finite_led_pairs": valid_led,
        "n_valid_continuous_windows": valid_continuous,
        "n_correlation_pairs": int(np.count_nonzero(finite)),
        "t0_alignment_error_max_abs_mV": float(np.max(np.abs(t0_error))) if t0_error.size else float("nan"),
        "t0_alignment_error_rms_mV": float(np.sqrt(np.mean(t0_error**2))) if t0_error.size else float("nan"),
        "correlation": stats,
        "per_voltage": per_voltage,
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=True) + "\n", encoding="utf-8")

    print(
        f"{dataset_name} | {args.stage} | continuous LED alignment | threshold={threshold_mV:g} mV | "
        f"grid={step_ps:.6g} ps | events={len(records)} | finite LED={valid_led} | valid windows={valid_continuous}"
    )
    print(
        f"  no-anchor LED error | C={calibration_bias_ps:.6g} ps ({calibration_source}) | "
        f"Pearson r={stats['pearson_r']:+.5f} | Spearman rho={stats['spearman_rho']:+.5f} | "
        f"R^2={stats['linear_r_squared']:.6g}"
    )
    if t0_error.size:
        print(
            f"  interpolation check | max |V(t=0)-threshold|={np.max(np.abs(t0_error)):.3e} mV | "
            f"RMS={np.sqrt(np.mean(t0_error**2)):.3e} mV"
        )
    print(f"  outputs: {output}")


if __name__ == "__main__":
    main()
