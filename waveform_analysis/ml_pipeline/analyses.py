from __future__ import annotations

import copy
import csv
import gc
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from utils_fit import fit_ctr_ps

from .common import atomic_json, canonical_json
from .concatenate import concatenate_prepared_datasets
from .dataset import PreparedDataset, load_prepared_dataset
from .models import get_model
from .prepared_data import prepare_ml_dataset
from .reporting import LABELS
from .sample_mask import SAMPLE_CONSTANT_FRACTION, dataset_training_sample_mask
from .splits import semantic_seed
from .train import predict_indices, search_model, selected_model
from .view import calibrated_led, corrected_timing_residual, model_target, target_family, waveform_view


@dataclass
class _ThresholdPoint:
    threshold_mV: float
    dataset_name: str
    prepared_dir: Path
    event_keys: tuple[Any, ...]
    model_residual_ps: np.ndarray
    led_residual_ps: np.ndarray
    validation_ctr_ps: float
    selected_parameters_json: str
    retained_events: int
    validation_events: int


@dataclass(frozen=True)
class ThresholdScanResult:
    datasets: list[PreparedDataset]
    manifest: dict[str, Any]


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _dataset_name(dataset: PreparedDataset) -> str:
    if bool(dataset.manifest.get("concatenated", False)):
        return str(dataset.manifest.get("dataset_name", "concatenated"))
    return Path(dataset.manifest["source"]).stem


def _threshold_label(value: float) -> str:
    return f"{float(value):g}".replace("-", "m").replace(".", "p") + "mV"


def _threshold_config(
    base_config: dict[str, Any],
    threshold_mV: float,
    analysis_dir: Path,
) -> dict[str, Any]:
    config = copy.deepcopy(base_config)
    threshold = float(threshold_mV)
    label = _threshold_label(threshold)
    config["standard_methods"]["led_thresholds_mV"] = [threshold]
    config["preprocessing"]["prepared_dir"] = str(
        Path(base_config["preprocessing"]["prepared_dir"]).resolve()
        / "_led_threshold_scan"
        / label
    )
    config["experiment"]["output_dir"] = str(
        (analysis_dir / "diagnostics" / label).resolve()
    )
    if bool(config["experiment"].get("concatenate_datasets", False)):
        config["experiment"]["fixed_led_threshold_mV"] = threshold
    return config


def _event_keys(dataset: PreparedDataset, indices: np.ndarray) -> tuple[Any, ...]:
    indices = np.asarray(indices, dtype=np.int64)
    if bool(dataset.manifest.get("concatenated", False)):
        source = np.load(
            dataset.directory / str(dataset.manifest["source_dataset_file"]),
            mmap_mode="r",
        )
        event = np.load(
            dataset.directory / str(dataset.manifest["source_event_index_file"]),
            mmap_mode="r",
        )
        return tuple((str(source[i]), int(event[i])) for i in indices)
    values = np.asarray(dataset.event_index[indices], dtype=np.int64)
    return tuple(int(value) for value in values)


def _common_event_keys(points: list[_ThresholdPoint]) -> tuple[Any, ...]:
    if not points:
        return ()
    common = set(points[0].event_keys)
    for point in points[1:]:
        common.intersection_update(point.event_keys)
    return tuple(sorted(common, key=str))


def _aligned(values: np.ndarray, keys: tuple[Any, ...], common: tuple[Any, ...]) -> np.ndarray:
    lookup = {key: index for index, key in enumerate(keys)}
    positions = np.asarray([lookup[key] for key in common], dtype=np.int64)
    return np.asarray(values, dtype=np.float64)[positions]


def _evaluate_threshold_candidate(
    dataset: PreparedDataset,
    config: dict[str, Any],
    model_name: str,
    logger,
) -> _ThresholdPoint:
    mode = str(config["mode"])
    dataset_name = _dataset_name(dataset)
    validation = np.asarray(dataset.validation, dtype=np.int64)
    sample_mask = dataset_training_sample_mask(dataset, mode)
    spec = get_model(model_name)
    search = search_model(
        spec,
        config["models"][model_name],
        config,
        dataset,
        mode,
        seed=semantic_seed(
            int(config["validation"]["seed"]),
            dataset_name,
            mode,
            model_name,
            "search",
        ),
        dataset_name=dataset_name,
        sample_mask=sample_mask,
        logger=logger,
    )
    fitted = selected_model(search)
    prediction, _time, _pair = predict_indices(
        spec,
        fitted,
        dataset,
        mode,
        validation,
    )
    target = model_target(dataset, mode)
    model_residual = corrected_timing_residual(target[validation], prediction)
    led_residual = calibrated_led(dataset, mode)[validation]
    point = _ThresholdPoint(
        threshold_mV=float(dataset.manifest["led_threshold_mV"][target_family(mode)]),
        dataset_name=dataset_name,
        prepared_dir=dataset.directory,
        event_keys=_event_keys(dataset, validation),
        model_residual_ps=np.asarray(model_residual, dtype=np.float64),
        led_residual_ps=np.asarray(led_residual, dtype=np.float64),
        validation_ctr_ps=float(search.best.score),
        selected_parameters_json=canonical_json(search.best.candidate),
        retained_events=int(dataset.n_events),
        validation_events=int(validation.size),
    )
    search.best.artifact = None
    del fitted
    gc.collect()
    return point


def _plot_threshold_results(output_dir: Path, rows: list[dict[str, Any]], model_name: str) -> list[Path]:
    import matplotlib.pyplot as plt

    generated: list[Path] = []
    for dataset_name in sorted({row["dataset"] for row in rows}):
        subset = sorted(
            [row for row in rows if row["dataset"] == dataset_name],
            key=lambda row: float(row["threshold_mV"]),
        )
        threshold = np.asarray([row["threshold_mV"] for row in subset], dtype=float)
        led = np.asarray([row["led_validation_ctr_ps"] for row in subset], dtype=float)
        model = np.asarray([row["model_validation_ctr_ps"] for row in subset], dtype=float)
        improvement = np.asarray([row["relative_improvement_pct"] for row in subset], dtype=float)

        fig, ax = plt.subplots(figsize=(7.6, 5.0))
        ax.plot(threshold, led, marker="o", label="LED")
        ax.plot(threshold, model, marker="o", label=LABELS.get(model_name, model_name))
        ax.set_xlabel("LED threshold [mV]")
        ax.set_ylabel("Common-validation CTR [ps]")
        ax.set_title(dataset_name)
        ax.grid(True, alpha=0.2)
        ax.legend(loc="best")
        fig.tight_layout()
        path = output_dir / f"ctr_vs_led_threshold_{dataset_name}.pdf"
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path)
        plt.close(fig)
        generated.append(path)

        fig, ax = plt.subplots(figsize=(7.6, 5.0))
        ax.plot(threshold, improvement, marker="o")
        ax.axhline(0.0, linestyle="--", linewidth=1.0)
        ax.set_xlabel("LED threshold [mV]")
        ax.set_ylabel("Relative CTR improvement [%]")
        ax.set_title(dataset_name)
        ax.grid(True, alpha=0.2)
        fig.tight_layout()
        path = output_dir / f"relative_improvement_vs_led_threshold_{dataset_name}.pdf"
        fig.savefig(path)
        plt.close(fig)
        generated.append(path)
    return generated


def run_led_threshold_scan(
    preprocessed: list[Any],
    config: dict[str, Any],
    output_dir: Path,
    logger,
    progress,
    *,
    rebuild: bool,
) -> ThresholdScanResult:
    analysis = config["analyses"]["led_threshold_scan"]
    model_name = str(analysis["selection_model"])
    thresholds = sorted({float(value) for value in config["standard_methods"]["led_thresholds_mV"]})
    concatenate = bool(config["experiment"].get("concatenate_datasets", False))
    points: dict[str, list[_ThresholdPoint]] = {}
    failures: list[dict[str, Any]] = []

    logger.info(
        "LED threshold scan | candidates=%s mV | selection model=%s | blind data not used",
        ", ".join(f"{value:g}" for value in thresholds),
        LABELS.get(model_name, model_name),
    )

    if concatenate:
        dataset_name = str(config["experiment"].get("concatenated_dataset_name", "concatenated"))
        for threshold in thresholds:
            candidate_config = _threshold_config(config, threshold, output_dir)
            label = f"threshold scan | {dataset_name} | {threshold:g} mV"
            try:
                with progress.task("threshold_scan", label):
                    prepared_sources = [
                        prepare_ml_dataset(
                            source,
                            candidate_config,
                            rebuild=rebuild,
                            logger=logger,
                            log_summary=False,
                        )
                        for source in preprocessed
                    ]
                    dataset = concatenate_prepared_datasets(
                        prepared_sources,
                        Path(candidate_config["preprocessing"]["prepared_dir"]) / dataset_name,
                        candidate_config,
                        name=dataset_name,
                        rebuild=rebuild,
                        logger=None,
                    )
                    point = _evaluate_threshold_candidate(
                        dataset,
                        candidate_config,
                        model_name,
                        logger,
                    )
                points.setdefault(dataset_name, []).append(point)
                logger.info(
                    "Threshold result | %s | %.6g mV | validation CTR=%.3f ps | retained=%d",
                    dataset_name,
                    threshold,
                    point.validation_ctr_ps,
                    point.retained_events,
                )
            except Exception as exc:
                failures.append(
                    {
                        "dataset": dataset_name,
                        "threshold_mV": threshold,
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                )
    else:
        for threshold in thresholds:
            candidate_config = _threshold_config(config, threshold, output_dir)
            for source in preprocessed:
                dataset_name = Path(source.manifest["source"]).stem
                label = f"threshold scan | {dataset_name} | {threshold:g} mV"
                try:
                    with progress.task("threshold_scan", label):
                        dataset = prepare_ml_dataset(
                            source,
                            candidate_config,
                            rebuild=rebuild,
                            logger=logger,
                            log_summary=False,
                        )
                        point = _evaluate_threshold_candidate(
                            dataset,
                            candidate_config,
                            model_name,
                            logger,
                        )
                    points.setdefault(dataset_name, []).append(point)
                    logger.info(
                        "Threshold result | %s | %.6g mV | validation CTR=%.3f ps | retained=%d",
                        dataset_name,
                        threshold,
                        point.validation_ctr_ps,
                        point.retained_events,
                    )
                except Exception as exc:
                    failures.append(
                        {
                            "dataset": dataset_name,
                            "threshold_mV": threshold,
                            "error": f"{type(exc).__name__}: {exc}",
                        }
                    )

    fit_config = dict(config.get("fit") or {})
    minimum = int(fit_config.get("min_events", 100))
    seed = int(config["validation"]["seed"])
    rows: list[dict[str, Any]] = []
    selected_rows: list[dict[str, Any]] = []
    selected_datasets: list[PreparedDataset] = []
    selected_thresholds: dict[str, float] = {}

    for dataset_name, dataset_points in points.items():
        if not dataset_points:
            raise RuntimeError(f"{dataset_name}: no LED-threshold candidate completed")
        common = _common_event_keys(dataset_points)
        if len(common) < minimum:
            raise RuntimeError(
                f"{dataset_name}: only {len(common)} validation events are common across "
                f"successful LED thresholds; need at least {minimum}"
            )

        scored: list[tuple[float, float, _ThresholdPoint, dict[str, Any]]] = []
        for point in sorted(dataset_points, key=lambda item: item.threshold_mV):
            model_residual = _aligned(point.model_residual_ps, point.event_keys, common)
            led_residual = _aligned(point.led_residual_ps, point.event_keys, common)
            model_ctr = fit_ctr_ps(
                model_residual,
                fit_config,
                seed=semantic_seed(
                    seed,
                    dataset_name,
                    model_name,
                    "threshold_scan",
                    f"{point.threshold_mV:g}",
                ),
                bootstrap=False,
            ).ctr_ps
            led_ctr = fit_ctr_ps(
                led_residual,
                fit_config,
                seed=semantic_seed(
                    seed,
                    dataset_name,
                    "led",
                    "threshold_scan",
                    f"{point.threshold_mV:g}",
                ),
                bootstrap=False,
            ).ctr_ps
            improvement = 100.0 * (led_ctr - model_ctr) / led_ctr
            row = {
                "dataset": dataset_name,
                "threshold_mV": point.threshold_mV,
                "selection_model": model_name,
                "common_validation_events": len(common),
                "threshold_validation_events": point.validation_events,
                "threshold_retained_events": point.retained_events,
                "led_validation_ctr_ps": float(led_ctr),
                "model_validation_ctr_ps": float(model_ctr),
                "relative_improvement_pct": float(improvement),
                "candidate_search_validation_ctr_ps": point.validation_ctr_ps,
                "selected_parameters_json": point.selected_parameters_json,
            }
            rows.append(row)
            scored.append((float(model_ctr), point.threshold_mV, point, row))

        _score, selected_threshold, selected_point, selected_row = min(
            scored,
            key=lambda item: (item[0], item[1]),
        )
        selected_thresholds[dataset_name] = float(selected_threshold)
        selected_datasets.append(load_prepared_dataset(selected_point.prepared_dir))
        selected_rows.append(
            {
                "dataset": dataset_name,
                "selection_model": model_name,
                "selected_threshold_mV": float(selected_threshold),
                "common_validation_events": len(common),
                "validation_led_ctr_ps": selected_row["led_validation_ctr_ps"],
                "validation_model_ctr_ps": selected_row["model_validation_ctr_ps"],
                "validation_relative_improvement_pct": selected_row["relative_improvement_pct"],
                "threshold_selected_using": "common_validation_model_ctr",
                "blind_used_for_threshold_selection": False,
            }
        )
        logger.info(
            "Threshold selected | %s | %.6g mV | common-validation CTR=%.3f ps | events=%d",
            dataset_name,
            selected_threshold,
            selected_row["model_validation_ctr_ps"],
            len(common),
        )

    if not selected_datasets:
        raise RuntimeError("LED threshold scan produced no selectable dataset")

    csv_dir = output_dir / "csv"
    plot_dir = output_dir / "plots"
    threshold_csv = csv_dir / "threshold_scan.csv"
    selected_csv = csv_dir / "selected_thresholds.csv"
    _write_csv(threshold_csv, rows)
    _write_csv(selected_csv, selected_rows)
    if failures:
        _write_csv(csv_dir / "failed_thresholds.csv", failures)
    _plot_threshold_results(plot_dir, rows, model_name)

    manifest = {
        "enabled": True,
        "selection_model": model_name,
        "candidate_thresholds_mV": thresholds,
        "selected_thresholds_mV": selected_thresholds,
        "selection_population": "intersection of validation events retained by every successful candidate threshold",
        "selection_metric": "validation_ctr",
        "blind_used_for_threshold_selection": False,
        "successful_points": len(rows),
        "failed_points": len(failures),
        "output_dir": str(output_dir.resolve()),
    }
    atomic_json(output_dir / "manifest.json", manifest)
    return ThresholdScanResult(selected_datasets, manifest)


def _window_mask(
    time_ps: np.ndarray,
    base_mask: np.ndarray,
    right_limit_ns: float,
) -> tuple[np.ndarray, float, int]:
    time = np.asarray(time_ps, dtype=np.float64).reshape(-1)
    mask = np.asarray(base_mask, dtype=bool).reshape(-1)
    if time.size != mask.size:
        raise ValueError("Window scan time axis and sample mask have different lengths")
    dt_ps = float(np.median(np.diff(time))) if time.size > 1 else 0.0
    tolerance_ps = max(1e-6, 0.51 * abs(dt_ps))
    requested_ps = float(right_limit_ns) * 1000.0
    if requested_ps > float(np.max(time)) + tolerance_ps:
        raise ValueError(
            f"Requested right limit {right_limit_ns:g} ns exceeds the prepared input window"
        )
    temporal = time <= requested_ps + tolerance_ps
    combined = np.asarray(mask & temporal, dtype=bool)
    if np.count_nonzero(combined) < 2:
        raise ValueError("Temporal window leaves fewer than two retained input samples")
    effective = float(np.max(time[combined]) / 1000.0)
    return combined, effective, int(np.count_nonzero(temporal))


def _plot_window_results(output_dir: Path, rows: list[dict[str, Any]]) -> list[Path]:
    import matplotlib.pyplot as plt

    generated: list[Path] = []
    for dataset_name in sorted({row["dataset"] for row in rows}):
        subset = [row for row in rows if row["dataset"] == dataset_name]
        fig, ax = plt.subplots(figsize=(7.6, 5.0))
        for model_name in list(dict.fromkeys(row["model"] for row in subset)):
            model_rows = sorted(
                [row for row in subset if row["model"] == model_name],
                key=lambda row: float(row["right_limit_ns"]),
            )
            x = np.asarray([row["right_limit_ns"] for row in model_rows], dtype=float)
            y = np.asarray([row["blind_ctr_ps"] for row in model_rows], dtype=float)
            yerr = np.asarray(
                [row["blind_ctr_uncertainty_ps"] for row in model_rows],
                dtype=float,
            )
            ax.errorbar(
                x,
                y,
                yerr=yerr,
                marker="o",
                capsize=3,
                label=LABELS.get(model_name, model_name),
            )
        led_ctr = float(subset[0]["led_blind_ctr_ps"])
        led_err = float(subset[0]["led_blind_ctr_uncertainty_ps"])
        ax.axhline(led_ctr, linestyle="--", linewidth=1.2, label="LED")
        if np.isfinite(led_err) and led_err > 0:
            ax.axhspan(led_ctr - led_err, led_ctr + led_err, alpha=0.12)
        ax.set_xlabel("ML window right limit [ns]")
        ax.set_ylabel("Blind-test CTR [ps]")
        ax.set_title(dataset_name)
        ax.grid(True, alpha=0.2)
        ax.legend(loc="best")
        fig.tight_layout()
        path = output_dir / f"blind_ctr_vs_window_{dataset_name}.pdf"
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path)
        plt.close(fig)
        generated.append(path)
    return generated


def run_window_scan(
    datasets: list[PreparedDataset],
    config: dict[str, Any],
    output_dir: Path,
    logger,
    progress,
    final_metrics: dict[str, Any],
) -> dict[str, Any]:
    analysis = config["analyses"]["window_scan"]
    limits = list(analysis["right_limits_ns"])
    models = list(analysis["models"])
    mode = str(config["mode"])
    fit_config = dict(config.get("fit") or {})
    seed = int(config["validation"]["seed"])
    rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []

    logger.info(
        "ML window scan | right limits=%s ns | models=%s | target and splits fixed",
        ", ".join(f"{value:g}" for value in limits),
        ", ".join(LABELS.get(name, name) for name in models),
    )

    for dataset in datasets:
        dataset_name = _dataset_name(dataset)
        training = np.asarray(dataset.training, dtype=np.int64)
        test = np.asarray(dataset.test, dtype=np.int64)
        view = waveform_view(dataset, mode, training[:1])
        time_ps = np.asarray(view.time_ps, dtype=np.float64)
        base_mask = dataset_training_sample_mask(dataset, mode)
        left_ns = float(dataset.manifest["ml_input"]["window_ns"]["start"])
        threshold_mV = float(dataset.manifest["led_threshold_mV"][target_family(mode)])
        led_row = final_metrics["led"][dataset_name]
        target = model_target(dataset, mode)

        logger.info(
            "Window dataset | %s | LED=%.6g mV | base samples=%d/%d",
            dataset_name,
            threshold_mV,
            int(np.count_nonzero(base_mask)),
            int(base_mask.size),
        )

        for right_ns in limits:
            try:
                combined_mask, effective_right_ns, temporal_count = _window_mask(
                    time_ps,
                    base_mask,
                    right_ns,
                )
            except Exception as exc:
                for model_name in models:
                    label = f"window scan | {dataset_name} | {model_name} | {right_ns:g} ns"
                    progress.complete(
                        "window_scan",
                        label,
                        success=False,
                        note=f"{type(exc).__name__}: {exc}",
                    )
                    failures.append(
                        {
                            "dataset": dataset_name,
                            "right_limit_ns": right_ns,
                            "model": model_name,
                            "error": f"{type(exc).__name__}: {exc}",
                        }
                    )
                continue

            for model_name in models:
                label = f"window scan | {dataset_name} | {model_name} | {right_ns:g} ns"
                final_key = (dataset_name, model_name)
                reused = np.array_equal(combined_mask, base_mask) and final_key in final_metrics["models"]
                try:
                    if reused:
                        metric = final_metrics["models"][final_key]
                        validation_ctr = float(metric["validation_ctr_ps"])
                        test_row = metric["test_row"]
                        selected_parameters_json = metric["selected_parameters_json"]
                        progress.complete(
                            "window_scan",
                            label,
                            note="reused final full-window fit",
                        )
                    else:
                        with progress.task("window_scan", label):
                            spec = get_model(model_name)
                            search = search_model(
                                spec,
                                config["models"][model_name],
                                config,
                                dataset,
                                mode,
                                seed=semantic_seed(
                                    seed,
                                    dataset_name,
                                    mode,
                                    model_name,
                                    "search",
                                ),
                                dataset_name=dataset_name,
                                sample_mask=combined_mask,
                                logger=logger,
                            )
                            fitted = selected_model(search)
                            prediction, _masked_time, _masked_pair = predict_indices(
                                spec,
                                fitted,
                                dataset,
                                mode,
                                test,
                            )
                            residual = corrected_timing_residual(target[test], prediction)
                            ctr = fit_ctr_ps(
                                residual,
                                fit_config,
                                seed=semantic_seed(
                                    seed,
                                    dataset_name,
                                    mode,
                                    model_name,
                                    "test",
                                ),
                                bootstrap=True,
                            )
                            validation_ctr = float(search.best.score)
                            selected_parameters_json = canonical_json(search.best.candidate)
                            test_row = {
                                "ctr_ps": float(ctr.ctr_ps),
                                "ctr_uncertainty_ps": float(ctr.ctr_error_ps),
                                "bootstrap_samples": int(ctr.bootstrap_samples),
                                "bootstrap_successful": int(ctr.bootstrap_successful),
                                "n": int(ctr.n_valid),
                            }
                            search.best.artifact = None
                            del fitted
                            gc.collect()

                    blind_ctr = float(test_row["ctr_ps"])
                    blind_error = float(test_row["ctr_uncertainty_ps"])
                    led_ctr = float(led_row["ctr_ps"])
                    improvement = 100.0 * (led_ctr - blind_ctr) / led_ctr
                    rows.append(
                        {
                            "dataset": dataset_name,
                            "mode": mode,
                            "model": model_name,
                            "led_threshold_mV": threshold_mV,
                            "window_start_ns": left_ns,
                            "right_limit_ns": float(right_ns),
                            "effective_right_limit_ns": effective_right_ns,
                            "window_width_ns": float(right_ns - left_ns),
                            "grid_samples_within_window": temporal_count,
                            "input_samples_after_masks": int(np.count_nonzero(combined_mask)),
                            "input_samples_full_window": int(base_mask.size),
                            "constant_mask_fraction": SAMPLE_CONSTANT_FRACTION,
                            "validation_ctr_ps": validation_ctr,
                            "selected_parameters_json": selected_parameters_json,
                            "blind_ctr_ps": blind_ctr,
                            "blind_ctr_uncertainty_ps": blind_error,
                            "bootstrap_samples": int(test_row["bootstrap_samples"]),
                            "bootstrap_successful": int(test_row["bootstrap_successful"]),
                            "blind_events": int(test_row["n"]),
                            "led_blind_ctr_ps": led_ctr,
                            "led_blind_ctr_uncertainty_ps": float(led_row["ctr_uncertainty_ps"]),
                            "relative_improvement_vs_led_pct": float(improvement),
                            "reused_final_fit": bool(reused),
                            "target_recomputed": False,
                            "split_recomputed": False,
                        }
                    )
                    logger.info(
                        "Window result | %s | %s | right=%g ns | blind CTR=%.3f ± %.3f ps",
                        dataset_name,
                        LABELS.get(model_name, model_name),
                        right_ns,
                        blind_ctr,
                        blind_error,
                    )
                except Exception as exc:
                    failures.append(
                        {
                            "dataset": dataset_name,
                            "right_limit_ns": right_ns,
                            "model": model_name,
                            "error": f"{type(exc).__name__}: {exc}",
                        }
                    )

    if not rows:
        raise RuntimeError("ML window scan produced no successful result")

    csv_dir = output_dir / "csv"
    plot_dir = output_dir / "plots"
    _write_csv(csv_dir / "window_scan.csv", rows)
    if failures:
        _write_csv(csv_dir / "failed_windows.csv", failures)
    _plot_window_results(plot_dir, rows)

    manifest = {
        "enabled": True,
        "right_limits_ns": limits,
        "models": models,
        "uses_selected_led_threshold": True,
        "target_recomputed": False,
        "splits_recomputed": False,
        "blind_role": "post-selection sensitivity analysis only",
        "successful_points": len(rows),
        "failed_points": len(failures),
        "output_dir": str(output_dir.resolve()),
    }
    atomic_json(output_dir / "manifest.json", manifest)
    return manifest
