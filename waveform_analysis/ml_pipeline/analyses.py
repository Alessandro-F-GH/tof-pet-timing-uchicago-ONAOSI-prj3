from __future__ import annotations

import copy
import csv
import gc
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from utils_fit import fit_ctr_ps

from .common import atomic_json, canonical_json, read_json
from .concatenate import concatenate_prepared_datasets
from .dataset import PreparedDataset, load_prepared_dataset
from .models import get_model
from .prepared_data import prepare_ml_dataset
from .plot_style import (
    DOUBLE_COLUMN,
    LABELS,
    SINGLE_COLUMN,
    clean_axis,
    model_style,
    paper_context,
    save_figure,
)
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
    selected_searches: dict[tuple[str, str], Any]


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



def _threshold_checkpoint_path(output_dir: Path, dataset_name: str, threshold_mV: float) -> Path:
    safe_dataset = str(dataset_name).replace("/", "_").replace("\\", "_")
    return output_dir / "checkpoints" / safe_dataset / f"{_threshold_label(threshold_mV)}.npz"


def _save_threshold_checkpoint(output_dir: Path, point: _ThresholdPoint) -> Path:
    path = _threshold_checkpoint_path(output_dir, point.dataset_name, point.threshold_mV)
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = np.asarray(
        [json.dumps(key, separators=(",", ":")) for key in point.event_keys],
        dtype="U256",
    )
    temporary = path.with_name(f".{path.name}.tmp.npz")
    np.savez_compressed(
        temporary,
        threshold_mV=np.asarray([point.threshold_mV], dtype=np.float64),
        dataset_name=np.asarray([point.dataset_name]),
        prepared_dir=np.asarray([str(point.prepared_dir)]),
        event_keys=keys,
        model_residual_ps=np.asarray(point.model_residual_ps, dtype=np.float64),
        led_residual_ps=np.asarray(point.led_residual_ps, dtype=np.float64),
        validation_ctr_ps=np.asarray([point.validation_ctr_ps], dtype=np.float64),
        selected_parameters_json=np.asarray([point.selected_parameters_json]),
        retained_events=np.asarray([point.retained_events], dtype=np.int64),
        validation_events=np.asarray([point.validation_events], dtype=np.int64),
    )
    temporary.replace(path)
    return path


def _load_threshold_checkpoint(
    output_dir: Path,
    dataset_name: str,
    threshold_mV: float,
) -> _ThresholdPoint | None:
    path = _threshold_checkpoint_path(output_dir, dataset_name, threshold_mV)
    if not path.is_file():
        return None
    try:
        with np.load(path, allow_pickle=False) as data:
            raw_keys = [json.loads(str(value)) for value in data["event_keys"]]
            event_keys = tuple(
                tuple(value) if isinstance(value, list) else value
                for value in raw_keys
            )
            point = _ThresholdPoint(
                threshold_mV=float(data["threshold_mV"][0]),
                dataset_name=str(data["dataset_name"][0]),
                prepared_dir=Path(str(data["prepared_dir"][0])),
                event_keys=event_keys,
                model_residual_ps=np.asarray(data["model_residual_ps"], dtype=np.float64),
                led_residual_ps=np.asarray(data["led_residual_ps"], dtype=np.float64),
                validation_ctr_ps=float(data["validation_ctr_ps"][0]),
                selected_parameters_json=str(data["selected_parameters_json"][0]),
                retained_events=int(data["retained_events"][0]),
                validation_events=int(data["validation_events"][0]),
            )
    except (OSError, ValueError, KeyError, json.JSONDecodeError, IndexError):
        return None
    if not point.prepared_dir.is_dir():
        return None
    return point


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
) -> tuple[_ThresholdPoint, Any]:
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
    return point, search


def _plot_threshold_results(output_dir: Path, rows: list[dict[str, Any]], model_name: str) -> list[Path]:
    import matplotlib.pyplot as plt

    generated: list[Path] = []
    with paper_context():
        for dataset_name in sorted({row["dataset"] for row in rows}):
            subset = sorted(
                [row for row in rows if row["dataset"] == dataset_name],
                key=lambda row: float(row["threshold_mV"]),
            )
            threshold = np.asarray([row["threshold_mV"] for row in subset], dtype=float)
            led = np.asarray([row["led_validation_ctr_ps"] for row in subset], dtype=float)
            model = np.asarray([row["model_validation_ctr_ps"] for row in subset], dtype=float)
            improvement = np.asarray([row["relative_improvement_pct"] for row in subset], dtype=float)
            selected_index = int(np.nanargmin(model))

            fig, ax = plt.subplots(figsize=SINGLE_COLUMN)
            ax.plot(
                threshold, led,
                label="LED",
                **model_style("led"),
            )
            ax.plot(
                threshold, model,
                label=LABELS.get(model_name, model_name),
                **model_style(model_name),
            )
            ax.scatter(
                [threshold[selected_index]],
                [model[selected_index]],
                s=34,
                facecolors="none",
                edgecolors=model_style(model_name)["color"],
                linewidths=1.1,
                zorder=5,
            )
            ax.set_xlabel("LED threshold [mV]")
            ax.set_ylabel("Validation CTR [ps]")
            ax.legend(loc="best")
            clean_axis(ax, grid="y")
            fig.tight_layout()
            path = save_figure(fig, output_dir / f"ctr_vs_led_threshold_{dataset_name}.pdf")
            plt.close(fig)
            generated.append(path)

            fig, ax = plt.subplots(figsize=SINGLE_COLUMN)
            ax.plot(
                threshold,
                improvement,
                **model_style(model_name),
            )
            ax.axhline(0.0, color="#7F7F7F", linestyle=":", linewidth=0.9)
            ax.scatter(
                [threshold[selected_index]],
                [improvement[selected_index]],
                s=34,
                facecolors="none",
                edgecolors=model_style(model_name)["color"],
                linewidths=1.1,
                zorder=5,
            )
            ax.set_xlabel("LED threshold [mV]")
            ax.set_ylabel("CTR improvement [%]")
            clean_axis(ax, grid="y")
            fig.tight_layout()
            path = save_figure(
                fig,
                output_dir / f"relative_improvement_vs_led_threshold_{dataset_name}.pdf",
            )
            plt.close(fig)
            generated.append(path)
    return generated

def _load_completed_threshold_scan(
    preprocessed: list[Any],
    config: dict[str, Any],
    output_dir: Path,
) -> ThresholdScanResult | None:
    manifest_path = output_dir / "manifest.json"
    selected_path = output_dir / "csv" / "selected_thresholds.csv"
    threshold_path = output_dir / "csv" / "threshold_scan.csv"
    if not (manifest_path.is_file() and selected_path.is_file() and threshold_path.is_file()):
        return None
    try:
        manifest = read_json(manifest_path)
        selected_rows = _read_csv(selected_path)
    except (OSError, ValueError, KeyError, csv.Error):
        return None

    model_name = str(config["analyses"]["led_threshold_scan"]["selection_model"])
    configured_thresholds = sorted(
        {float(value) for value in config["standard_methods"]["led_thresholds_mV"]}
    )
    stored_thresholds = sorted(
        float(value) for value in manifest.get("candidate_thresholds_mV", [])
    )
    if (
        not bool(manifest.get("enabled", False))
        or str(manifest.get("selection_model", "")) != model_name
        or stored_thresholds != configured_thresholds
    ):
        return None

    concatenate = bool(config["experiment"].get("concatenate_datasets", False))
    if concatenate:
        expected = {
            str(config["experiment"].get("concatenated_dataset_name", "concatenated"))
        }
    else:
        expected = {
            Path(source.manifest["source"]).stem
            for source in preprocessed
        }
    selected = {
        str(row.get("dataset")): float(row["selected_threshold_mV"])
        for row in selected_rows
        if row.get("dataset") and row.get("selected_threshold_mV") not in {None, ""}
    }
    if set(selected) != expected:
        return None

    datasets: list[PreparedDataset] = []
    for dataset_name in sorted(expected):
        threshold = selected[dataset_name]
        candidate_config = _threshold_config(config, threshold, output_dir)
        if concatenate:
            prepared_dir = (
                Path(candidate_config["preprocessing"]["prepared_dir"]) / dataset_name
            )
        else:
            prepared_dir = (
                Path(candidate_config["preprocessing"]["prepared_dir"]) / dataset_name
            )
        try:
            datasets.append(load_prepared_dataset(prepared_dir))
        except Exception:
            return None
    return ThresholdScanResult(datasets, manifest, {})


def run_led_threshold_scan(
    preprocessed: list[Any],
    config: dict[str, Any],
    output_dir: Path,
    logger,
    progress,
    *,
    rebuild: bool,
    resume: bool = False,
) -> ThresholdScanResult:
    analysis = config["analyses"]["led_threshold_scan"]
    model_name = str(analysis["selection_model"])
    thresholds = sorted({float(value) for value in config["standard_methods"]["led_thresholds_mV"]})
    concatenate = bool(config["experiment"].get("concatenate_datasets", False))
    mode = str(config["mode"])
    family = target_family(mode)
    fit_config = dict(config.get("fit") or {})
    seed = int(config["validation"]["seed"])
    failures: list[dict[str, Any]] = []

    if resume and not rebuild:
        completed = _load_completed_threshold_scan(
            preprocessed,
            config,
            output_dir,
        )
        if completed is not None:
            logger.info(
                "LED threshold scan | reused completed scan | datasets=%d",
                len(completed.datasets),
            )
            for _dataset in completed.datasets:
                for _threshold in thresholds:
                    progress.complete(
                        "led_scan",
                        f"threshold scan | {_dataset_name(_dataset)} | {_threshold:g} mV",
                        note="reused completed scan",
                        announce=False,
                    )
            return completed

    logger.info(
        "LED threshold scan | candidates=%s mV | selection model=%s | blind data not used",
        ", ".join(f"{value:g}" for value in thresholds),
        LABELS.get(model_name, model_name),
    )

    # Phase 1: prepare every threshold-specific dataset without fitting ML models.
    # This makes the common validation population known before the expensive scan.
    prepared_candidates: dict[str, list[tuple[float, dict[str, Any], PreparedDataset]]] = {}

    if concatenate:
        dataset_name = str(config["experiment"].get("concatenated_dataset_name", "concatenated"))
        for threshold in thresholds:
            candidate_config = _threshold_config(config, threshold, output_dir)
            label = f"threshold scan | {dataset_name} | {threshold:g} mV"
            try:
                prepared_sources = [
                    prepare_ml_dataset(
                        source,
                        candidate_config,
                        rebuild=rebuild,
                        logger=logger,
                        log_summary=False,
                        write_diagnostics=False,
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
                prepared_candidates.setdefault(dataset_name, []).append(
                    (threshold, candidate_config, dataset)
                )
            except Exception as exc:
                progress.complete(
                    "led_scan",
                    label,
                    note=f"prepare failed: {type(exc).__name__}: {exc}",
                    announce=False,
                )
                failures.append(
                    {
                        "dataset": dataset_name,
                        "threshold_mV": threshold,
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                )
    else:
        for source in preprocessed:
            dataset_name = Path(source.manifest["source"]).stem
            for threshold in thresholds:
                candidate_config = _threshold_config(config, threshold, output_dir)
                label = f"threshold scan | {dataset_name} | {threshold:g} mV"
                try:
                    dataset = prepare_ml_dataset(
                        source,
                        candidate_config,
                        rebuild=rebuild,
                        logger=logger,
                        log_summary=False,
                        write_diagnostics=False,
                    )
                    prepared_candidates.setdefault(dataset_name, []).append(
                        (threshold, candidate_config, dataset)
                    )
                except Exception as exc:
                    progress.complete(
                        "led_scan",
                        label,
                        note=f"prepare failed: {type(exc).__name__}: {exc}",
                        announce=False,
                    )
                    failures.append(
                        {
                            "dataset": dataset_name,
                            "threshold_mV": threshold,
                            "error": f"{type(exc).__name__}: {exc}",
                        }
                    )

    rows: list[dict[str, Any]] = []
    selected_rows: list[dict[str, Any]] = []
    selected_datasets: list[PreparedDataset] = []
    selected_thresholds: dict[str, float] = {}
    selected_searches: dict[tuple[str, str], Any] = {}
    initial_led_thresholds: dict[str, float] = {}

    # Phase 2: for each dataset, fit the development-best LED threshold first,
    # then scan every other threshold exactly once.
    for dataset_name, candidates in prepared_candidates.items():
        if not candidates:
            raise RuntimeError(f"{dataset_name}: no LED-threshold candidate could be prepared")

        def development_led_score(item):
            threshold, _candidate_config, dataset = item
            score = float(dataset.manifest["led_development_ctr_ps"][family])
            return score, float(threshold)

        best_led_candidate = min(candidates, key=development_led_score)
        best_led_threshold = float(best_led_candidate[0])
        initial_led_thresholds[dataset_name] = best_led_threshold
        logger.info(
            "LED scan order | %s | fit development-best LED first: %.6g mV",
            dataset_name,
            best_led_threshold,
        )

        ordered = [best_led_candidate] + [
            item for item in sorted(candidates, key=lambda item: item[0])
            if not np.isclose(float(item[0]), best_led_threshold, rtol=0.0, atol=1e-12)
        ]

        points: dict[float, _ThresholdPoint] = {}
        searches: dict[float, Any] = {}
        datasets_by_threshold = {float(item[0]): item[2] for item in candidates}
        configs_by_threshold = {float(item[0]): item[1] for item in candidates}

        for order_index, (threshold, candidate_config, dataset) in enumerate(ordered):
            threshold = float(threshold)
            label = f"threshold scan | {dataset_name} | {threshold:g} mV"
            point = None
            search = None

            # Always fit the development-best LED point first so that its trained
            # model is available for reuse. Other completed checkpoints can be
            # reused during resume without retraining.
            if order_index != 0 and not rebuild:
                point = _load_threshold_checkpoint(output_dir, dataset_name, threshold)

            try:
                if point is not None:
                    progress.complete(
                        "led_scan",
                        label,
                        note="reused checkpoint",
                        announce=False,
                    )
                else:
                    with progress.task(
                        "led_scan",
                        label,
                        announce_start=False,
                        announce_finish=False,
                    ):
                        point, search = _evaluate_threshold_candidate(
                            dataset,
                            candidate_config,
                            model_name,
                            logger,
                        )
                        _save_threshold_checkpoint(output_dir, point)
            except Exception as exc:
                failures.append(
                    {
                        "dataset": dataset_name,
                        "threshold_mV": threshold,
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                )
                continue

            points[threshold] = point
            if search is not None:
                searches[threshold] = search
            logger.info(
                "LED scan | %s | %.6g mV | validation CTR=%.3f ps | retained=%d | %s",
                dataset_name,
                threshold,
                point.validation_ctr_ps,
                point.retained_events,
                progress.stage_text("led_scan"),
            )

        if not points:
            raise RuntimeError(f"{dataset_name}: every LED-threshold model fit failed")
        common = _common_event_keys(list(points.values()))
        if len(common) < 2:
            raise RuntimeError(
                f"{dataset_name}: fewer than two validation events are common across successful LED thresholds"
            )

        scored: list[tuple[float, float, _ThresholdPoint, dict[str, Any]]] = []
        for threshold in sorted(points):
            point = points[threshold]
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
                    f"{threshold:g}",
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
                    f"{threshold:g}",
                ),
                bootstrap=False,
            ).ctr_ps
            improvement = 100.0 * (led_ctr - model_ctr) / led_ctr
            row = {
                "dataset": dataset_name,
                "threshold_mV": threshold,
                "selection_model": model_name,
                "common_validation_events": len(common),
                "threshold_validation_events": point.validation_events,
                "threshold_retained_events": point.retained_events,
                "led_validation_ctr_ps": float(led_ctr),
                "model_validation_ctr_ps": float(model_ctr),
                "relative_improvement_pct": float(improvement),
                "candidate_search_validation_ctr_ps": point.validation_ctr_ps,
                "selected_parameters_json": point.selected_parameters_json,
                "development_best_led_threshold": bool(
                    np.isclose(threshold, best_led_threshold, rtol=0.0, atol=1e-12)
                ),
            }
            rows.append(row)
            scored.append((float(model_ctr), threshold, point, row))

        _score, selected_threshold, selected_point, selected_row = min(
            scored,
            key=lambda item: (item[0], item[1]),
        )
        selected_threshold = float(selected_threshold)

        # On an interrupted/resumed run the winning point may have come from a
        # numerical checkpoint. Materialize only that one model if necessary.
        selected_search = searches.get(selected_threshold)
        if selected_search is None:
            logger.info(
                "LED scan | %s | materializing selected %.6g mV model from checkpoint",
                dataset_name,
                selected_threshold,
            )
            selected_point, selected_search = _evaluate_threshold_candidate(
                datasets_by_threshold[selected_threshold],
                configs_by_threshold[selected_threshold],
                model_name,
                logger,
            )
            _save_threshold_checkpoint(output_dir, selected_point)

        # Keep only the winning trained artifact for direct final evaluation.
        for threshold, search in searches.items():
            if threshold != selected_threshold:
                search.best.artifact = None
        gc.collect()

        selected_thresholds[dataset_name] = selected_threshold
        selected_datasets.append(datasets_by_threshold[selected_threshold])
        selected_searches[(dataset_name, model_name)] = selected_search
        selected_rows.append(
            {
                "dataset": dataset_name,
                "selection_model": model_name,
                "development_best_led_threshold_mV": best_led_threshold,
                "selected_threshold_mV": selected_threshold,
                "common_validation_events": len(common),
                "validation_led_ctr_ps": selected_row["led_validation_ctr_ps"],
                "validation_model_ctr_ps": selected_row["model_validation_ctr_ps"],
                "validation_relative_improvement_pct": selected_row["relative_improvement_pct"],
                "threshold_selected_using": "common_validation_model_ctr",
                "blind_used_for_threshold_selection": False,
            }
        )
        logger.info(
            "Threshold selected | %s | %.6g mV | common-validation CTR=%.3f ps | events=%d%s",
            dataset_name,
            selected_threshold,
            selected_row["model_validation_ctr_ps"],
            len(common),
            " | reusing already-fitted model",
        )

    if not selected_datasets:
        raise RuntimeError("LED threshold scan produced no selectable dataset")

    csv_dir = output_dir / "csv"
    _write_csv(csv_dir / "threshold_scan.csv", rows)
    _write_csv(csv_dir / "selected_thresholds.csv", selected_rows)
    if failures:
        _write_csv(csv_dir / "failed_thresholds.csv", failures)

    manifest = {
        "enabled": True,
        "selection_model": model_name,
        "candidate_thresholds_mV": thresholds,
        "development_best_led_thresholds_mV": initial_led_thresholds,
        "selected_thresholds_mV": selected_thresholds,
        "scan_order_policy": "fit development-best LED threshold first, then skip it in the remaining ML threshold scan",
        "selection_population": "intersection of validation events retained by every successful candidate threshold",
        "selection_metric": "validation_ctr",
        "blind_used_for_threshold_selection": False,
        "successful_points": len(rows),
        "failed_points": len(failures),
        "output_dir": str(output_dir.resolve()),
    }
    atomic_json(output_dir / "manifest.json", manifest)
    return ThresholdScanResult(selected_datasets, manifest, selected_searches)


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
    temporal = time <= requested_ps + tolerance_ps
    combined = np.asarray(mask & temporal, dtype=bool)
    if np.count_nonzero(combined) < 2:
        raise ValueError("Temporal window leaves fewer than two retained input samples")
    effective = float(np.max(time[combined]) / 1000.0)
    return combined, effective, int(np.count_nonzero(temporal))


def _plot_window_results(output_dir: Path, rows: list[dict[str, Any]]) -> list[Path]:
    import matplotlib.pyplot as plt

    generated: list[Path] = []
    with paper_context():
        for dataset_name in sorted({row["dataset"] for row in rows}):
            subset = [row for row in rows if row["dataset"] == dataset_name]
            fig, ax = plt.subplots(figsize=SINGLE_COLUMN)
            for model_index, model_name in enumerate(dict.fromkeys(row["model"] for row in subset)):
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
                    capsize=2.5,
                    label=LABELS.get(model_name, model_name),
                    **model_style(model_name, model_index),
                )
            led_ctr = float(subset[0]["led_blind_ctr_ps"])
            led_err = float(subset[0]["led_blind_ctr_uncertainty_ps"])
            led_style = model_style("led")
            ax.axhline(
                led_ctr,
                color=led_style["color"],
                linestyle=led_style["linestyle"],
                linewidth=1.0,
                label="LED",
            )
            if np.isfinite(led_err) and led_err > 0:
                ax.axhspan(led_ctr - led_err, led_ctr + led_err, color="#7F7F7F", alpha=0.10)
            ax.set_xlabel("Window right limit [ns]")
            ax.set_ylabel("CTR [ps]")
            ax.legend(loc="best")
            clean_axis(ax, grid="y")
            fig.tight_layout()
            path = save_figure(fig, output_dir / f"blind_ctr_vs_window_{dataset_name}.pdf")
            plt.close(fig)
            generated.append(path)
    return generated


def _read_csv(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def make_analysis_plots(
    run_dir: str | Path,
    plot_root: str | Path,
) -> list[Path]:
    """Rebuild integrated-analysis figures from persisted CSV results."""
    run = Path(run_dir).resolve()
    destination = Path(plot_root).resolve()
    generated: list[Path] = []

    threshold_root = run / "analyses" / "led_threshold"
    threshold_rows = _read_csv(threshold_root / "csv" / "threshold_scan.csv")
    if threshold_rows:
        manifest_path = threshold_root / "manifest.json"
        manifest = read_json(manifest_path) if manifest_path.is_file() else {}
        model_name = str(
            manifest.get("selection_model")
            or threshold_rows[0].get("selection_model")
            or "cnn"
        )
        generated.extend(
            _plot_threshold_results(
                destination / "led_threshold",
                threshold_rows,
                model_name,
            )
        )

    window_root = run / "analyses" / "window"
    window_rows = _read_csv(window_root / "csv" / "window_scan.csv")
    if window_rows:
        generated.extend(
            _plot_window_results(
                destination / "window_scan",
                window_rows,
            )
        )
    return generated

def run_window_scan(
    datasets: list[PreparedDataset],
    config: dict[str, Any],
    output_dir: Path,
    logger,
    progress,
    final_metrics: dict[str, Any],
    *,
    resume: bool = False,
) -> dict[str, Any]:
    analysis = config["analyses"]["window_scan"]
    limits = list(analysis["right_limits_ns"])
    models = list(analysis["models"])
    mode = str(config["mode"])
    fit_config = dict(config.get("fit") or {})
    seed = int(config["validation"]["seed"])
    csv_dir = output_dir / "csv"
    window_csv = csv_dir / "window_scan.csv"
    failed_csv = csv_dir / "failed_windows.csv"
    loaded_rows: list[dict[str, Any]] = _read_csv(window_csv) if resume else []
    unique_rows: dict[tuple[str, str, float], dict[str, Any]] = {}
    for row in loaded_rows:
        if not row.get("dataset") or not row.get("model") or row.get("right_limit_ns") in {None, ""}:
            continue
        key = (
            str(row.get("dataset")),
            str(row.get("model")),
            float(row.get("right_limit_ns")),
        )
        unique_rows[key] = row
    rows: list[dict[str, Any]] = list(unique_rows.values())
    failures: list[dict[str, Any]] = []
    completed = set(unique_rows)
    if resume and len(rows) != len(loaded_rows):
        _write_csv(window_csv, rows)

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
        base_right_ns = float(dataset.manifest["ml_input"]["window_ns"]["end"])
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
            is_base_window = np.isclose(
                float(right_ns),
                base_right_ns,
                rtol=0.0,
                atol=max(1e-12, 1e-9 * max(1.0, abs(base_right_ns))),
            )
            try:
                if is_base_window:
                    combined_mask = np.asarray(base_mask, dtype=bool).copy()
                    effective_right_ns = float(np.max(time_ps[combined_mask]) / 1000.0)
                    temporal_count = int(base_mask.size)
                else:
                    combined_mask, effective_right_ns, temporal_count = _window_mask(
                        time_ps,
                        base_mask,
                        right_ns,
                    )
            except Exception as exc:
                for model_name in models:
                    label = f"window scan | {dataset_name} | {model_name} | {right_ns:g} ns"
                    progress.complete(
                        f"window_scan:{model_name}",
                        label,
                        note=f"skipped: {type(exc).__name__}: {exc}",
                    )
                    failures.append(
                        {
                            "dataset": dataset_name,
                            "right_limit_ns": right_ns,
                            "model": model_name,
                            "error": f"{type(exc).__name__}: {exc}",
                        }
                    )
                    _write_csv(failed_csv, failures)
                continue

            for model_name in models:
                label = f"window scan | {dataset_name} | {model_name} | {right_ns:g} ns"
                point_key = (dataset_name, model_name, float(right_ns))
                if point_key in completed:
                    progress.complete(
                        f"window_scan:{model_name}",
                        label,
                        note="reused window checkpoint",
                    )
                    continue
                final_key = (dataset_name, model_name)
                reused = (
                    (is_base_window or np.array_equal(combined_mask, base_mask))
                    and final_key in final_metrics["models"]
                )
                try:
                    if reused:
                        metric = final_metrics["models"][final_key]
                        validation_ctr = float(metric["validation_ctr_ps"])
                        test_row = metric["test_row"]
                        selected_parameters_json = metric["selected_parameters_json"]
                        progress.complete(
                            f"window_scan:{model_name}",
                            label,
                            note="reused final full-window fit",
                        )
                    else:
                        with progress.task(f"window_scan:{model_name}", label):
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
                    completed.add(point_key)
                    _write_csv(window_csv, rows)
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
                    _write_csv(failed_csv, failures)

    if not rows:
        raise RuntimeError("ML window scan produced no successful result")

    expected = {
        (str(_dataset_name(dataset)), str(model_name), float(right_ns))
        for dataset in datasets
        for model_name in models
        for right_ns in limits
    }
    missing = sorted(expected - completed, key=lambda item: (item[0], item[1], item[2]))
    if failures:
        _write_csv(failed_csv, failures)
    elif failed_csv.is_file():
        failed_csv.unlink()

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
        "missing_points": len(missing),
        "output_dir": str(output_dir.resolve()),
    }
    atomic_json(output_dir / "manifest.json", manifest)
    if missing:
        preview = ", ".join(
            f"{dataset}/{model}/{right:g}ns"
            for dataset, model, right in missing[:8]
        )
        suffix = "" if len(missing) <= 8 else f", ... (+{len(missing) - 8})"
        raise RuntimeError(
            f"ML window scan incomplete: {len(missing)} configured point(s) missing: "
            f"{preview}{suffix}"
        )
    return manifest
