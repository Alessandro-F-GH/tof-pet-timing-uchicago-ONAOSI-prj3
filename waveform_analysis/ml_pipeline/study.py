from utils_fit import fit_ctr_ps

from __future__ import annotations

import copy
import csv
import gc
import json
import logging
import shutil
from pathlib import Path
from typing import Any

import numpy as np

from .analyses import run_blind_led_threshold_scan, run_led_threshold_scan, run_window_scan
from .common import canonical_hash, read_json, voltage_from_name
from .concatenate import concatenate_prepared_datasets
from .config import discover_root_files, load_config, public_config
from .data import preprocess_selected
from .event_selection import select_events
from .plot_rebuild import rebuild_study_plots
from .models import get_model
from .prepared_data import prepare_ml_dataset
from .progress import ProgressTracker
from .reporting import LABELS
from .sample_mask import (
    SAMPLE_CONSTANT_FRACTION,
    apply_sample_mask,
    apply_sample_mask_to_time,
    dataset_training_sample_mask,
)
from .splits import semantic_seed
from .stats import ctr_estimate, format_residual_summary, residual_summary
from .storage import RunStore
from .train import predict_indices, save_model, search_model, selected_model
from .view import (
    calibrated_led,
    corrected_timing_residual,
    inverse_pair,
    model_target,
    standard_delta,
    target_family,
    waveform_view,
)


def _logger(run_dir: Path):
    logger = logging.getLogger(f"waveform-study:{run_dir}")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    logger.propagate = False
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    for handler in (
        logging.StreamHandler(),
        logging.FileHandler(run_dir / "study.log", encoding="utf-8"),
    ):
        handler.setFormatter(fmt)
        logger.addHandler(handler)
    return logger



def _csv_rows(path: Path) -> list[dict[str, str]]:
    import csv

    if not path.is_file():
        return []
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def _assert_resume_config_matches(config: dict[str, Any], run_dir: Path) -> None:
    """Refuse to mix artifacts from different resolved study configurations."""
    manifest_path = run_dir / "manifest.json"
    if not manifest_path.is_file():
        return
    try:
        manifest = read_json(manifest_path)
    except Exception as exc:
        raise RuntimeError(
            f"Cannot resume study with unreadable manifest: {manifest_path}"
        ) from exc
    stored_config = manifest.get("config")
    if not isinstance(stored_config, dict):
        raise RuntimeError(
            f"Cannot resume study because manifest has no resolved config: {manifest_path}"
        )
    stored_hash = canonical_hash(stored_config)
    current_hash = str(config.get("_config_fingerprint", ""))
    if stored_hash != current_hash:
        raise RuntimeError(
            "Cannot resume study with a different configuration. "
            "Use the original config, choose a new output directory, or use --overwrite."
        )


def _completed_run_matches(config: dict[str, Any], run_dir: Path) -> bool:
    """Return True only when the existing run is complete for this exact config."""
    manifest_path = run_dir / "manifest.json"
    results_path = run_dir / "csv" / "results.csv"
    if not manifest_path.is_file() or not results_path.is_file():
        return False

    try:
        manifest = read_json(manifest_path)
    except Exception:
        return False

    stored_config = manifest.get("config")
    if not isinstance(stored_config, dict):
        return False
    if canonical_hash(stored_config) != str(config.get("_config_fingerprint", "")):
        return False

    # Verify persisted outputs even when status="complete". This protects
    # against a stale/corrupt manifest or files removed after completion.
    roots = discover_root_files(config)
    concatenate = bool(config["experiment"].get("concatenate_datasets", False))
    expected_datasets = 1 if concatenate else len(roots)
    datasets = manifest.get("datasets")
    if not isinstance(datasets, dict) or len(datasets) != expected_datasets:
        return False

    rows = _csv_rows(results_path)
    if not rows:
        return False
    available = {
        (str(row.get("dataset")), str(row.get("method")), str(row.get("stage")))
        for row in rows
    }
    for dataset_name in datasets:
        required = {
            (dataset_name, "led", "development_selection"),
            (dataset_name, "led", "train"),
            (dataset_name, "led", "test"),
        }
        if bool(config["cfd"]):
            required.update(
                {
                    (dataset_name, "cfd", "development_selection"),
                    (dataset_name, "cfd", "train"),
                    (dataset_name, "cfd", "test"),
                }
            )
        for model_name in config["models"]:
            required.update(
                {
                    (dataset_name, model_name, "validation"),
                    (dataset_name, model_name, "train"),
                    (dataset_name, model_name, "test"),
                }
            )
        if not required.issubset(available):
            return False

    analyses = manifest.get("analyses")
    if not isinstance(analyses, dict):
        return False

    led_enabled = bool(config["analyses"]["led_threshold_scan"]["enabled"])
    led_manifest = analyses.get("led_threshold_scan")
    if not isinstance(led_manifest, dict) or bool(led_manifest.get("enabled")) != led_enabled:
        return False
    if led_enabled:
        selected = led_manifest.get("selected_thresholds_mV")
        if not isinstance(selected, dict) or len(selected) != expected_datasets:
            return False

    window_cfg = config["analyses"]["window_scan"]
    window_enabled = bool(window_cfg["enabled"])
    window_manifest = analyses.get("window_scan")
    if not isinstance(window_manifest, dict) or bool(window_manifest.get("enabled")) != window_enabled:
        return False
    if window_enabled:
        window_rows = _csv_rows(run_dir / "analyses" / "window" / "csv" / "window_scan.csv")
        expected_points = (
            expected_datasets
            * len(window_cfg["right_limits_ns"])
            * len(window_cfg["models"])
        )
        combinations = {
            (
                str(row.get("dataset")),
                str(row.get("model")),
                float(row.get("right_limit_ns")),
            )
            for row in window_rows
            if row.get("dataset")
            and row.get("model")
            and row.get("right_limit_ns") not in {None, ""}
        }
        if len(combinations) != expected_points:
            return False
        for dataset_name in datasets:
            for model_name in window_cfg["models"]:
                for right_ns in window_cfg["right_limits_ns"]:
                    if (dataset_name, str(model_name), float(right_ns)) not in combinations:
                        return False

    return True



def _metric_row(config, name, voltage, mode, method, residual, population_n, seed, logger, stage="test"):
    values = np.asarray(residual, dtype=float)
    finite = values[np.isfinite(values)]
    context = f"{name}/{mode}/{method}/{stage}"
    try:
        result = ctr_estimate(
            finite,
            config.get("fit"),
            seed=int(seed),
            bootstrap=True,
        )
    except ValueError as exc:
        detail = format_residual_summary(residual_summary(values))
        logger.error("CTR unavailable | %s | reason=%s | %s", context, exc, detail)
        raise RuntimeError(f"{context}: CTR unavailable: {exc}; {detail}") from exc
    return {
        "dataset": name,
        "voltage_V": voltage,
        "mode": mode,
        "method": method,
        "stage": stage,
        "ctr_ps": float(result.ctr_ps),
        "ctr_uncertainty_ps": float(result.ctr_error_ps),
        "center_ps": float(result.center_ps),
        "coverage_fraction": float(result.coverage_fraction),
        "interval_events": int(result.interval_events),
        "interval_low_ps": float(result.interval_low_ps),
        "interval_high_ps": float(result.interval_high_ps),
        "interval_width_ps": float(result.interval_width_ps),
        "gaussian_equivalent_scale": float(result.gaussian_equivalent_scale),
        "bootstrap_samples": int(result.bootstrap_samples),
        "bootstrap_successful": int(result.bootstrap_successful),
        "n": int(result.n_valid),
        "population_n": int(population_n),
        "crossing_efficiency": float(result.n_valid / max(1, int(population_n))),
    }


def _selection_row(name, voltage, mode, method, score, parameters, metric):
    return {
        "dataset": name,
        "voltage_V": voltage,
        "mode": mode,
        "method": method,
        "stage": "development_selection" if method in {"led", "cfd"} else "validation",
        "selection_score": float(score),
        "selection_metric": metric,
        "ctr_ps": float(score) if metric in {"development_ctr", "validation_ctr"} else float("nan"),
        "ctr_uncertainty_ps": float("nan"),
        "center_ps": float("nan"),
        "n": 0,
        "population_n": 0,
        "crossing_efficiency": float("nan"),
        "parameters_json": json.dumps(parameters, sort_keys=True),
    }


def _dataset_name(dataset):
    if bool(dataset.manifest.get("concatenated", False)):
        return str(dataset.manifest.get("dataset_name", "concatenated"))
    return Path(dataset.manifest["source"]).stem


def _dataset_voltage(dataset, name):
    if bool(dataset.manifest.get("concatenated", False)):
        return float("nan")
    values = np.asarray(dataset.bias_voltage_V, dtype=float)
    finite = values[np.isfinite(values)]
    return float(np.median(finite)) if finite.size else voltage_from_name(name)


def _prepare_without_threshold_scan(preprocessed, config, rebuild, logger, progress):
    concatenate = bool(config["experiment"].get("concatenate_datasets", False))
    prep_config = copy.deepcopy(config)
    if concatenate:
        fixed_led = float(config["experiment"]["fixed_led_threshold_mV"])
        prep_config["standard_methods"]["led_thresholds_mV"] = [fixed_led]
        logger.info("Fixed LED threshold for concatenated study | %.6g mV", fixed_led)

    prepared = []
    for source in preprocessed:
        label = f"prepare ML dataset | {Path(source.manifest['source']).name}"
        with progress.task("led_prepare", label):
            prepared.append(
                prepare_ml_dataset(
                    source,
                    prep_config,
                    rebuild=rebuild,
                    logger=logger,
                )
            )

    if not concatenate:
        return prepared

    name = str(config["experiment"].get("concatenated_dataset_name", "concatenated"))
    with progress.task("led_prepare", f"concatenate ML dataset | {name}"):
        return [
            concatenate_prepared_datasets(
                prepared,
                Path(config["preprocessing"]["prepared_dir"]) / name,
                prep_config,
                name=name,
                rebuild=rebuild,
                logger=logger,
            )
        ]


def _base_manifest(config, concatenate, roots, analyses):
    coverage = float(config["fit"].get("coverage_fraction", 0.90))
    manifest = {
        "schema_version": 13,
        "status": "running",
        "config_fingerprint": str(config.get("_config_fingerprint", "")),
        "protocol": "single_mode_configured_analyses_holdout",
        "mode": str(config["mode"]),
        "concatenate_datasets": concatenate,
        "test_used_for_model_selection": False,
        "model_selection_metric": "validation_ctr",
        "selected_model_policy": "validation-selected trained model used directly without refit",
        "sample_mask_policy": {
            "derived_from": "training_split_only",
            "constant_fraction": SAMPLE_CONSTANT_FRACTION,
            "discard_rule": (
                "discard temporal sample when both input channels independently have one exact "
                "normalized float32 value in at least 99% of training events"
            ),
            "shared_across_input_channels": True,
            "shared_across_models_within_dataset": True,
            "validation_and_test_do_not_define_mask": True,
        },
        "training_data_policy": {"uses_full_training_split": True},
        "ml_target": "delta_t_led - true_tof - calibration_bias",
        "corrected_residual": "ml_target - paired_model_prediction",
        "prediction_limit_ps": float(config["ml_output"]["max_abs_ps"]),
        "ctr_metric": "gaussian_equivalent_shortest_coverage_interval",
        "ctr_coverage_fraction": coverage,
        "ctr_definition": "scale(p) * shortest empirical interval containing ceil(p*N) finite residuals",
        "ctr_uses_all_finite_residuals": True,
        "ctr_uncertainty": "event_bootstrap_ctr_std",
        "ctr_bootstrap_samples": int(config["fit"]["bootstrap_samples"]),
        "config": public_config(config),
        "analyses": analyses,
        "source_count": len(roots),
        "datasets": {},
    }
    return manifest


def _dataset_manifest(dataset, sample_count, retained_samples, threshold, mode):
    family = target_family(mode)
    value = {
        "prepared_dir": str(dataset.directory),
        "split": dataset.manifest["split"],
        "led_threshold_mV": dataset.manifest["led_threshold_mV"],
        "led_training_mean_ps": dataset.manifest["led_training_mean_ps"],
        "cfd_fraction": dataset.manifest["cfd_fraction"],
        "subsampling": int(dataset.manifest["ml_input"]["subsampling"]),
        "sample_mask": {
            "derived_from": "training_split_only",
            "constant_fraction": SAMPLE_CONSTANT_FRACTION,
            "shared_across_input_channels": True,
            "input_samples_before": sample_count,
            "input_samples_after": retained_samples,
            "input_samples_removed": sample_count - retained_samples,
        },
        "target_definition": dataset.manifest.get(
            "target_definition",
            "delta_t_led - true_tof - calibration_bias",
        ),
        "corrected_definition": "target - paired_model_prediction",
        "concatenated": bool(dataset.manifest.get("concatenated", False)),
    }
    if bool(dataset.manifest.get("concatenated", False)):
        value["source_datasets"] = dataset.manifest.get("source_datasets", [])
        value["fixed_led_threshold_mV"] = threshold
    value["selected_led_threshold_mV"] = float(dataset.manifest["led_threshold_mV"][family])
    return value


def _evaluate_final_datasets(
    datasets,
    config,
    store,
    logger,
    progress,
    manifest,
    *,
    prefit_searches=None,
):
    rows = store.read_results() if store.resume else []
    existing = {
        (str(row.get("dataset")), str(row.get("method")), str(row.get("stage")))
        for row in rows
    }

    def existing_row(dataset_name, method, stage):
        return next(
            (
                row for row in rows
                if str(row.get("dataset")) == str(dataset_name)
                and str(row.get("method")) == str(method)
                and str(row.get("stage")) == str(stage)
            ),
            None,
        )
    final_metrics: dict[str, Any] = {"led": {}, "models": {}}
    prefit_searches = dict(prefit_searches or {})
    seed = int(config["validation"]["seed"])
    mode = str(config["mode"])

    for dataset in datasets:
        name = _dataset_name(dataset)
        voltage = _dataset_voltage(dataset, name)
        family = target_family(mode)
        threshold = float(dataset.manifest["led_threshold_mV"][family])
        store.save_split(name, dataset)

        logger.info(
            "Final dataset | %s | LED=%.6g mV | train=%d | validation=%d | blind=%d",
            name,
            threshold,
            dataset.training.size,
            dataset.validation.size,
            dataset.test.size,
        )

        if (name, "led", "development_selection") not in existing:
            rows.append(
                _selection_row(
                    name,
                    voltage,
                    mode,
                    "led",
                    dataset.manifest["led_development_ctr_ps"][family],
                    {"threshold_mV": threshold},
                    "development_ctr",
                )
            )
            existing.add((name, "led", "development_selection"))
        if config["cfd"] and family in dataset.manifest["cfd_fraction"]:
            fraction = float(dataset.manifest["cfd_fraction"][family])
            if (name, "cfd", "development_selection") not in existing:
                rows.append(
                    _selection_row(
                        name,
                        voltage,
                        mode,
                        "cfd",
                        dataset.manifest["cfd_development_ctr_ps"][family],
                        {"fraction": fraction},
                        "development_ctr",
                    )
                )
                existing.add((name, "cfd", "development_selection"))

        sample_mask = dataset_training_sample_mask(dataset, mode)
        sample_count = int(sample_mask.size)
        retained_samples = int(np.count_nonzero(sample_mask))
        logger.info(
            "Input mask | %s | retained=%d/%d samples | train-only constant threshold=%.1f%%",
            name,
            retained_samples,
            sample_count,
            100.0 * SAMPLE_CONSTANT_FRACTION,
        )

        target = model_target(dataset, mode)

        led_reference = calibrated_led(dataset, mode)
        for stage, indices in (
            ("train", np.asarray(dataset.training, dtype=np.int64)),
            ("test", np.asarray(dataset.test, dtype=np.int64)),
        ):
            led = led_reference[indices]
            led_row = existing_row(name, "led", stage)
            if led_row is None:
                led_row = _metric_row(
                    config,
                    name,
                    voltage,
                    mode,
                    "led",
                    led,
                    indices.size,
                    semantic_seed(seed, name, mode, "led", stage),
                    logger,
                    stage=stage,
                )
                rows.append(led_row)
                existing.add((name, "led", stage))
                store.save_residuals(name, "led", led, stage=stage)
            if stage == "test":
                final_metrics["led"][name] = led_row

            if config["cfd"]:
                cfd_row = existing_row(name, "cfd", stage)
                if cfd_row is None:
                    led_mean = float(dataset.manifest["led_training_mean_ps"][family])
                    cfd = standard_delta(dataset, mode, "cfd")[indices] - led_mean
                    cfd_row = _metric_row(
                        config,
                        name,
                        voltage,
                        mode,
                        "cfd",
                        cfd,
                        indices.size,
                        semantic_seed(seed, name, mode, "cfd", stage),
                        logger,
                        stage=stage,
                    )
                    rows.append(cfd_row)
                    existing.add((name, "cfd", stage))
                    store.save_residuals(name, "cfd", cfd, stage=stage)

        for model_name, model_config in config["models"].items():
            label = f"final model | {name} | {LABELS.get(model_name, model_name)}"
            search = None
            fitted = None
            validation_row = existing_row(name, model_name, "validation")
            train_row = existing_row(name, model_name, "train")
            test_row_existing = existing_row(name, model_name, "test")
            artifacts = store.root / "artifacts" / name
            completed = (
                store.resume
                and validation_row is not None
                and train_row is not None
                and test_row_existing is not None
                and (store.root / "models" / name / model_name).is_dir()
                and (artifacts / f"{model_name}_train_model_output_ps.npy").is_file()
                and (artifacts / f"{model_name}_test_model_output_ps.npy").is_file()
                and (artifacts / f"{model_name}_train_residuals_ps.npy").is_file()
                and (artifacts / f"{model_name}_test_residuals_ps.npy").is_file()
            )
            if completed:
                progress.complete(
                    f"final_model:{model_name}",
                    label,
                    note="reused completed result",
                )
                final_metrics["models"][(name, model_name)] = {
                    "validation_ctr_ps": float(validation_row["selection_score"]),
                    "selected_parameters_json": str(validation_row.get("parameters_json") or "{}"),
                    "test_row": test_row_existing,
                }
                logger.info(
                    "Final result reused | %s | %s | validation CTR=%.3f ps | blind CTR=%.3f ± %.3f ps",
                    name,
                    LABELS.get(model_name, model_name),
                    float(validation_row["selection_score"]),
                    float(test_row_existing["ctr_ps"]),
                    float(test_row_existing["ctr_uncertainty_ps"]),
                )
                continue
            with progress.task(f"final_model:{model_name}", label):
                spec = get_model(model_name)
                search = prefit_searches.pop((name, model_name), None)
                if search is not None:
                    logger.info(
                        "Final model | %s | %s | reusing fit from LED-threshold scan",
                        name,
                        LABELS.get(model_name, model_name),
                    )
                else:
                    search = search_model(
                        spec,
                        model_config,
                        config,
                        dataset,
                        mode,
                        seed=semantic_seed(seed, name, mode, model_name, "search"),
                        dataset_name=name,
                        sample_mask=sample_mask,
                        logger=logger,
                    )
                fitted = selected_model(search)
                selected_parameters = dict(search.best.candidate or {})
                save_model(
                    spec,
                    fitted,
                    store.model_dir(name, model_name),
                    selected_parameters,
                )
                store.save_search(name, model_name, search.as_dict())
                rows.append(
                    _selection_row(
                        name,
                        voltage,
                        mode,
                        model_name,
                        search.best.score,
                        selected_parameters,
                        "validation_ctr",
                    )
                )

                xai = config.get("reporting", {}).get("xai", {}) or {}
                if bool(xai.get("enabled", True)) and spec.explain is not None:
                    limit = min(dataset.development.size, int(xai.get("max_events", 1024)))
                    chosen = dataset.development[:limit]
                    xai_view = waveform_view(dataset, mode, chosen)
                    normalized = apply_sample_mask(
                        xai_view.materialize(),
                        fitted.sample_mask,
                    )
                    time_ps = apply_sample_mask_to_time(
                        xai_view.time_ps,
                        fitted.sample_mask,
                    )
                    importance = spec.explain(fitted.artifact, normalized)
                    physical = inverse_pair(dataset, mode, normalized)
                    store.save_xai(
                        name,
                        model_name,
                        time_ps=time_ps,
                        importance=importance,
                        example_pair_mV=physical[0],
                    )

                test_row = None
                for stage, indices in (
                    ("train", np.asarray(dataset.training, dtype=np.int64)),
                    ("test", np.asarray(dataset.test, dtype=np.int64)),
                ):
                    prediction, _time, _pair = predict_indices(
                        spec,
                        fitted,
                        dataset,
                        mode,
                        indices,
                    )
                    store.save_model_output(
                        name,
                        model_name,
                        prediction,
                        stage=stage,
                    )
                    residual = corrected_timing_residual(target[indices], prediction)
                    metric_row = _metric_row(
                        config,
                        name,
                        voltage,
                        mode,
                        model_name,
                        residual,
                        indices.size,
                        semantic_seed(seed, name, mode, model_name, stage),
                        logger,
                        stage=stage,
                    )
                    rows.append(metric_row)
                    store.save_residuals(
                        name,
                        model_name,
                        residual,
                        stage=stage,
                    )
                    if stage == "test":
                        test_row = metric_row

                if test_row is None:
                    raise RuntimeError(f"{name}/{model_name}: blind-test metric was not produced")
                final_metrics["models"][(name, model_name)] = {
                    "validation_ctr_ps": float(search.best.score),
                    "selected_parameters_json": json.dumps(selected_parameters, sort_keys=True),
                    "test_row": test_row,
                }

            logger.info(
                "Final result | %s | %s | validation CTR=%.3f ps | blind CTR=%.3f ± %.3f ps",
                name,
                LABELS.get(model_name, model_name),
                final_metrics["models"][(name, model_name)]["validation_ctr_ps"],
                float(final_metrics["models"][(name, model_name)]["test_row"]["ctr_ps"]),
                float(final_metrics["models"][(name, model_name)]["test_row"]["ctr_uncertainty_ps"]),
            )
            if search is not None:
                search.best.artifact = None
            del fitted
            gc.collect()
            store.write_results(rows)

        manifest["datasets"][name] = _dataset_manifest(
            dataset,
            sample_count,
            retained_samples,
            threshold,
            mode,
        )
        store.write_results(rows)
        store.write_manifest(manifest)

    return rows, final_metrics


def _run_standard_study(
    config_or_path: dict[str, Any] | str | Path,
    *,
    overwrite: bool = False,
    resume: bool = False,
    rebuild_preprocessing: bool = False,
) -> Path:
    config = load_config(config_or_path) if not isinstance(config_or_path, dict) else config_or_path
    store = RunStore(
        config["experiment"]["output_dir"],
        overwrite=overwrite,
        resume=resume,
    )
    logger = _logger(store.root)
    if resume:
        _assert_resume_config_matches(config, store.root)
    if resume and _completed_run_matches(config, store.root):
        logger.info("Study already complete | %s | nothing to resume", store.root)
        return store.root

    roots = discover_root_files(config)
    if not roots:
        raise FileNotFoundError("No ROOT files matched the configured source")

    concatenate = bool(config["experiment"].get("concatenate_datasets", False))
    dataset_count = 1 if concatenate else len(roots)
    threshold_cfg = config["analyses"]["led_threshold_scan"]
    window_cfg = config["analyses"]["window_scan"]
    threshold_enabled = bool(threshold_cfg["enabled"])
    window_enabled = bool(window_cfg["enabled"])
    threshold_count = len(config["standard_methods"]["led_thresholds_mV"]) if threshold_enabled else 0
    prepare_count = 0 if threshold_enabled else len(roots) + (1 if concatenate else 0)
    window_models = list(window_cfg.get("models", [])) if window_enabled else []
    window_limits = list(window_cfg.get("right_limits_ns", [])) if window_enabled else []

    plan = {
        "selection": len(roots),
        "native_preprocess": len(roots),
    }
    if threshold_enabled:
        plan["led_scan"] = dataset_count * threshold_count
    elif prepare_count:
        plan["led_prepare"] = prepare_count
    for model_name in config["models"]:
        plan[f"final_model:{model_name}"] = dataset_count
    for model_name in window_models:
        plan[f"window_scan:{model_name}"] = dataset_count * len(window_limits)
    progress = ProgressTracker(logger, plan)

    logger.info(
        "%s | mode=%s | source files=%d | final datasets=%d | models=%s",
        "Study resume" if resume else "Study",
        config["mode"],
        len(roots),
        dataset_count,
        ", ".join(LABELS.get(name, name) for name in config["models"]),
    )
    logger.info(
        "Configured analyses | LED threshold scan=%s%s | ML window scan=%s%s",
        "on" if threshold_enabled else "off",
        (
            f" ({LABELS.get(str(threshold_cfg.get('selection_model')), str(threshold_cfg.get('selection_model')))})"
            if threshold_enabled
            else ""
        ),
        "on" if window_enabled else "off",
        (
            f" ({len(window_limits)} limits × {len(window_models)} models)"
            if window_enabled
            else ""
        ),
    )

    preprocessed = []
    for root in roots:
        with progress.task("selection", f"event selection | {root.name}"):
            selection = select_events(
                root,
                config,
                rebuild=rebuild_preprocessing,
                logger=logger,
            )
        with progress.task("native_preprocess", f"native preprocessing | {root.name}"):
            preprocessed.append(
                preprocess_selected(
                    root,
                    selection,
                    config,
                    rebuild=rebuild_preprocessing,
                    logger=logger,
                )
            )

    analyses_manifest: dict[str, Any] = {}
    prefit_searches: dict[tuple[str, str], Any] = {}
    if threshold_enabled:
        threshold_result = run_led_threshold_scan(
            preprocessed,
            config,
            store.root / "analyses" / "led_threshold",
            logger,
            progress,
            rebuild=rebuild_preprocessing,
            resume=resume,
        )
        datasets = threshold_result.datasets
        analyses_manifest["led_threshold_scan"] = threshold_result.manifest
        prefit_searches = threshold_result.selected_searches
    else:
        datasets = _prepare_without_threshold_scan(
            preprocessed,
            config,
            rebuild_preprocessing,
            logger,
            progress,
        )
        family = target_family(str(config["mode"]))
        analyses_manifest["led_threshold_scan"] = {
            "enabled": False,
            "selection_policy": "development LED CTR during ML dataset preparation",
            "selected_thresholds_mV": {
                _dataset_name(dataset): float(dataset.manifest["led_threshold_mV"][family])
                for dataset in datasets
            },
        }

    manifest = _base_manifest(
        config,
        concatenate,
        roots,
        analyses_manifest,
    )
    if concatenate:
        family = target_family(str(config["mode"]))
        manifest["concatenated_dataset"] = {
            "name": _dataset_name(datasets[0]),
            "selected_led_threshold_mV": float(
                datasets[0].manifest["led_threshold_mV"][family]
            ),
            "source_count": len(roots),
        }
    store.write_manifest(manifest)

    _rows, final_metrics = _evaluate_final_datasets(
        datasets,
        config,
        store,
        logger,
        progress,
        manifest,
        prefit_searches=prefit_searches,
    )

    if window_enabled:
        analyses_manifest["window_scan"] = run_window_scan(
            datasets,
            config,
            store.root / "analyses" / "window",
            logger,
            progress,
            final_metrics,
            resume=resume,
        )
    else:
        analyses_manifest["window_scan"] = {"enabled": False}
    manifest["analyses"] = analyses_manifest
    store.write_manifest(manifest)

    logger.info("Rendering study plots")
    generated = rebuild_study_plots(store.root)
    logger.info("Study plots | files=%d | %s", len(generated), store.root)

    manifest["status"] = "complete"
    store.write_manifest(manifest)
    logger.info("Study complete | %s | elapsed=%s", store.root, progress.elapsed_text)
    return store.root


def _prepare_experiment_root(
    output_dir: str | Path,
    *,
    overwrite: bool,
    resume: bool,
) -> Path:
    root = Path(output_dir).resolve()
    if overwrite and resume:
        raise ValueError("overwrite and resume are mutually exclusive")
    if overwrite and root.exists():
        shutil.rmtree(root)
    if root.exists() and any(root.iterdir()) and not resume:
        raise FileExistsError(
            f"Run directory is not empty: {root}. "
            "Use --resume to continue it or --overwrite to replace it."
        )
    root.mkdir(parents=True, exist_ok=True)
    return root


def _paired_model_comparison_rows(
    run_dir: Path,
    window_name: str,
    config: dict[str, Any],
) -> list[dict[str, Any]]:
    results_path = run_dir / "csv" / "results.csv"
    if not results_path.is_file():
        return []
    with results_path.open(encoding="utf-8", newline="") as stream:
        results = list(csv.DictReader(stream))
    datasets = sorted(
        {
            str(row["dataset"])
            for row in results
            if row.get("stage") == "test"
            and row.get("method") in {"mlp", "onishi_cnn"}
        },
        key=voltage_from_name,
    )
    fit_config = dict(config.get("fit") or {})
    samples = int(fit_config.get("bootstrap_samples", 0))
    seed = int(config["validation"]["seed"])
    rows: list[dict[str, Any]] = []
    for dataset in datasets:
        mlp_path = run_dir / "artifacts" / dataset / "mlp_test_residuals_ps.npy"
        onishi_path = run_dir / "artifacts" / dataset / "onishi_cnn_test_residuals_ps.npy"
        if not (mlp_path.is_file() and onishi_path.is_file()):
            continue
        mlp = np.asarray(np.load(mlp_path), dtype=np.float64).reshape(-1)
        onishi = np.asarray(np.load(onishi_path), dtype=np.float64).reshape(-1)
        if mlp.shape != onishi.shape:
            raise ValueError(
                f"{dataset}/{window_name}: paired model residual shapes differ"
            )
        finite = np.isfinite(mlp) & np.isfinite(onishi)
        mlp = mlp[finite]
        onishi = onishi[finite]
        if mlp.size < 2:
            raise ValueError(
                f"{dataset}/{window_name}: fewer than two common blind events"
            )
        mlp_ctr = fit_ctr_ps(mlp, fit_config, bootstrap=False).ctr_ps
        onishi_ctr = fit_ctr_ps(onishi, fit_config, bootstrap=False).ctr_ps
        delta = float(onishi_ctr - mlp_ctr)
        relative = float(100.0 * delta / onishi_ctr)
        rng = np.random.default_rng(
            semantic_seed(seed, dataset, window_name, "mlp_vs_onishi_paired")
        )
        boot_delta = []
        boot_relative = []
        for _ in range(samples):
            idx = rng.integers(0, mlp.size, size=mlp.size)
            try:
                mlp_b = fit_ctr_ps(
                    mlp[idx], fit_config, bootstrap=False
                ).ctr_ps
                onishi_b = fit_ctr_ps(
                    onishi[idx], fit_config, bootstrap=False
                ).ctr_ps
            except ValueError:
                continue
            if not (
                np.isfinite(mlp_b)
                and np.isfinite(onishi_b)
                and onishi_b > 0
            ):
                continue
            d = float(onishi_b - mlp_b)
            boot_delta.append(d)
            boot_relative.append(100.0 * d / onishi_b)
        rows.append(
            {
                "window": window_name,
                "dataset": dataset,
                "voltage_V": voltage_from_name(dataset),
                "blind_events": int(mlp.size),
                "mlp_ctr_ps": float(mlp_ctr),
                "onishi_cnn_ctr_ps": float(onishi_ctr),
                "onishi_minus_mlp_ctr_ps": delta,
                "paired_bootstrap_uncertainty_ps": (
                    float(np.std(boot_delta, ddof=1))
                    if len(boot_delta) > 1
                    else float("nan")
                ),
                "mlp_improvement_over_onishi_percent": relative,
                "paired_bootstrap_uncertainty_percent": (
                    float(np.std(boot_relative, ddof=1))
                    if len(boot_relative) > 1
                    else float("nan")
                ),
                "paired_bootstrap_successful": len(boot_delta),
                "comparison_population": "blind",
            }
        )
    return rows


def _write_model_comparison_plot(
    root: Path,
    rows: list[dict[str, Any]],
) -> list[Path]:
    import matplotlib.pyplot as plt

    generated: list[Path] = []
    plot_dir = root / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    for window in sorted({str(row["window"]) for row in rows}):
        subset = sorted(
            [row for row in rows if str(row["window"]) == window],
            key=lambda row: float(row["voltage_V"]),
        )
        if not subset:
            continue
        x = np.asarray([float(row["voltage_V"]) for row in subset], dtype=float)
        y = np.asarray(
            [float(row["mlp_improvement_over_onishi_percent"]) for row in subset],
            dtype=float,
        )
        err = np.asarray(
            [float(row["paired_bootstrap_uncertainty_percent"]) for row in subset],
            dtype=float,
        )
        fig, ax = plt.subplots(figsize=(7.0, 3.35))
        ax.errorbar(
            x,
            y,
            yerr=np.where(np.isfinite(err), err, 0.0),
            marker="o",
            capsize=2.5,
        )
        ax.axhline(0.0, color="#7F7F7F", ls=":", lw=0.9)
        ax.set_xlabel("Bias voltage [V]")
        ax.set_ylabel("MLP improvement over Onishi CNN [%]")
        ax.grid(axis="y", linewidth=0.5, alpha=0.22)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        fig.tight_layout()
        target = plot_dir / f"paired_model_comparison_{window}.pdf"
        fig.savefig(target, bbox_inches="tight", pad_inches=0.03)
        plt.close(fig)
        generated.append(target)
    return generated


def _run_model_comparison_experiment(
    config: dict[str, Any],
    *,
    overwrite: bool,
    resume: bool,
    rebuild_preprocessing: bool,
) -> Path:
    root = _prepare_experiment_root(
        config["experiment"]["output_dir"],
        overwrite=overwrite,
        resume=resume,
    )
    fixed_led = float(config["experiment"]["fixed_led_threshold_mV"])
    windows = dict(config["experiment"]["windows"])
    base_prepared = Path(config["preprocessing"]["prepared_dir"]).resolve()

    subruns: dict[str, str] = {}
    paired_rows: list[dict[str, Any]] = []
    for index, window_name in enumerate(("onishi", "wide")):
        sub = copy.deepcopy(config)
        sub["experiment"]["type"] = "standard"
        sub["experiment"]["output_dir"] = str(root / window_name)
        sub["experiment"]["name"] = (
            f"{config['experiment'].get('name', 'model_comparison')}_{window_name}"
        )
        sub["standard_methods"]["led_thresholds_mV"] = [fixed_led]
        sub["ml_input"]["window_ns"] = {
            "start": float(windows[window_name]["start"]),
            "end": float(windows[window_name]["end"]),
        }
        sub["preprocessing"]["prepared_dir"] = str(
            base_prepared / "_model_comparison" / window_name
        )
        sub["analyses"] = {
            "led_threshold_scan": {
                "enabled": False,
                "selection_model": "mlp",
            },
            "window_scan": {
                "enabled": False,
                "right_limits_ns": [],
                "models": [],
            },
        }
        sub["_config_fingerprint"] = canonical_hash(public_config(sub))
        subrun = _run_standard_study(
            sub,
            overwrite=False,
            resume=resume,
            rebuild_preprocessing=(
                rebuild_preprocessing if index == 0 else False
            ),
        )
        subruns[window_name] = str(subrun)
        paired_rows.extend(
            _paired_model_comparison_rows(subrun, window_name, sub)
        )

    csv_dir = root / "csv"
    csv_dir.mkdir(parents=True, exist_ok=True)
    paired_csv = csv_dir / "paired_model_comparison.csv"
    if paired_rows:
        fields = list(dict.fromkeys(key for row in paired_rows for key in row))
        with paired_csv.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows(paired_rows)
        _write_model_comparison_plot(root, paired_rows)

    manifest = {
        "schema_version": 1,
        "status": "complete",
        "experiment_type": "model_comparison",
        "models": {
            "proposed": "mlp",
            "reference": "onishi_cnn",
        },
        "model_labels": {
            "mlp": "Antisymmetric MLP",
            "onishi_cnn": "Onishi CNN",
        },
        "windows_ns": windows,
        "fixed_led_threshold_mV": fixed_led,
        "hyperparameter_selection_population": "validation",
        "hyperparameter_selection_metric": "validation_ctr",
        "final_evaluation_population": "blind",
        "refit_after_validation_selection": False,
        "paired_model_comparison": "blind paired event bootstrap",
        "subruns": subruns,
        "config": public_config(config),
    }
    (root / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return root


def _run_threshold_scan_experiment(
    config: dict[str, Any],
    *,
    overwrite: bool,
    resume: bool,
    rebuild_preprocessing: bool,
) -> Path:
    root = _prepare_experiment_root(
        config["experiment"]["output_dir"],
        overwrite=overwrite,
        resume=resume,
    )
    logger = _logger(root)
    roots = discover_root_files(config)
    target_voltage = float(config["experiment"]["voltage_V"])
    roots = [
        path
        for path in roots
        if np.isfinite(voltage_from_name(path.stem))
        and np.isclose(
            voltage_from_name(path.stem),
            target_voltage,
            rtol=0.0,
            atol=1e-9,
        )
    ]
    if not roots:
        raise FileNotFoundError(
            f"No ROOT file matched threshold-scan voltage {target_voltage:g} V"
        )

    plan = {
        "selection": len(roots),
        "native_preprocess": len(roots),
        "threshold_scan": len(roots)
        * len(config["standard_methods"]["led_thresholds_mV"]),
    }
    progress = ProgressTracker(logger, plan)
    preprocessed = []
    for source in roots:
        with progress.task("selection", f"event selection | {source.name}"):
            selection = select_events(
                source,
                config,
                rebuild=rebuild_preprocessing,
                logger=logger,
            )
        with progress.task(
            "native_preprocess",
            f"native preprocessing | {source.name}",
        ):
            preprocessed.append(
                preprocess_selected(
                    source,
                    selection,
                    config,
                    rebuild=rebuild_preprocessing,
                    logger=logger,
                )
            )

    run_blind_led_threshold_scan(
        preprocessed,
        config,
        root,
        logger,
        progress,
        rebuild=rebuild_preprocessing,
        resume=resume,
    )
    logger.info(
        "Threshold scan complete | voltage=%g V | %s",
        target_voltage,
        root,
    )
    return root


def run_study(
    config_or_path: dict[str, Any] | str | Path,
    *,
    overwrite: bool = False,
    resume: bool = False,
    rebuild_preprocessing: bool = False,
) -> Path:
    config = (
        load_config(config_or_path)
        if not isinstance(config_or_path, dict)
        else config_or_path
    )
    experiment_type = str(
        config.get("experiment", {}).get("type", "standard")
    ).lower()
    if experiment_type == "model_comparison":
        return _run_model_comparison_experiment(
            config,
            overwrite=overwrite,
            resume=resume,
            rebuild_preprocessing=rebuild_preprocessing,
        )
    if experiment_type == "threshold_scan":
        return _run_threshold_scan_experiment(
            config,
            overwrite=overwrite,
            resume=resume,
            rebuild_preprocessing=rebuild_preprocessing,
        )
    return _run_standard_study(
        config,
        overwrite=overwrite,
        resume=resume,
        rebuild_preprocessing=rebuild_preprocessing,
    )
