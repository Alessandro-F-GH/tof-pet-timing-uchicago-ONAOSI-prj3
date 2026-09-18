from __future__ import annotations

import copy
import gc
import json
import logging
import shutil
from pathlib import Path
from typing import Any

import numpy as np

from .analyses import run_blind_led_threshold_scan
from .common import canonical_hash, read_json, voltage_from_name
from .concatenate import concatenate_prepared_datasets
from .config import discover_root_files, load_config, public_config
from .data import preprocess_selected
from .event_selection import select_events
from .plot_rebuild import rebuild_study_plots
from .models import get_model
from .prepared_data import prepare_ml_dataset
from .progress import ProgressTracker
from .reporting import LABELS, plot_model_study_windows
from .selection_outputs import ensure_selection_outputs
from .sample_mask import (
    SAMPLE_CONSTANT_FRACTION,
    apply_sample_mask,
    apply_sample_mask_to_time,
    dataset_training_sample_mask,
)
from .splits import semantic_seed
from .stats import ctr_estimate, format_residual_summary, residual_summary
from .storage import RunStore
from .train import fit_fixed_model, predict_indices, save_model, search_model, selected_model
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
                    (dataset_name, model_name, "train"),
                    (dataset_name, model_name, "test"),
                }
            )
            required.add(
                (
                    dataset_name,
                    model_name,
                    "fixed_configuration" if model_name == "onishi_cnn" else "validation",
                )
            )
        if not required.issubset(available):
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


def _fixed_configuration_row(name, voltage, mode, method, parameters):
    return {
        "dataset": name,
        "voltage_V": voltage,
        "mode": mode,
        "method": method,
        "stage": "fixed_configuration",
        "selection_score": float("nan"),
        "selection_metric": "fixed_reference_configuration",
        "ctr_ps": float("nan"),
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


def _prepare_datasets(preprocessed, config, rebuild, logger, progress):
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


def _base_manifest(config, concatenate, roots):
    coverage = float(config["fit"].get("coverage_fraction", 0.90))
    manifest = {
        "schema_version": 13,
        "status": "running",
        "config_fingerprint": str(config.get("_config_fingerprint", "")),
        "protocol": "single_mode_holdout",
        "mode": str(config["mode"]),
        "concatenate_datasets": concatenate,
        "test_used_for_model_selection": False,
        "model_selection_metric": "validation_ctr_for_tuned_models_only",
        "model_policies": {
            "mlp": "grid search on validation CTR; selected trained checkpoint used directly without refit",
            "onishi_cnn": "fixed Onishi reference configuration; fit on training+validation; no model selection; blind held out",
        },
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
        "training_data_policy": {"mlp": "training only", "onishi_cnn": "training plus validation"},
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
            "Dataset | %s | LED=%.6g mV | train=%d | validation=%d | blind=%d",
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
            "Input | %s | retained=%d/%d samples | constant-threshold=%.1f%%",
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
            label = f"{name} | {LABELS.get(model_name, model_name)}"
            fixed_reference = model_name == "onishi_cnn"
            selection_stage = "fixed_configuration" if fixed_reference else "validation"
            selection_row = existing_row(name, model_name, selection_stage)
            train_row = existing_row(name, model_name, "train")
            test_row_existing = existing_row(name, model_name, "test")
            artifacts = store.root / "artifacts" / name
            model_dir = store.root / "models" / name / model_name
            xai_enabled = bool(
                (config.get("reporting", {}).get("xai", {}) or {}).get("enabled", True)
            )
            completed = (
                store.resume
                and selection_row is not None
                and train_row is not None
                and test_row_existing is not None
                and (model_dir / "model.pt").is_file()
                and (model_dir / "metadata.json").is_file()
                and (store.root / "search" / name / f"{model_name}.json").is_file()
                and (artifacts / f"{model_name}_train_model_output_ps.npy").is_file()
                and (artifacts / f"{model_name}_test_model_output_ps.npy").is_file()
                and (artifacts / f"{model_name}_train_residuals_ps.npy").is_file()
                and (artifacts / f"{model_name}_test_residuals_ps.npy").is_file()
                and (
                    not xai_enabled
                    or (artifacts / f"{model_name}_xai.npz").is_file()
                )
            )
            if completed:
                progress.complete(
                    f"final_model:{model_name}",
                    label,
                    note="resume",
                    announce=False,
                )
                final_metrics["models"][(name, model_name)] = {
                    "selection_stage": selection_stage,
                    "validation_ctr_ps": (
                        float(selection_row["selection_score"])
                        if not fixed_reference
                        else float("nan")
                    ),
                    "selected_parameters_json": str(
                        selection_row.get("parameters_json") or "{}"
                    ),
                    "test_row": test_row_existing,
                }
                if fixed_reference:
                    logger.info(
                        "Result | %s | %s | resumed | blind CTR=%.3f ± %.3f ps",
                        name,
                        LABELS.get(model_name, model_name),
                        float(test_row_existing["ctr_ps"]),
                        float(test_row_existing["ctr_uncertainty_ps"]),
                    )
                else:
                    logger.info(
                        "Result | %s | %s | resumed | blind CTR=%.3f ± %.3f ps",
                        name,
                        LABELS.get(model_name, model_name),
                        float(test_row_existing["ctr_ps"]),
                        float(test_row_existing["ctr_uncertainty_ps"]),
                    )
                continue

            search = None
            fitted = None
            with progress.task(
                f"final_model:{model_name}",
                label,
                announce_start=False,
                announce_finish=False,
            ):
                spec = get_model(model_name)
                if fixed_reference:
                    fixed_log_parameters = {
                        **dict(spec.candidates(model_config)[0] or {}),
                        "epochs": int((model_config.get("training") or {}).get("epochs", 100)),
                        "lr_decay_epochs": list(
                            (model_config.get("training") or {}).get(
                                "lr_decay_epochs", [30, 60]
                            )
                        ),
                        "lr_decay_factor": float(
                            (model_config.get("training") or {}).get(
                                "lr_decay_factor", 0.1
                            )
                        ),
                        "architecture": dict(model_config.get("architecture") or {}),
                    }
                    logger.info(
                        "Train | %s | %s | params=%s | train+validation=%d",
                        name,
                        LABELS.get(model_name, model_name),
                        json.dumps(fixed_log_parameters, sort_keys=True),
                        int(dataset.development.size),
                    )
                    fitted, selected_parameters = fit_fixed_model(
                        spec,
                        model_config,
                        config,
                        dataset,
                        mode,
                        seed=semantic_seed(seed, name, mode, model_name, "fixed_fit"),
                        sample_mask=sample_mask,
                        logger=logger,
                    )
                    rows.append(
                        _fixed_configuration_row(
                            name,
                            voltage,
                            mode,
                            model_name,
                            selected_parameters,
                        )
                    )
                    store.save_search(
                        name,
                        model_name,
                        {
                            "policy": "fixed_reference_configuration",
                            "validation_used_for_selection": False,
                            "parameters": selected_parameters,
                        },
                    )
                    validation_ctr = float("nan")
                else:
                    search = prefit_searches.pop((name, model_name), None)
                    if search is not None:
                        logger.info(
                            "Model selection | %s | %s | reusing saved validation-selected fit",
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
                    validation_ctr = float(search.best.score)
                    logger.info(
                        "Selected | %s | %s | validation CTR=%.3f ps | params=%s",
                        name,
                        LABELS.get(model_name, model_name),
                        validation_ctr,
                        json.dumps(selected_parameters, sort_keys=True),
                    )
                    rows.append(
                        _selection_row(
                            name,
                            voltage,
                            mode,
                            model_name,
                            validation_ctr,
                            selected_parameters,
                            "validation_ctr",
                        )
                    )
                    store.save_search(name, model_name, search.as_dict())

                save_model(
                    spec,
                    fitted,
                    store.model_dir(name, model_name),
                    selected_parameters,
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
                    "selection_stage": selection_stage,
                    "validation_ctr_ps": validation_ctr,
                    "selected_parameters_json": json.dumps(selected_parameters, sort_keys=True),
                    "test_row": test_row,
                }

            if fixed_reference:
                logger.info(
                    "Result | %s | %s | blind CTR=%.3f ± %.3f ps",
                    name,
                    LABELS.get(model_name, model_name),
                    float(test_row["ctr_ps"]),
                    float(test_row["ctr_uncertainty_ps"]),
                )
            else:
                logger.info(
                    "Result | %s | %s | blind CTR=%.3f ± %.3f ps",
                    name,
                    LABELS.get(model_name, model_name),
                    float(test_row["ctr_ps"]),
                    float(test_row["ctr_uncertainty_ps"]),
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
        for root in discover_root_files(config):
            selection = select_events(
                root,
                config,
                rebuild=False,
                logger=logger,
            )
            ensure_selection_outputs(root, selection, config, logger)
        logger.info("Study already complete | %s | selection diagnostics ensured", store.root)
        return store.root

    roots = discover_root_files(config)
    if not roots:
        raise FileNotFoundError("No ROOT files matched the configured source")

    concatenate = bool(config["experiment"].get("concatenate_datasets", False))
    dataset_count = 1 if concatenate else len(roots)
    prepare_count = len(roots) + (1 if concatenate else 0)
    plan = {
        "selection": len(roots),
        "native_preprocess": len(roots),
        "led_prepare": prepare_count,
    }
    for model_name in config["models"]:
        plan[f"final_model:{model_name}"] = dataset_count
    progress = ProgressTracker(logger, plan)

    logger.info(
        "%s | mode=%s | datasets=%d | models=%s",
        "Resume" if resume else "Study",
        config["mode"],
        dataset_count,
        ", ".join(LABELS.get(name, name) for name in config["models"]),
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
            ensure_selection_outputs(root, selection, config, logger)
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

    datasets = _prepare_datasets(
        preprocessed,
        config,
        rebuild_preprocessing,
        logger,
        progress,
    )
    prefit_searches: dict[tuple[str, str], Any] = {}

    manifest = _base_manifest(
        config,
        concatenate,
        roots,
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

    generated = rebuild_study_plots(store.root)
    logger.info("Plots | generated=%d", len(generated))

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


def _check_experiment_resume_config(
    config: dict[str, Any],
    root: Path,
    *,
    resume: bool,
) -> None:
    if not resume:
        return
    path = root / "manifest.json"
    if not path.is_file():
        return
    manifest = read_json(path)
    stored = manifest.get("config")
    if not isinstance(stored, dict):
        raise RuntimeError(
            f"Cannot resume experiment with missing resolved config: {path}"
        )
    if canonical_hash(stored) != str(config.get("_config_fingerprint", "")):
        raise RuntimeError(
            "Cannot resume experiment with a different configuration. "
            "Use the original config or --overwrite."
        )


def _model_study_compatibility(
    config: dict[str, Any],
    windows: dict[str, Any],
    fixed_led: float,
) -> dict[str, Any]:
    data = copy.deepcopy(config["data"])
    data.pop("root_folder", None)
    preprocessing = copy.deepcopy(config["preprocessing"])
    for key in ("selection_store_dir", "preprocessed_dir", "prepared_dir"):
        preprocessing.pop(key, None)
    payload = {
        "mode": str(config["mode"]),
        "data": data,
        "preprocessing": preprocessing,
        "validation": copy.deepcopy(config["validation"]),
        "fit": copy.deepcopy(config["fit"]),
        "standard_methods": {
            "fixed_led_threshold_mV": float(fixed_led),
            "led_minimum_crossing_efficiency": float(
                config["standard_methods"].get(
                    "led_minimum_crossing_efficiency", 0.95
                )
            ),
            "led_coincidence_window_ns": float(
                config["standard_methods"].get(
                    "led_coincidence_window_ns", 2.0
                )
            ),
        },
        "ml_input": {
            "windows": copy.deepcopy(windows),
            "subsampling": int(config["ml_input"].get("subsampling", 1)),
        },
        "ml_output": copy.deepcopy(config["ml_output"]),
    }
    return {
        "signature": canonical_hash(payload),
        "payload": payload,
    }


def _run_model_study_experiment(
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
    _check_experiment_resume_config(config, root, resume=resume)

    model_name = next(iter(config["models"]))
    fixed_led = float(config["experiment"]["fixed_led_threshold_mV"])
    windows = copy.deepcopy(config["ml_input"]["windows"])
    compatibility = _model_study_compatibility(config, windows, fixed_led)

    logger.info(
        "%s | model study | model=%s | mode=%s | LED=%g mV | windows=%s",
        "Resume" if resume else "Experiment",
        LABELS.get(model_name, model_name),
        config["mode"],
        fixed_led,
        ",".join(windows),
    )

    running_manifest = {
        "schema_version": 2,
        "status": "running",
        "experiment_type": "model_study",
        "model": model_name,
        "model_label": LABELS.get(model_name, model_name),
        "windows_ns": windows,
        "fixed_led_threshold_mV": fixed_led,
        "compatibility": compatibility,
        "config_fingerprint": str(config.get("_config_fingerprint", "")),
        "config": public_config(config),
    }
    (root / "manifest.json").write_text(
        json.dumps(running_manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    base_prepared = Path(config["preprocessing"]["prepared_dir"]).resolve()
    subruns: dict[str, str] = {}
    for index, (window_name, window) in enumerate(windows.items()):
        sub = copy.deepcopy(config)
        sub["experiment"]["type"] = "standard"
        sub["experiment"]["output_dir"] = str(root / window_name)
        sub["experiment"]["name"] = (
            f"{config['experiment'].get('name', model_name)}_{window_name}"
        )
        sub["standard_methods"]["led_thresholds_mV"] = [fixed_led]
        sub["ml_input"] = {
            "window_ns": {
                "start": float(window["start"]),
                "end": float(window["end"]),
            },
            "subsampling": int(config["ml_input"].get("subsampling", 1)),
        }
        sub["preprocessing"]["prepared_dir"] = str(
            base_prepared / "_model_study" / window_name
        )
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

    manifest = {
        **running_manifest,
        "status": "complete",
        "subruns": subruns,
        "final_evaluation_population": "blind",
        "model_selection": (
            "validation_ctr"
            if model_name == "mlp"
            else "fixed_reference_configuration"
        ),
        "config": public_config(config),
    }
    (root / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    summary_paths: list[Path] = []
    plot_model_study_windows(root, manifest, summary_paths)
    logger.info(
        "Model study summary plots | generated=%d",
        len(summary_paths),
    )
    logger.info("Model study complete | model=%s | %s", model_name, root)
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
    _check_experiment_resume_config(config, root, resume=resume)
    running_manifest = {
        "schema_version": 1,
        "status": "running",
        "experiment_type": "threshold_scan",
        "config_fingerprint": str(config.get("_config_fingerprint", "")),
        "config": public_config(config),
    }
    (root / "manifest.json").write_text(
        json.dumps(running_manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
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
            ensure_selection_outputs(source, selection, config, logger)
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

    scan_manifest = run_blind_led_threshold_scan(
        preprocessed,
        config,
        root,
        logger,
        progress,
        rebuild=rebuild_preprocessing,
        resume=resume,
    )
    final_manifest = {
        **running_manifest,
        **scan_manifest,
        "status": "complete",
        "config_fingerprint": str(config.get("_config_fingerprint", "")),
        "config": public_config(config),
    }
    (root / "manifest.json").write_text(
        json.dumps(final_manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
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
    if experiment_type == "model_study":
        return _run_model_study_experiment(
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
