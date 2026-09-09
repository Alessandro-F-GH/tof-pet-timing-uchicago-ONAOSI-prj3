from __future__ import annotations

import json, logging
from pathlib import Path
from typing import Any

import numpy as np

from .common import voltage_from_name
from .config import discover_root_files, load_config, public_config
from .data import preprocess_selected
from .event_selection import select_events
from .models import get_model
from .prepared_data import prepare_ml_dataset
from .selection_outputs import ensure_selection_outputs
from .splits import semantic_seed
from .stats import ctr_estimate, format_residual_summary, residual_summary
from .storage import RunStore
from .train import predict_indices, save_model, search_model, selected_model
from .view import calibrated_led, corrected_timing_residual, inverse_pair, model_target, standard_delta, target_family


def _logger(run_dir: Path):
    logger = logging.getLogger(f"waveform-study:{run_dir}")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    for h in (logging.StreamHandler(), logging.FileHandler(run_dir / "study.log", encoding="utf-8")):
        h.setFormatter(fmt)
        logger.addHandler(h)
    return logger


def _metric_row(config, name, voltage, mode, method, residual, population_n, seed, logger, stage="test"):
    values = np.asarray(residual, dtype=float)
    finite = values[np.isfinite(values)]
    minimum = int((config.get("fit") or {}).get("min_events", 100))
    context = f"{name}/{mode}/{method}/{stage}"
    summary = residual_summary(values)
    if finite.size < minimum:
        detail = format_residual_summary(summary)
        logger.error("CTR metric unavailable | %s | too few finite residuals (need %d) | %s", context, minimum, detail)
        raise RuntimeError(f"{context}: only {finite.size} finite residuals; {detail}")
    try:
        result = ctr_estimate(finite, config.get("fit"), seed=int(seed), bootstrap=True)
    except ValueError as exc:
        detail = format_residual_summary(summary)
        logger.error("Histogram FWHM CTR unavailable | %s | reason=%s | %s", context, exc, detail)
        raise RuntimeError(f"{context}: histogram FWHM CTR unavailable: {exc}; {detail}") from exc
    return {
        "dataset": name,
        "voltage_V": voltage,
        "mode": mode,
        "method": method,
        "stage": stage,
        "ctr_ps": float(result.ctr_ps),
        "ctr_uncertainty_ps": float(result.ctr_error_ps),
        "center_ps": float(result.center_ps),
        "fwhm_left_ps": float(result.left_half_ps),
        "fwhm_right_ps": float(result.right_half_ps),
        "bin_width_ps": float(result.bin_width_ps),
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
        "ctr_ps": float(score) if metric == "development_histogram_fwhm" else float("nan"),
        "ctr_uncertainty_ps": float("nan"),
        "center_ps": float("nan"),
        "n": 0,
        "population_n": 0,
        "crossing_efficiency": float("nan"),
        "parameters_json": json.dumps(parameters, sort_keys=True),
    }


def _prepare_one(root, config, rebuild, logger):
    selection = select_events(root, config, rebuild=rebuild, logger=logger)
    ensure_selection_outputs(root, selection, config, logger)
    preprocessed = preprocess_selected(root, selection, config, rebuild=rebuild, logger=logger)
    return prepare_ml_dataset(preprocessed, config, rebuild=rebuild, logger=logger)


def _dataset_voltage(dataset, name):
    values = np.asarray(dataset.bias_voltage_V, dtype=float)
    finite = values[np.isfinite(values)]
    return float(np.median(finite)) if finite.size else voltage_from_name(name)


def run_study(
    config_or_path: dict[str, Any] | str | Path,
    *,
    overwrite: bool = False,
    rebuild_preprocessing: bool = False,
) -> Path:
    config = load_config(config_or_path) if not isinstance(config_or_path, dict) else config_or_path
    store = RunStore(config["experiment"]["output_dir"], overwrite=overwrite)
    logger = _logger(store.root)
    roots = discover_root_files(config)
    if not roots:
        raise FileNotFoundError("No ROOT files matched the configured source")

    datasets = [_prepare_one(root, config, rebuild_preprocessing, logger) for root in roots]
    rows = []
    seed = int(config["validation"]["seed"])
    mode = str(config["mode"])
    manifest = {
        "schema_version": 7,
        "protocol": "single_mode_validation_selected_model_holdout",
        "mode": mode,
        "test_used_for_selection": False,
        "model_selection_metric": "validation_rmse",
        "selected_model_policy": "use_validation_selected_trained_model_without_refit",
        "model_architecture_constraint": "y_theta(s1,s2)=g_theta(s1)-g_theta(s2)",
        "ml_target": "delta_t_led - delta_delta_anchor - true_tof - calibration_bias",
        "corrected_residual": "ml_target - paired_model_prediction",
        "prediction_limit_ps": float(config["ml_output"]["max_abs_ps"]),
        "ctr_metric": "fixed_bin_histogram_fwhm",
        "ctr_bin_width_ps": float(config["fit"]["bin_width_ps"]),
        "ctr_uncertainty": "event_bootstrap_fwhm_std",
        "ctr_bootstrap_samples": int(config["fit"]["bootstrap_samples"]),
        "config": public_config(config),
        "datasets": {},
    }

    for dataset in datasets:
        name = Path(dataset.manifest["source"]).stem
        voltage = _dataset_voltage(dataset, name)
        store.save_split(name, dataset)
        fitted_models = {}
        family = target_family(mode)
        threshold = float(dataset.manifest["led_threshold_mV"][family])
        rows.append(_selection_row(name, voltage, mode, "led", dataset.manifest["led_development_ctr_ps"][family], {"threshold_mV": threshold}, "development_histogram_fwhm"))
        if config["cfd"] and family in dataset.manifest["cfd_fraction"]:
            fraction = float(dataset.manifest["cfd_fraction"][family])
            rows.append(_selection_row(name, voltage, mode, "cfd", dataset.manifest["cfd_development_ctr_ps"][family], {"fraction": fraction}, "development_histogram_fwhm"))

        logger.info("ML dataset %s | mode=%s | training=%d | validation=%d | test=%d", name, mode, dataset.training.size, dataset.validation.size, dataset.test.size)

        for model_name, model_config in config["models"].items():
            spec = get_model(model_name)
            search = search_model(
                spec, model_config, config, dataset, mode,
                seed=semantic_seed(seed, name, mode, model_name, "search"),
                dataset_name=name,
                logger=logger,
            )
            fitted = selected_model(search)
            save_model(spec, fitted, store.model_dir(name, model_name), search.best.candidate)
            store.save_search(name, model_name, search.as_dict())
            fitted_models[model_name] = fitted
            rows.append(_selection_row(name, voltage, mode, model_name, search.best.score, search.best.candidate, "validation_rmse"))
            logger.info(
                "Selected dataset=%s | %s/%s | validation RMSE %.6g ps | using selected trained model without refit | output clipped to ±%.0f ps | %s",
                name, mode, model_name, search.best.score, float(config["ml_output"]["max_abs_ps"]), search.best.candidate,
            )

            xai = config.get("reporting", {}).get("xai", {}) or {}
            if bool(xai.get("enabled", True)) and spec.explain is not None:
                limit = min(dataset.development.size, int(xai.get("max_events", 1024)))
                chosen = dataset.development[:limit]
                _prediction, time_ps, normalized = predict_indices(spec, fitted, dataset, mode, chosen)
                importance = spec.explain(fitted.artifact, normalized)
                physical = inverse_pair(dataset, mode, normalized)
                store.save_xai(name, model_name, time_ps=time_ps, importance=importance, example_pair_mV=physical[0])

        led_reference = calibrated_led(dataset, mode)
        target = model_target(dataset, mode)
        for stage, indices in (("train", np.asarray(dataset.training, dtype=np.int64)), ("test", np.asarray(dataset.test, dtype=np.int64))):
            led = led_reference[indices]
            rows.append(_metric_row(config, name, voltage, mode, "led", led, indices.size, semantic_seed(seed, name, mode, "led", stage), logger, stage=stage))
            store.save_residuals(name, "led", led, stage=stage)

            if config["cfd"]:
                led_mean = float(dataset.manifest["led_training_mean_ps"][family])
                cfd = standard_delta(dataset, mode, "cfd")[indices] - led_mean
                rows.append(_metric_row(config, name, voltage, mode, "cfd", cfd, indices.size, semantic_seed(seed, name, mode, "cfd", stage), logger, stage=stage))
                store.save_residuals(name, "cfd", cfd, stage=stage)

            for model_name, fitted in fitted_models.items():
                spec = get_model(model_name)
                prediction, _time, _pair = predict_indices(spec, fitted, dataset, mode, indices)
                store.save_model_output(name, model_name, prediction, stage=stage)
                residual = corrected_timing_residual(target[indices], prediction)
                rows.append(_metric_row(config, name, voltage, mode, model_name, residual, indices.size, semantic_seed(seed, name, mode, model_name, stage), logger, stage=stage))
                store.save_residuals(name, model_name, residual, stage=stage)

        manifest["datasets"][name] = {
            "prepared_dir": str(dataset.directory),
            "split": dataset.manifest["split"],
            "led_threshold_mV": dataset.manifest["led_threshold_mV"],
            "led_training_mean_ps": dataset.manifest["led_training_mean_ps"],
            "cfd_fraction": dataset.manifest["cfd_fraction"],
            "subsampling": int(dataset.manifest["ml_input"]["subsampling"]),
            "target_definition": "delta_t_led - delta_delta_anchor - true_tof - calibration_bias",
            "corrected_definition": "target - paired_model_prediction",
        }
        store.write_results(rows)
        store.write_manifest(manifest)

    logger.info("Study complete | %s", store.root)
    return store.root
