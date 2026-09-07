from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .config import discover_root_files, load_config, public_config
from .models import get_model
from .prepared_data import prepare_file_dataset
from .search import SearchResult
from .splits import make_split, semantic_seed
from .standard_methods import evaluate_cfd, evaluate_led, select_cfd, select_led
from .stats import bootstrap_ctr, ctr_fwhm
from .storage import RunStore
from .train import FittedModel, predict_indices, refit_selected, save_model, search_model
from .view import target_family


@dataclass
class StandardSelection:
    led: SearchResult
    cfd: SearchResult | None
    development_led_ps: np.ndarray
    training_led_ps: np.ndarray
    validation_led_ps: np.ndarray


@dataclass
class FinalModel:
    fitted: FittedModel
    search: SearchResult


def _logger(run_dir: Path) -> logging.Logger:
    logger = logging.getLogger(f"waveform-study:{run_dir}")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    for handler in (logging.StreamHandler(), logging.FileHandler(run_dir / "study.log", encoding="utf-8")):
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger


def _metric_row(
    config: dict[str, Any],
    dataset_name: str,
    voltage: float,
    mode: str,
    method: str,
    stage: str,
    residual_ps: np.ndarray,
    *,
    parameters: Any,
    population_n: int,
    seed: int,
) -> dict[str, Any]:
    residual = np.asarray(residual_ps, dtype=np.float64)
    finite = residual[np.isfinite(residual)]
    if finite.size < int((config.get("fit") or {}).get("min_events", 20)):
        raise RuntimeError(f"{dataset_name}/{mode}/{method}: too few finite {stage} residuals")
    bootstrap = int(config.get("reporting", {}).get("ctr_uncertainty_bootstrap_samples", 500)) if stage == "blind" else 0
    metric = bootstrap_ctr(finite, bootstrap, seed, config.get("fit")) if bootstrap else ctr_fwhm(finite, config.get("fit"))
    return {
        "dataset": dataset_name,
        "voltage_V": voltage,
        "mode": mode,
        "method": method,
        "stage": stage,
        "ctr_ps": float(metric.ctr_ps),
        "ctr_uncertainty_ps": float(metric.uncertainty_ps),
        "center_ps": float(metric.center_ps),
        "n": int(metric.n),
        "population_n": int(population_n),
        "crossing_efficiency": float(metric.n / max(1, int(population_n))),
        "parameters_json": json.dumps(parameters, sort_keys=True),
    }


def _validation_row(dataset_name, voltage, mode, method, search, n, parameters=None):
    return {
        "dataset": dataset_name,
        "voltage_V": voltage,
        "mode": mode,
        "method": method,
        "stage": "validation",
        "ctr_ps": float(search.best.score),
        "ctr_uncertainty_ps": float("nan"),
        "center_ps": float("nan"),
        "n": int(n),
        "population_n": int(n),
        "crossing_efficiency": 1.0,
        "parameters_json": json.dumps(search.best.candidate if parameters is None else parameters, sort_keys=True),
    }


def _prepare_datasets(config: dict[str, Any], rebuild: bool, logger: logging.Logger):
    roots = discover_root_files(config)
    if not roots:
        raise FileNotFoundError(
            f"No ROOT files match {config['data'].get('root_glob', '*.root')} in {config['data']['root_folder']}"
        )
    return [prepare_file_dataset(config, root, rebuild=rebuild, logger=logger) for root in roots]


def run_study(
    config_or_path: dict[str, Any] | str | Path,
    *,
    overwrite: bool = False,
    rebuild_preprocessing: bool = False,
) -> Path:
    config = load_config(config_or_path) if not isinstance(config_or_path, dict) else config_or_path
    store = RunStore(config["experiment"]["output_dir"], overwrite=overwrite)
    logger = _logger(store.root)
    datasets = _prepare_datasets(config, rebuild_preprocessing, logger)
    rows: list[dict[str, Any]] = []
    seed = int(config["validation"].get("seed", 20260813))
    modes = list(config["channel_modes"])
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "protocol": "development_blind_then_training_validation_holdout",
        "ctr_metric": "direct_smoothed_histogram_fwhm",
        "blind_selection_use": False,
        "config": public_config(config),
        "config_fingerprint": config.get("_config_fingerprint"),
        "datasets": {},
    }

    for dataset in datasets:
        name = str(dataset.manifest.get("name", dataset.directory.name))
        voltage = float(np.nanmedian(dataset.bias_voltage_V)) if dataset.bias_voltage_V.size else float("nan")
        split = make_split(
            dataset.n_events,
            blind_fraction=float(config["validation"]["blind_fraction"]),
            validation_fraction=float(config["validation"]["validation_fraction"]),
            seed=semantic_seed(seed, name),
        )
        store.save_split(name, split)
        logger.info("Dataset %s | %s", name, split.as_dict())

        needed_families = sorted({target_family(mode) for mode in modes})
        standards: dict[str, StandardSelection] = {}
        for family in needed_families:
            led_search = select_led(
                config, dataset, family, split.training, split.validation,
                semantic_seed(seed, name, family, "led"),
            )
            cfd_needed = any(
                target_family(mode) == family and bool((config["modes"].get(mode) or {}).get("cfd", True))
                for mode in modes
            )
            cfd_search = (
                select_cfd(
                    config, dataset, family, split.training, split.validation,
                    semantic_seed(seed, name, family, "cfd"),
                )
                if cfd_needed else None
            )
            led_threshold = float(led_search.best.candidate)
            standards[family] = StandardSelection(
                led=led_search,
                cfd=cfd_search,
                development_led_ps=evaluate_led(config, dataset, family, split.development, led_threshold),
                training_led_ps=evaluate_led(config, dataset, family, split.training, led_threshold),
                validation_led_ps=evaluate_led(config, dataset, family, split.validation, led_threshold),
            )
            store.save_search(name, family, "led", led_search.as_dict())
            if cfd_search is not None:
                store.save_search(name, family, "cfd", cfd_search.as_dict())
            logger.info(
                "Standards %s | LED %.6g mV -> %.3f ps%s",
                family,
                led_threshold,
                led_search.best.score,
                "" if cfd_search is None else f" | CFD {float(cfd_search.best.candidate):.4f} -> {cfd_search.best.score:.3f} ps",
            )

        # Phase 1: all model selection and final development refits. Blind indices
        # are deliberately not passed to any function in this phase.
        final_models: dict[tuple[str, str], FinalModel] = {}
        for mode in modes:
            family = target_family(mode)
            standard = standards[family]
            rows.append(_validation_row(name, voltage, mode, "led", standard.led, split.validation.size))
            if bool((config["modes"].get(mode) or {}).get("cfd", True)) and standard.cfd is not None:
                rows.append(_validation_row(name, voltage, mode, "cfd", standard.cfd, split.validation.size))

            for model_name, model_config in config["models"].items():
                spec = get_model(model_name)
                search = search_model(
                    spec,
                    model_config,
                    config,
                    dataset,
                    mode,
                    split.training,
                    split.validation,
                    standard.training_led_ps,
                    standard.validation_led_ps,
                    seed=semantic_seed(seed, name, mode, model_name, "search"),
                )
                fitted = refit_selected(
                    spec,
                    model_config,
                    config,
                    dataset,
                    mode,
                    split.development,
                    standard.development_led_ps,
                    search.best,
                    seed=semantic_seed(seed, name, mode, model_name, "final"),
                )
                save_model(spec, fitted, store.model_dir(name, mode, model_name), search.best.candidate)
                store.save_search(name, mode, model_name, search.as_dict())
                final_models[(mode, model_name)] = FinalModel(fitted, search)
                rows.append(_validation_row(name, voltage, mode, model_name, search, split.validation.size))
                logger.info(
                    "Selected %s/%s | validation CTR %.3f ps | %s",
                    mode, model_name, search.best.score, search.best.candidate,
                )

                xai_cfg = config.get("reporting", {}).get("xai", {}) or {}
                if bool(xai_cfg.get("enabled", True)) and spec.explain is not None:
                    baseline = standard.development_led_ps
                    valid = np.flatnonzero(np.isfinite(baseline))
                    limit = min(valid.size, int(xai_cfg.get("max_events", 1024)))
                    chosen = split.development[valid[:limit]]
                    _prediction, time_ps, pair = predict_indices(
                        spec, fitted, config, dataset, mode, chosen, search.best.candidate
                    )
                    importance = spec.explain(fitted.artifact, fitted.normalization.transform(pair))
                    store.save_xai(
                        name, mode, model_name,
                        time_ps=time_ps,
                        importance=importance,
                        example_pair_mV=pair[0],
                    )

        # Phase 2: only now is the permanent blind population touched.
        for mode in modes:
            family = target_family(mode)
            standard = standards[family]
            led_threshold = float(standard.led.best.candidate)
            blind_led = evaluate_led(config, dataset, family, split.blind, led_threshold)
            led_residual = blind_led - float(dataset.true_tof_ps)
            rows.append(
                _metric_row(
                    config, name, voltage, mode, "led", "blind", led_residual,
                    parameters={"threshold_mV": led_threshold},
                    population_n=split.blind.size,
                    seed=semantic_seed(seed, name, mode, "led", "blind"),
                )
            )
            store.save_residuals(name, mode, "led", led_residual)

            if bool((config["modes"].get(mode) or {}).get("cfd", True)) and standard.cfd is not None:
                fraction = float(standard.cfd.best.candidate)
                blind_cfd = evaluate_cfd(config, dataset, family, split.blind, fraction)
                cfd_residual = blind_cfd - float(dataset.true_tof_ps)
                rows.append(
                    _metric_row(
                        config, name, voltage, mode, "cfd", "blind", cfd_residual,
                        parameters={"fraction": fraction},
                        population_n=split.blind.size,
                        seed=semantic_seed(seed, name, mode, "cfd", "blind"),
                    )
                )
                store.save_residuals(name, mode, "cfd", cfd_residual)

            led_valid = np.isfinite(blind_led)
            for model_name in config["models"]:
                final = final_models[(mode, model_name)]
                spec = get_model(model_name)
                blind_indices = split.blind[led_valid]
                correction, _time, _pair = predict_indices(
                    spec,
                    final.fitted,
                    config,
                    dataset,
                    mode,
                    blind_indices,
                    final.search.best.candidate,
                )
                residual = blind_led[led_valid] + correction - float(dataset.true_tof_ps)
                rows.append(
                    _metric_row(
                        config, name, voltage, mode, model_name, "blind", residual,
                        parameters=final.search.best.candidate,
                        population_n=split.blind.size,
                        seed=semantic_seed(seed, name, mode, model_name, "blind"),
                    )
                )
                store.save_residuals(name, mode, model_name, residual)

        manifest["datasets"][name] = {
            "path": str(dataset.directory),
            "n_events": dataset.n_events,
            "split": split.as_dict(),
            "standards": {
                family: {
                    "led_threshold_mV": float(selection.led.best.candidate),
                    "cfd_fraction": None if selection.cfd is None else float(selection.cfd.best.candidate),
                }
                for family, selection in standards.items()
            },
        }
        store.write_results(rows)
        store.write_manifest(manifest)

    logger.info("Study complete | %s", store.root)
    return store.root
