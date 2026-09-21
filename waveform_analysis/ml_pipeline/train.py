from __future__ import annotations

import copy
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from utils_fit import fit_ctr_ps

from .models.spec import ModelSpec
from .sample_mask import apply_sample_mask, apply_sample_mask_to_time, training_sample_mask
from .search import SearchResult, select_candidate
from .storage import atomic_json
from .view import model_target, waveform_view


@dataclass
class FittedModel:
    artifact: Any
    metadata: dict[str, Any]
    output_max_abs_ps: float | None = None
    sample_mask: np.ndarray | None = None


def _fit_once(
    spec,
    model_config,
    parameters,
    train_x,
    train_target,
    *,
    seed,
    output_max_abs_ps=None,
    input_time_ps=None,
    sample_mask=None,
    logger=None,
):
    runtime_config = copy.deepcopy(model_config)
    runtime_config["_prediction_max_abs_ps"] = None if output_max_abs_ps is None else float(output_max_abs_ps)
    if logger is not None:
        runtime_config["_logger"] = logger
    if input_time_ps is not None:
        runtime_config["_input_time_ps"] = np.asarray(input_time_ps, dtype=np.float64)
    artifact = spec.fit(
        parameters,
        np.asarray(train_x, dtype=np.float32),
        np.asarray(train_target, dtype=np.float64),
        seed=seed,
        config=runtime_config,
    )
    metadata = dict(getattr(artifact, "metadata", {}) or {})
    metadata["output_max_abs_ps"] = None if output_max_abs_ps is None else float(output_max_abs_ps)
    metadata["target_definition"] = "delta_t_led - true_tof - calibration_bias"
    return FittedModel(
        artifact,
        metadata,
        None if output_max_abs_ps is None else float(output_max_abs_ps),
        None if sample_mask is None else np.asarray(sample_mask, dtype=bool).copy(),
    )


def predict_model(spec: ModelSpec, fitted: FittedModel, pair: np.ndarray) -> np.ndarray:
    model_input = apply_sample_mask(pair, fitted.sample_mask)
    values = np.asarray(spec.predict(fitted.artifact, model_input), dtype=np.float64)
    if fitted.output_max_abs_ps is not None:
        values = np.clip(values, -float(fitted.output_max_abs_ps), float(fitted.output_max_abs_ps))
    return values



def fit_development_model(
    spec,
    model_config,
    config,
    dataset,
    mode: str,
    parameters,
    *,
    seed: int,
    sample_mask: np.ndarray | None = None,
    logger=None,
    selection_performed: bool,
) -> FittedModel:
    """Fit the final model on the full development population.

    Models that implement internal early stopping may carve their configured
    holdout from this development population. The permanent blind/test split is
    never used for fitting or early stopping.
    """
    fit_indices = np.asarray(dataset.development, dtype=np.int64)
    fit_view = waveform_view(dataset, mode, fit_indices)
    fit_x_full = fit_view.materialize()
    training_indices = np.asarray(dataset.training, dtype=np.int64)
    if sample_mask is None:
        mask_view = waveform_view(dataset, mode, training_indices)
        sample_mask = training_sample_mask(mask_view.materialize())
    sample_mask = np.asarray(sample_mask, dtype=bool).reshape(-1)
    if sample_mask.size != fit_x_full.shape[-1]:
        raise ValueError(
            f"Sample mask has {sample_mask.size} entries but waveform has "
            f"{fit_x_full.shape[-1]} samples"
        )
    if not np.any(sample_mask):
        raise ValueError("Sample mask removes every waveform sample")

    fit_x = apply_sample_mask(fit_x_full, sample_mask)
    masked_time_ps = apply_sample_mask_to_time(fit_view.time_ps, sample_mask)
    fit_target = model_target(dataset, mode)[fit_indices]

    runtime_model_config = copy.deepcopy(model_config)
    runtime_model_config["_early_stopping_seed"] = int(seed)
    fitted = _fit_once(
        spec,
        runtime_model_config,
        dict(parameters or {}),
        fit_x,
        fit_target,
        seed=seed,
        output_max_abs_ps=float(config["ml_output"]["max_abs_ps"]),
        input_time_ps=masked_time_ps,
        sample_mask=sample_mask,
        logger=logger,
    )
    fitted.metadata.update(
        {
            "sample_mask_definition": (
                "shared temporal mask derived from the full training split; discard a sample "
                "when both detector channels independently have one exact normalized float32 "
                "value in at least 99% of training events"
            ),
            "sample_mask_constant_fraction": 0.99,
            "sample_mask_training_events": int(training_indices.size),
            "input_samples_before_mask": int(sample_mask.size),
            "input_samples_after_mask": int(np.count_nonzero(sample_mask)),
            "input_samples_removed": int(
                sample_mask.size - np.count_nonzero(sample_mask)
            ),
            "training_events": int(fit_target.size),
            "fit_population": "development",
            "development_events": int(fit_target.size),
            "validation_used_for_training": True,
            "validation_used_for_selection": bool(selection_performed),
            "model_selection_performed": bool(selection_performed),
            "refit_after_selection": bool(selection_performed),
            "selection_protocol": (
                "validation_ctr_model_selection_then_development_refit"
                if selection_performed
                else "single_candidate_selection_skipped_development_fit"
            ),
        }
    )
    return fitted


def fit_fixed_model(
    spec,
    model_config,
    config,
    dataset,
    mode: str,
    *,
    seed: int,
    sample_mask: np.ndarray | None = None,
    logger=None,
) -> tuple[FittedModel, dict[str, Any]]:
    """Fit one fixed-reference candidate on the full development population."""
    candidates = list(spec.candidates(model_config))
    if len(candidates) != 1:
        raise ValueError(
            f"Fixed-reference model {spec.name!r} must expose exactly one candidate, "
            f"got {len(candidates)}"
        )
    parameters = dict(candidates[0] or {})
    fitted = fit_development_model(
        spec,
        model_config,
        config,
        dataset,
        mode,
        parameters,
        seed=seed,
        sample_mask=sample_mask,
        logger=logger,
        selection_performed=False,
    )
    fitted.metadata.update(
        {
            "selection_protocol": "fixed_reference_configuration_no_model_selection",
            "validation_used_for_selection": False,
            "model_selection_performed": False,
            "refit_after_selection": False,
        }
    )
    return fitted, parameters

def search_model(
    spec,
    model_config,
    config,
    dataset,
    mode: str,
    *,
    seed: int,
    dataset_name: str | None = None,
    sample_mask: np.ndarray | None = None,
    logger=None,
) -> SearchResult:
    training = np.asarray(dataset.training, dtype=np.int64)
    validation = np.asarray(dataset.validation, dtype=np.int64)
    train_view = waveform_view(dataset, mode, training)
    validation_view = waveform_view(dataset, mode, validation)
    train_x_full = train_view.materialize()
    validation_x_full = validation_view.materialize()
    if sample_mask is None:
        sample_mask = training_sample_mask(train_x_full)
    sample_mask = np.asarray(sample_mask, dtype=bool).reshape(-1)
    if sample_mask.size != train_x_full.shape[-1]:
        raise ValueError(
            f"Sample mask has {sample_mask.size} entries but waveform has {train_x_full.shape[-1]} samples"
        )
    if not np.any(sample_mask):
        raise ValueError("Sample mask removes every waveform sample")
    train_x = apply_sample_mask(train_x_full, sample_mask)
    validation_x = apply_sample_mask(validation_x_full, sample_mask)
    masked_time_ps = apply_sample_mask_to_time(train_view.time_ps, sample_mask)
    target = model_target(dataset, mode)
    train_target = target[training]
    validation_target = target[validation]
    output_limit = float(config["ml_output"]["max_abs_ps"])
    fit_config = dict(config.get("fit") or {})

    def fit_candidate(parameters, candidate_seed):
        model_parameters = dict(parameters)
        runtime_model_config = copy.deepcopy(model_config)
        runtime_model_config["_early_stopping_seed"] = int(seed)
        fitted = _fit_once(
            spec,
            runtime_model_config,
            model_parameters,
            train_x,
            train_target,
            seed=candidate_seed,
            output_max_abs_ps=output_limit,
            input_time_ps=masked_time_ps,
            sample_mask=sample_mask,
            logger=logger,
        )
        fitted.metadata.update(
            {
                "sample_mask_definition": (
                    "shared temporal mask derived from the full training split; discard a sample "
                    "when both detector channels independently have one exact normalized float32 "
                    "value in at least 99% of training events"
                ),
                "sample_mask_constant_fraction": 0.99,
                "sample_mask_training_events": int(train_x_full.shape[0]),
                "input_samples_before_mask": int(sample_mask.size),
                "input_samples_after_mask": int(np.count_nonzero(sample_mask)),
                "input_samples_removed": int(sample_mask.size - np.count_nonzero(sample_mask)),
                "training_events": int(train_target.size),
            }
        )
        return fitted

    def predict_candidate(_parameters, fitted):
        return validation_target - predict_model(spec, fitted, validation_x)

    def score_candidate(residual):
        values = np.asarray(residual, dtype=np.float64)
        finite = values[np.isfinite(values)]
        if finite.size != values.size or finite.size == 0:
            raise ValueError("Validation CTR requires finite residuals for every validation event")
        return float(
            fit_ctr_ps(
                finite,
                fit_config,
                seed=seed,
                bootstrap=False,
            ).ctr_ps
        )

    context = str(dataset_name or "dataset")
    model_label = "Antisymmetric MLP" if spec.name == "mlp" else spec.name

    def on_start(number, total, candidate):
        return None

    def on_result(number, total, result):
        if logger is None or total <= 1:
            return
        parameters = dict(result.candidate or {})
        parameters_json = json.dumps(parameters, sort_keys=True)
        if result.error is None:
            logger.info(
                "Candidate | %s | %s | %d/%d | validation CTR=%.3f ps | params=%s",
                context,
                model_label,
                number,
                total,
                result.score,
                parameters_json,
            )
        else:
            logger.warning(
                "Candidate | %s | %s | %d/%d | FAILED | params=%s | %s",
                context,
                model_label,
                number,
                total,
                parameters_json,
                result.error,
            )

    candidates = list(spec.candidates(model_config))
    if logger is not None and len(candidates) > 1:
        logger.info(
            "Model selection | %s | %s | candidates=%d",
            context,
            model_label,
            len(candidates),
        )

    return select_candidate(
        candidates,
        fit_candidate=fit_candidate,
        predict_candidate=predict_candidate,
        score_candidate=score_candidate,
        seed=seed,
        on_candidate_start=on_start,
        on_candidate_result=on_result,
    )


def predict_indices(spec, fitted, dataset, mode, indices):
    view = waveform_view(dataset, mode, np.asarray(indices, dtype=np.int64))
    pair = apply_sample_mask(view.materialize(), fitted.sample_mask)
    time_ps = apply_sample_mask_to_time(view.time_ps, fitted.sample_mask)
    return predict_model(spec, fitted, pair), time_ps, pair


def save_model(spec, fitted, directory: Path, parameters):
    directory.mkdir(parents=True, exist_ok=True)
    spec.save(fitted.artifact, directory)
    sample_mask_file = None
    if fitted.sample_mask is not None:
        sample_mask_file = "sample_mask.npy"
        np.save(directory / sample_mask_file, np.asarray(fitted.sample_mask, dtype=bool))
    prediction_definition = fitted.metadata.get(
        "prediction_definition",
        "paired waveform-dependent timing correction y_theta = g(s1)-g(s2) [ps]",
    )
    atomic_json(
        directory / "metadata.json",
        {
            "model": spec.name,
            "parameters": parameters,
            "training": fitted.metadata,
            "sample_mask_file": sample_mask_file,
            "selection_protocol": fitted.metadata.get(
                "selection_protocol",
                "development_fit_with_internal_early_stopping",
            ),
            "prediction_definition": prediction_definition,
        },
    )
