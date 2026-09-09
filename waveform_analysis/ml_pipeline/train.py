from __future__ import annotations

import copy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .models.spec import ModelSpec
from .search import SearchResult, select_candidate
from .storage import atomic_json
from .view import model_target, waveform_view


@dataclass
class FittedModel:
    artifact: Any
    metadata: dict[str, Any]
    output_max_abs_ps: float | None = None


def _fit_once(
    spec,
    model_config,
    parameters,
    train_x,
    train_target,
    *,
    seed,
    validation_x=None,
    validation_target=None,
    output_max_abs_ps=None,
):
    runtime_config = copy.deepcopy(model_config)
    runtime_config["_prediction_max_abs_ps"] = None if output_max_abs_ps is None else float(output_max_abs_ps)
    artifact = spec.fit(
        parameters,
        np.asarray(train_x, dtype=np.float32),
        np.asarray(train_target, dtype=np.float64),
        seed=seed,
        config=runtime_config,
        validation_x=None if validation_x is None else np.asarray(validation_x, dtype=np.float32),
        validation_target=None if validation_target is None else np.asarray(validation_target, dtype=np.float64),
    )
    metadata = dict(getattr(artifact, "metadata", {}) or {})
    metadata["output_max_abs_ps"] = None if output_max_abs_ps is None else float(output_max_abs_ps)
    metadata["target_definition"] = "delta_t_led - delta_anchor_shift - true_tof - calibration_bias"
    return FittedModel(artifact, metadata, None if output_max_abs_ps is None else float(output_max_abs_ps))


def predict_model(spec: ModelSpec, fitted: FittedModel, pair: np.ndarray) -> np.ndarray:
    values = np.asarray(spec.predict(fitted.artifact, np.asarray(pair, dtype=np.float32)), dtype=np.float64)
    if fitted.output_max_abs_ps is not None:
        values = np.clip(values, -float(fitted.output_max_abs_ps), float(fitted.output_max_abs_ps))
    return values


def _rmse(values):
    residual = np.asarray(values, dtype=np.float64)
    return float(np.sqrt(np.mean(residual**2)))


def search_model(
    spec,
    model_config,
    config,
    dataset,
    mode: str,
    *,
    seed: int,
    dataset_name: str | None = None,
    logger=None,
) -> SearchResult:
    training = np.asarray(dataset.training, dtype=np.int64)
    validation = np.asarray(dataset.validation, dtype=np.int64)
    train_x = waveform_view(dataset, mode, training).materialize()
    validation_x = waveform_view(dataset, mode, validation).materialize()
    target = model_target(dataset, mode)
    train_target = target[training]
    validation_target = target[validation]
    output_limit = float(config["ml_output"]["max_abs_ps"])
    dataset_label = str(dataset_name or dataset.directory.name)

    def fit_candidate(parameters, candidate_seed):
        return _fit_once(
            spec,
            model_config,
            parameters,
            train_x,
            train_target,
            seed=candidate_seed,
            validation_x=validation_x,
            validation_target=validation_target,
            output_max_abs_ps=output_limit,
        )

    def predict_candidate(_parameters, fitted):
        return validation_target - predict_model(spec, fitted, validation_x)

    def on_start(number, total, candidate):
        if logger is not None:
            logger.info(
                "Training dataset=%s | %s/%s | candidate %d/%d | %s",
                dataset_label,
                mode,
                spec.name,
                number,
                total,
                candidate,
            )

    def on_result(number, total, result):
        if logger is None:
            return
        if result.error is None:
            logger.info(
                "Validation dataset=%s | %s/%s | candidate %d/%d | slide-target RMSE %.6g ps | output clipped to ±%.0f ps | %s",
                dataset_label,
                mode,
                spec.name,
                number,
                total,
                result.score,
                output_limit,
                result.candidate,
            )
        else:
            logger.warning(
                "Candidate failed dataset=%s | %s/%s | candidate %d/%d | %s | %s",
                dataset_label,
                mode,
                spec.name,
                number,
                total,
                result.candidate,
                result.error,
            )

    return select_candidate(
        spec.candidates(model_config),
        fit_candidate=fit_candidate,
        predict_candidate=predict_candidate,
        score_candidate=_rmse,
        seed=seed,
        on_candidate_start=on_start,
        on_candidate_result=on_result,
    )


def selected_model(search: SearchResult) -> FittedModel:
    """Return the validation-selected model exactly as trained during search."""
    fitted = search.best.artifact
    if not isinstance(fitted, FittedModel):
        raise RuntimeError("Selected candidate no longer contains its trained model artifact")
    return fitted


def predict_indices(spec, fitted, dataset, mode, indices):
    view = waveform_view(dataset, mode, np.asarray(indices, dtype=np.int64))
    pair = view.materialize()
    return predict_model(spec, fitted, pair), view.time_ps, pair


def save_model(spec, fitted, directory: Path, parameters):
    directory.mkdir(parents=True, exist_ok=True)
    spec.save(fitted.artifact, directory)
    atomic_json(
        directory / "metadata.json",
        {
            "model": spec.name,
            "parameters": parameters,
            "training": fitted.metadata,
            "selection_protocol": "validation_selected_model_used_directly_without_refit",
            "prediction_definition": "paired waveform-dependent timing correction y_theta = g(s1)-g(s2) [ps]",
        },
    )
