from __future__ import annotations

import copy
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
    validation_x=None,
    validation_target=None,
    output_max_abs_ps=None,
    input_time_ps=None,
    sample_mask=None,
):
    runtime_config = copy.deepcopy(model_config)
    runtime_config["_prediction_max_abs_ps"] = None if output_max_abs_ps is None else float(output_max_abs_ps)
    if input_time_ps is not None:
        runtime_config["_input_time_ps"] = np.asarray(input_time_ps, dtype=np.float64)
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
    dataset_label = str(dataset_name or dataset.directory.name)

    def fit_candidate(parameters, candidate_seed):
        model_parameters = dict(parameters)
        runtime_model_config = copy.deepcopy(model_config)
        runtime_model_config["_fit_config"] = fit_config
        fitted = _fit_once(
            spec,
            runtime_model_config,
            model_parameters,
            train_x,
            train_target,
            seed=candidate_seed,
            validation_x=validation_x,
            validation_target=validation_target,
            output_max_abs_ps=output_limit,
            input_time_ps=masked_time_ps,
            sample_mask=sample_mask,
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
                "training_uses_full_split": True,
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

    def on_start(number, total, candidate):
        if logger is not None:
            logger.debug(
                "  %d/%d | starting | %s",
                number,
                total,
                "default" if not candidate else candidate,
            )

    def on_result(number, total, result):
        if logger is None:
            return
        parameters = "default" if not result.candidate else result.candidate
        if result.error is None:
            sigma_suffix = ""
            if "sigma_max_ps" in result.metadata:
                sigma_suffix = f" | selected sigma_max={float(result.metadata['sigma_max_ps']):g} ps"
            logger.info(
                "  %d/%d | CTR=%.6g ps | %s%s",
                number,
                total,
                result.score,
                parameters,
                sigma_suffix,
            )
        else:
            logger.warning(
                "  %d/%d | FAILED | %s | %s",
                number,
                total,
                parameters,
                result.error,
            )

    candidates = list(spec.candidates(model_config))
    if logger is not None:
        sigma_thresholds = list(
            (model_config.get("parameters") or {}).get("sigma_max_ps") or []
        )
        if spec.name == "cnn_heteroscedastic" and sigma_thresholds:
            logger.info(
                "%s search | training_candidates=%d | sigma_thresholds=%d | values=%s",
                spec.name,
                len(candidates),
                len(sigma_thresholds),
                sigma_thresholds,
            )
        else:
            logger.info("%s search | candidates=%d", spec.name, len(candidates))

    return select_candidate(
        candidates,
        fit_candidate=fit_candidate,
        predict_candidate=predict_candidate,
        score_candidate=score_candidate,
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
            "selection_protocol": "full_training_split_validation_ctr_selected_model_used_directly_without_refit",
            "prediction_definition": prediction_definition,
        },
    )
