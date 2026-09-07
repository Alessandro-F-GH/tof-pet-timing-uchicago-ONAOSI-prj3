from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .models.spec import ModelSpec
from .search import SearchResult, select_candidate
from .stats import ctr_fwhm
from .storage import atomic_json
from .view import waveform_view


@dataclass(frozen=True)
class Normalization:
    mode: str
    mean: float | np.ndarray
    scale: float | np.ndarray

    @classmethod
    def fit(cls, pair: np.ndarray, mode: str) -> "Normalization":
        x = np.asarray(pair, dtype=np.float64)
        if x.ndim != 3 or x.shape[1] != 2:
            raise ValueError("Waveform pair must have shape [event, detector, sample]")
        if mode == "global":
            mean = float(np.mean(x))
            scale = max(float(np.std(x)), 1e-6)
            return cls(mode, mean, scale)
        if mode == "feature":
            mean = np.mean(x, axis=(0, 1))
            scale = np.std(x, axis=(0, 1))
            scale = np.where(scale > 1e-6, scale, 1.0)
            return cls(mode, mean.astype(np.float64), scale.astype(np.float64))
        raise ValueError(f"Unknown normalization mode: {mode}")

    def transform(self, pair: np.ndarray) -> np.ndarray:
        return ((np.asarray(pair, dtype=np.float32) - self.mean) / self.scale).astype(np.float32)

    def as_dict(self) -> dict[str, Any]:
        def value(item):
            array = np.asarray(item)
            return float(array) if array.ndim == 0 else array.tolist()
        return {"mode": self.mode, "mean": value(self.mean), "scale": value(self.scale)}


@dataclass
class FittedModel:
    artifact: Any
    normalization: Normalization
    metadata: dict[str, Any]


@dataclass
class ModelSelection:
    search: SearchResult
    final: FittedModel
    candidate: dict[str, Any]


def representation_candidates(
    spec: ModelSpec,
    model_config: dict[str, Any],
    windows: list[dict[str, Any]],
    subsampling_factors: list[int],
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for parameters in spec.candidates(model_config):
        for window in windows:
            for factor in subsampling_factors:
                output.append(
                    {
                        "parameters": parameters,
                        "window_id": str(window["id"]),
                        "start_ns": float(window["start_ns"]),
                        "end_ns": float(window["end_ns"]),
                        "subsampling": int(factor),
                    }
                )
    return output


def _materialize(
    dataset,
    mode: str,
    indices: np.ndarray,
    candidate: dict[str, Any],
    preprocessing: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray]:
    view = waveform_view(
        dataset,
        mode,
        indices,
        start_ns=float(candidate["start_ns"]),
        end_ns=float(candidate["end_ns"]),
        subsampling=int(candidate["subsampling"]),
        preprocessing=preprocessing,
    )
    return view.materialize(), view.time_ps


def _fit_once(
    spec: ModelSpec,
    model_config: dict[str, Any],
    candidate: dict[str, Any],
    train_x: np.ndarray,
    train_target: np.ndarray,
    *,
    seed: int,
    validation_x: np.ndarray | None = None,
    validation_target: np.ndarray | None = None,
    final_epochs: int | None = None,
) -> FittedModel:
    normalization = Normalization.fit(train_x, spec.normalization)
    x_train = normalization.transform(train_x)
    x_validation = None if validation_x is None else normalization.transform(validation_x)
    artifact = spec.fit(
        candidate["parameters"],
        x_train,
        train_target,
        seed=seed,
        config=model_config,
        validation_x=x_validation,
        validation_target=validation_target,
        final_epochs=final_epochs,
    )
    return FittedModel(artifact, normalization, dict(getattr(artifact, "metadata", {}) or {}))


def predict_model(spec: ModelSpec, fitted: FittedModel, pair: np.ndarray) -> np.ndarray:
    return np.asarray(spec.predict(fitted.artifact, fitted.normalization.transform(pair)), dtype=np.float64)


def search_model(
    spec: ModelSpec,
    model_config: dict[str, Any],
    config: dict[str, Any],
    dataset,
    mode: str,
    training_indices: np.ndarray,
    validation_indices: np.ndarray,
    training_baseline_ps: np.ndarray,
    validation_baseline_ps: np.ndarray,
    *,
    seed: int,
) -> SearchResult:
    true_tof = float(dataset.true_tof_ps)
    train_baseline = np.asarray(training_baseline_ps, dtype=np.float64)
    val_baseline = np.asarray(validation_baseline_ps, dtype=np.float64)
    train_valid = np.isfinite(train_baseline)
    val_valid = np.isfinite(val_baseline)
    if np.count_nonzero(train_valid) < 10 or np.count_nonzero(val_valid) < 10:
        raise RuntimeError("Too few finite optimized-LED events for model search")
    train_indices = np.asarray(training_indices, dtype=np.int64)[train_valid]
    val_indices = np.asarray(validation_indices, dtype=np.int64)[val_valid]
    train_target = true_tof - train_baseline[train_valid]
    val_target = true_tof - val_baseline[val_valid]

    candidates = representation_candidates(
        spec,
        model_config,
        config["windows_ns"],
        [int(value) for value in config["preprocessing"].get("subsampling_factors", [1])],
    )
    materialized: dict[tuple[str, int, str], tuple[np.ndarray, np.ndarray]] = {}

    def data(candidate: dict[str, Any], split: str) -> tuple[np.ndarray, np.ndarray]:
        key = (str(candidate["window_id"]), int(candidate["subsampling"]), split)
        if key not in materialized:
            indices = train_indices if split == "train" else val_indices
            materialized[key] = _materialize(dataset, mode, indices, candidate, config["preprocessing"])
        return materialized[key]

    def fit_candidate(candidate, _train_data, candidate_seed):
        train_x, _ = data(candidate, "train")
        val_x, _ = data(candidate, "validation")
        return _fit_once(
            spec,
            model_config,
            candidate,
            train_x,
            train_target,
            seed=candidate_seed,
            validation_x=val_x,
            validation_target=val_target,
        )

    def predict_candidate(candidate, fitted, _validation_data):
        val_x, _ = data(candidate, "validation")
        correction = predict_model(spec, fitted, val_x)
        return val_baseline[val_valid] + correction - true_tof

    return select_candidate(
        candidates,
        train_indices,
        val_indices,
        fit_candidate=fit_candidate,
        predict_candidate=predict_candidate,
        score_candidate=lambda residual: ctr_fwhm(residual, config.get("fit")).ctr_ps,
        seed=seed,
    )


def refit_selected(
    spec: ModelSpec,
    model_config: dict[str, Any],
    config: dict[str, Any],
    dataset,
    mode: str,
    development_indices: np.ndarray,
    development_baseline_ps: np.ndarray,
    selected,
    *,
    seed: int,
) -> FittedModel:
    baseline = np.asarray(development_baseline_ps, dtype=np.float64)
    valid = np.isfinite(baseline)
    indices = np.asarray(development_indices, dtype=np.int64)[valid]
    target = float(dataset.true_tof_ps) - baseline[valid]
    x, _time = _materialize(dataset, mode, indices, selected.candidate, config["preprocessing"])
    epochs = selected.metadata.get("best_epoch") if selected.metadata else None
    return _fit_once(
        spec,
        model_config,
        selected.candidate,
        x,
        target,
        seed=seed,
        final_epochs=None if epochs is None else int(epochs),
    )


def predict_indices(spec, fitted, config, dataset, mode, indices, candidate) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x, time_ps = _materialize(dataset, mode, np.asarray(indices, dtype=np.int64), candidate, config["preprocessing"])
    return predict_model(spec, fitted, x), time_ps, x


def save_model(spec: ModelSpec, fitted: FittedModel, directory: Path, candidate: dict[str, Any]) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    spec.save(fitted.artifact, directory)
    atomic_json(
        directory / "metadata.json",
        {
            "model": spec.name,
            "candidate": candidate,
            "normalization": fitted.normalization.as_dict(),
            "training": fitted.metadata,
        },
    )
