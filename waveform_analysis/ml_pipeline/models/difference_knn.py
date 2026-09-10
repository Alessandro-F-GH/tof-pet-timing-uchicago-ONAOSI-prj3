from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.neighbors import KNeighborsRegressor

from .spec import ModelSpec


def _difference(pair: np.ndarray) -> np.ndarray:
    """Exact feature representation used by LinearSVR: normalized s1 - s2."""
    x = np.asarray(pair, dtype=np.float32)
    if x.ndim != 3 or x.shape[1] != 2:
        raise ValueError(f"difference_knn expects [event, detector=2, sample], got {x.shape}")
    return np.ascontiguousarray(x[:, 0, :] - x[:, 1, :], dtype=np.float32)


def candidates(config: dict[str, Any]) -> list[dict[str, Any]]:
    parameters = config.get("parameters", {})
    return [
        {"n_neighbors": int(k)}
        for k in parameters.get("n_neighbors", [5, 10, 20, 50, 100])
    ]


@dataclass
class DifferenceKNNArtifact:
    model: KNeighborsRegressor
    metadata: dict[str, Any]


def fit(
    params,
    train_x,
    train_target,
    *,
    seed,
    config,
    validation_x=None,
    validation_target=None,
):
    del seed, validation_x, validation_target
    x = _difference(train_x)
    y = np.asarray(train_target, dtype=np.float64).reshape(-1)
    if x.shape[0] != y.size:
        raise ValueError(f"Training inputs/targets differ in length: {x.shape[0]} != {y.size}")

    n_neighbors = int(params["n_neighbors"])
    if n_neighbors < 1:
        raise ValueError("n_neighbors must be >= 1")
    if n_neighbors > x.shape[0]:
        raise ValueError(f"n_neighbors={n_neighbors} exceeds training events={x.shape[0]}")

    model = KNeighborsRegressor(
        n_neighbors=n_neighbors,
        weights="uniform",
        algorithm="brute",
        metric="euclidean",
        n_jobs=int(config.get("n_jobs", -1)),
    )
    model.fit(x, y)
    metadata = {
        "input_definition": "normalized prepared waveform difference d(t)=s1(t)-s2(t)",
        "input_equivalence": "exact feature representation used by linear_svr",
        "metric": "euclidean",
        "algorithm": "brute",
        "weights": "uniform",
        "n_neighbors": n_neighbors,
        "prediction_definition": "mean timing-correction target of the k nearest training waveform differences [ps]",
        "training_events": int(x.shape[0]),
        "features": int(x.shape[1]),
    }
    return DifferenceKNNArtifact(model=model, metadata=metadata)


def predict(artifact: DifferenceKNNArtifact, normalized_pair: np.ndarray) -> np.ndarray:
    return np.asarray(artifact.model.predict(_difference(normalized_pair)), dtype=np.float64)


def save(artifact: DifferenceKNNArtifact, path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact.model, path / "model.joblib")


MODEL_SPEC = ModelSpec(
    name="difference_knn",
    candidates=candidates,
    fit=fit,
    predict=predict,
    save=save,
    explain=None,
)
