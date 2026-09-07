from __future__ import annotations

import itertools
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.svm import LinearSVR

from .spec import ModelSpec


@dataclass
class LinearSVRArtifact:
    model: LinearSVR
    metadata: dict[str, Any]


def candidates(config: dict[str, Any]) -> list[dict[str, Any]]:
    parameters = config.get("parameters", {})
    c_values = parameters.get("C", [1.0, 100.0])
    epsilon_values = parameters.get("epsilon_ps", [10.0, 60.0])
    return [
        {"C": float(c_value), "epsilon_ps": float(epsilon)}
        for c_value, epsilon in itertools.product(c_values, epsilon_values)
    ]


def fit(
    params: dict[str, Any],
    train_x: np.ndarray,
    train_target: np.ndarray,
    *,
    seed: int,
    config: dict[str, Any],
    validation_x: np.ndarray | None = None,
    validation_target: np.ndarray | None = None,
    final_epochs: int | None = None,
) -> LinearSVRArtifact:
    del validation_x, validation_target, final_epochs
    x = np.asarray(train_x, dtype=np.float64)
    if x.ndim != 3 or x.shape[1] != 2:
        raise ValueError("Linear SVR expects [event, detector, sample]")
    model = LinearSVR(
        C=float(params["C"]),
        epsilon=float(params["epsilon_ps"]),
        fit_intercept=False,
        loss=str(config.get("loss", "epsilon_insensitive")),
        tol=float(config.get("tolerance", 1e-3)),
        max_iter=int(config.get("max_iterations", 10000)),
        dual=config.get("dual", "auto"),
        random_state=int(seed),
    )
    model.fit(x[:, 0, :] - x[:, 1, :], np.asarray(train_target, dtype=np.float64))
    return LinearSVRArtifact(model, {})


def predict(artifact: LinearSVRArtifact, normalized_pair: np.ndarray) -> np.ndarray:
    x = np.asarray(normalized_pair, dtype=np.float64)
    return np.asarray(artifact.model.predict(x[:, 0, :] - x[:, 1, :]), dtype=np.float64)


def save(artifact: LinearSVRArtifact, path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact.model, path / "model.joblib")


def explain(artifact: LinearSVRArtifact, normalized_pair: np.ndarray) -> np.ndarray:
    del normalized_pair
    return np.abs(np.asarray(artifact.model.coef_, dtype=np.float64))


MODEL_SPEC = ModelSpec(
    name="linear_svr",
    normalization="feature",
    candidates=candidates,
    fit=fit,
    predict=predict,
    save=save,
    explain=explain,
)
