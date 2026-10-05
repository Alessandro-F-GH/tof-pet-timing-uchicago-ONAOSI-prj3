from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.linear_model import Ridge


@dataclass
class LinearTransformArtifact:
    metadata: dict[str, Any]


@dataclass
class LinearRidgeArtifact:
    regressor: Ridge
    metadata: dict[str, Any]


def finite_grid_values(raw, default):
    if raw is None:
        return list(default)
    if isinstance(raw, dict):
        kind = str(raw.get("type", "")).strip().lower()
        if kind == "fixed":
            return [raw["value"]]
        if kind == "categorical":
            return list(raw["choices"])
        raise ValueError("linear Ridge candidates require a finite fixed/categorical alpha grid")
    if isinstance(raw, (list, tuple, np.ndarray)):
        return list(raw)
    return [raw]


def candidates(config):
    parameters = config.get("parameters", {})
    alphas = [
        float(value)
        for value in finite_grid_values(
            parameters.get("ridge_alpha"),
            [0.01, 0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0, 100000.0],
        )
    ]
    if any((not np.isfinite(alpha)) or alpha <= 0 for alpha in alphas):
        raise ValueError("linear Ridge ridge_alpha values must be finite and positive")
    return [{"ridge_alpha": alpha} for alpha in alphas]


def transform_parameters(params, config):
    del params, config
    return {}


def validate_pair(pair: np.ndarray, model_name: str) -> np.ndarray:
    values = np.asarray(pair, dtype=np.float64)
    if values.ndim != 3 or values.shape[1] != 2:
        raise ValueError(
            f"{model_name} expects [event, detector=2, time], got {values.shape}"
        )
    if not np.all(np.isfinite(values)):
        raise ValueError(f"{model_name} input contains non-finite values")
    return values


def fit_ridge(
    params,
    train_x,
    train_target,
    *,
    model_name: str,
    fit_intercept: bool,
    metadata: dict[str, Any],
):
    x = np.asarray(train_x, dtype=np.float64)
    y = np.asarray(train_target, dtype=np.float64).reshape(-1)
    if x.ndim != 2:
        raise ValueError(
            f"{model_name} expects transformed [event, feature] input, got {x.shape}"
        )
    if x.shape[0] != y.size:
        raise ValueError(f"{model_name} input and target must contain the same number of events")
    if not np.all(np.isfinite(x)) or not np.all(np.isfinite(y)):
        raise ValueError(f"{model_name} input/target contains non-finite values")

    alpha = float(params["ridge_alpha"])
    if not np.isfinite(alpha) or alpha <= 0:
        raise ValueError(f"{model_name} ridge_alpha must be finite and positive")

    regressor = Ridge(
        alpha=alpha,
        fit_intercept=bool(fit_intercept),
        solver="lsqr",
        tol=1e-4,
    )
    regressor.fit(x, y)

    return LinearRidgeArtifact(
        regressor=regressor,
        metadata={
            **metadata,
            "ridge_alpha": alpha,
            "fit_intercept": bool(fit_intercept),
            "solver": "lsqr",
            "tol": 1e-4,
            "training_events": int(y.size),
            "feature_count": int(x.shape[1]),
        },
    )


def predict(artifact: LinearRidgeArtifact, features: np.ndarray) -> np.ndarray:
    x = np.asarray(features, dtype=np.float64)
    if x.ndim != 2:
        raise ValueError(f"linear Ridge expects [event, feature] input, got {x.shape}")
    return np.asarray(artifact.regressor.predict(x), dtype=np.float64).reshape(-1)


def explain(artifact: LinearRidgeArtifact, features: np.ndarray) -> np.ndarray:
    x = np.asarray(features, dtype=np.float64)
    if x.ndim != 2:
        raise ValueError(f"linear Ridge expects [event, feature] input, got {x.shape}")
    coefficients = np.asarray(artifact.regressor.coef_, dtype=np.float64).reshape(-1)
    if coefficients.size != x.shape[1]:
        raise ValueError("linear Ridge coefficient count does not match feature count")
    return np.abs(coefficients)


def save(artifact: LinearRidgeArtifact, path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    with (path / "model.pkl").open("wb") as stream:
        pickle.dump(artifact, stream, protocol=pickle.HIGHEST_PROTOCOL)
