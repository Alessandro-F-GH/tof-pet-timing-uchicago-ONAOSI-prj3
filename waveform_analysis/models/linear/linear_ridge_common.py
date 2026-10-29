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


def candidates(config: dict[str, Any]) -> list[dict[str, float]]:
    """Expose the configured alpha grid through the standard model interface."""
    parameter = (config.get("parameters") or {}).get("ridge_alpha")
    if not isinstance(parameter, dict):
        raise ValueError("Ridge model space must define parameters.ridge_alpha")
    kind = parameter.get("type")
    if kind == "categorical":
        values = parameter.get("choices")
    elif kind == "fixed":
        values = [parameter.get("value")]
    else:
        raise ValueError("Grid-based Ridge requires fixed or categorical ridge_alpha")
    if not isinstance(values, list) or not values:
        raise ValueError("ridge_alpha must resolve to a non-empty grid")
    alphas = np.asarray(values, dtype=np.float64)
    if alphas.ndim != 1 or np.any(~np.isfinite(alphas)) or np.any(alphas <= 0):
        raise ValueError("ridge_alpha values must be finite and positive")
    return [{"ridge_alpha": float(alpha)} for alpha in alphas]


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
    config: dict[str, Any],
):
    x = np.asarray(train_x, dtype=np.float64)
    y = np.asarray(train_target, dtype=np.float64).reshape(-1)
    if x.ndim != 2:
        raise ValueError(
            f"{model_name} expects transformed [event, feature] input, got {x.shape}"
        )
    if x.shape[0] != y.size:
        raise ValueError(
            f"{model_name} input and target must contain the same number of events"
        )
    if not np.all(np.isfinite(x)) or not np.all(np.isfinite(y)):
        raise ValueError(f"{model_name} input/target contains non-finite values")

    del config
    if y.size < 1:
        raise ValueError(f"{model_name} Ridge requires at least one training event")
    if not isinstance(params, dict) or "ridge_alpha" not in params:
        raise ValueError(f"{model_name} requires a selected ridge_alpha parameter")
    alpha = float(params["ridge_alpha"])
    if not np.isfinite(alpha) or alpha <= 0:
        raise ValueError(f"{model_name} ridge_alpha must be finite and positive")
    regressor = Ridge(alpha=alpha, fit_intercept=bool(fit_intercept))
    regressor.fit(x, y)

    return LinearRidgeArtifact(
        regressor=regressor,
        metadata={
            **metadata,
            "ridge_alpha": alpha,
            "fit_intercept": bool(fit_intercept),
            "selection_method": "development_cv",
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
