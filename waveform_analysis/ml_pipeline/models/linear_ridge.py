from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.linear_model import Ridge

from .spec import FeatureTransformSpec, ModelSpec


@dataclass
class DifferenceTransformArtifact:
    metadata: dict[str, Any]


@dataclass
class LinearRidgeArtifact:
    regressor: Ridge
    metadata: dict[str, Any]


def _finite_grid_values(raw, default):
    if raw is None:
        return list(default)
    if isinstance(raw, dict):
        kind = str(raw.get("type", "")).strip().lower()
        if kind == "fixed":
            return [raw["value"]]
        if kind == "categorical":
            return list(raw["choices"])
        raise ValueError("linear_ridge candidates require a finite fixed/categorical alpha grid")
    if isinstance(raw, (list, tuple, np.ndarray)):
        return list(raw)
    return [raw]


def candidates(config):
    parameters = config.get("parameters", {})
    alphas = [
        float(value)
        for value in _finite_grid_values(
            parameters.get("ridge_alpha"),
            [0.01, 0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0, 100000.0],
        )
    ]
    if any((not np.isfinite(alpha)) or alpha <= 0 for alpha in alphas):
        raise ValueError("linear_ridge ridge_alpha values must be finite and positive")
    return [{"ridge_alpha": alpha} for alpha in alphas]


def _difference(pair: np.ndarray) -> np.ndarray:
    values = np.asarray(pair, dtype=np.float64)
    if values.ndim != 3 or values.shape[1] != 2:
        raise ValueError(
            f"linear_ridge expects [event, detector=2, time], got {values.shape}"
        )
    difference = values[:, 0, :] - values[:, 1, :]
    if not np.all(np.isfinite(difference)):
        raise ValueError("linear_ridge input contains non-finite values")
    return np.ascontiguousarray(difference)


def _difference_parameters(params, config):
    del params, config
    return {}


def fit_difference_transform(parameters, train_x, *, seed, config):
    del parameters, seed, config
    difference = _difference(train_x)
    artifact = DifferenceTransformArtifact(
        metadata={
            "definition": "deterministic sample-wise detector difference s1-s2",
            "feature_count": int(difference.shape[1]),
            "training_events": int(difference.shape[0]),
        }
    )
    return artifact, difference


def transform_difference(artifact: DifferenceTransformArtifact, pair: np.ndarray) -> np.ndarray:
    del artifact
    return _difference(pair)


def fit(params, train_x, train_target, *, seed, config):
    del seed
    x = np.asarray(train_x, dtype=np.float64)
    y = np.asarray(train_target, dtype=np.float64).reshape(-1)
    if x.ndim != 2:
        raise ValueError(
            f"linear_ridge expects cached [event, time] difference features, got {x.shape}"
        )
    if x.shape[0] != y.size:
        raise ValueError("linear_ridge input and target must contain the same number of events")
    if not np.all(np.isfinite(x)) or not np.all(np.isfinite(y)):
        raise ValueError("linear_ridge input/target contains non-finite values")

    alpha = float(params["ridge_alpha"])
    if not np.isfinite(alpha) or alpha <= 0:
        raise ValueError("linear_ridge ridge_alpha must be finite and positive")

    training = config.get("training", {})
    solver = str(training.get("solver", "lsqr")).strip().lower()
    tol = float(training.get("tol", 1e-6))
    max_iter_raw = training.get("max_iter")
    max_iter = None if max_iter_raw is None else int(max_iter_raw)
    if tol <= 0 or not np.isfinite(tol):
        raise ValueError("linear_ridge training.tol must be finite and positive")
    if max_iter is not None and max_iter < 1:
        raise ValueError("linear_ridge training.max_iter must be >= 1 when provided")

    # No intercept: swapping the two detectors negates the difference features,
    # so this is exactly the shared linear scorer g(s1)-g(s2), g(s)=w^T s.
    regressor = Ridge(
        alpha=alpha,
        fit_intercept=False,
        solver=solver,
        tol=tol,
        max_iter=max_iter,
    )
    regressor.fit(x, y)

    return LinearRidgeArtifact(
        regressor=regressor,
        metadata={
            "input_definition": "cached sample-wise difference of normalized detector waveforms: s1-s2",
            "prediction_definition": "ridge linear correction w^T(s1-s2) = g(s1)-g(s2) [ps]",
            "detector_swap_antisymmetry_enforced": True,
            "equivalent_formulation": "shared linear scorer g(s1)-g(s2), with g(s)=w^T s",
            "ridge_alpha": alpha,
            "fit_intercept": False,
            "solver": solver,
            "tol": tol,
            "max_iter": max_iter,
            "training_events": int(y.size),
            "feature_count": int(x.shape[1]),
        },
    )


def predict(artifact: LinearRidgeArtifact, features: np.ndarray) -> np.ndarray:
    x = np.asarray(features, dtype=np.float64)
    if x.ndim != 2:
        raise ValueError(f"linear_ridge expects [event, time] features, got {x.shape}")
    return np.asarray(artifact.regressor.predict(x), dtype=np.float64).reshape(-1)


def explain(artifact: LinearRidgeArtifact, features: np.ndarray) -> np.ndarray:
    x = np.asarray(features, dtype=np.float64)
    if x.ndim != 2:
        raise ValueError(f"linear_ridge expects [event, time] features, got {x.shape}")
    coefficients = np.asarray(artifact.regressor.coef_, dtype=np.float64).reshape(-1)
    return np.abs(coefficients)


def save(artifact: LinearRidgeArtifact, path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    with (path / "model.pkl").open("wb") as stream:
        pickle.dump(artifact, stream, protocol=pickle.HIGHEST_PROTOCOL)


DIFFERENCE_TRANSFORM = FeatureTransformSpec(
    name="paired_waveform_difference",
    parameters=_difference_parameters,
    fit_transform=fit_difference_transform,
    transform=transform_difference,
)


MODEL_SPEC = ModelSpec(
    name="linear_ridge",
    candidates=candidates,
    fit=fit,
    predict=predict,
    save=save,
    explain=explain,
    preserve_temporal_grid=True,
    estimator_formulation="shared",
    feature_transform=DIFFERENCE_TRANSFORM,
)
