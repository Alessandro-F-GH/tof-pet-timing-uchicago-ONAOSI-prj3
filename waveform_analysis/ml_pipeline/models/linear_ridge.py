from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.linear_model import Ridge

from .spec import ModelSpec


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


def fit(params, train_x, train_target, *, seed, config):
    del seed  # deterministic closed-form/iterative linear fit
    x = _difference(train_x)
    y = np.asarray(train_target, dtype=np.float64).reshape(-1)
    if x.shape[0] != y.size:
        raise ValueError("linear_ridge input and target must contain the same number of events")
    if not np.all(np.isfinite(y)):
        raise ValueError("linear_ridge target contains non-finite values")

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

    # No intercept: f(s1-s2) is then exactly odd under detector exchange.
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
            "input_definition": "sample-wise difference of the two normalized detector waveforms: s1 - s2",
            "prediction_definition": "ridge linear correction w^T(s1-s2) [ps]",
            "detector_swap_antisymmetry_enforced": True,
            "equivalent_formulation": "shared linear scorer g(s1)-g(s2), with g(s)=w^T s",
            "ridge_alpha": alpha,
            "fit_intercept": False,
            "solver": solver,
            "tol": tol,
            "max_iter": max_iter,
            "training_events": int(y.size),
            "input_samples": int(x.shape[1]),
        },
    )


def predict(artifact: LinearRidgeArtifact, pair: np.ndarray) -> np.ndarray:
    prediction = artifact.regressor.predict(_difference(pair))
    return np.asarray(prediction, dtype=np.float64).reshape(-1)


def explain(artifact: LinearRidgeArtifact, pair: np.ndarray) -> np.ndarray:
    # The coefficient magnitude is the exact global linear sensitivity per retained sample.
    _difference(pair)  # validate the same input contract used for prediction
    coefficients = np.asarray(artifact.regressor.coef_, dtype=np.float64).reshape(-1)
    return np.abs(coefficients)


def save(artifact: LinearRidgeArtifact, path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    with (path / "model.pkl").open("wb") as stream:
        pickle.dump(artifact, stream, protocol=pickle.HIGHEST_PROTOCOL)


MODEL_SPEC = ModelSpec(
    name="linear_ridge",
    candidates=candidates,
    fit=fit,
    predict=predict,
    save=save,
    explain=explain,
    # Although implemented from the pair difference, this is mathematically
    # g(s1)-g(s2) with one shared linear scorer g. Keep it in the shared family
    # so shared/direct reporting reflects the actual hypothesis class.
    estimator_formulation="shared",
)
