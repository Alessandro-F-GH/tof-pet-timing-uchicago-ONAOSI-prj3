from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.linear_model import RidgeCV
from waveform_analysis.core.ridge import ridge_cv_config


@dataclass
class LinearTransformArtifact:
    metadata: dict[str, Any]


@dataclass
class LinearRidgeArtifact:
    regressor: RidgeCV
    metadata: dict[str, Any]


def candidates(config: dict[str, Any]) -> list[dict[str, float]]:
    """Inspect the effective lambda grid; RidgeCV selects it internally.

    This callback satisfies ModelSpec's shared interface and never launches
    an outer candidate search for the two linear estimators.
    """
    return [{"ridge_alpha": alpha} for alpha in ridge_cv_config(config).alphas]


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

    del params  # Lambda is selected internally, never by manual/Optuna candidates.
    settings = ridge_cv_config(config)
    if y.size < 2:
        raise ValueError(
            f"{model_name} RidgeCV requires at least two development events"
        )
    regressor = RidgeCV(
        alphas=settings.alphas,
        fit_intercept=bool(fit_intercept),
        cv=settings.cv,
        scoring=settings.scoring,
        gcv_mode=settings.gcv_mode,
    )
    regressor.fit(x, y)

    return LinearRidgeArtifact(
        regressor=regressor,
        metadata={
            **metadata,
            "ridge_alpha": float(regressor.alpha_),
            "fit_intercept": bool(fit_intercept),
            "selection_method": "ridge_cv",
            "ridge_cv": settings.as_dict(),
            "internal_cv_mse_ps2": -float(regressor.best_score_),
            "internal_cv": "leave_one_out" if settings.cv is None else "k_fold",
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


# Retain serialized identities while legacy imports resolve to this module.
from waveform_analysis.core.compat import (
    preserve_legacy_identity as _preserve_legacy_identity,
)

_preserve_legacy_identity(
    __name__, "waveform_analysis.ml_pipeline.models._linear_ridge_common"
)
