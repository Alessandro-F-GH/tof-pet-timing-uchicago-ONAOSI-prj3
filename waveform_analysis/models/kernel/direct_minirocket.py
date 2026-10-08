from __future__ import annotations

import itertools
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from waveform_analysis.models.spec import FeatureTransformSpec, ModelSpec


@dataclass
class DirectMiniRocketTransformArtifact:
    transformer: Any
    scaler: StandardScaler
    metadata: dict[str, Any]


@dataclass
class DirectMiniRocketArtifact:
    regressor: Ridge
    metadata: dict[str, Any]


def candidates(config):
    p = config.get("parameters", {})
    kernels = [int(v) for v in p.get("num_kernels", [10000])]
    alphas = [float(v) for v in p.get("ridge_alpha", [1.0])]
    rows = []
    for num_kernels, alpha in itertools.product(kernels, alphas):
        if num_kernels < 84:
            raise ValueError("direct_minirocket num_kernels must be >= 84")
        if alpha <= 0:
            raise ValueError("direct_minirocket ridge_alpha must be positive")
        rows.append({"num_kernels": num_kernels, "ridge_alpha": alpha})
    return rows


def _transform_parameters(params, config):
    execution = config.get("transform", {})
    n_jobs = int(execution.get("n_jobs", -1))
    if n_jobs == 0:
        raise ValueError("direct_minirocket transform.n_jobs cannot be zero")
    return {"num_kernels": int(params["num_kernels"]), "n_jobs": n_jobs}


def _transformer(parameters, seed):
    try:
        from sktime.transformations.rocket import MiniRocketMultivariate
    except ImportError as exc:
        raise ImportError(
            "direct_minirocket requires the optional 'sktime' dependency. "
            "Install waveform_analysis requirements before running this model."
        ) from exc
    return MiniRocketMultivariate(
        num_kernels=int(parameters["num_kernels"]),
        n_jobs=int(parameters.get("n_jobs", -1)),
        random_state=int(seed),
    )


def _array(features) -> np.ndarray:
    if hasattr(features, "to_numpy"):
        features = features.to_numpy()
    return np.asarray(features, dtype=np.float32)


def fit_transform(parameters, train_x, *, seed, config):
    del config
    x = np.asarray(train_x, dtype=np.float32)
    if x.ndim != 3 or x.shape[1] != 2:
        raise ValueError(
            f"direct_minirocket expects [event, detector=2, time], got {x.shape}"
        )
    if x.shape[-1] < 9:
        raise ValueError("direct_minirocket requires at least 9 temporal samples")

    transformer = _transformer(parameters, seed)
    features = _array(transformer.fit_transform(x))
    scaler = StandardScaler(with_mean=False)
    scaler.fit(features)
    scaled = np.asarray(scaler.transform(features, copy=False), dtype=np.float32)
    artifact = DirectMiniRocketTransformArtifact(
        transformer=transformer,
        scaler=scaler,
        metadata={
            "input_definition": "paired two-channel waveform processed jointly by MiniRocketMultivariate",
            "num_kernels": int(parameters["num_kernels"]),
            "n_jobs": int(parameters.get("n_jobs", -1)),
            "feature_count": int(scaled.shape[1]),
            "feature_dtype": str(scaled.dtype),
            "training_events": int(scaled.shape[0]),
            "training_seed": int(seed),
        },
    )
    return artifact, scaled


def transform(
    artifact: DirectMiniRocketTransformArtifact,
    normalized_pair: np.ndarray,
) -> np.ndarray:
    x = np.asarray(normalized_pair, dtype=np.float32)
    features = _array(artifact.transformer.transform(x))
    return np.asarray(artifact.scaler.transform(features, copy=False), dtype=np.float32)


def save_transform(artifact: DirectMiniRocketTransformArtifact, path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    with (path / "transform.pkl").open("wb") as stream:
        pickle.dump(artifact, stream, protocol=pickle.HIGHEST_PROTOCOL)


def fit(params, train_x, train_target, *, seed, config):
    del seed, config
    features = np.asarray(train_x, dtype=np.float32)
    y = np.asarray(train_target, dtype=np.float64)
    if features.ndim != 2:
        raise ValueError(
            "direct_minirocket Ridge expects transformed [event, feature] input, "
            f"got {features.shape}"
        )
    if features.shape[0] != y.size:
        raise ValueError(
            "direct_minirocket transformed input and target must contain the same number of events"
        )

    regressor = Ridge(alpha=float(params["ridge_alpha"]), solver="lsqr", tol=1e-4)
    regressor.fit(features, y)
    return DirectMiniRocketArtifact(
        regressor=regressor,
        metadata={
            "input_definition": "cached MiniRocketMultivariate features from the paired normalized detector waveforms",
            "prediction_definition": "joint pair transform Phi(s1,s2) followed by Ridge regression",
            "detector_swap_antisymmetry_enforced": False,
            "num_kernels": int(params["num_kernels"]),
            "ridge_alpha": float(params["ridge_alpha"]),
            "training_events": int(y.size),
            "feature_count": int(features.shape[1]),
            "feature_dtype": str(features.dtype),
        },
    )


def predict(artifact: DirectMiniRocketArtifact, features: np.ndarray) -> np.ndarray:
    return np.asarray(
        artifact.regressor.predict(np.asarray(features, dtype=np.float32)),
        dtype=np.float64,
    )


def save(artifact: DirectMiniRocketArtifact, path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    with (path / "model.pkl").open("wb") as stream:
        pickle.dump(artifact, stream, protocol=pickle.HIGHEST_PROTOCOL)


FEATURE_TRANSFORM = FeatureTransformSpec(
    name="direct_minirocket_multivariate",
    parameters=_transform_parameters,
    fit_transform=fit_transform,
    transform=transform,
    save=save_transform,
)


MODEL_SPEC = ModelSpec(
    name="direct_minirocket",
    candidates=candidates,
    fit=fit,
    predict=predict,
    save=save,
    estimator_formulation="direct",
    feature_transform=FEATURE_TRANSFORM,
)


# Retain serialized identities while legacy imports resolve to this module.
from waveform_analysis.core.compat import (
    preserve_legacy_identity as _preserve_legacy_identity,
)

_preserve_legacy_identity(
    __name__, "waveform_analysis.ml_pipeline.models.direct_minirocket"
)
