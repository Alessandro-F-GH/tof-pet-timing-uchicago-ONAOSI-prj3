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
class SharedMiniRocketTransformArtifact:
    transformer: Any
    scaler: StandardScaler
    metadata: dict[str, Any]


@dataclass
class SharedMiniRocketArtifact:
    regressor: Ridge
    metadata: dict[str, Any]


def candidates(config):
    p = config.get("parameters", {})
    kernels = [int(v) for v in p.get("num_kernels", [10000])]
    alphas = [float(v) for v in p.get("ridge_alpha", [1.0])]
    rows = []
    for num_kernels, alpha in itertools.product(kernels, alphas):
        if num_kernels < 84:
            raise ValueError("shared_minirocket num_kernels must be >= 84")
        if alpha <= 0:
            raise ValueError("shared_minirocket ridge_alpha must be positive")
        rows.append({"num_kernels": num_kernels, "ridge_alpha": alpha})
    return rows


def _transform_parameters(params, config):
    execution = config.get("transform", {})
    n_jobs = int(execution.get("n_jobs", -1))
    if n_jobs == 0:
        raise ValueError("shared_minirocket transform.n_jobs cannot be zero")
    return {"num_kernels": int(params["num_kernels"]), "n_jobs": n_jobs}


def _transformer(parameters, seed):
    try:
        from sktime.transformations.rocket import MiniRocket
    except ImportError as exc:
        raise ImportError(
            "shared_minirocket requires the optional 'sktime' dependency. "
            "Install waveform_analysis requirements before running this model."
        ) from exc
    return MiniRocket(
        num_kernels=int(parameters["num_kernels"]),
        n_jobs=int(parameters.get("n_jobs", -1)),
        random_state=int(seed),
    )


def _array(features) -> np.ndarray:
    if hasattr(features, "to_numpy"):
        features = features.to_numpy()
    return np.asarray(features, dtype=np.float32)


def _single_channel_input(values: np.ndarray) -> np.ndarray:
    x = np.asarray(values, dtype=np.float32)
    if x.ndim != 2:
        raise ValueError(
            f"shared_minirocket single-channel input must be [event,time], got {x.shape}"
        )
    return x[:, None, :]


def _pair(values: np.ndarray) -> np.ndarray:
    x = np.asarray(values, dtype=np.float32)
    if x.ndim != 3 or x.shape[1] != 2:
        raise ValueError(
            f"shared_minirocket expects [event, detector=2, time], got {x.shape}"
        )
    return x


def fit_transform(parameters, train_x, *, seed, config):
    del config
    pair = _pair(train_x)
    if pair.shape[-1] < 9:
        raise ValueError("shared_minirocket requires at least 9 temporal samples")

    pooled = np.concatenate([pair[:, 0, :], pair[:, 1, :]], axis=0)
    transformer = _transformer(parameters, seed)
    pooled_features = _array(transformer.fit_transform(_single_channel_input(pooled)))

    scaler = StandardScaler(with_mean=False)
    scaler.fit(pooled_features)

    n = pair.shape[0]
    z1 = np.asarray(scaler.transform(pooled_features[:n], copy=False), dtype=np.float32)
    z2 = np.asarray(scaler.transform(pooled_features[n:], copy=False), dtype=np.float32)
    difference = np.asarray(z1 - z2, dtype=np.float32)

    artifact = SharedMiniRocketTransformArtifact(
        transformer=transformer,
        scaler=scaler,
        metadata={
            "input_definition": "same fitted univariate MiniRocket applied independently to both detector waveforms",
            "feature_definition": "phi(s1)-phi(s2)",
            "num_kernels": int(parameters["num_kernels"]),
            "n_jobs": int(parameters.get("n_jobs", -1)),
            "feature_count": int(difference.shape[1]),
            "feature_dtype": str(difference.dtype),
            "training_events": int(n),
            "pooled_transform_fit_waveforms": int(pooled.shape[0]),
            "training_seed": int(seed),
        },
    )
    return artifact, difference


def transform(
    artifact: SharedMiniRocketTransformArtifact,
    normalized_pair: np.ndarray,
) -> np.ndarray:
    pair = _pair(normalized_pair)
    z1 = _array(artifact.transformer.transform(_single_channel_input(pair[:, 0, :])))
    z2 = _array(artifact.transformer.transform(_single_channel_input(pair[:, 1, :])))
    z1 = np.asarray(artifact.scaler.transform(z1, copy=False), dtype=np.float32)
    z2 = np.asarray(artifact.scaler.transform(z2, copy=False), dtype=np.float32)
    return np.asarray(z1 - z2, dtype=np.float32)


def save_transform(artifact: SharedMiniRocketTransformArtifact, path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    with (path / "transform.pkl").open("wb") as stream:
        pickle.dump(artifact, stream, protocol=pickle.HIGHEST_PROTOCOL)


def fit(params, train_x, train_target, *, seed, config):
    del seed, config
    features = np.asarray(train_x, dtype=np.float32)
    y = np.asarray(train_target, dtype=np.float64)
    if features.ndim != 2:
        raise ValueError(
            "shared_minirocket Ridge expects transformed [event, feature] input, "
            f"got {features.shape}"
        )
    if features.shape[0] != y.size:
        raise ValueError(
            "shared_minirocket transformed input and target must contain the same number of events"
        )

    regressor = Ridge(
        alpha=float(params["ridge_alpha"]),
        fit_intercept=False,
        solver="lsqr",
        tol=1e-4,
    )
    regressor.fit(features, y)
    return SharedMiniRocketArtifact(
        regressor=regressor,
        metadata={
            "input_definition": "cached shared MiniRocket feature difference phi(s1)-phi(s2)",
            "prediction_definition": "w^T[phi(s1)-phi(s2)] = g(s1)-g(s2)",
            "detector_swap_antisymmetry_enforced": True,
            "num_kernels": int(params["num_kernels"]),
            "ridge_alpha": float(params["ridge_alpha"]),
            "fit_intercept": False,
            "training_events": int(y.size),
            "feature_count": int(features.shape[1]),
            "feature_dtype": str(features.dtype),
        },
    )


def predict(artifact: SharedMiniRocketArtifact, features: np.ndarray) -> np.ndarray:
    return np.asarray(
        artifact.regressor.predict(np.asarray(features, dtype=np.float32)),
        dtype=np.float64,
    )


def save(artifact: SharedMiniRocketArtifact, path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    with (path / "model.pkl").open("wb") as stream:
        pickle.dump(artifact, stream, protocol=pickle.HIGHEST_PROTOCOL)


FEATURE_TRANSFORM = FeatureTransformSpec(
    name="shared_minirocket_univariate",
    parameters=_transform_parameters,
    fit_transform=fit_transform,
    transform=transform,
    save=save_transform,
)


MODEL_SPEC = ModelSpec(
    name="shared_minirocket",
    candidates=candidates,
    fit=fit,
    predict=predict,
    save=save,
    estimator_formulation="shared",
    feature_transform=FEATURE_TRANSFORM,
)


# Retain serialized identities while legacy imports resolve to this module.
from waveform_analysis.core.compat import (
    preserve_legacy_identity as _preserve_legacy_identity,
)

_preserve_legacy_identity(
    __name__, "waveform_analysis.ml_pipeline.models.shared_minirocket"
)
