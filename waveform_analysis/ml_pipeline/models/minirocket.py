from __future__ import annotations

import itertools
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.linear_model import Ridge

from .spec import ModelSpec


@dataclass
class MiniRocketArtifact:
    transformer: Any
    regressor: Ridge
    metadata: dict[str, Any]


def candidates(config):
    p = config.get("parameters", {})
    kernels = [int(v) for v in p.get("num_kernels", [10000])]
    alphas = [float(v) for v in p.get("ridge_alpha", [1.0])]
    jobs = [int(v) for v in p.get("n_jobs", [1])]
    rows = []
    for num_kernels, alpha, n_jobs in itertools.product(kernels, alphas, jobs):
        if num_kernels < 84:
            raise ValueError("MiniRocket num_kernels must be >= 84")
        if alpha <= 0:
            raise ValueError("MiniRocket ridge_alpha must be positive")
        if n_jobs == 0:
            raise ValueError("MiniRocket n_jobs cannot be zero")
        rows.append({
            "num_kernels": num_kernels,
            "ridge_alpha": alpha,
            "n_jobs": n_jobs,
        })
    return rows


def _transformer(params, seed):
    try:
        from sktime.transformations.panel.rocket import MiniRocketMultivariate
    except ImportError as exc:
        raise ImportError(
            "minirocket requires the optional 'sktime' dependency. "
            "Install waveform_analysis requirements before running this model."
        ) from exc
    return MiniRocketMultivariate(
        num_kernels=int(params["num_kernels"]),
        n_jobs=int(params.get("n_jobs", 1)),
        random_state=int(seed),
    )


def _array(features) -> np.ndarray:
    if hasattr(features, "to_numpy"):
        features = features.to_numpy()
    return np.asarray(features, dtype=np.float64)


def fit(params, train_x, train_target, *, seed, config):
    x = np.asarray(train_x, dtype=np.float32)
    y = np.asarray(train_target, dtype=np.float64)
    if x.ndim != 3 or x.shape[1] != 2:
        raise ValueError(f"minirocket expects [event, detector=2, time], got {x.shape}")
    if x.shape[0] != y.size:
        raise ValueError("minirocket train_x and train_target must contain the same number of events")
    transformer = _transformer(params, seed)
    features = _array(transformer.fit_transform(x))
    regressor = Ridge(alpha=float(params["ridge_alpha"]))
    regressor.fit(features, y)
    return MiniRocketArtifact(
        transformer=transformer,
        regressor=regressor,
        metadata={
            "input_definition": "paired normalized detector waveforms as one multivariate two-channel time series",
            "prediction_definition": "MiniRocket multivariate features followed by Ridge regression",
            "detector_swap_antisymmetry_enforced": False,
            "num_kernels": int(params["num_kernels"]),
            "ridge_alpha": float(params["ridge_alpha"]),
            "n_jobs": int(params.get("n_jobs", 1)),
            "training_events": int(y.size),
            "feature_count": int(features.shape[1]),
            "training_seed": int(seed),
        },
    )


def predict(artifact: MiniRocketArtifact, normalized_pair: np.ndarray) -> np.ndarray:
    x = np.asarray(normalized_pair, dtype=np.float32)
    features = _array(artifact.transformer.transform(x))
    return np.asarray(artifact.regressor.predict(features), dtype=np.float64)


def save(artifact: MiniRocketArtifact, path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    with (path / "model.pkl").open("wb") as stream:
        pickle.dump(artifact, stream, protocol=pickle.HIGHEST_PROTOCOL)


MODEL_SPEC = ModelSpec(
    name="minirocket",
    candidates=candidates,
    fit=fit,
    predict=predict,
    save=save,
    preserve_temporal_grid=True,
    estimator_formulation="direct",
)
