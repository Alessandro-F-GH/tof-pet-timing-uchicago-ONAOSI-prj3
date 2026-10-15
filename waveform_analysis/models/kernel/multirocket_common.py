"""Control-fitted MultiRocket features followed by Ridge regression."""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from importlib.metadata import version
from pathlib import Path
import pickle
from typing import Any

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from waveform_analysis.models.spec import FeatureTransformSpec, ModelSpec

MULTIROCKET_MODELS = {"direct_multirocket", "shared_multirocket"}
AEON_VERSION = "1.6.0"
TRANSFORM_VERSION = 1


def _integer(value, name, minimum=1):
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < minimum:
        raise ValueError(f"MultiRocket {name} must be an integer >= {minimum}")
    return int(value)


def transform_parameters(params, config):
    result = {
        "n_kernels": _integer(params["n_kernels"], "n_kernels", 84),
        "max_dilations_per_kernel": _integer(params["max_dilations_per_kernel"], "max_dilations_per_kernel"),
        "n_features_per_kernel": _integer(params["n_features_per_kernel"], "n_features_per_kernel"),
        "normalise": params["normalise"],
    }
    if result["n_features_per_kernel"] != 4:
        raise ValueError("MultiRocket requires 4 pooling features per kernel")
    if not isinstance(result["normalise"], bool):
        raise ValueError("MultiRocket normalise must be boolean")
    execution = config.get("transform", {})
    n_jobs = execution.get("n_jobs", 2)
    if isinstance(n_jobs, bool) or not isinstance(n_jobs, (int, np.integer)) or n_jobs == 0:
        raise ValueError("MultiRocket transform.n_jobs must be a nonzero integer")
    result.update(
        n_jobs=int(n_jobs), batch_size=_integer(execution.get("batch_size", 128), "transform.batch_size"),
        aeon_version=AEON_VERSION, transform_version=TRANSFORM_VERSION,
    )
    return result


@contextmanager
def _runtime(n_jobs, transformer=None):
    """Contain aeon's NumPy seeding and restore Numba threads on failure too."""
    from numba import config as numba_config, get_num_threads, set_num_threads
    from aeon.utils.validation import check_n_jobs

    state, threads = np.random.get_state(), get_num_threads()
    effective = min(check_n_jobs(n_jobs), numba_config.NUMBA_NUM_THREADS)
    old_jobs = transformer._n_jobs if transformer is not None else None
    try:
        set_num_threads(effective)
        if transformer is not None:
            transformer._n_jobs = effective
        yield effective
    finally:
        if transformer is not None:
            transformer._n_jobs = old_jobs
        set_num_threads(threads)
        np.random.set_state(state)


@dataclass
class MultiRocketTransformArtifact:
    transformer: Any
    scaler: StandardScaler
    shared: bool
    metadata: dict[str, Any]
    time_ps: np.ndarray | None = None
    sample_indices: np.ndarray | None = None


@dataclass
class MultiRocketArtifact:
    regressor: Ridge
    metadata: dict[str, Any]


def _pair(values):
    pair = np.asarray(values, dtype=np.float32)
    if pair.ndim != 3 or pair.shape[1] != 2 or len(pair) < 1 or not np.all(np.isfinite(pair)):
        raise ValueError("MultiRocket expects finite [event, detector=2, time] input")
    # The differenced representation also needs a full 9-sample kernel grid.
    if pair.shape[-1] < 10:
        raise ValueError("MultiRocket requires at least 10 retained samples for raw and differenced inputs")
    return pair


def _batches(transformer, values, batch_size):
    for start in range(0, len(values), batch_size):
        stop = min(start + batch_size, len(values))
        raw = np.asarray(transformer.transform(values[start:stop]), dtype=np.float32)
        if raw.ndim != 2 or raw.shape[0] != stop - start or not np.all(np.isfinite(raw)):
            raise RuntimeError("MultiRocket produced invalid tabular features")
        yield start, stop, raw


def _scaled(transformer, scaler, values, batch_size):
    matrix = None
    for start, stop, raw in _batches(transformer, values, batch_size):
        if matrix is None:
            matrix = np.empty((len(values), raw.shape[1]), dtype=np.float32)
        matrix[start:stop] = scaler.transform(raw, copy=False)
    return matrix


def fit_transform(parameters, train_x, *, seed, config, shared):
    try:
        from aeon.transformations.collection.convolution_based import MultiRocket
    except ImportError as exc:
        raise ImportError("MultiRocket requires aeon; install waveform_analysis/requirements.txt") from exc
    actual_version = version("aeon")
    if actual_version != AEON_VERSION:
        raise ImportError(f"MultiRocket requires aeon=={AEON_VERSION}; found {actual_version}")
    pair = _pair(train_x)
    values = np.concatenate([pair[:, :1], pair[:, 1:]], axis=0) if shared else pair
    with _runtime(parameters["n_jobs"]) as effective_jobs:
        transformer = MultiRocket(
            **{key: parameters[key] for key in ("n_kernels", "max_dilations_per_kernel", "n_features_per_kernel", "normalise")},
            n_jobs=effective_jobs, random_state=int(seed),
        )
        transformer.fit(values)  # Control only, without timing targets.
        scaler = StandardScaler(with_mean=False)
        feature_count = None
        for _, _, raw in _batches(transformer, values, parameters["batch_size"]):
            scaler.partial_fit(raw)
            feature_count = raw.shape[1]
        # The frozen-cache caller discards control features. Avoid allocating a
        # potentially multi-GB control matrix solely to satisfy that callback.
        if config.get("_frozen_transform_only", False):
            features = np.empty((0, feature_count), dtype=np.float32)
        else:
            scaled = _scaled(transformer, scaler, values, parameters["batch_size"])
            features = scaled[:len(pair)] - scaled[len(pair):] if shared else scaled
    artifact = MultiRocketTransformArtifact(transformer, scaler, shared, {
        "aeon_version": AEON_VERSION, "transform_version": TRANSFORM_VERSION,
        "training_seed": int(seed), "training_events": len(pair), "transform_fit_waveforms": len(values),
        "input_samples": pair.shape[-1], "feature_count": feature_count, "feature_dtype": "float32",
        "feature_definition": "phi(s1)-phi(s2)" if shared else "phi(s1,s2)",
        "scaling": "control_standard_deviation", "n_jobs": parameters["n_jobs"],
        "batch_size": parameters["batch_size"], "normalise": parameters["normalise"],
    })
    return artifact, np.asarray(features, dtype=np.float32)


def transform(artifact, values):
    pair = _pair(values)
    if pair.shape[-1] != artifact.metadata["input_samples"]:
        raise ValueError("MultiRocket input length differs from its frozen control grid")
    with _runtime(artifact.metadata["n_jobs"], artifact.transformer):
        if artifact.shared:
            first = _scaled(artifact.transformer, artifact.scaler, pair[:, :1], artifact.metadata["batch_size"])
            # Fill detector differences directly, avoiding two full feature matrices.
            for start, stop, raw in _batches(artifact.transformer, pair[:, 1:], artifact.metadata["batch_size"]):
                first[start:stop] -= artifact.scaler.transform(raw, copy=False)
            return first
        return _scaled(artifact.transformer, artifact.scaler, pair, artifact.metadata["batch_size"])


def save_transform(artifact, path):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    with (path / "transform.pkl").open("wb") as stream:
        pickle.dump(artifact, stream, protocol=pickle.HIGHEST_PROTOCOL)
    # Bias/dilation/channel parameters for both waveform representations.
    arrays = {f"{name}_parameter_{i}": np.asarray(value) for name, parameters in
              (("raw", artifact.transformer.parameter), ("difference", artifact.transformer.parameter1))
              for i, value in enumerate(parameters)}
    arrays.update(
        scaler_scale=artifact.scaler.scale_, scaler_mean=artifact.scaler.mean_, scaler_variance=artifact.scaler.var_,
        shared=np.asarray(artifact.shared), aeon_version=np.asarray(AEON_VERSION),
        transform_version=np.asarray(TRANSFORM_VERSION), training_seed=np.asarray(artifact.metadata["training_seed"]),
        feature_count=np.asarray(artifact.metadata["feature_count"]),
        normalise=np.asarray(artifact.metadata["normalise"]),
        transform_identity=np.asarray(artifact.metadata.get("transform_identity", "")),
        control_analysis_protocol_identity=np.asarray(artifact.metadata.get("control_analysis_protocol_identity", "")),
    )
    if artifact.time_ps is not None:
        arrays.update(time_ps=artifact.time_ps, sample_indices=artifact.sample_indices)
    np.savez_compressed(path / "multirocket.npz", **arrays)


def fit(params, train_x, train_target, *, seed, config, shared):
    del seed
    x, y = np.asarray(train_x, dtype=np.float32), np.asarray(train_target, dtype=np.float64)
    alpha = float(params["ridge_alpha"])
    if not np.isfinite(alpha) or alpha <= 0:
        raise ValueError("MultiRocket ridge_alpha must be finite and positive")
    if x.ndim != 2 or y.ndim != 1 or len(x) != len(y) or x.shape[1] < 1:
        raise ValueError("MultiRocket Ridge requires [event, feature] input and one target per event")
    # The engine supplies an owned training-row copy. Center that buffer in
    # place instead of allocating another full feature matrix inside Ridge.
    # Standalone callers retain the default non-mutating sklearn behavior.
    regressor = Ridge(
        alpha=alpha, fit_intercept=not shared, solver="lsqr", tol=1e-4,
        copy_X=not config.get("_fit_input_owned", False),
    ).fit(x, y)
    return MultiRocketArtifact(regressor, {
        "ridge_alpha": alpha, "training_events": len(y), "feature_count": x.shape[1],
        "fit_intercept": not shared, "detector_swap_antisymmetry_enforced": shared,
    })


def predict(artifact, features):
    return np.asarray(artifact.regressor.predict(np.asarray(features, np.float32)), dtype=np.float64)


def save(artifact, path):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    with (path / "model.pkl").open("wb") as stream:
        pickle.dump(artifact, stream, protocol=pickle.HIGHEST_PROTOCOL)


def candidates(config):
    from waveform_analysis.engine.search import fixed_parameters, grid_candidates, optimization_config
    return grid_candidates(config) if optimization_config(config).strategy == "grid" else [fixed_parameters(config)]


def make_spec(name, fit_transform_callback, fit_callback):
    return ModelSpec(
        name=name, candidates=candidates, fit=fit_callback, predict=predict, save=save,
        artifact_format="pickle", requires_frozen_features=True,
        bounded_transform_batches=True, inplace_fit_input=True,
        estimator_formulation="shared" if name == "shared_multirocket" else "direct",
        feature_transform=FeatureTransformSpec(
            name=f"{name}_aeon_{AEON_VERSION}_v{TRANSFORM_VERSION}", parameters=transform_parameters,
            fit_transform=fit_transform_callback, transform=transform, save=save_transform,
        ),
    )
