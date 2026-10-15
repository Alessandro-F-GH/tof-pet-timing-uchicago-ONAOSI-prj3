"""Frozen Hydra features with control-fitted sparse scaling and Ridge regression."""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from importlib.metadata import version
from pathlib import Path
import pickle
from typing import Any

import numpy as np
from sklearn.linear_model import Ridge

from waveform_analysis.models.spec import FeatureTransformSpec, ModelSpec

HYDRA_MODELS = {"direct_hydra", "shared_hydra"}
AEON_VERSION = "1.6.0"
HYDRA_TRANSFORM_VERSION = 1


def _positive_integer(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < 1:
        raise ValueError(f"Hydra {name} must be a positive integer")
    return int(value)


def transform_parameters(params, config):
    # The frozen cache supplies only fixed parameters; a missing value here
    # rejects transform tuning rather than fitting transforms on CV folds.
    result = {
        key: _positive_integer(params[key], key)
        for key in ("n_kernels", "n_groups", "max_num_channels")
    }
    if result["n_groups"] > 1 and result["n_groups"] % 2:
        raise ValueError("Hydra n_groups must be 1 or even; aeon truncates odd groups")
    execution = config.get("transform", {})
    n_jobs = execution.get("n_jobs", 2)
    if isinstance(n_jobs, bool) or not isinstance(n_jobs, (int, np.integer)) or n_jobs == 0:
        raise ValueError("Hydra transform.n_jobs must be a nonzero integer")
    result.update(
        n_jobs=int(n_jobs),
        batch_size=_positive_integer(execution.get("batch_size", 128), "transform.batch_size"),
        aeon_version=AEON_VERSION,
        transform_version=HYDRA_TRANSFORM_VERSION,
    )
    return result


@contextmanager
def _torch_runtime(n_jobs):
    """Contain Hydra's global seeding/thread changes, including failed calls."""
    import torch
    from aeon.utils.validation import check_n_jobs

    previous_threads = torch.get_num_threads()
    devices = list(range(torch.cuda.device_count())) if torch.cuda.is_initialized() else []
    with torch.random.fork_rng(devices=devices), torch.no_grad():
        try:
            torch.set_num_threads(check_n_jobs(n_jobs))
            yield
        finally:
            torch.set_num_threads(previous_threads)


def _tensor(features):
    import torch

    values = torch.as_tensor(np.asarray(features, dtype=np.float32))
    if values.ndim != 2 or not torch.isfinite(values).all():
        raise ValueError("Hydra sparse scaling requires finite [event, feature] input")
    return values


@dataclass
class HydraSparseScaler:
    """aeon 1.6.0 reference scaling, including its sample SD and zero mask.

    Store NumPy arrays for portable serialization. Computation remains float32
    PyTorch to match the reference rather than changing reduction conventions.
    """

    mu: np.ndarray | None = None
    sigma: np.ndarray | None = None
    epsilon: np.ndarray | None = None

    def fit(self, features):
        x = _tensor(features).clamp(0).sqrt()
        if x.shape[0] < 2:
            raise ValueError("Hydra sparse scaling requires at least two control waveforms")
        epsilon = (x == 0).float().mean(0) ** 4 + 1e-8
        self.mu = x.mean(0).numpy().copy()
        self.sigma = (x.std(0) + epsilon).numpy().copy()
        self.epsilon = epsilon.numpy().copy()
        return self

    def transform(self, features):
        import torch

        if self.mu is None or self.sigma is None:
            raise ValueError("Hydra sparse scaler has not been fitted")
        x = _tensor(features).clamp(0).sqrt()
        if x.shape[1] != len(self.mu):
            raise ValueError("Hydra feature count differs from the frozen scaler")
        scaled = ((x - torch.as_tensor(self.mu)) * (x != 0)) / torch.as_tensor(self.sigma)
        if not torch.isfinite(scaled).all():
            raise ValueError("Hydra sparse scaling produced non-finite features")
        return scaled.numpy()


@dataclass
class HydraTransformArtifact:
    transformer: Any
    scaler: HydraSparseScaler
    shared: bool
    metadata: dict[str, Any]
    time_ps: np.ndarray | None = None
    sample_indices: np.ndarray | None = None


@dataclass
class HydraArtifact:
    regressor: Ridge
    metadata: dict[str, Any]


def _pair(values):
    x = np.asarray(values, dtype=np.float32)
    if x.ndim != 3 or x.shape[1] != 2 or x.shape[0] < 1 or not np.all(np.isfinite(x)):
        raise ValueError("Hydra expects finite [event, detector=2, time] input")
    if x.shape[-1] < 9:
        raise ValueError("Hydra requires at least 9 retained temporal samples")
    return x


def _raw_features(transformer, values, batch_size):
    """Bound convolution intermediates; keep only the final tabular matrix."""
    matrix = None
    for start in range(0, len(values), batch_size):
        stop = min(start + batch_size, len(values))
        raw = transformer.transform(values[start:stop]).detach().numpy()
        if raw.ndim != 2 or raw.shape[0] != stop - start or not np.all(np.isfinite(raw)):
            raise RuntimeError("Hydra returned invalid features")
        if matrix is None:
            matrix = np.empty((len(values), raw.shape[1]), dtype=np.float32)
        matrix[start:stop] = raw
    return matrix


def fit_transform(parameters, train_x, *, seed, config, shared):
    del config
    try:
        from aeon.transformations.collection.convolution_based import HydraTransformer
    except ImportError as exc:
        raise ImportError("Hydra requires aeon and torch; install waveform_analysis/requirements.txt") from exc
    actual_version = version("aeon")
    if actual_version != AEON_VERSION:
        raise ImportError(f"Hydra requires aeon=={AEON_VERSION}; found {actual_version}")
    pair = _pair(train_x)
    x = np.concatenate([pair[:, :1], pair[:, 1:]], axis=0) if shared else pair
    batch_size = int(parameters["batch_size"])
    with _torch_runtime(parameters["n_jobs"]):
        transformer = HydraTransformer(
            **{key: parameters[key] for key in ("n_kernels", "n_groups", "max_num_channels", "n_jobs")},
            random_state=int(seed), output_type="tensor",
        )
        # Initialize on control dimensions; never pass regression targets.
        transformer.fit(x)
        raw = _raw_features(transformer, x, batch_size)
        scaler = HydraSparseScaler().fit(raw)
        scaled = scaler.transform(raw)
    features = scaled[:len(pair)] - scaled[len(pair):] if shared else scaled
    artifact = HydraTransformArtifact(transformer, scaler, shared, {
        "aeon_version": AEON_VERSION,
        "transform_version": HYDRA_TRANSFORM_VERSION,
        "training_seed": int(seed), "training_events": len(pair),
        "transform_fit_waveforms": len(x), "input_samples": pair.shape[-1],
        "feature_count": features.shape[1], "feature_dtype": str(features.dtype),
        "feature_definition": "phi(s1)-phi(s2)" if shared else "phi(s1,s2)",
        "scaling": "hydra_reference_sparse", "n_jobs": parameters["n_jobs"],
        "batch_size": batch_size,
    })
    return artifact, np.asarray(features, dtype=np.float32)


def transform(artifact, values):
    pair = _pair(values)
    if pair.shape[-1] != artifact.metadata["input_samples"]:
        raise ValueError("Hydra input length differs from its frozen control input grid")
    x = np.concatenate([pair[:, :1], pair[:, 1:]], axis=0) if artifact.shared else pair
    with _torch_runtime(artifact.metadata["n_jobs"]):
        raw = _raw_features(artifact.transformer, x, artifact.metadata["batch_size"])
        scaled = artifact.scaler.transform(raw)
    features = scaled[:len(pair)] - scaled[len(pair):] if artifact.shared else scaled
    return np.asarray(features, dtype=np.float32)


def export_transform(artifact):
    """Export kernels and the exact dilation/branch/statistic/group feature order."""
    hydra = artifact.transformer._hydra
    dilations = hydra.dilations.numpy()
    d, b, h, k = len(dilations), hydra.divisor, hydra.h, hydra.k
    count = d * b * 2 * h * k
    if count != artifact.metadata["feature_count"]:
        raise RuntimeError("Unsupported Hydra feature layout")
    arrays = {
        "weights": hydra.W.numpy(),
        "channel_indices": np.stack([indices.numpy() for indices in hydra.idx]),
        "dilations": dilations, "paddings": hydra.paddings.numpy(),
        "feature_column": np.arange(count, dtype=np.int32),
        "feature_dilation": np.repeat(dilations, b * 2 * h * k),
        "feature_branch": np.tile(np.repeat(np.arange(b), 2 * h * k), d),
        "feature_statistic": np.tile(np.repeat(["max_response_sum", "min_win_count"], h * k), d * b),
        "feature_group": np.tile(np.repeat(np.arange(h), k), d * b * 2),
        "feature_kernel": np.tile(np.arange(k), d * b * 2 * h),
        "scaler_mu": artifact.scaler.mu, "scaler_sigma": artifact.scaler.sigma,
        "scaler_epsilon": artifact.scaler.epsilon,
        "shared": np.asarray(artifact.shared), "aeon_version": np.asarray(AEON_VERSION),
        "training_seed": np.asarray(artifact.metadata["training_seed"]),
        "transform_version": np.asarray(HYDRA_TRANSFORM_VERSION),
        "transform_identity": np.asarray(artifact.metadata.get("transform_identity", "")),
        "control_analysis_protocol_identity": np.asarray(artifact.metadata.get("control_analysis_protocol_identity", "")),
    }
    if artifact.time_ps is not None:
        arrays.update(time_ps=artifact.time_ps, sample_indices=artifact.sample_indices)
    return arrays


def save_transform(artifact, path):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    with (path / "transform.pkl").open("wb") as stream:
        pickle.dump(artifact, stream, protocol=pickle.HIGHEST_PROTOCOL)
    np.savez_compressed(path / "hydra.npz", **export_transform(artifact))


def fit(params, train_x, train_target, *, seed, config, shared):
    del seed, config
    x, y = np.asarray(train_x, dtype=np.float32), np.asarray(train_target, dtype=np.float64)
    alpha = float(params["ridge_alpha"])
    if not np.isfinite(alpha) or alpha <= 0:
        raise ValueError("Hydra ridge_alpha must be finite and positive")
    if x.ndim != 2 or y.ndim != 1 or len(x) != len(y) or x.shape[1] < 1:
        raise ValueError("Hydra Ridge requires [event, feature] input and one target per event")
    regressor = Ridge(alpha=alpha, fit_intercept=not shared, solver="lsqr", tol=1e-4).fit(x, y)
    return HydraArtifact(regressor, {
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
        estimator_formulation="shared" if name == "shared_hydra" else "direct",
        feature_transform=FeatureTransformSpec(
            name=f"{name}_aeon_{AEON_VERSION}_v{HYDRA_TRANSFORM_VERSION}",
            parameters=transform_parameters, fit_transform=fit_transform_callback,
            transform=transform, save=save_transform,
        ),
    )
