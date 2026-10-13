"""Control-fitted random dilated shapelets followed by Ridge regression."""
from __future__ import annotations

from dataclasses import dataclass, field
from importlib.metadata import version
from pathlib import Path
import pickle

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from waveform_analysis.models.spec import FeatureTransformSpec, ModelSpec

RDST_MODELS = {"direct_rdst", "shared_rdst"}
AEON_VERSION = "1.6.0"
FEATURE_NAMES = ("minimum_distance", "match_location", "occurrence_count")


@dataclass
class RDSTTransformArtifact:
    transformer: object
    scaler: StandardScaler
    shared: bool
    metadata: dict
    time_ps: np.ndarray | None = None
    sample_indices: np.ndarray | None = None


@dataclass
class RDSTArtifact:
    regressor: Ridge
    metadata: dict = field(default_factory=dict)


def transform_parameters(params, config):
    # All transform settings must be fixed in the model space. The frozen-cache
    # parser deliberately raises on a categorical transform parameter.
    result = {key: params[key] for key in (
        "max_shapelets", "shapelet_lengths", "proba_normalization",
        "threshold_percentiles", "alpha_similarity", "use_prime_dilations",
    )}
    if isinstance(result["max_shapelets"], bool) or int(result["max_shapelets"]) != result["max_shapelets"] or result["max_shapelets"] < 1:
        raise ValueError("RDST max_shapelets must be a positive integer")
    lengths = result["shapelet_lengths"]
    if not isinstance(lengths, (list, tuple)) or not lengths or any(
        isinstance(n, bool) or int(n) != n or n < 2 for n in lengths
    ):
        raise ValueError("RDST shapelet_lengths must contain integers >= 2")
    result["shapelet_lengths"] = [int(n) for n in lengths]
    for name in ("proba_normalization", "alpha_similarity"):
        value = float(result[name])
        if not np.isfinite(value) or not 0 <= value <= 1:
            raise ValueError(f"RDST {name} must be in [0,1]")
        result[name] = value
    percentiles = np.asarray(result["threshold_percentiles"], dtype=float)
    if percentiles.shape != (2,) or not np.all(np.isfinite(percentiles)) or not 0 <= percentiles[0] <= percentiles[1] <= 100:
        raise ValueError("RDST threshold_percentiles must be an ordered pair in [0,100]")
    result["threshold_percentiles"] = percentiles.tolist()
    if not isinstance(result["use_prime_dilations"], bool):
        raise ValueError("RDST use_prime_dilations must be boolean")
    n_jobs = config.get("transform", {}).get("n_jobs", 2)
    if isinstance(n_jobs, bool) or int(n_jobs) != n_jobs or int(n_jobs) == 0:
        raise ValueError("RDST transform.n_jobs must be a nonzero integer")
    result["n_jobs"] = int(n_jobs)
    # Version is part of frozen-transform identity, not an aeon constructor arg.
    result["aeon_version"] = AEON_VERSION
    return result


def _pair(values):
    x = np.asarray(values, dtype=np.float32)
    if x.ndim != 3 or x.shape[1] != 2 or x.shape[0] < 1 or not np.all(np.isfinite(x)):
        raise ValueError("RDST expects finite [event, detector=2, time] input")
    return x


def fit_transform(parameters, train_x, *, seed, config, shared):
    del config
    from aeon.transformations.collection.shapelet_based import RandomDilatedShapeletTransform

    if version("aeon") != AEON_VERSION:
        raise ImportError(f"RDST requires aeon=={AEON_VERSION} for its persisted shapelet layout")
    pair = _pair(train_x)
    if max(parameters["shapelet_lengths"]) > pair.shape[-1]:
        raise ValueError("RDST shapelet_lengths exceed retained temporal samples")
    x = np.concatenate([pair[:, :1, :], pair[:, 1:, :]], axis=0) if shared else pair
    kwargs = {k: v for k, v in parameters.items() if k != "aeon_version"}
    transformer = RandomDilatedShapeletTransform(random_state=int(seed), **kwargs)
    # No regression targets: aeon interprets y as class labels.
    raw = np.asarray(transformer.fit_transform(x), dtype=np.float32)
    shapelets = export_shapelets(transformer)
    if raw.shape != (len(x), 3 * len(shapelets["lengths"])) or not np.all(np.isfinite(raw)):
        raise RuntimeError("Unexpected RDST feature layout or non-finite features")
    scaler = StandardScaler(with_mean=False).fit(raw)
    scaled = np.asarray(scaler.transform(raw), dtype=np.float32)
    features = scaled[:len(pair)] - scaled[len(pair):] if shared else scaled
    artifact = RDSTTransformArtifact(transformer, scaler, shared, {
        "aeon_version": AEON_VERSION, "training_seed": int(seed),
        "training_events": len(pair), "transform_fit_waveforms": len(x),
        "shapelet_count": int(transformer.n_shapelets_),
        "feature_count": int(features.shape[1]), "feature_names": list(FEATURE_NAMES),
        "feature_definition": "phi(s1)-phi(s2)" if shared else "phi(s1,s2)",
    })
    return artifact, np.asarray(features, dtype=np.float32)


def transform(artifact, values):
    pair = _pair(values)
    if artifact.shared:
        n = len(pair)
        pooled = np.concatenate([pair[:, :1], pair[:, 1:]], axis=0)
        z = np.asarray(artifact.scaler.transform(artifact.transformer.transform(pooled)), dtype=np.float32)
        return np.asarray(z[:n] - z[n:], dtype=np.float32)
    return np.asarray(artifact.scaler.transform(artifact.transformer.transform(pair)), dtype=np.float32)


def export_shapelets(transformer):
    """Adapt aeon 1.6.0's array tuple to named, pickle-free arrays."""
    names = ("values", "startpoints", "lengths", "dilations", "thresholds",
             "normalizes", "means", "stds", "classes")
    if len(transformer.shapelets_) != len(names):
        raise RuntimeError("Unsupported aeon shapelets_ layout")
    arrays = {k: np.asarray(v).copy() for k, v in zip(names, transformer.shapelets_)}
    n = int(transformer.n_shapelets_)
    if n < 1 or any(len(v) != n for v in arrays.values()) or arrays["values"].ndim != 3:
        raise RuntimeError("Inconsistent RDST shapelet arrays")
    arrays["shapelet_id"] = np.arange(n, dtype=np.int32)
    arrays["feature_columns"] = np.arange(3 * n, dtype=np.int32).reshape(n, 3)
    arrays["feature_names"] = np.asarray(FEATURE_NAMES)
    return arrays


def save_transform(artifact, path):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    with (path / "transform.pkl").open("wb") as stream:
        pickle.dump(artifact, stream, protocol=pickle.HIGHEST_PROTOCOL)
    arrays = export_shapelets(artifact.transformer)
    arrays.update(
        scaler_scale=artifact.scaler.scale_, scaler_variance=artifact.scaler.var_,
        aeon_version=np.asarray(AEON_VERSION),
        transform_identity=np.asarray(artifact.metadata.get("transform_identity", "")),
        control_analysis_protocol_identity=np.asarray(artifact.metadata.get("control_analysis_protocol_identity", "")),
        shared=np.asarray(artifact.shared),
    )
    if artifact.time_ps is not None:
        arrays["time_ps"] = artifact.time_ps
        arrays["sample_indices"] = artifact.sample_indices
    np.savez_compressed(path / "shapelets.npz", **arrays)


def fit(params, train_x, train_target, *, seed, config, shared):
    del seed, config
    x, y = np.asarray(train_x, dtype=np.float32), np.asarray(train_target, dtype=np.float64)
    alpha = float(params["ridge_alpha"])
    if not np.isfinite(alpha) or alpha <= 0:
        raise ValueError("RDST ridge_alpha must be finite and positive")
    if x.ndim != 2 or y.ndim != 1 or x.shape[0] != len(y) or x.shape[1] % 3:
        raise ValueError("RDST Ridge expects [event, 3*shapelet] features and one target per event")
    regressor = Ridge(alpha=alpha, fit_intercept=not shared, solver="lsqr", tol=1e-4).fit(x, y)
    return RDSTArtifact(regressor, {
        "ridge_alpha": alpha, "training_events": len(y), "feature_count": x.shape[1],
        "detector_swap_antisymmetry_enforced": shared, "fit_intercept": not shared,
    })


def predict(artifact, features):
    return np.asarray(artifact.regressor.predict(np.asarray(features, np.float32)), dtype=np.float64)


def save(artifact, path):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    with (path / "model.pkl").open("wb") as stream:
        pickle.dump(artifact, stream, protocol=pickle.HIGHEST_PROTOCOL)


def make_spec(name, fit_transform_callback, fit_callback):
    return ModelSpec(
        name=name, candidates=candidates, fit=fit_callback, predict=predict, save=save,
        estimator_formulation="shared" if name == "shared_rdst" else "direct",
        feature_transform=FeatureTransformSpec(
            name=f"{name}_aeon_{AEON_VERSION}", parameters=transform_parameters,
            fit_transform=fit_transform_callback, transform=transform, save=save_transform,
        ),
    )


def candidates(config):
    from waveform_analysis.engine.search import grid_candidates, fixed_parameters, optimization_config
    return grid_candidates(config) if optimization_config(config).strategy == "grid" else [fixed_parameters(config)]
