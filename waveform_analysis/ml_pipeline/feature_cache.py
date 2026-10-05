from __future__ import annotations

import gc
import json
import os
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .common import atomic_json, canonical_hash
from .sample_mask import apply_sample_mask, training_sample_mask
from .splits import semantic_seed
from .train import FittedFeatureTransform
from .view import waveform_view


_CACHE_SCHEMA_VERSION = 2
_MINIROCKET_TRANSFORMS = {
    "direct_minirocket_multivariate",
    "shared_minirocket_univariate",
}


@dataclass(frozen=True)
class PersistentFeatureCache:
    directory: Path
    features: np.ndarray
    transform: FittedFeatureTransform
    sample_mask: np.ndarray
    metadata: dict[str, Any]

    def rows(self, indices) -> np.ndarray:
        idx = np.asarray(indices, dtype=np.int64)
        return np.asarray(self.features[idx], dtype=np.float32)


def is_minirocket_transform(feature_transform) -> bool:
    return (
        feature_transform is not None
        and str(feature_transform.name) in _MINIROCKET_TRANSFORMS
    )


def _training_sample_mask(dataset, mode, fit_indices) -> np.ndarray:
    idx = np.asarray(fit_indices, dtype=np.int64)
    pair = waveform_view(dataset, mode, idx).materialize()
    try:
        return training_sample_mask(pair)
    finally:
        del pair


def transform_identity(
    spec,
    model_space,
    dataset,
    mode,
    fit_indices,
    parameters,
    *,
    seed_base,
):
    if spec.feature_transform is None:
        raise ValueError("feature transform is required")

    idx = np.asarray(fit_indices, dtype=np.int64)
    transform_parameters = dict(
        spec.feature_transform.parameters(dict(parameters or {}), model_space) or {}
    )
    parameter_hash = canonical_hash(transform_parameters)
    transform_seed = semantic_seed(
        int(seed_base),
        "feature_transform",
        spec.feature_transform.name,
        parameter_hash,
    )
    protocol_identity = str(
        dataset.manifest.get("analysis_protocol_identity")
        or dataset.manifest["analysis_population_identity"]
    )
    sample_mask = _training_sample_mask(dataset, mode, idx)
    scope_key = canonical_hash(
        {
            "protocol": protocol_identity,
            "mode": mode,
            "indices": idx.tolist(),
            "sample_mask": sample_mask.tolist(),
        }
    )
    identity = canonical_hash(
        {
            "name": spec.feature_transform.name,
            "parameters": transform_parameters,
            "seed": transform_seed,
            "scope": scope_key,
        }
    )
    return identity, transform_parameters, transform_seed, sample_mask


def cache_directory(dataset, transform_name: str, identity: str) -> Path:
    return (
        Path(dataset.directory).resolve()
        / "feature_cache"
        / str(transform_name)
        / identity[:16]
    )


def _load_transform(spec, path: Path, metadata: dict[str, Any]) -> FittedFeatureTransform:
    with path.open("rb") as stream:
        artifact = pickle.load(stream)
    return FittedFeatureTransform(
        spec.feature_transform,
        artifact,
        str(metadata["transform_identity"]),
        dict(metadata["transform_parameters"]),
        int(metadata["transform_seed"]),
        {},
    )


def load_feature_cache(
    spec,
    model_space,
    dataset,
    mode,
    fit_indices,
    parameters,
    *,
    seed_base,
    logger=None,
):
    if not is_minirocket_transform(spec.feature_transform):
        return None

    identity, transform_parameters, transform_seed, sample_mask = transform_identity(
        spec,
        model_space,
        dataset,
        mode,
        fit_indices,
        parameters,
        seed_base=seed_base,
    )
    directory = cache_directory(dataset, spec.feature_transform.name, identity)
    metadata_path = directory / "manifest.json"
    features_path = directory / "features.npy"
    transform_path = directory / "transform.pkl"
    if not (metadata_path.is_file() and features_path.is_file() and transform_path.is_file()):
        return None

    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None

    expected = {
        "schema_version": _CACHE_SCHEMA_VERSION,
        "transform_identity": identity,
        "transform_name": spec.feature_transform.name,
        "transform_parameters": transform_parameters,
        "transform_seed": int(transform_seed),
        "event_population_identity": str(dataset.manifest["event_population_identity"]),
        "analysis_protocol_identity": str(dataset.manifest["analysis_protocol_identity"]),
        "mode": str(mode),
        "n_events": int(dataset.n_events),
        "sample_mask_identity": canonical_hash(sample_mask.tolist()),
        "input_samples_before_mask": int(sample_mask.size),
        "input_samples_after_mask": int(sample_mask.sum()),
        "dtype": "float32",
    }
    if any(metadata.get(key) != value for key, value in expected.items()):
        return None

    features = np.load(features_path, mmap_mode="r")
    feature_count = int(metadata.get("feature_count", -1))
    if (
        features.dtype != np.float32
        or features.ndim != 2
        or features.shape != (int(dataset.n_events), feature_count)
    ):
        return None

    transform = _load_transform(spec, transform_path, metadata)
    if logger is not None:
        logger.info(
            "Feature cache reused | %s | id=%s | events=%d | features=%d | samples=%d/%d | size=%.2f GiB",
            spec.feature_transform.name,
            identity[:12],
            features.shape[0],
            features.shape[1],
            int(sample_mask.sum()),
            int(sample_mask.size),
            features.nbytes / (1024**3),
        )
    return PersistentFeatureCache(directory, features, transform, sample_mask, metadata)


def _save_transform(path: Path, artifact) -> None:
    tmp = path.with_name(f".{path.name}.tmp")
    try:
        with tmp.open("wb") as stream:
            pickle.dump(artifact, stream, protocol=pickle.HIGHEST_PROTOCOL)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink()


def build_feature_cache(
    spec,
    model_space,
    config,
    dataset,
    fit_indices,
    parameters,
    feature_transform: FittedFeatureTransform,
    *,
    seed_base,
    fit_features=None,
    logger=None,
):
    if not is_minirocket_transform(spec.feature_transform):
        raise ValueError("persistent feature caching is only enabled for MiniRocket transforms")

    idx_fit = np.asarray(fit_indices, dtype=np.int64)
    expected_identity, _, _, sample_mask = transform_identity(
        spec,
        model_space,
        dataset,
        config["mode"],
        idx_fit,
        parameters,
        seed_base=int(seed_base),
    )
    identity = str(feature_transform.identity)
    if identity != expected_identity:
        raise RuntimeError(
            "MiniRocket feature-transform identity does not match the training-derived sample mask"
        )

    directory = cache_directory(dataset, spec.feature_transform.name, identity)
    directory.mkdir(parents=True, exist_ok=True)
    features_path = directory / "features.npy"
    metadata_path = directory / "manifest.json"
    transform_path = directory / "transform.pkl"

    existing = load_feature_cache(
        spec,
        model_space,
        dataset,
        config["mode"],
        idx_fit,
        parameters,
        seed_base=int(seed_base),
        logger=logger,
    )
    if existing is not None:
        return existing

    if fit_features is not None:
        fit_features = np.asarray(fit_features, dtype=np.float32)
        if fit_features.ndim != 2 or fit_features.shape[0] != idx_fit.size:
            raise ValueError("fit feature cache shape does not match fit indices")
        feature_count = int(fit_features.shape[1])
    else:
        feature_count = int(
            getattr(feature_transform.artifact, "metadata", {}).get("feature_count", 0)
        )
        if feature_count < 1:
            raise RuntimeError("MiniRocket transform does not expose feature_count")

    temp_path = directory / ".features.npy.tmp"
    if temp_path.exists():
        temp_path.unlink()
    matrix = np.lib.format.open_memmap(
        temp_path,
        mode="w+",
        dtype=np.float32,
        shape=(int(dataset.n_events), feature_count),
    )

    completed = np.zeros(int(dataset.n_events), dtype=bool)
    if fit_features is not None and idx_fit.size:
        matrix[idx_fit] = fit_features
        completed[idx_fit] = True

    remaining = np.flatnonzero(~completed).astype(np.int64, copy=False)
    chunk_size = int(config.get("runtime", {}).get("prediction_chunk_size", 4096))
    total_remaining = int(remaining.size)
    if logger is not None:
        logger.info(
            "Feature cache build | %s | id=%s | events=%d | prefilled=%d | remaining=%d | features=%d | samples=%d/%d | dtype=float32 | chunk=%d",
            spec.feature_transform.name,
            identity[:12],
            int(dataset.n_events),
            int(idx_fit.size if fit_features is not None else 0),
            total_remaining,
            feature_count,
            int(sample_mask.sum()),
            int(sample_mask.size),
            chunk_size,
        )

    try:
        for start in range(0, total_remaining, chunk_size):
            stop = min(start + chunk_size, total_remaining)
            chunk_indices = remaining[start:stop]
            pair = waveform_view(dataset, config["mode"], chunk_indices).materialize()
            pair = apply_sample_mask(pair, sample_mask)
            transformed = np.asarray(feature_transform.apply(pair), dtype=np.float32)
            if transformed.shape != (chunk_indices.size, feature_count):
                raise RuntimeError(
                    "Feature transform cache chunk has an unexpected shape: "
                    f"{transformed.shape}"
                )
            matrix[chunk_indices] = transformed
            del transformed, pair
            if logger is not None:
                logger.info(
                    "Feature cache progress | %s | %d/%d transformed (%.1f%%)",
                    spec.feature_transform.name,
                    stop,
                    total_remaining,
                    100.0 * stop / max(1, total_remaining),
                )

        matrix.flush()
        del matrix
        gc.collect()
        os.replace(temp_path, features_path)
        _save_transform(transform_path, feature_transform.artifact)
        metadata = {
            "schema_version": _CACHE_SCHEMA_VERSION,
            "transform_identity": identity,
            "transform_name": spec.feature_transform.name,
            "transform_parameters": dict(feature_transform.parameters),
            "transform_seed": int(feature_transform.seed),
            "event_population_identity": str(dataset.manifest["event_population_identity"]),
            "analysis_protocol_identity": str(dataset.manifest["analysis_protocol_identity"]),
            "mode": str(config["mode"]),
            "fit_indices_identity": canonical_hash(idx_fit.tolist()),
            "fit_events": int(idx_fit.size),
            "n_events": int(dataset.n_events),
            "feature_count": feature_count,
            "sample_mask_identity": canonical_hash(sample_mask.tolist()),
            "input_samples_before_mask": int(sample_mask.size),
            "input_samples_after_mask": int(sample_mask.sum()),
            "dtype": "float32",
            "feature_file": str(features_path.resolve()),
            "transform_file": str(transform_path.resolve()),
        }
        atomic_json(metadata_path, metadata)
    finally:
        try:
            del matrix
        except UnboundLocalError:
            pass
        if temp_path.exists():
            temp_path.unlink()

    features = np.load(features_path, mmap_mode="r")
    feature_transform.cache.clear()
    if logger is not None:
        logger.info(
            "Feature cache complete | %s | id=%s | events=%d | features=%d | samples=%d/%d | size=%.2f GiB",
            spec.feature_transform.name,
            identity[:12],
            features.shape[0],
            features.shape[1],
            int(sample_mask.sum()),
            int(sample_mask.size),
            features.nbytes / (1024**3),
        )
    return PersistentFeatureCache(directory, features, feature_transform, sample_mask, metadata)
