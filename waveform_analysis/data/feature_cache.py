from __future__ import annotations

import json
import os
import pickle
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from waveform_analysis.core.io import atomic_json, canonical_hash
from waveform_analysis.data.splits import semantic_seed
from waveform_analysis.engine.train import FittedFeatureTransform
from waveform_analysis.data.view import waveform_view

_CACHE_VERSION = 3
_FROZEN_INPUT_MODELS = {
    "direct_linear_ridge",
    "shared_linear_ridge",
    "direct_minirocket",
    "shared_minirocket",
}
_MINIROCKET_MODELS = {"direct_minirocket", "shared_minirocket"}


@dataclass(frozen=True)
class FrozenTransform:
    transform: FittedFeatureTransform
    identity: str
    fit_role: str
    input_samples: int


@dataclass(frozen=True)
class FrozenFeatureSet:
    features: np.ndarray
    sample_mask: np.ndarray
    transform: FittedFeatureTransform
    identity: str
    role: str

    def rows(self, indices) -> np.ndarray:
        return np.asarray(
            self.features[np.asarray(indices, dtype=np.int64)],
            dtype=np.float32,
        )


def uses_frozen_model_input(spec) -> bool:
    return spec.name in _FROZEN_INPUT_MODELS and spec.feature_transform is not None


def _fixed_candidate(model_space: dict) -> dict:
    return {
        str(name): raw.get("value")
        for name, raw in (model_space.get("parameters") or {}).items()
        if isinstance(raw, dict) and str(raw.get("type", "")).lower() == "fixed"
    }


def _transform_parameters(spec, model_space: dict) -> dict:
    candidate = _fixed_candidate(model_space)
    try:
        return dict(spec.feature_transform.parameters(candidate, model_space) or {})
    except KeyError as exc:
        raise ValueError(
            f"{spec.name} feature-transform parameter {exc.args[0]!r} must be fixed "
            "during the downstream model search"
        ) from exc


def _pair(dataset, mode, indices=None) -> np.ndarray:
    if indices is None:
        indices = np.arange(dataset.n_events, dtype=np.int64)
    return np.asarray(
        waveform_view(dataset, mode, np.asarray(indices, dtype=np.int64)).materialize(),
        dtype=np.float32,
    )


def _atomic_pickle(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp")
    try:
        with tmp.open("wb") as stream:
            pickle.dump(value, stream, protocol=pickle.HIGHEST_PROTOCOL)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink()


def _transform_root(cache_root, mode, window, model_name, identity):
    return (
        Path(cache_root).resolve()
        / "model_inputs"
        / ("energy" if mode == "energy_to_energy" else "timing")
        / str(window)
        / str(model_name)
        / identity[:16]
    )


def prepare_frozen_transform(
    spec,
    model_space,
    config,
    *,
    control_dataset,
    development_dataset,
    cache_root,
    logger=None,
) -> FrozenTransform | None:
    if not uses_frozen_model_input(spec):
        return None

    mode = str(config["mode"])
    window = str(config.get("window_name") or "window")
    transform_parameters = _transform_parameters(spec, model_space)
    fit_role = "control" if spec.name in _MINIROCKET_MODELS else "deterministic"
    fit_identity = (
        str(control_dataset.manifest["analysis_protocol_identity"])
        if fit_role == "control"
        else "no_fit"
    )
    seed = semantic_seed(
        int(config["seed"]),
        mode,
        window,
        spec.feature_transform.name,
        "frozen_transform",
    )
    input_samples = int(
        waveform_view(
            development_dataset,
            mode,
            np.asarray([0], dtype=np.int64),
        ).time_ps.size
    )
    identity = canonical_hash(
        {
            "schema_version": _CACHE_VERSION,
            "model": spec.name,
            "transform": spec.feature_transform.name,
            "parameters": transform_parameters,
            "seed": int(seed),
            "fit_role": fit_role,
            "fit_identity": fit_identity,
            "input_samples": input_samples,
        }
    )
    directory = _transform_root(cache_root, mode, window, spec.name, identity)
    manifest_path = directory / "transform.json"
    artifact_path = directory / "transform.pkl"

    if manifest_path.is_file() and artifact_path.is_file():
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            if manifest.get("identity") == identity:
                with artifact_path.open("rb") as stream:
                    artifact = pickle.load(stream)
                wrapper = FittedFeatureTransform(
                    spec.feature_transform,
                    artifact,
                    identity,
                    transform_parameters,
                    int(seed),
                    {},
                )
                if logger:
                    logger.info(
                        "Model transform reused | %s | fit=%s",
                        spec.name.replace("_", " ").title(),
                        fit_role,
                    )
                return FrozenTransform(wrapper, identity, fit_role, input_samples)
        except (OSError, ValueError, json.JSONDecodeError, pickle.PickleError):
            pass

    directory.mkdir(parents=True, exist_ok=True)
    if fit_role == "control":
        fit_pair = _pair(control_dataset, mode)
        if logger:
            logger.info(
                "Model transform fit | %s | control only | events=%d",
                spec.name.replace("_", " ").title(),
                int(control_dataset.n_events),
            )
    else:
        fit_pair = _pair(
            development_dataset,
            mode,
            np.asarray([0], dtype=np.int64),
        )

    artifact, _ = spec.feature_transform.fit_transform(
        transform_parameters,
        fit_pair,
        seed=int(seed),
        config=model_space,
    )
    del fit_pair
    metadata = getattr(artifact, "metadata", None)
    if isinstance(metadata, dict):
        metadata["transform_fit_role"] = fit_role
        metadata["input_samples"] = input_samples

    _atomic_pickle(artifact_path, artifact)
    atomic_json(
        manifest_path,
        {
            "schema_version": _CACHE_VERSION,
            "identity": identity,
            "model": spec.name,
            "transform": spec.feature_transform.name,
            "parameters": transform_parameters,
            "seed": int(seed),
            "fit_role": fit_role,
            "fit_identity": fit_identity,
            "input_samples": input_samples,
        },
    )
    wrapper = FittedFeatureTransform(
        spec.feature_transform,
        artifact,
        identity,
        transform_parameters,
        int(seed),
        {},
    )
    return FrozenTransform(wrapper, identity, fit_role, input_samples)


def prepare_frozen_features(
    spec,
    frozen_transform: FrozenTransform,
    config,
    dataset,
    *,
    role,
    cache_root,
    logger=None,
) -> FrozenFeatureSet:
    mode = str(config["mode"])
    window = str(config.get("window_name") or "window")
    dataset_identity = str(dataset.manifest["analysis_protocol_identity"])
    identity = canonical_hash(
        {
            "schema_version": _CACHE_VERSION,
            "transform_identity": frozen_transform.identity,
            "dataset_identity": dataset_identity,
            "role": str(role),
        }
    )
    directory = _transform_root(
        cache_root,
        mode,
        window,
        spec.name,
        frozen_transform.identity,
    )
    path = directory / f"{role}_{identity[:12]}.npy"
    metadata_path = directory / f"{role}_{identity[:12]}.json"
    if path.is_file() and metadata_path.is_file():
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            features = np.load(path, mmap_mode="r")
            if metadata.get("identity") == identity and features.shape[0] == int(
                dataset.n_events
            ):
                if logger:
                    logger.info(
                        "Model input reused | %s | %s | events=%d | features=%d",
                        spec.name.replace("_", " ").title(),
                        role,
                        features.shape[0],
                        features.shape[1],
                    )
                return FrozenFeatureSet(
                    features,
                    np.ones(frozen_transform.input_samples, dtype=bool),
                    frozen_transform.transform,
                    identity,
                    str(role),
                )
        except (OSError, ValueError, json.JSONDecodeError):
            pass

    directory.mkdir(parents=True, exist_ok=True)
    chunk_size = int(config.get("runtime", {}).get("prediction_chunk_size", 4096))
    first = frozen_transform.transform.apply(
        _pair(dataset, mode, np.asarray([0], dtype=np.int64))
    )
    feature_count = int(np.asarray(first).shape[1])
    tmp = path.with_name(f".{path.name}.tmp")
    if tmp.exists():
        tmp.unlink()
    matrix = np.lib.format.open_memmap(
        tmp,
        mode="w+",
        dtype=np.float32,
        shape=(int(dataset.n_events), feature_count),
    )
    try:
        for start in range(0, int(dataset.n_events), chunk_size):
            stop = min(start + chunk_size, int(dataset.n_events))
            indices = np.arange(start, stop, dtype=np.int64)
            transformed = np.asarray(
                frozen_transform.transform.apply(_pair(dataset, mode, indices)),
                dtype=np.float32,
            )
            if transformed.shape != (stop - start, feature_count):
                raise RuntimeError(
                    f"{spec.name} transformed feature shape changed: "
                    f"{transformed.shape}"
                )
            matrix[start:stop] = transformed
        matrix.flush()
        del matrix
        os.replace(tmp, path)
    finally:
        try:
            del matrix
        except UnboundLocalError:
            pass
        if tmp.exists():
            tmp.unlink()

    atomic_json(
        metadata_path,
        {
            "schema_version": _CACHE_VERSION,
            "identity": identity,
            "transform_identity": frozen_transform.identity,
            "dataset_identity": dataset_identity,
            "role": str(role),
            "events": int(dataset.n_events),
            "feature_count": feature_count,
        },
    )
    features = np.load(path, mmap_mode="r")
    if logger:
        logger.info(
            "Model input ready | %s | %s | events=%d | features=%d",
            spec.name.replace("_", " ").title(),
            role,
            features.shape[0],
            features.shape[1],
        )
    return FrozenFeatureSet(
        features,
        np.ones(frozen_transform.input_samples, dtype=bool),
        frozen_transform.transform,
        identity,
        str(role),
    )


# Retain serialized identities while legacy imports resolve to this module.
from waveform_analysis.core.compat import (
    preserve_legacy_identity as _preserve_legacy_identity,
)

_preserve_legacy_identity(__name__, "waveform_analysis.ml_pipeline.feature_cache")
