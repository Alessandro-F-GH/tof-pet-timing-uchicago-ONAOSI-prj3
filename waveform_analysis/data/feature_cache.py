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
from waveform_analysis.data.view import model_sample_mask, waveform_view
from waveform_analysis.signal.sample_mask import apply_sample_mask
_CACHE_VERSION = 4


@dataclass(frozen=True)
class FrozenTransform:
    transform: FittedFeatureTransform
    identity: str
    fit_role: str
    input_samples: int
    sample_mask: np.ndarray


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
    return spec.feature_transform is not None


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
    return apply_sample_mask(
        waveform_view(dataset, mode, np.asarray(indices, dtype=np.int64)).materialize(),
        model_sample_mask(dataset, mode),
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
    fit_role = spec.feature_transform.fit_role
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
    sample_mask = model_sample_mask(development_dataset, mode)
    input_samples = int(sample_mask.sum())
    if fit_role == "control" and not np.array_equal(sample_mask, model_sample_mask(control_dataset, mode)):
        raise ValueError("Control and development model masks differ")
    identity = canonical_hash(
        {
            "schema_version": _CACHE_VERSION,
            "sample_mask": sample_mask.tolist(),
            "sample_mask_identity": development_dataset.manifest.get("sample_mask_identity"),
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
                return FrozenTransform(wrapper, identity, fit_role, input_samples, sample_mask)
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
        config=({**model_space, "_frozen_transform_only": True}
                if spec.bounded_transform_batches else model_space),
    )
    del fit_pair
    if spec.requires_frozen_features:
        view = waveform_view(control_dataset, mode, np.asarray([0], dtype=np.int64))
        artifact.time_ps = np.asarray(view.time_ps, dtype=np.float64)[sample_mask]
        artifact.sample_indices = np.flatnonzero(sample_mask)
    metadata = getattr(artifact, "metadata", None)
    if isinstance(metadata, dict):
        metadata["transform_fit_role"] = fit_role
        metadata["input_samples"] = input_samples
        if spec.requires_frozen_features:
            metadata["transform_identity"] = identity
            metadata["control_analysis_protocol_identity"] = fit_identity

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
    return FrozenTransform(wrapper, identity, fit_role, input_samples, sample_mask)


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
    if not np.array_equal(model_sample_mask(dataset, mode), frozen_transform.sample_mask):
        raise ValueError("Dataset mask differs from the frozen model transform mask")
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
                    np.asarray(frozen_transform.sample_mask, dtype=bool),
                    frozen_transform.transform,
                    identity,
                    str(role),
                )
        except (OSError, ValueError, json.JSONDecodeError):
            pass

    directory.mkdir(parents=True, exist_ok=True)
    chunk_size = int(config.get("runtime", {}).get("prediction_chunk_size", 4096))
    if spec.bounded_transform_batches:
        # Bound the returned feature matrix, not only MultiRocket's internal
        # convolution batches: 4096 default-budget rows occupy about 0.8 GB.
        chunk_size = min(chunk_size, int(frozen_transform.transform.artifact.metadata["batch_size"]))
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
        np.asarray(frozen_transform.sample_mask, dtype=bool),
        frozen_transform.transform,
        identity,
        str(role),
    )
