from __future__ import annotations

import os
import pickle
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from waveform_analysis.core.io import canonical_hash
from waveform_analysis.core.artifact_identity import ProductRecipe
from waveform_analysis.data.artifact_store import ArtifactStore
from waveform_analysis.data.named_cache import (alias_exists, load_named_reference, transform_alias, feature_alias, get_or_build_named, cache_root_from_config)
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
    # Legacy combined matrices are retained as one mmap. New composite caches
    # keep their feature families in separate read-only mmaps.
    _features: np.ndarray | None
    sample_mask: np.ndarray
    transform: FittedFeatureTransform
    identity: str
    role: str
    feature_blocks: tuple[np.ndarray, ...] = ()
    block_names: tuple[str, ...] = ()

    @property
    def features(self) -> np.ndarray:
        if self._features is not None:
            return self._features
        # Compatibility accessor for diagnostics/tests; training uses rows()
        # and therefore never materializes the full concatenated matrix.
        return np.concatenate(self.feature_blocks, axis=1)

    @property
    def shape(self) -> tuple[int, int]:
        if self._features is not None:
            return self._features.shape
        return (self.feature_blocks[0].shape[0],
                sum(block.shape[1] for block in self.feature_blocks))

    def rows(self, indices) -> np.ndarray:
        indices = np.asarray(indices, dtype=np.int64)
        if self._features is not None:
            return np.asarray(self._features[indices], dtype=np.float32)
        return np.concatenate(
            [np.asarray(block[indices], dtype=np.float32)
             for block in self.feature_blocks],
            axis=1,
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


def prepare_frozen_transform(
    spec, model_space, config, *, control_dataset, development_dataset,
    cache_root, logger=None, legacy_transform_path=None,
) -> FrozenTransform | None:
    """Load or fit one immutable feature transform in the shared artifact store."""
    if not uses_frozen_model_input(spec):
        return None

    mode = str(config["mode"])
    transform_parameters = _transform_parameters(spec, model_space)
    fit_role = spec.feature_transform.fit_role
    fit_identity = (
        str(control_dataset.manifest["analysis_protocol_identity"])
        if fit_role == "control" else "no_fit"
    )
    seed = semantic_seed(
        int(config["seed"]), mode, config.get("window_name"),
        spec.feature_transform.name, "frozen_transform",
    )
    sample_mask = model_sample_mask(development_dataset, mode)
    input_samples = int(sample_mask.sum())
    if fit_role == "control" and not np.array_equal(
        sample_mask, model_sample_mask(control_dataset, mode)
    ):
        raise ValueError("Control and development model masks differ")

    recipe = ProductRecipe(
        "feature-transforms",
        {
            "schema_version": _CACHE_VERSION,
            "model": spec.feature_cache_key or spec.name,
            "transform": spec.feature_transform.name,
            "parameters": transform_parameters,
            "seed": int(seed),
            "fit_role": fit_role,
            "fit_identity": fit_identity,
            "sample_mask_identity": development_dataset.manifest.get("sample_mask_identity"),
            "sample_mask": sample_mask.tolist(),
            "input_samples": input_samples,
            "mode": mode,
            "window_ns": config["window_ns"],
        },
        implementation=f"{spec.feature_transform.name}:frozen-transform-v{_CACHE_VERSION}",
    )
    store = ArtifactStore(config["preprocessing"].get("artifact_root", Path(cache_root) / "artifact_store"))

    def build(directory, effective_recipe=None):
        if legacy_transform_path is not None:
            source = Path(legacy_transform_path).expanduser().resolve()
            if not source.is_file():
                raise FileNotFoundError(f"Legacy fitted transformation not found: {source}")
            with source.open("rb") as stream:
                artifact = pickle.load(stream)
            if not hasattr(artifact, "metadata"):
                raise ValueError(f"Legacy transformation has no metadata: {source}")
        else:
            if fit_role == "control":
                fit_pair = _pair(control_dataset, mode)
            else:
                fit_pair = _pair(development_dataset, mode, np.asarray([0], dtype=np.int64))
            try:
                artifact, _ = spec.feature_transform.fit_transform(
                    transform_parameters, fit_pair, seed=int(seed),
                    config=({**model_space, "_frozen_transform_only": True}
                            if spec.bounded_transform_batches else model_space),
                )
            finally:
                del fit_pair
        if spec.requires_frozen_features:
            view = waveform_view(control_dataset, mode, np.asarray([0], dtype=np.int64))
            artifact.time_ps = np.asarray(view.time_ps, dtype=np.float64)[sample_mask]
            artifact.sample_indices = np.flatnonzero(sample_mask)
        metadata = getattr(artifact, "metadata", None)
        if isinstance(metadata, dict):
            metadata["transform_fit_role"] = fit_role
            metadata["input_samples"] = input_samples
            metadata["transform_identity"] = (effective_recipe or recipe).product_id
            metadata["control_analysis_protocol_identity"] = fit_identity
        with (directory / "transform.pkl").open("wb") as stream:
            pickle.dump(artifact, stream, protocol=pickle.HIGHEST_PROTOCOL)
            stream.flush()
            os.fsync(stream.fileno())

    alias = transform_alias(cache_root_from_config(config, cache_root), config, spec.feature_transform.name, fit_role)
    if config.get("reuse_prepared_inputs", False):
        existing = store.lookup(recipe)
        if existing.status != "hit":
            raise FileNotFoundError(f"Missing cached frozen transform for {spec.feature_transform.name}")
        if alias_exists(alias):
            aliased = load_named_reference(store, alias, expected_type=recipe.product_type)
            if aliased.product_id != existing.reference.product_id:
                raise ValueError("Cached transform alias does not match requested configuration")
    reference = get_or_build_named(store, recipe, alias, build, metadata={
        "fit_role": fit_role, "model": spec.name, "transform": spec.feature_transform.name,
    })
    with (reference.payload_path("transform.pkl")).open("rb") as stream:
        artifact = pickle.load(stream)
    if logger:
        logger.info(
            "Model transform artifact | %s | fit=%s | id=%s",
            spec.name.replace("_", " ").title(), fit_role, reference.product_id[:12],
        )
    wrapper = FittedFeatureTransform(
        spec.feature_transform, artifact, reference.product_id,
        transform_parameters, int(seed), {},
    )
    return FrozenTransform(wrapper, reference.product_id, fit_role, input_samples, sample_mask)


def prepare_frozen_features(
    spec, frozen_transform: FrozenTransform, config, dataset, *,
    role, cache_root, logger=None,
) -> FrozenFeatureSet:
    """Materialize immutable feature matrices once per transform and population."""
    mode = str(config["mode"])
    if not np.array_equal(model_sample_mask(dataset, mode), frozen_transform.sample_mask):
        raise ValueError("Dataset mask differs from the frozen model transform mask")
    dataset_identity = str(dataset.manifest["analysis_protocol_identity"])
    recipe = ProductRecipe(
        "feature-matrices",
        {
            "schema_version": _CACHE_VERSION,
            "transform_identity": frozen_transform.identity,
            "dataset_identity": dataset_identity,
            "model": spec.feature_cache_key or spec.name,
            "formulation": spec.estimator_formulation,
            "role": str(role),
            "mode": mode,
        },
        parents=(frozen_transform.identity,),
        implementation=(
            f"{spec.feature_transform.name}:feature-matrix-v{_CACHE_VERSION}"
        ),
    )
    store = ArtifactStore(config["preprocessing"].get(
        "artifact_root", str(Path(cache_root).resolve() / "artifact_store")
    ))

    # Prefer a legacy combined matrix when one already exists. Its recipe is
    # intentionally unchanged so caches created before block storage remain valid.
    combined = store.lookup(recipe)
    if config.get("reuse_prepared_inputs", False) and combined.status != "hit":
        raise FileNotFoundError(f"Missing cached innovation feature matrix for {spec.name} ({role})")
    if combined.status == "hit":
        features = store.open_array(combined.reference, "features.npy")
        if features.shape[0] != int(dataset.n_events) or features.ndim != 2:
            raise ValueError("Cached feature matrix has incompatible shape")
        if logger:
            logger.info(
                "Model input artifact | %s | %s | events=%d | features=%d | id=%s",
                spec.name.replace("_", " ").title(), role, features.shape[0],
                features.shape[1], combined.reference.product_id[:12],
            )
        return FrozenFeatureSet(
            features, np.asarray(frozen_transform.sample_mask, dtype=bool),
            frozen_transform.transform, combined.reference.product_id, str(role),
        )

    transform_spec = spec.feature_transform
    if transform_spec.block_names and transform_spec.transform_block is not None:
        block_arrays = []
        block_ids = []
        chunk_size = int(config.get("runtime", {}).get("prediction_chunk_size", 4096))
        if spec.bounded_transform_batches:
            chunk_size = min(
                chunk_size,
                int(frozen_transform.transform.artifact.metadata["batch_size"]),
            )
        for block_name in transform_spec.block_names:
            block_recipe = ProductRecipe(
                "feature-blocks",
                {
                    "schema_version": 1,
                    "transform_identity": frozen_transform.identity,
                    "dataset_identity": dataset_identity,
                    "model": spec.name,
                    "formulation": spec.estimator_formulation,
                    "role": str(role),
                    "mode": mode,
                    "block": block_name,
                },
                parents=(frozen_transform.identity,),
                implementation=(
                    f"{transform_spec.name}:{spec.name}:{block_name}:feature-block-v2"
                ),
            )

            def build_block(directory, effective_recipe=None, block_name=block_name):
                first = np.asarray(
                    transform_spec.transform_block(
                        frozen_transform.transform.artifact,
                        _pair(dataset, mode, np.asarray([0], dtype=np.int64)),
                        block_name,
                    ),
                    dtype=np.float32,
                )
                if first.ndim != 2 or first.shape[0] != 1 or not np.all(np.isfinite(first)):
                    raise RuntimeError(f"{spec.name}/{block_name} produced invalid features")
                count = int(first.shape[1])
                matrix = np.lib.format.open_memmap(
                    directory / "features.npy", mode="w+", dtype=np.float32,
                    shape=(int(dataset.n_events), count),
                )
                try:
                    matrix[0:1] = first
                    for start in range(1, int(dataset.n_events), chunk_size):
                        stop = min(start + chunk_size, int(dataset.n_events))
                        transformed = np.asarray(
                            transform_spec.transform_block(
                                frozen_transform.transform.artifact,
                                _pair(dataset, mode, np.arange(start, stop, dtype=np.int64)),
                                block_name,
                            ),
                            dtype=np.float32,
                        )
                        if transformed.shape != (stop - start, count) or not np.all(np.isfinite(transformed)):
                            raise RuntimeError(f"{spec.name}/{block_name} produced invalid feature batches")
                        matrix[start:stop] = transformed
                    matrix.flush()
                finally:
                    mmap = getattr(matrix, "_mmap", None)
                    if mmap is not None:
                        mmap.close()
                    del matrix

            block_alias = feature_alias(cache_root_from_config(config, cache_root), config, (spec.feature_cache_key or spec.name), config.get(str(role), {"root_file": dataset.manifest.get("dataset_source", role)}), str(role), block_name)
            block_ref = get_or_build_named(
                store, block_recipe, block_alias, build_block,
                metadata={"model": spec.name, "block": block_name,
                          "role": str(role), "events": int(dataset.n_events),
                          "dataset_identity": dataset_identity},
            )
            block = store.open_array(block_ref, "features.npy")
            if block.ndim != 2 or block.shape[0] != int(dataset.n_events):
                raise ValueError(f"Cached feature block {block_name} has incompatible shape")
            block_arrays.append(block)
            block_ids.append(block_ref.product_id)
        identity = canonical_hash({
            "transform_identity": frozen_transform.identity,
            "dataset_identity": dataset_identity,
            "role": str(role), "mode": mode,
            "blocks": dict(zip(transform_spec.block_names, block_ids)),
        })
        if logger:
            logger.info(
                "Model input blocks | %s | %s | events=%d | blocks=%s",
                spec.name.replace("_", " ").title(), role, int(dataset.n_events),
                ", ".join(f"{name}={array.shape[1]}" for name, array in zip(transform_spec.block_names, block_arrays)),
            )
        return FrozenFeatureSet(
            None, np.asarray(frozen_transform.sample_mask, dtype=bool),
            frozen_transform.transform, identity, str(role),
            tuple(block_arrays), tuple(transform_spec.block_names),
        )

    def build(directory, effective_recipe=None):
        chunk_size = int(config.get("runtime", {}).get("prediction_chunk_size", 4096))
        if spec.bounded_transform_batches:
            chunk_size = min(
                chunk_size,
                int(frozen_transform.transform.artifact.metadata["batch_size"]),
            )
        first = np.asarray(frozen_transform.transform.apply(
            _pair(dataset, mode, np.asarray([0], dtype=np.int64))
        ), dtype=np.float32)
        if first.ndim != 2 or not np.all(np.isfinite(first)):
            raise RuntimeError(f"{spec.name} produced an invalid feature matrix")
        feature_count = int(first.shape[1])
        path = directory / "features.npy"
        matrix = np.lib.format.open_memmap(
            path, mode="w+", dtype=np.float32,
            shape=(int(dataset.n_events), feature_count),
        )
        try:
            matrix[0:1] = first
            for start in range(1, int(dataset.n_events), chunk_size):
                stop = min(start + chunk_size, int(dataset.n_events))
                indices = np.arange(start, stop, dtype=np.int64)
                transformed = np.asarray(
                    frozen_transform.transform.apply(_pair(dataset, mode, indices)),
                    dtype=np.float32,
                )
                if transformed.shape != (stop - start, feature_count):
                    raise RuntimeError(
                        f"{spec.name} transformed feature shape changed: {transformed.shape}"
                    )
                matrix[start:stop] = transformed
            matrix.flush()
        finally:
            mmap = getattr(matrix, "_mmap", None)
            if mmap is not None:
                mmap.close()
            del matrix

    alias = feature_alias(cache_root_from_config(config, cache_root), config, spec.name, config.get(str(role), {"root_file": dataset.manifest.get("dataset_source", role)}), str(role))
    reference = get_or_build_named(store, recipe, alias, build, metadata={
        "model": spec.name, "role": str(role), "events": int(dataset.n_events),
        "dataset_identity": dataset_identity,
    })
    features = store.open_array(reference, "features.npy")
    if features.shape[0] != int(dataset.n_events) or features.ndim != 2:
        raise ValueError("Cached feature matrix has incompatible shape")
    if logger:
        logger.info(
            "Model input artifact | %s | %s | events=%d | features=%d | id=%s",
            spec.name.replace("_", " ").title(), role, features.shape[0],
            features.shape[1], reference.product_id[:12],
        )
    return FrozenFeatureSet(
        features, np.asarray(frozen_transform.sample_mask, dtype=bool),
        frozen_transform.transform, reference.product_id, str(role),
    )
