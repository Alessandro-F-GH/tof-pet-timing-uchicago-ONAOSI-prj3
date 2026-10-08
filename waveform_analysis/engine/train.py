from __future__ import annotations

import copy
import gc
import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from waveform_analysis.core.io import canonical_hash
from waveform_analysis.signal.sample_mask import (
    apply_sample_mask,
    apply_sample_mask_to_time,
    training_sample_mask,
)
from waveform_analysis.data.splits import semantic_seed
from waveform_analysis.data.view import model_target, waveform_view


_DEFAULT_PREDICTION_CHUNK_SIZE = 4096


@dataclass
class FittedFeatureTransform:
    spec: Any
    artifact: Any
    identity: str
    parameters: dict[str, Any]
    seed: int
    cache: dict[str, np.ndarray] = field(default_factory=dict)

    def apply(self, values):
        transformed = np.asarray(
            self.spec.transform(self.artifact, np.asarray(values, np.float32))
        )
        if transformed.ndim < 2 or transformed.shape[0] != np.asarray(values).shape[0]:
            raise RuntimeError(
                f"Feature transform {self.spec.name} changed the event axis"
            )
        if not np.all(np.isfinite(transformed)):
            raise RuntimeError(
                f"Feature transform {self.spec.name} produced non-finite values"
            )
        return transformed


class FeatureTransformCache:
    """Keep only one fitted feature transform and its transformed fit input in RAM."""

    def __init__(self):
        self._items = {}

    def clear(self):
        self._items.clear()

    def prepare(self, spec, parameters, values, *, seed_base, scope_key, config):
        transform_parameters = dict(
            spec.parameters(dict(parameters or {}), config) or {}
        )
        parameter_hash = canonical_hash(transform_parameters)
        transform_seed = semantic_seed(
            int(seed_base), "feature_transform", spec.name, parameter_hash
        )
        identity = canonical_hash(
            {
                "name": spec.name,
                "parameters": transform_parameters,
                "seed": transform_seed,
                "scope": str(scope_key),
            }
        )
        cached = self._items.get(identity)
        fit_key = f"fit:{scope_key}"
        if cached is not None:
            return cached, cached.cache[fit_key]

        # A different transform configuration supersedes the previous one.
        # This preserves reuse across candidates that share the same transform
        # parameters (for example MiniRocket ridge-alpha scans) without letting
        # transformed matrices accumulate across the whole optimization.
        self.clear()

        artifact, transformed = spec.fit_transform(
            transform_parameters,
            np.asarray(values, np.float32),
            seed=transform_seed,
            config=config,
        )
        transformed = np.asarray(transformed)
        if transformed.ndim < 2 or transformed.shape[0] != np.asarray(values).shape[0]:
            raise RuntimeError(f"Feature transform {spec.name} changed the event axis")
        if not np.all(np.isfinite(transformed)):
            raise RuntimeError(
                f"Feature transform {spec.name} produced non-finite values"
            )

        wrapper = FittedFeatureTransform(
            spec,
            artifact,
            identity,
            transform_parameters,
            transform_seed,
            {fit_key: transformed},
        )
        self._items[identity] = wrapper
        logger = config.get("_logger")
        if logger is not None and bool(getattr(spec, "log_fit", True)):
            logger.info(
                "Feature transform fitted | %s | id=%s | params=%s | events=%d | features=%s",
                spec.name,
                identity[:12],
                transform_parameters,
                transformed.shape[0],
                tuple(transformed.shape[1:]),
            )
        return wrapper, transformed


@dataclass(frozen=True)
class PreparedFitInput:
    x: np.ndarray
    y: np.ndarray
    time_ps: np.ndarray
    sample_mask: np.ndarray
    scope_key: str


class FitInputCache:
    def __init__(self):
        self._items = {}

    def clear(self):
        self._items.clear()

    def prepare(self, spec, dataset, mode, indices):
        idx = np.asarray(indices, np.int64)
        protocol_identity = str(
            dataset.manifest.get("analysis_protocol_identity")
            or dataset.manifest["analysis_population_identity"]
        )
        key = canonical_hash(
            {
                "protocol": protocol_identity,
                "mode": mode,
                "indices": idx.tolist(),
                "preserve_temporal_grid": bool(spec.preserve_temporal_grid),
            }
        )
        cached = self._items.get(key)
        if cached is not None:
            return cached

        view = waveform_view(dataset, mode, idx)
        xfull = view.materialize()
        mask = (
            np.ones(xfull.shape[-1], bool)
            if spec.preserve_temporal_grid
            else training_sample_mask(xfull)
        )
        x = apply_sample_mask(xfull, mask)
        time = apply_sample_mask_to_time(view.time_ps, mask)
        y = model_target(dataset, mode)[idx]
        scope_key = canonical_hash(
            {
                "protocol": protocol_identity,
                "mode": mode,
                "indices": idx.tolist(),
                "sample_mask": np.asarray(mask, bool).tolist(),
            }
        )
        prepared = PreparedFitInput(
            np.asarray(x, np.float32),
            np.asarray(y, np.float64),
            np.asarray(time, np.float64),
            np.asarray(mask, bool),
            scope_key,
        )
        self._items[key] = prepared
        return prepared


@dataclass
class FittedModel:
    artifact: Any
    metadata: dict[str, Any]
    output_max_abs_ps: float | None = None
    sample_mask: np.ndarray | None = None
    feature_transform: FittedFeatureTransform | None = None


def release_training_memory():
    """Release cyclic Python objects and unused CUDA allocator blocks."""
    gc.collect()
    try:
        import torch
    except ImportError:
        return
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _fit_once(
    spec,
    model_config,
    parameters,
    x,
    y,
    *,
    seed,
    output_limit,
    input_time_ps,
    sample_mask,
    logger=None,
):
    cfg = copy.deepcopy(model_config)
    cfg["_prediction_max_abs_ps"] = float(output_limit)
    cfg["_input_time_ps"] = np.asarray(input_time_ps, np.float64)
    if logger is not None:
        cfg["_logger"] = logger
    artifact = spec.fit(
        parameters,
        np.asarray(x, np.float32),
        np.asarray(y, np.float64),
        seed=int(seed),
        config=cfg,
    )
    metadata = dict(getattr(artifact, "metadata", {}) or {})
    metadata["estimator_formulation"] = spec.estimator_formulation
    return FittedModel(
        artifact,
        metadata,
        float(output_limit),
        np.asarray(sample_mask, bool),
    )


def fit_on_indices(
    spec,
    model_config,
    config,
    dataset,
    indices,
    parameters,
    *,
    seed,
    transform_seed_base=None,
    feature_transform_cache=None,
    fit_input_cache=None,
    logger=None,
    frozen_features=None,
):
    idx = np.asarray(indices, np.int64)
    if frozen_features is not None:
        x = frozen_features.rows(idx)
        y = np.asarray(model_target(dataset, config["mode"])[idx], dtype=np.float64)
        mask = np.asarray(frozen_features.sample_mask, dtype=bool)
        time = apply_sample_mask_to_time(
            waveform_view(
                dataset, config["mode"], np.asarray([0], dtype=np.int64)
            ).time_ps,
            mask,
        )
        prepared = PreparedFitInput(x, y, time, mask, frozen_features.identity)
    else:
        cache = fit_input_cache if fit_input_cache is not None else FitInputCache()
        prepared = cache.prepare(spec, dataset, config["mode"], idx)
        x = prepared.x
        y = prepared.y
        time = prepared.time_ps
        mask = prepared.sample_mask

    cfg = copy.deepcopy(model_config)
    cfg["_early_stopping_seed"] = int(seed)
    cfg["_input_time_ps"] = np.asarray(time, np.float64)
    if logger is not None:
        cfg["_logger"] = logger

    feature_transform = (
        frozen_features.transform if frozen_features is not None else None
    )
    if spec.feature_transform is not None and frozen_features is None:
        transform_cache = (
            feature_transform_cache
            if feature_transform_cache is not None
            else FeatureTransformCache()
        )
        feature_transform, x = transform_cache.prepare(
            spec.feature_transform,
            parameters,
            x,
            seed_base=int(seed if transform_seed_base is None else transform_seed_base),
            scope_key=prepared.scope_key,
            config=cfg,
        )

    fitted = _fit_once(
        spec,
        cfg,
        dict(parameters or {}),
        x,
        y,
        seed=seed,
        output_limit=config["ml_output"]["max_abs_ps"],
        input_time_ps=time,
        sample_mask=mask,
        logger=logger,
    )
    fitted.feature_transform = feature_transform
    fitted.metadata.update(
        {
            "training_events": int(y.size),
            "sample_mask_training_events": int(y.size),
            "input_samples_before_mask": int(mask.size),
            "input_samples_after_mask": int(mask.sum()),
            "prediction_chunk_size": int(
                config.get("runtime", {}).get(
                    "prediction_chunk_size", _DEFAULT_PREDICTION_CHUNK_SIZE
                )
            ),
        }
    )
    if feature_transform is not None:
        fitted.metadata.update(
            {
                "feature_transform": feature_transform.spec.name,
                "feature_transform_identity": feature_transform.identity,
                "feature_transform_parameters": feature_transform.parameters,
                "feature_transform_seed": int(feature_transform.seed),
            }
        )
    return fitted


def _model_input(fitted, dataset, mode, indices, *, swapped=False):
    idx = np.asarray(indices, np.int64)
    view = waveform_view(dataset, mode, idx)
    pair = apply_sample_mask(view.materialize(), fitted.sample_mask)
    if swapped:
        pair = pair[:, ::-1, :]
    pair = np.ascontiguousarray(pair)
    if fitted.feature_transform is None:
        return pair
    return fitted.feature_transform.apply(pair)


def _prediction_from_input(spec, fitted, values):
    prediction = np.asarray(spec.predict(fitted.artifact, values), np.float64).reshape(
        -1
    )
    if fitted.output_max_abs_ps is not None:
        prediction = np.clip(
            prediction,
            -fitted.output_max_abs_ps,
            fitted.output_max_abs_ps,
        )
    return prediction


def _resolve_prediction_chunk_size(fitted, chunk_size):
    value = (
        fitted.metadata.get("prediction_chunk_size", _DEFAULT_PREDICTION_CHUNK_SIZE)
        if chunk_size is None
        else chunk_size
    )
    value = int(value)
    if value < 1:
        raise ValueError("prediction chunk size must be >= 1")
    return value


def _predict_indices(
    spec,
    fitted,
    dataset,
    mode,
    indices,
    *,
    swapped,
    chunk_size=None,
    frozen_features=None,
):
    idx = np.asarray(indices, np.int64).reshape(-1)
    if not idx.size:
        return np.empty(0, dtype=np.float64)

    chunk = _resolve_prediction_chunk_size(fitted, chunk_size)
    output = np.empty(idx.size, dtype=np.float64)
    for start in range(0, idx.size, chunk):
        stop = min(start + chunk, idx.size)
        if frozen_features is not None and not swapped:
            values = frozen_features.rows(idx[start:stop])
        else:
            values = _model_input(
                fitted, dataset, mode, idx[start:stop], swapped=swapped
            )
        prediction = _prediction_from_input(spec, fitted, values)
        if prediction.size != stop - start:
            raise RuntimeError(
                f"Model prediction changed the event axis: expected {stop - start}, got {prediction.size}"
            )
        output[start:stop] = prediction
    return output


def predict_indices(
    spec, fitted, dataset, mode, indices, *, chunk_size=None, frozen_features=None
):
    return _predict_indices(
        spec,
        fitted,
        dataset,
        mode,
        indices,
        swapped=False,
        chunk_size=chunk_size,
        frozen_features=frozen_features,
    )


def detector_swap_rmse(
    spec,
    fitted,
    dataset,
    mode,
    indices,
    *,
    forward_prediction=None,
    chunk_size=None,
):
    if spec.estimator_formulation == "shared":
        return 0.0
    forward = (
        np.asarray(forward_prediction, np.float64).reshape(-1)
        if forward_prediction is not None
        else predict_indices(
            spec,
            fitted,
            dataset,
            mode,
            indices,
            chunk_size=chunk_size,
        )
    )
    reverse = _predict_indices(
        spec,
        fitted,
        dataset,
        mode,
        indices,
        swapped=True,
        chunk_size=chunk_size,
    )
    if forward.shape != reverse.shape:
        raise RuntimeError("Detector-swap diagnostic prediction shape mismatch")
    epsilon = forward + reverse
    if not np.all(np.isfinite(epsilon)):
        raise RuntimeError("Detector-swap diagnostic produced non-finite predictions")
    return float(np.sqrt(np.mean(np.square(epsilon))))


def save_model(spec, fitted, directory, parameters):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    spec.save(fitted.artifact, directory)
    if (
        fitted.feature_transform is not None
        and fitted.feature_transform.spec.save is not None
    ):
        fitted.feature_transform.spec.save(
            fitted.feature_transform.artifact,
            directory / "feature_transform",
        )
    if fitted.sample_mask is not None:
        np.save(directory / "sample_mask.npy", np.asarray(fitted.sample_mask, bool))


def _load_pickle_feature_transform(spec, directory):
    tpath = directory / "feature_transform" / "transform.pkl"
    if not tpath.is_file():
        return None
    with tpath.open("rb") as stream:
        artifact = pickle.load(stream)
    return FittedFeatureTransform(
        spec.feature_transform, artifact, "reloaded", {}, 0, {}
    )


def _loaded_metadata(artifact, config):
    metadata = dict(getattr(artifact, "metadata", {}) or {})
    metadata["prediction_chunk_size"] = int(
        config.get("runtime", {}).get(
            "prediction_chunk_size", _DEFAULT_PREDICTION_CHUNK_SIZE
        )
    )
    return metadata


def saved_model_complete(spec, directory) -> bool:
    directory = Path(directory)
    if not directory.is_dir() or not (directory / "sample_mask.npy").is_file():
        return False
    pickle_models = {
        "direct_linear_ridge",
        "shared_linear_ridge",
        "direct_minirocket",
        "shared_minirocket",
    }
    if spec.name in pickle_models:
        if not (directory / "model.pkl").is_file():
            return False
        if spec.name in {"direct_minirocket", "shared_minirocket"}:
            return (directory / "feature_transform" / "transform.pkl").is_file()
        return True
    return (directory / "model.pt").is_file()


def load_fitted_model(spec, directory, parameters, config):
    directory = Path(directory)
    mask = np.load(directory / "sample_mask.npy").astype(bool)
    output_limit = float(config["ml_output"]["max_abs_ps"])

    if spec.name in {
        "direct_linear_ridge",
        "shared_linear_ridge",
        "direct_minirocket",
        "shared_minirocket",
    }:
        with (directory / "model.pkl").open("rb") as stream:
            artifact = pickle.load(stream)
        if spec.name in {"direct_linear_ridge", "shared_linear_ridge"}:
            from waveform_analysis.models.linear.linear_ridge_common import (
                LinearTransformArtifact,
            )

            feature_transform = FittedFeatureTransform(
                spec.feature_transform,
                LinearTransformArtifact({"reloaded": True}),
                "reloaded",
                {},
                0,
                {},
            )
        else:
            feature_transform = _load_pickle_feature_transform(spec, directory)
        return FittedModel(
            artifact,
            _loaded_metadata(artifact, config),
            output_limit,
            mask,
            feature_transform,
        )

    import torch

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    payload = torch.load(
        directory / "model.pt",
        map_location=device,
        weights_only=False,
    )
    metadata = dict(payload.get("metadata", {}) or {})
    metadata["prediction_chunk_size"] = int(
        config.get("runtime", {}).get(
            "prediction_chunk_size", _DEFAULT_PREDICTION_CHUNK_SIZE
        )
    )
    n = int(mask.sum())
    p = dict(parameters or {})

    if spec.name == "antisymmetric_mlp":
        from waveform_analysis.engine.neural_training import MLPArtifact
        from waveform_analysis.models.neural.antisymmetric_mlp import AntisymmetricMLP

        model = AntisymmetricMLP(n, p["architecture"], p["activation"])
        artifact_type = MLPArtifact
    elif spec.name == "direct_mlp":
        from waveform_analysis.engine.neural_training import MLPArtifact
        from waveform_analysis.models.neural.direct_mlp import DirectPairMLP

        model = DirectPairMLP(n, p["architecture"], p["activation"])
        artifact_type = MLPArtifact
    elif spec.name == "locally_connected_mlp":
        from waveform_analysis.engine.neural_training import MLPArtifact
        from waveform_analysis.models.neural.locally_connected_mlp import (
            SharedLocallyConnectedScorer,
        )

        model = SharedLocallyConnectedScorer(
            n,
            p["architecture"],
            p["activation"],
            layer1_kernel_samples=p["layer1_kernel_samples"],
            layer1_stride_samples=p["layer1_stride_samples"],
            layer2_kernel_positions=p["layer2_kernel_positions"],
            layer2_stride_positions=p["layer2_stride_positions"],
            max_correction_ps=p["max_correction_ps"],
        )
        artifact_type = MLPArtifact
    elif spec.name in {"shared_cnn1d", "independent_cnn1d"}:
        from waveform_analysis.models.neural.cnn1d_common import (
            IndependentCNN1D,
            SharedCNN1D,
        )
        from waveform_analysis.engine.neural_training import MLPArtifact

        cls = SharedCNN1D if spec.name == "shared_cnn1d" else IndependentCNN1D
        model = cls(
            n,
            p["architecture"],
            p["activation"],
            conv_channels=p["conv_channels"],
            kernel_samples=p["kernel_samples"],
        )
        artifact_type = MLPArtifact
    elif spec.name == "onishi_cnn":
        from waveform_analysis.models.neural.onishi_cnn import (
            OnishiCNNArtifact,
            OnishiPairedCNN,
        )

        model = OnishiPairedCNN(config["model"]["space"].get("architecture", {}))
        with torch.no_grad():
            model(torch.zeros((1, 2, n), dtype=torch.float32))
        artifact_type = OnishiCNNArtifact
    else:
        raise ValueError(f"Inference reload is not implemented for model {spec.name}")

    model.load_state_dict(payload["state_dict"])
    model.to(device)
    model.eval()
    artifact = artifact_type(model, str(device), metadata)
    return FittedModel(artifact, metadata, output_limit, mask, None)


# Retain serialized identities while legacy imports resolve to this module.
from waveform_analysis.core.compat import (
    preserve_legacy_identity as _preserve_legacy_identity,
)

_preserve_legacy_identity(__name__, "waveform_analysis.ml_pipeline.train")
