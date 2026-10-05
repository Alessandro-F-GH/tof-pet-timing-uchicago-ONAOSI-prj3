from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np

from waveform_analysis.ml_pipeline.feature_cache import (
    build_feature_cache,
    load_feature_cache,
)
from waveform_analysis.ml_pipeline.models.spec import FeatureTransformSpec
from waveform_analysis.ml_pipeline.study_cached import _detector_swap_enabled
from waveform_analysis.ml_pipeline.train import FittedFeatureTransform


class DummyDataset:
    def __init__(self, directory: Path, n_events=8):
        self.directory = directory
        self.directory.mkdir(parents=True, exist_ok=True)
        self.manifest = {
            "analysis_protocol_identity": "protocol-A",
            "analysis_population_identity": "protocol-A",
            "event_population_identity": "population-A",
        }
        values = np.arange(n_events * 2 * 4, dtype=np.float32).reshape(n_events, 2, 4)
        self.energy_windows = values
        self.timing_windows = None
        self.energy_time_ps = np.arange(4, dtype=np.float64)
        self.timing_time_ps = None
        self.n_events = n_events


def _spec():
    def parameters(params, config):
        del config
        return {"num_kernels": int(params["num_kernels"]), "n_jobs": -1}

    def transform(artifact, values):
        x = np.asarray(values, dtype=np.float32)
        scale = float(artifact.scale)
        return np.stack(
            [x[:, 0, :].mean(axis=1), x[:, 1, :].mean(axis=1)],
            axis=1,
        ).astype(np.float32) * scale

    feature_transform = FeatureTransformSpec(
        name="direct_minirocket_multivariate",
        parameters=parameters,
        fit_transform=lambda *args, **kwargs: None,
        transform=transform,
        save=None,
    )
    return SimpleNamespace(
        feature_transform=feature_transform,
        preserve_temporal_grid=True,
    )


def test_feature_cache_is_float32_memmap_and_reusable(tmp_path):
    dataset = DummyDataset(tmp_path / "prepared")
    spec = _spec()
    model_space = {"transform": {"n_jobs": -1}}
    parameters = {"num_kernels": 10000}
    fit_indices = np.asarray([0, 2, 4], dtype=np.int64)
    transform = FittedFeatureTransform(
        spec.feature_transform,
        SimpleNamespace(scale=2.0, metadata={"feature_count": 2}),
        "expected-identity",
        {"num_kernels": 10000, "n_jobs": -1},
        123,
        {},
    )

    # Use the actual identity expected by the cache loader.
    from waveform_analysis.ml_pipeline.feature_cache import transform_identity

    identity, transform_parameters, transform_seed, _ = transform_identity(
        spec,
        model_space,
        dataset,
        "energy_to_energy",
        fit_indices,
        parameters,
        seed_base=11,
    )
    transform.identity = identity
    transform.parameters = transform_parameters
    transform.seed = transform_seed
    fit_features = transform.apply(dataset.energy_windows[fit_indices])

    config = {
        "mode": "energy_to_energy",
        "runtime": {"prediction_chunk_size": 2},
    }
    cache = build_feature_cache(
        spec,
        model_space,
        config,
        dataset,
        fit_indices,
        parameters,
        transform,
        seed_base=11,
        fit_features=fit_features,
    )
    assert isinstance(cache.features, np.memmap)
    assert cache.features.dtype == np.float32
    assert cache.features.shape == (dataset.n_events, 2)

    loaded = load_feature_cache(
        spec,
        model_space,
        dataset,
        "energy_to_energy",
        fit_indices,
        parameters,
        seed_base=11,
    )
    assert loaded is not None
    np.testing.assert_allclose(loaded.features, cache.features)


def test_detector_swap_diagnostic_is_opt_in():
    assert _detector_swap_enabled({}) is False
    assert _detector_swap_enabled({"diagnostics": {}}) is False
    assert _detector_swap_enabled({"diagnostics": {"detector_swap": True}}) is True
