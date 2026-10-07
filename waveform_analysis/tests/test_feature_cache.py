from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from waveform_analysis.ml_pipeline.feature_cache import (
    prepare_frozen_features,
    prepare_frozen_transform,
)
from waveform_analysis.ml_pipeline.models import get_model


class DummyDataset:
    def __init__(self, directory, identity, n_events=12, n_samples=16):
        self.directory = directory
        self.manifest = {
            "analysis_protocol_identity": identity,
            "analysis_population_identity": identity,
            "event_population_identity": identity,
            "dataset_source": f"{identity}.root",
        }
        rng = np.random.default_rng(abs(hash(identity)) % (2**32))
        self.energy_windows = rng.normal(size=(n_events, 2, n_samples)).astype(np.float32)
        self.timing_windows = None
        self.energy_time_ps = np.arange(n_samples, dtype=np.float64)
        self.timing_time_ps = None
        self.energy_target_ps = rng.normal(size=n_events)
        self.timing_target_ps = None
        self.event_index = np.arange(n_events, dtype=np.int64)
        self.n_events = n_events


def _config(tmp_path):
    return {
        "mode": "energy_to_energy",
        "window_name": "test",
        "seed": 1001,
        "runtime": {"prediction_chunk_size": 5},
        "preprocessing": {"cache_dir": str(tmp_path)},
    }


def test_shared_linear_difference_is_materialized_once_per_dataset(tmp_path):
    spec = get_model("shared_linear_ridge")
    space = {
        "parameters": {
            "ridge_alpha": {"type": "float", "low": 1e-6, "high": 10.0, "log": True}
        }
    }
    control = DummyDataset(tmp_path / "control", "control")
    development = DummyDataset(tmp_path / "development", "development")
    config = _config(tmp_path)

    frozen = prepare_frozen_transform(
        spec,
        space,
        config,
        control_dataset=control,
        development_dataset=development,
        cache_root=tmp_path,
    )
    features = prepare_frozen_features(
        spec,
        frozen,
        config,
        development,
        role="development",
        cache_root=tmp_path,
    )
    expected = development.energy_windows[:, 0, :] - development.energy_windows[:, 1, :]
    np.testing.assert_allclose(features.features, expected, rtol=1e-6, atol=1e-6)
    assert frozen.fit_role == "deterministic"

    reused = prepare_frozen_features(
        spec,
        frozen,
        config,
        development,
        role="development",
        cache_root=tmp_path,
    )
    assert reused.identity == features.identity


def test_minirocket_transform_is_defined_as_control_fitted():
    # Avoid importing the optional sktime dependency; this checks the protocol
    # contract through the registered model and fixed transform parameters.
    spec = get_model("direct_minirocket")
    assert spec.feature_transform.name == "direct_minirocket_multivariate"
    assert spec.feature_transform.parameters(
        {"num_kernels": 10000},
        {"transform": {"n_jobs": -1}},
    ) == {"num_kernels": 10000, "n_jobs": -1}
