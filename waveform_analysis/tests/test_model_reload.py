from __future__ import annotations

from pathlib import Path

import numpy as np

from waveform_analysis.models import get_model
from waveform_analysis.engine.train import (
    FittedModel,
    FittedFeatureTransform,
    load_fitted_model,
    save_model,
    saved_model_complete,
)


def _ridge_roundtrip(tmp_path: Path, model_name: str):
    spec = get_model(model_name)
    rng = np.random.default_rng(7)
    x = rng.normal(size=(40, 6))
    y = rng.normal(size=40)
    artifact = spec.fit({"ridge_alpha": 0.1}, x, y, seed=1, config={})
    transform_artifact = spec.feature_transform.fit_transform({}, rng.normal(size=(40, 2, 3)), seed=1, config={})[0]
    transform = FittedFeatureTransform(spec.feature_transform, transform_artifact, "id", {}, 1, {})
    fitted = FittedModel(
        artifact=artifact,
        metadata={},
        output_max_abs_ps=2000.0,
        sample_mask=np.ones(3, dtype=bool),
        feature_transform=transform,
    )
    directory = tmp_path / model_name
    save_model(spec, fitted, directory, {"ridge_alpha": 0.1})
    assert saved_model_complete(spec, directory)
    loaded = load_fitted_model(
        spec,
        directory,
        {"ridge_alpha": 0.1},
        {"ml_output": {"max_abs_ps": 2000.0}, "runtime": {"prediction_chunk_size": 32}},
    )
    assert loaded.feature_transform is not None
    np.testing.assert_array_equal(loaded.sample_mask, np.ones(3, dtype=bool))


def test_direct_linear_ridge_roundtrip(tmp_path):
    _ridge_roundtrip(tmp_path, "direct_linear_ridge")


def test_shared_linear_ridge_roundtrip(tmp_path):
    _ridge_roundtrip(tmp_path, "shared_linear_ridge")


def test_incomplete_model_is_detected(tmp_path):
    spec = get_model("direct_linear_ridge")
    directory = tmp_path / "model"
    directory.mkdir()
    np.save(directory / "sample_mask.npy", np.ones(3, dtype=bool))
    assert not saved_model_complete(spec, directory)
