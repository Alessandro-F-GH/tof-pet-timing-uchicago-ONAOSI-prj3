from __future__ import annotations

from pathlib import Path

import numpy as np

from waveform_analysis.models import get_model
from waveform_analysis.data.storage import RunStore
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



def test_run_store_pins_model_outside_result_tree(tmp_path):
    spec = get_model("direct_linear_ridge")
    rng = np.random.default_rng(19)
    artifact = spec.fit(
        {"ridge_alpha": 0.1},
        rng.normal(size=(24, 6)),
        rng.normal(size=24),
        seed=5,
        config={},
    )
    fitted = FittedModel(
        artifact=artifact,
        metadata={"training_events": 24},
        output_max_abs_ps=2000.0,
        sample_mask=np.ones(3, dtype=bool),
        feature_transform=None,
    )
    run = RunStore(
        tmp_path / "results" / "run",
        artifact_root=tmp_path / "artifact_store",
    )
    reference = run.save_model_artifact(
        spec, fitted, {"ridge_alpha": 0.1},
        final_fit_fingerprint="a" * 64,
    )
    assert run.model_dir == reference.path
    assert reference.path.is_dir()
    assert not (run.root / "model").exists()
    assert run.model_reference_path.is_file()
    assert (reference.path / "model.pkl").is_file()


def test_saved_model_requires_its_bundled_feature_transform(tmp_path):
    spec = get_model("direct_minirocket")
    directory = tmp_path / "incomplete"
    directory.mkdir()
    (directory / "model.pkl").write_bytes(b"placeholder")
    np.save(directory / "sample_mask.npy", np.ones(3, dtype=bool))
    assert not saved_model_complete(spec, directory)

    # A model saved through the normal artifact path is self-contained.
    import pickle
    transform_artifact = spec.feature_transform.fit_transform(
        {"num_kernels": 84, "n_jobs": 1},
        np.random.default_rng(31).normal(size=(12, 2, 16)).astype(np.float32),
        seed=9, config={},
    )[0]
    from waveform_analysis.engine.train import FittedFeatureTransform
    model_artifact = spec.fit(
        {"ridge_alpha": 0.1},
        np.random.default_rng(32).normal(size=(12, 84)).astype(np.float32),
        np.random.default_rng(33).normal(size=12),
        seed=9, config={},
    )
    fitted = FittedModel(
        artifact=model_artifact, metadata={}, output_max_abs_ps=2000.0,
        sample_mask=np.ones(16, dtype=bool),
        feature_transform=FittedFeatureTransform(
            spec.feature_transform, transform_artifact, "transform-id", {}, 9, {}
        ),
    )
    save_model(spec, fitted, directory, {"ridge_alpha": 0.1})
    assert (directory / "feature_transform" / "transform.pkl").is_file()
    assert saved_model_complete(spec, directory)
    loaded = load_fitted_model(
        spec, directory, {"ridge_alpha": 0.1},
        {"ml_output": {"max_abs_ps": 2000.0}, "runtime": {"prediction_chunk_size": 32}},
    )
    assert loaded.feature_transform is not None
