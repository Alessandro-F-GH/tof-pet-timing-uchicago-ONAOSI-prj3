"""Hydra reference scaling, frozen scientific roles, persistence and XAI."""
from copy import deepcopy
from dataclasses import replace
from waveform_analysis.data.results import read_run_manifest

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from waveform_analysis.data.feature_cache import prepare_frozen_features, prepare_frozen_transform
from waveform_analysis.engine.train import fit_on_indices, load_fitted_model, predict_indices, save_model, saved_model_complete
from waveform_analysis.models import get_model
from waveform_analysis.models.kernel.hydra_common import HydraSparseScaler, _torch_runtime
from waveform_analysis.tests.test_feature_cache import DummyDataset, _config

ROOT = Path(__file__).resolve().parents[1]


def space(name):
    value = json.loads((ROOT / f"config/model_spaces/{name}.json").read_text())
    value["parameters"]["n_kernels"]["value"] = 4
    value["parameters"]["n_groups"]["value"] = 4
    value["parameters"]["ridge_alpha"]["choices"] = [0.1, 10.0]
    value["transform"].update(n_jobs=1, batch_size=7)
    return value


def test_sparse_scaler_matches_reference_and_preserves_zeros():
    from aeon.classification.convolution_based._hydra import _SparseScaler

    control = np.array([[0, -3, 4, 1], [0, 0, 9, 1], [0, 4, 0, 1]], dtype=np.float32)
    explained = np.array([[0, -1, 16, 1], [0, 9, 0, 4]], dtype=np.float32)
    reference = _SparseScaler()
    reference.fit(torch.from_numpy(control))
    scaler = HydraSparseScaler().fit(control)
    np.testing.assert_array_equal(scaler.mu, reference.mu.numpy())
    np.testing.assert_array_equal(scaler.sigma, reference.sigma.numpy())
    np.testing.assert_array_equal(scaler.epsilon, reference.epsilon.numpy())
    actual = scaler.transform(explained)
    np.testing.assert_array_equal(actual, reference.transform(torch.from_numpy(explained)).numpy())
    assert np.all(np.isfinite(actual))
    assert np.all(actual[explained <= 0] == 0)
    assert scaler.transform(control)[:, 3].tolist() == [0, 0, 0]
    with pytest.raises(ValueError, match="two control"):
        HydraSparseScaler().fit(control[:1])


def test_torch_runtime_restores_state_after_failure():
    state, threads = torch.get_rng_state().clone(), torch.get_num_threads()
    with pytest.raises(RuntimeError, match="simulated"):
        with _torch_runtime(1):
            torch.manual_seed(999)
            torch.randn(10)
            raise RuntimeError("simulated Hydra failure")
    assert torch.get_num_threads() == threads
    assert torch.equal(torch.get_rng_state(), state)


@pytest.mark.parametrize("name", ["direct_hydra", "shared_hydra"])
def test_control_fit_alpha_grid_reuse_reference_reload_and_chunking(name, tmp_path, monkeypatch):
    from aeon.transformations.collection.convolution_based import HydraTransformer
    from aeon.classification.convolution_based._hydra import _SparseScaler

    spec, model_space = get_model(name), space(name)
    control = DummyDataset(tmp_path / "control", "control", n_events=20)
    development = DummyDataset(tmp_path / "development", "development", n_events=24)
    calls = []
    original_fit = HydraTransformer._fit
    def fit(self, X, y=None):
        x = X
        assert y is None
        expected = control.energy_windows
        if name == "shared_hydra":
            expected = np.concatenate([expected[:, :1], expected[:, 1:]], axis=0)
        np.testing.assert_array_equal(x, expected)
        calls.append(x.copy())
        return original_fit(self, x, y)
    monkeypatch.setattr(HydraTransformer, "_fit", fit)
    config = {**_config(tmp_path), "ml_output": {"max_abs_ps": 2000.0}}
    state, threads = torch.get_rng_state().clone(), torch.get_num_threads()
    frozen = prepare_frozen_transform(spec, model_space, config, control_dataset=control,
                                      development_dataset=development, cache_root=tmp_path)
    assert frozen.fit_role == "control"
    assert torch.get_num_threads() == threads and torch.equal(torch.get_rng_state(), state)
    transform = frozen.transform.artifact
    # Verify actual Hydra features and sparse scaling against the library's
    # unbatched reference on the same frozen kernels and control population.
    with _torch_runtime(1):
        raw = transform.transformer.transform(calls[0])
        scaler = _SparseScaler()
        scaler.fit(raw)
        np.testing.assert_allclose(transform.scaler.mu, scaler.mu.numpy(), rtol=2e-6, atol=2e-6)
        pair = development.energy_windows
        pooled = np.concatenate([pair[:, :1], pair[:, 1:]], axis=0) if transform.shared else pair
        reference = scaler.transform(transform.transformer.transform(pooled)).numpy()
        if transform.shared:
            reference = reference[:len(pair)] - reference[len(pair):]
    features = prepare_frozen_features(spec, frozen, config, development,
                                       role="development", cache_root=tmp_path)
    np.testing.assert_allclose(features.features, reference, rtol=2e-5, atol=2e-5)
    candidates = spec.candidates(model_space)
    assert len(candidates) == 2
    fitted = fit_on_indices(spec, model_space, config, development, np.arange(24), candidates[0],
                             seed=4, frozen_features=features)
    fit_on_indices(spec, model_space, config, development, np.arange(24), candidates[1],
                   seed=4, frozen_features=features)
    reused = prepare_frozen_transform(spec, model_space, config, control_dataset=control,
                                      development_dataset=development, cache_root=tmp_path)
    assert reused.identity == frozen.identity and len(calls) == 1
    prediction = predict_indices(spec, fitted, development, config["mode"], np.arange(24), chunk_size=5)
    other_chunks = predict_indices(spec, fitted, development, config["mode"], np.arange(24), chunk_size=24)
    np.testing.assert_allclose(prediction, other_chunks, atol=1e-5, rtol=1e-5)
    assert torch.get_num_threads() == threads and torch.equal(torch.get_rng_state(), state)
    if transform.shared:
        reverse = frozen.transform.apply(development.energy_windows[:, ::-1])
        np.testing.assert_allclose(spec.predict(fitted.artifact, reverse), -other_chunks, atol=1e-5)
        assert fitted.artifact.regressor.intercept_ == 0
    else:
        assert fitted.artifact.regressor.fit_intercept
    directory = tmp_path / "model"
    save_model(spec, fitted, directory, candidates[0])
    assert saved_model_complete(spec, directory)
    with np.load(directory / "feature_transform/hydra.npz", allow_pickle=False) as saved:
        np.testing.assert_array_equal(saved["weights"], transform.transformer._hydra.W.numpy())
        assert len(saved["feature_column"]) == features.features.shape[1]
        assert saved["transform_identity"].item() == frozen.identity
        np.testing.assert_array_equal(saved["sample_indices"], np.arange(16))
    restored = load_fitted_model(spec, directory, candidates[0], config)
    np.testing.assert_array_equal(predict_indices(spec, restored, development, config["mode"], np.arange(24), chunk_size=5), prediction)
    with pytest.raises(ValueError, match="frozen control"):
        fit_on_indices(spec, model_space, config, development, np.arange(24), candidates[0], seed=4)


def test_hydra_transform_parameters_must_remain_fixed(tmp_path):
    model_space = space("direct_hydra")
    model_space["parameters"]["n_groups"] = {"type": "categorical", "choices": [4, 8]}
    with pytest.raises(ValueError, match="must be fixed"):
        prepare_frozen_transform(get_model("direct_hydra"), model_space, _config(tmp_path),
                                 control_dataset=None, development_dataset=None, cache_root=tmp_path)


@pytest.mark.parametrize("key,value", [("n_kernels", 0), ("n_groups", 3), ("max_num_channels", -1)])
def test_invalid_hydra_parameters(key, value):
    spec, model_space = get_model("direct_hydra"), space("direct_hydra")
    params = {k: v["value"] for k, v in model_space["parameters"].items() if v["type"] == "fixed"}
    params[key] = value
    with pytest.raises(ValueError, match="Hydra"):
        spec.feature_transform.parameters(params, model_space)
