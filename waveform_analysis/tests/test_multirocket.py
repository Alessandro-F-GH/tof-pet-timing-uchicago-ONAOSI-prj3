"""MultiRocket reference scaling, frozen scientific roles, persistence and XAI."""
from copy import deepcopy
from dataclasses import replace
from waveform_analysis.data.results import read_run_manifest

import json
from pathlib import Path

import numpy as np
import pytest

from waveform_analysis.data.feature_cache import prepare_frozen_features, prepare_frozen_transform
from waveform_analysis.engine.train import fit_on_indices, load_fitted_model, predict_indices, save_model, saved_model_complete
from waveform_analysis.models import get_model
from waveform_analysis.models.kernel.multirocket_common import _runtime
from waveform_analysis.tests.test_feature_cache import DummyDataset, _config

ROOT = Path(__file__).resolve().parents[1]


def space(name):
    value = json.loads((ROOT / f"config/model_spaces/{name}.json").read_text())
    value["parameters"]["n_kernels"]["value"] = 84
    value["parameters"]["max_dilations_per_kernel"]["value"] = 2
    value["parameters"]["ridge_alpha"]["choices"] = [0.1, 10.0]
    value["transform"].update(n_jobs=1, batch_size=7)
    return value


def assert_numpy_state(before, after):
    assert before[0] == after[0] and before[2:] == after[2:]
    np.testing.assert_array_equal(before[1], after[1])


def test_runtime_restores_state_after_failure():
    from numba import get_num_threads
    state, threads = np.random.get_state(), get_num_threads()
    with pytest.raises(RuntimeError, match="simulated"):
        with _runtime(1):
            np.random.seed(999)
            np.random.randn(10)
            raise RuntimeError("simulated MultiRocket failure")
    assert get_num_threads() == threads
    assert_numpy_state(state, np.random.get_state())


@pytest.mark.parametrize("name", ["direct_multirocket", "shared_multirocket"])
def test_control_fit_alpha_grid_reuse_reference_reload_and_chunking(name, tmp_path, monkeypatch):
    from aeon.transformations.collection.convolution_based import MultiRocket
    from sklearn.preprocessing import StandardScaler
    from numba import get_num_threads

    spec, model_space = get_model(name), space(name)
    control = DummyDataset(tmp_path / "control", "control", n_events=20)
    development = DummyDataset(tmp_path / "development", "development", n_events=24)
    calls = []
    original_fit = MultiRocket._fit
    def fit(self, X, y=None):
        x = X
        assert y is None
        expected = control.energy_windows
        if name == "shared_multirocket":
            expected = np.concatenate([expected[:, :1], expected[:, 1:]], axis=0)
        np.testing.assert_array_equal(x, expected)
        calls.append(x.copy())
        return original_fit(self, x, y)
    monkeypatch.setattr(MultiRocket, "_fit", fit)
    config = {**_config(tmp_path), "ml_output": {"max_abs_ps": 2000.0}}
    state, threads = np.random.get_state(), get_num_threads()
    frozen = prepare_frozen_transform(spec, model_space, config, control_dataset=control,
                                      development_dataset=development, cache_root=tmp_path)
    assert frozen.fit_role == "control"
    assert get_num_threads() == threads
    assert_numpy_state(state, np.random.get_state())
    transform = frozen.transform.artifact
    # Verify actual MultiRocket features and standard scaling against the library's
    # unbatched reference on the same frozen kernels and control population.
    with _runtime(1, transform.transformer):
        raw = transform.transformer.transform(calls[0])
        scaler = StandardScaler(with_mean=False).fit(raw)
        np.testing.assert_allclose(transform.scaler.mean_, scaler.mean_, rtol=2e-6, atol=2e-6)
        np.testing.assert_allclose(transform.scaler.scale_, scaler.scale_, rtol=2e-6, atol=2e-6)
        assert int(transform.scaler.n_samples_seen_) == len(calls[0])
        pair = development.energy_windows
        pooled = np.concatenate([pair[:, :1], pair[:, 1:]], axis=0) if transform.shared else pair
        reference = scaler.transform(transform.transformer.transform(pooled))
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
    assert get_num_threads() == threads
    assert_numpy_state(state, np.random.get_state())
    if transform.shared:
        reverse = frozen.transform.apply(development.energy_windows[:, ::-1])
        np.testing.assert_allclose(spec.predict(fitted.artifact, reverse), -other_chunks, atol=1e-5)
        assert fitted.artifact.regressor.intercept_ == 0
    else:
        assert fitted.artifact.regressor.fit_intercept
    directory = tmp_path / "model"
    save_model(spec, fitted, directory, candidates[0])
    assert saved_model_complete(spec, directory)
    with np.load(directory / "feature_transform/multirocket.npz", allow_pickle=False) as saved:
        np.testing.assert_array_equal(saved["raw_parameter_0"], transform.transformer.parameter[0])
        assert saved["feature_count"].item() == features.features.shape[1]
        np.testing.assert_array_equal(saved["scaler_scale"], transform.scaler.scale_)
        assert saved["transform_identity"].item() == frozen.identity
        np.testing.assert_array_equal(saved["sample_indices"], np.arange(16))
    restored = load_fitted_model(spec, directory, candidates[0], config)
    np.testing.assert_array_equal(predict_indices(spec, restored, development, config["mode"], np.arange(24), chunk_size=5), prediction)
    with pytest.raises(ValueError, match="frozen control"):
        fit_on_indices(spec, model_space, config, development, np.arange(24), candidates[0], seed=4)


def test_multirocket_transform_parameters_must_remain_fixed(tmp_path):
    model_space = space("direct_multirocket")
    model_space["parameters"]["n_kernels"] = {"type": "categorical", "choices": [84, 168]}
    with pytest.raises(ValueError, match="must be fixed"):
        prepare_frozen_transform(get_model("direct_multirocket"), model_space, _config(tmp_path),
                                 control_dataset=None, development_dataset=None, cache_root=tmp_path)


@pytest.mark.parametrize("key,value", [("n_kernels", 83), ("max_dilations_per_kernel", 0), ("n_features_per_kernel", 3)])
def test_invalid_multirocket_parameters(key, value):
    spec, model_space = get_model("direct_multirocket"), space("direct_multirocket")
    params = {k: v["value"] for k, v in model_space["parameters"].items() if v["type"] == "fixed"}
    params[key] = value
    with pytest.raises(ValueError, match="MultiRocket"):
        spec.feature_transform.parameters(params, model_space)
