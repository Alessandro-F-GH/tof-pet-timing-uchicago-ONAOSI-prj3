"""RDST numerical, scientific-role, persistence and XAI contracts."""
from copy import deepcopy
from dataclasses import replace
from waveform_analysis.data.results import read_run_manifest

import json
from pathlib import Path

import numpy as np
import pytest

from waveform_analysis.models import get_model
from waveform_analysis.data.feature_cache import prepare_frozen_transform, prepare_frozen_features
from waveform_analysis.engine.train import fit_on_indices, load_fitted_model, predict_indices, save_model, saved_model_complete
from waveform_analysis.tests.test_feature_cache import DummyDataset, _config

ROOT = Path(__file__).resolve().parents[1]


def space(name):
    value = json.loads((ROOT / f"config/model_spaces/{name}.json").read_text())
    value["parameters"]["max_shapelets"]["value"] = 8
    value["parameters"]["shapelet_lengths"]["value"] = [3, 5]
    value["parameters"]["ridge_alpha"]["choices"] = [0.1, 10.0]
    value["transform"]["n_jobs"] = 1
    return value


@pytest.mark.parametrize("name", ["direct_rdst", "shared_rdst"])
def test_control_fitting_grid_reuse_and_reload(name, tmp_path, monkeypatch):
    from aeon.transformations.collection.shapelet_based import RandomDilatedShapeletTransform
    spec, model_space = get_model(name), space(name)
    control = DummyDataset(tmp_path / "control", "control", n_events=20)
    development = DummyDataset(tmp_path / "development", "development", n_events=24)
    calls = []
    original = RandomDilatedShapeletTransform._fit

    def fit(self, x, y=None):
        assert y is None
        expected = control.energy_windows
        if name == "shared_rdst":
            expected = np.concatenate([expected[:, :1], expected[:, 1:]], axis=0)
        np.testing.assert_array_equal(x, expected)
        calls.append(len(x))
        return original(self, x, y)

    monkeypatch.setattr(RandomDilatedShapeletTransform, "_fit", fit)
    config = {**_config(tmp_path), "ml_output": {"max_abs_ps": 2000.0}}
    frozen = prepare_frozen_transform(spec, model_space, config, control_dataset=control,
                                      development_dataset=development, cache_root=tmp_path)
    assert frozen.fit_role == "control"
    features = prepare_frozen_features(spec, frozen, config, development,
                                       role="development", cache_root=tmp_path)
    assert features.features.shape[1] == 3 * frozen.transform.artifact.transformer.n_shapelets_
    candidates = spec.candidates(model_space)
    assert len(candidates) == 2
    fitted = fit_on_indices(spec, model_space, config, development, np.arange(24), candidates[0],
                             seed=4, frozen_features=features)
    # Changing alpha reuses the control transform and does not refit its scaler.
    fit_on_indices(spec, model_space, config, development, np.arange(24), candidates[1],
                   seed=4, frozen_features=features)
    again = prepare_frozen_transform(spec, model_space, config, control_dataset=control,
                                     development_dataset=development, cache_root=tmp_path)
    assert again.identity == frozen.identity and len(calls) == 1
    expected = predict_indices(spec, fitted, development, config["mode"], np.arange(24))
    if name == "shared_rdst":
        reverse = frozen.transform.apply(development.energy_windows[:, ::-1])
        np.testing.assert_allclose(spec.predict(fitted.artifact, reverse), -expected, atol=1e-6)
        assert fitted.artifact.regressor.intercept_ == 0
    beta = np.asarray(fitted.artifact.regressor.coef_, dtype=np.float64).reshape(-1, 3)
    assert beta.shape[1] == 3
    assert np.all(np.isfinite(beta))
    directory = tmp_path / "model"
    save_model(spec, fitted, directory, candidates[0])
    assert saved_model_complete(spec, directory)
    with np.load(directory / "feature_transform/shapelets.npz", allow_pickle=False) as saved:
        assert saved["feature_columns"].size > 0
        np.testing.assert_array_equal(saved["sample_indices"], np.arange(16))
    restored = load_fitted_model(spec, directory, candidates[0], config)
    np.testing.assert_array_equal(predict_indices(spec, restored, development, config["mode"], np.arange(24)), expected)
    with pytest.raises(ValueError, match="frozen control"):
        fit_on_indices(spec, model_space, config, development, np.arange(24), candidates[0], seed=4)


def test_transform_parameters_must_remain_fixed(tmp_path):
    model_space = space("direct_rdst")
    model_space["parameters"]["max_shapelets"] = {"type": "categorical", "choices": [8, 16]}
    with pytest.raises(ValueError, match="must be fixed"):
        prepare_frozen_transform(get_model("direct_rdst"), model_space, _config(tmp_path),
                                 control_dataset=None, development_dataset=None, cache_root=tmp_path)


@pytest.mark.parametrize("parameter,value", [
    ("max_shapelets", 0), ("shapelet_lengths", [1]),
    ("proba_normalization", 1.1), ("threshold_percentiles", [10, 5]),
    ("alpha_similarity", float("nan")), ("use_prime_dilations", "false"),
])
def test_invalid_transform_parameters(parameter, value):
    spec = get_model("direct_rdst")
    model_space = space("direct_rdst")
    params = {k: v["value"] for k, v in model_space["parameters"].items() if v["type"] == "fixed"}
    params[parameter] = value
    with pytest.raises(ValueError, match="RDST"):
        spec.feature_transform.parameters(params, model_space)
