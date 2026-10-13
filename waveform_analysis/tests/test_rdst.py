"""RDST numerical, scientific-role, persistence and XAI contracts."""
from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path

import numpy as np
import pytest

from waveform_analysis.models import get_model
from waveform_analysis.data.feature_cache import prepare_frozen_transform, prepare_frozen_features
from waveform_analysis.engine.train import fit_on_indices, load_fitted_model, predict_indices, save_model, saved_model_complete
from waveform_analysis.engine.shapelet_xai import coefficient_importance
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
def test_control_fitting_grid_reuse_reload_and_xai(name, tmp_path, monkeypatch):
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
    scores = coefficient_importance(fitted)
    beta = np.asarray(fitted.artifact.regressor.coef_, dtype=np.float64).reshape(-1, 3)
    np.testing.assert_array_equal(scores["shapelet_importance_ps"], np.abs(beta).sum(axis=1))
    np.testing.assert_allclose(scores["coefficients_raw"] * scores["control_feature_scale"], beta)
    assert np.all(np.diff(scores["shapelet_importance_ps"][scores["ranking"]]) <= 0)
    directory = tmp_path / "model"
    save_model(spec, fitted, directory, candidates[0])
    assert saved_model_complete(spec, directory)
    with np.load(directory / "feature_transform/shapelets.npz", allow_pickle=False) as saved:
        np.testing.assert_array_equal(saved["feature_columns"], scores["feature_columns"])
        np.testing.assert_array_equal(saved["sample_indices"], np.arange(16))
    restored = load_fitted_model(spec, directory, candidates[0], config)
    np.testing.assert_array_equal(predict_indices(spec, restored, development, config["mode"], np.arange(24)), expected)
    np.testing.assert_array_equal(coefficient_importance(restored)["ranking"], scores["ranking"])
    with pytest.raises(ValueError, match="frozen control"):
        fit_on_indices(spec, model_space, config, development, np.arange(24), candidates[0], seed=4)


def test_transform_parameters_must_remain_fixed(tmp_path):
    model_space = space("direct_rdst")
    model_space["parameters"]["max_shapelets"] = {"type": "categorical", "choices": [8, 16]}
    with pytest.raises(ValueError, match="must be fixed"):
        prepare_frozen_transform(get_model("direct_rdst"), model_space, _config(tmp_path),
                                 control_dataset=None, development_dataset=None, cache_root=tmp_path)


def test_shapelet_plots_use_masked_time_and_scaled_coefficients(tmp_path):
    from waveform_analysis.reporting.shapelets import plot_shapelets
    from waveform_analysis.core.config import load_batch_config
    cfg = load_batch_config(ROOT / "config/batches/test_ridge.json").runs[0]["plot_config"]
    arrays = {
        "ranking": np.array([0]), "rank": np.array([1]), "feature_names": np.array(["distance", "location", "count"]),
        "shapelet_importance_ps": np.array([6.0]), "lengths": np.array([3]), "dilations": np.array([2]),
        "startpoints": np.array([1]), "normalizes": np.array([True]), "thresholds": np.array([0.5]),
        "coefficients_scaled_ps": np.array([[1., -2., 3.]]), "coefficients_raw": np.array([[0.1, -1., 3.]]),
        "control_feature_scale": np.array([[10., 2., 1.]]), "time_ps": np.array([0., 10., 30., 50., 80., 100.]),
        "values": np.array([[[1., 0., -1.], [0., 1., 0.]]]), "estimator_formulation": np.array("direct"),
    }
    output = plot_shapelets(tmp_path, arrays, cfg)
    assert output.is_file()
    assert "coefficient_scaled_ps" in (tmp_path / "tables/shapelet_importance.csv").read_text()


def test_actual_shapelet_batch_grid_blind_xai_reporting_resume_and_replay(tmp_path, monkeypatch):
    from waveform_analysis.core.config import load_batch_config
    from waveform_analysis.engine import study, batch as batch_engine
    from waveform_analysis.engine.xai_replay import rerun_xai, PreparedCacheLookup
    from waveform_analysis.data.dataset import InputTransform
    from waveform_analysis.data.storage import RunStore

    resolved = load_batch_config(ROOT / "config/batches/benchmark_FBK.json")
    runs = []
    for name in ("direct_rdst", "shared_rdst"):
        c = deepcopy(resolved.runs[0])
        c.update(model={"name": name, "space": space(name)}, run_id=name, name=name,
                 output_dir=str(tmp_path / name), batch_output_dir=str(tmp_path))
        c["preprocessing"]["cache_dir"] = str(tmp_path / "cache")
        c["cross_validation"].update(folds=2, minimum_events_per_fold=20)
        c["cross_validation"]["pruning"]["enabled"] = False
        c["bootstrap"]["n_resamples"] = 3
        c["xai"]["enabled"] = True
        c["save_model"] = True
        runs.append(c)
    protocol = deepcopy(resolved.protocol)
    protocol["bootstrap"]["n_resamples"] = 3
    resolved = replace(resolved, output_dir=str(tmp_path), protocol=protocol, runs=tuple(runs))
    datasets = {}
    for seed, role in enumerate(("control", "development", "blind"), 314):
        d = DummyDataset(tmp_path / role, role, n_events=512, n_samples=16)
        rng = np.random.default_rng(seed)
        d.energy_windows = rng.normal(size=(512, 2, 16)).astype(np.float32)
        d.energy_target_ps = rng.normal(size=512)
        d.timing_windows = d.energy_windows
        d.timing_time_ps = d.energy_time_ps
        d.timing_target_ps = d.energy_target_ps * 80
        d.timing_led_time_ps = np.column_stack([d.timing_target_ps, np.zeros(d.n_events)])
        d.energy_led_time_ps = d.timing_led_time_ps
        d.energy_target_ps = d.timing_target_ps
        d.energy_transform = d.timing_transform = InputTransform(np.zeros((2, 1)), np.ones((2, 1)))
        d.bias_voltage_V = np.full(d.n_events, 48.)
        d.manifest.update(led_control_mean_ps=0., sample_mask_fit_role="control")
        d.sample_mask = np.ones(16, dtype=bool)
        datasets[role] = d
    fits = []
    original_fit = study._fit_final
    def fit(*args, **kwargs):
        fits.append(args[0].name)
        return original_fit(*args, **kwargs)
    monkeypatch.setattr(study, "_fit_final", fit)
    def prepare(config, role, *args, **kwargs):
        if role == "blind":
            assert config["model"]["name"] in fits
        return None, None, datasets[role]
    monkeypatch.setattr(study, "prepare_role_dataset", prepare)
    for module in (study, batch_engine):
        monkeypatch.setattr(module, "fit_control", lambda *a, **kw: ({}, tmp_path / "control"))
        monkeypatch.setattr(module, "publish_preprocessing_diagnostics", lambda *a, **kw: None)
    outputs = batch_engine.run_batch(resolved)
    assert fits == ["direct_rdst", "shared_rdst"]
    for output in outputs:
        store = RunStore(output)
        assert len(store.read_candidate_rows()) == 2
        assert (output / "tables/cv.csv").is_file()
        assert (output / "tables/shapelet_importance.csv").is_file()
        assert list((output / "plots/shapelets").glob("*.png"))
        assert json.loads(store.manifest_path.read_text())["blind_used_in_selection"] is False
        with np.load(store.xai_path) as scores:
            assert scores["method"].item() == "shapelet_coefficients"
    assert batch_engine.run_batch(resolved) == outputs
    assert len(fits) == 2
    # Replay must load the saved transform, not refit or even require prepared caches.
    monkeypatch.setattr(PreparedCacheLookup, "find", lambda *a, **kw: pytest.fail("Coefficient XAI must not load acquisitions/caches"))
    weights = [(output / "model/model.pkl").read_bytes() for output in outputs]
    assert len(rerun_xai(tmp_path)) == 2
    assert weights == [(output / "model/model.pkl").read_bytes() for output in outputs]
    (outputs[0] / "artifacts/pred.npz").unlink()
    study.run_study(runs[0])
    assert len(fits) == 2


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
