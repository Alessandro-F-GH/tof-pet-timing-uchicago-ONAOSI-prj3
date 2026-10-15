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


def test_actual_multirocket_batch_grid_blind_temporal_xai_reporting_and_resume(tmp_path, monkeypatch):
    from waveform_analysis.core.config import load_batch_config
    from waveform_analysis.engine import study, batch as batch_engine
    from waveform_analysis.engine.xai_replay import PreparedCacheLookup, rerun_xai
    from waveform_analysis.data.dataset import InputTransform
    from waveform_analysis.data.storage import RunStore

    resolved = load_batch_config(ROOT / "config/batches/test_multirocket.json")
    protocol = deepcopy(resolved.protocol)
    protocol["bootstrap"]["n_resamples"] = 3
    runs = []
    # The example may sweep multiple modes/windows. Keep one timing run per
    # formulation so test output directories and mask assertions stay distinct.
    representatives = {}
    for original in resolved.runs:
        if original["mode"] == "timing_to_timing":
            representatives.setdefault(original["model"]["name"], original)
    for original in representatives.values():
        name = original["model"]["name"]
        c = deepcopy(original)
        c.update(model={"name": name, "space": space(name)}, run_id=name, name=name,
                 output_dir=str(tmp_path / name), batch_output_dir=str(tmp_path))
        c["preprocessing"]["cache_dir"] = str(tmp_path / "cache")
        c["cross_validation"].update(folds=2, minimum_events_per_fold=20)
        c["cross_validation"]["pruning"]["enabled"] = False
        c["bootstrap"]["n_resamples"] = 3
        c["xai"].update(enabled=True, max_events=8, group_size_samples=4)
        runs.append(c)
    resolved = replace(resolved, output_dir=str(tmp_path), protocol=protocol, runs=tuple(runs))
    datasets = {}
    for seed, role in enumerate(("control", "development", "blind"), 314):
        d = DummyDataset(tmp_path / role, role, n_events=512, n_samples=20)
        rng = np.random.default_rng(seed)
        d.energy_windows = rng.uniform(size=(512, 2, 20)).astype(np.float32)
        d.timing_windows = d.energy_windows
        d.timing_time_ps = d.energy_time_ps
        d.energy_target_ps = d.timing_target_ps = rng.normal(size=512) * 80
        d.energy_led_time_ps = d.timing_led_time_ps = np.column_stack([d.timing_target_ps, np.zeros(d.n_events)])
        d.energy_transform = d.timing_transform = InputTransform(np.zeros((2, 1)), np.ones((2, 1)))
        d.bias_voltage_V = np.full(d.n_events, 48.)
        d.manifest.update(dataset_role=role, led_control_mean_ps=0., sample_mask_fit_role="control")
        # Exercise the real timing mask: shared control mask, applied before MultiRocket.
        d.sample_mask = np.ones(20, dtype=bool)
        d.sample_mask[[2, 8]] = False
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
    assert fits == ["direct_multirocket", "shared_multirocket"]
    for output in outputs:
        store = RunStore(output)
        assert len(store.read_candidate_rows()) == 2
        assert (output / "tables/cv.csv").is_file()
        assert (output / "plots/xai.png").is_file()
        assert read_run_manifest(store.root)["blind_used_in_selection"] is False
        with np.load(store.xai_path) as scores:
            assert scores["replacement_method"].item() == "control_sample_mean"
            assert scores["importance_ps"].shape == ((2, 20) if output.name == "direct_multirocket" else (20,))
        with np.load(store.model_dir / "feature_transform/multirocket.npz") as exported:
            np.testing.assert_array_equal(exported["sample_indices"], np.flatnonzero(datasets["control"].sample_mask))
    assert batch_engine.run_batch(resolved) == outputs
    assert len(fits) == 2
    monkeypatch.setattr(PreparedCacheLookup, "find", lambda self, root, role, *args: datasets[role])
    weights = [(output / "model/model.pkl").read_bytes() for output in outputs]
    assert len(rerun_xai(tmp_path, max_events=8, group_size_samples=4)) == 2
    assert weights == [(output / "model/model.pkl").read_bytes() for output in outputs]
    (outputs[0] / "artifacts/pred.npz").unlink()
    study.run_study(runs[0])
    assert len(fits) == 2


@pytest.mark.parametrize("name", ["direct_multirocket", "shared_multirocket"])
def test_default_feature_budget(name):
    spec = get_model(name)
    model_space = json.loads((ROOT / f"config/model_spaces/{name}.json").read_text())
    params = {key: value["value"] for key, value in model_space["parameters"].items() if value["type"] == "fixed"}
    parameters = spec.feature_transform.parameters(params, model_space)
    control = np.random.default_rng(421).uniform(size=(4, 2, 20)).astype(np.float32)
    artifact, features = spec.feature_transform.fit_transform(parameters, control, seed=41, config=model_space)
    assert features.shape == (4, 49728)
    assert np.all(np.isfinite(features))
    np.testing.assert_array_equal(spec.feature_transform.transform(artifact, control), features)


@pytest.mark.parametrize("name", ["direct_multirocket", "shared_multirocket"])
@pytest.mark.parametrize("memory_mapped", [False, True])
def test_engine_avoids_ridge_copy_without_mutating_cache(name, memory_mapped, tmp_path):
    from sklearn.linear_model import Ridge
    from waveform_analysis.data.feature_cache import FrozenFeatureSet
    from types import SimpleNamespace

    spec, model_space = get_model(name), space(name)
    dataset = DummyDataset(tmp_path, "memory", n_events=32, n_samples=16)
    matrix = np.random.default_rng(94).normal(size=(32, 20)).astype(np.float32)
    if memory_mapped:
        np.save(tmp_path / "features.npy", matrix)
        matrix = np.load(tmp_path / "features.npy", mmap_mode="r")
    before = matrix.copy()
    features = FrozenFeatureSet(matrix, np.ones(16, bool), SimpleNamespace(
        spec=spec.feature_transform, identity="test", parameters={}, seed=1), "test", "development")
    config = {**_config(tmp_path), "ml_output": {"max_abs_ps": 2000.0}}
    candidate = spec.candidates(model_space)[0]
    fitted = fit_on_indices(spec, model_space, config, dataset, np.arange(24), candidate,
                             seed=1, frozen_features=features)
    assert fitted.artifact.regressor.copy_X is False
    np.testing.assert_array_equal(matrix, before)
    reference = Ridge(alpha=candidate["ridge_alpha"], fit_intercept=name.startswith("direct"),
                      solver="lsqr", tol=1e-4).fit(before[:24], dataset.energy_target_ps[:24])
    np.testing.assert_array_equal(fitted.artifact.regressor.coef_, reference.coef_)
    np.testing.assert_array_equal(spec.predict(fitted.artifact, matrix[24:]), reference.predict(before[24:]))
    # The public fit callback must still protect a caller's reusable matrix.
    spec.fit(candidate, matrix, dataset.energy_target_ps, seed=1, config=model_space)
    np.testing.assert_array_equal(matrix, before)


def test_prediction_batches_are_bounded_by_transform_batch_size():
    from types import SimpleNamespace
    from sklearn.linear_model import Ridge

    spec = get_model('direct_multirocket')
    matrix = np.random.default_rng(82).normal(size=(50, 20)).astype(np.float32)
    regressor = Ridge(alpha=10).fit(matrix, np.arange(50, dtype=np.float32))
    calls = []
    def rows(indices):
        calls.append(len(indices))
        return matrix[indices]
    fitted = SimpleNamespace(
        metadata={'prediction_chunk_size': 4096}, output_max_abs_ps=None,
        feature_transform=SimpleNamespace(artifact=SimpleNamespace(metadata={'batch_size': 3})),
        artifact=SimpleNamespace(regressor=regressor),
    )
    actual = predict_indices(spec, fitted, None, 'energy_to_energy', np.arange(50),
                             chunk_size=4096, frozen_features=SimpleNamespace(rows=rows))
    assert max(calls) <= 3 and sum(calls) == 50
    np.testing.assert_allclose(actual, regressor.predict(matrix), atol=1e-5)
