"""Hydra reference scaling, frozen scientific roles, persistence and XAI."""
from copy import deepcopy
from dataclasses import replace
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


def test_actual_hydra_batch_grid_blind_temporal_xai_reporting_and_resume(tmp_path, monkeypatch):
    from waveform_analysis.core.config import load_batch_config
    from waveform_analysis.engine import study, batch as batch_engine
    from waveform_analysis.engine.xai_replay import PreparedCacheLookup, rerun_xai
    from waveform_analysis.data.dataset import InputTransform
    from waveform_analysis.data.storage import RunStore

    resolved = load_batch_config(ROOT / "config/batches/test_hydra.json")
    protocol = deepcopy(resolved.protocol)
    protocol["bootstrap"]["n_resamples"] = 3
    runs = []
    for original in resolved.runs:
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
        # Exercise the real timing mask: shared control mask, applied before Hydra.
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
    assert fits == ["direct_hydra", "shared_hydra"]
    for output in outputs:
        store = RunStore(output)
        assert len(store.read_candidate_rows()) == 2
        assert (output / "tables/cv.csv").is_file()
        assert (output / "plots/xai.png").is_file()
        assert json.loads(store.manifest_path.read_text())["blind_used_in_selection"] is False
        with np.load(store.xai_path) as scores:
            assert scores["replacement_method"].item() == "control_sample_mean"
            assert scores["importance_ps"].shape == ((2, 20) if output.name == "direct_hydra" else (20,))
        with np.load(store.model_dir / "feature_transform/hydra.npz") as exported:
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
