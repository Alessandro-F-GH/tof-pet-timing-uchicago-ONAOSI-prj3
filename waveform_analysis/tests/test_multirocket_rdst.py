"""Two-penalty mathematics and real control-fitted composite model workflow."""
from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path

import numpy as np
import pytest
from sklearn.linear_model import Ridge

from waveform_analysis.data.feature_cache import prepare_frozen_features, prepare_frozen_transform
from waveform_analysis.engine.train import fit_on_indices, load_fitted_model, predict_indices, save_model, saved_model_complete
from waveform_analysis.engine.xai import model_importance, model_xai_method
from waveform_analysis.models import get_model
from waveform_analysis.tests.test_feature_cache import DummyDataset, _config

ROOT = Path(__file__).resolve().parents[1]
NAMES = ("direct_multirocket_rdst", "shared_multirocket_rdst")


def space(name):
    value = json.loads((ROOT / f"config/model_spaces/{name}.json").read_text())
    for key, fixed in {"n_kernels": 84, "max_dilations_per_kernel": 2,
                       "max_shapelets": 4, "shapelet_lengths": [3, 5]}.items():
        value["parameters"][key]["value"] = fixed
    value["transform"].update(n_jobs=1, batch_size=7)
    return value


@pytest.mark.parametrize("name", NAMES)
@pytest.mark.parametrize("alphas", [(0.7, 9.0), (8.0, 0.3), (2.0, 2.0)])
def test_two_penalties_match_explicit_reference_and_protect_input(name, alphas, tmp_path):
    spec = get_model(name)
    rng = np.random.default_rng(41)
    matrix = rng.normal(size=(48, 7)).astype(np.float32)
    target = rng.normal(size=48) + (0 if name.startswith("shared_") else 4)
    np.save(tmp_path / "features.npy", matrix)
    cached = np.load(tmp_path / "features.npy", mmap_mode="r")
    config = {"_feature_transform_metadata": {
        "feature_blocks": {"multirocket": [0, 4], "rdst": [4, 7]}}}
    fitted = spec.fit(dict(zip(("multirocket_alpha", "rdst_alpha"), alphas)),
                      cached, target, seed=1, config=config)
    x = matrix.astype(np.float64)
    shared = name.startswith("shared_")
    centered = x if shared else x - x.mean(axis=0)
    y = target if shared else target - target.mean()
    penalties = np.array([alphas[0]] * 4 + [alphas[1]] * 3)
    beta = np.linalg.solve(centered.T @ centered + np.diag(penalties), centered.T @ y)
    intercept = 0 if shared else target.mean() - x.mean(axis=0) @ beta
    # The production LSQR fitter uses tol=1e-4, rather than an exact solve.
    np.testing.assert_allclose(fitted.regressor.coef_, beta, rtol=1e-3, atol=5e-4)
    np.testing.assert_allclose(spec.predict(fitted, cached), x @ beta + intercept, atol=1e-3)
    np.testing.assert_array_equal(cached, matrix)
    if alphas[0] == alphas[1]:
        reference = Ridge(alpha=alphas[0], fit_intercept=not shared, solver="lsqr", tol=1e-4).fit(matrix, target)
        np.testing.assert_array_equal(fitted.regressor.coef_, reference.coef_)


@pytest.mark.parametrize("key,value", [("multirocket_alpha", 0), ("rdst_alpha", -1),
                                       ("rdst_alpha", float("nan")), ("multirocket_alpha", float("inf"))])
def test_invalid_penalties(key, value):
    params = {"multirocket_alpha": 1., "rdst_alpha": 1., key: value}
    with pytest.raises(ValueError, match=key):
        get_model(NAMES[0]).fit(params, np.ones((3, 5)), np.ones(3), seed=1, config={})


@pytest.mark.parametrize("name", NAMES)
def test_control_only_transform_reuse_block_reference_and_reload(name, tmp_path, monkeypatch):
    from aeon.transformations.collection.convolution_based import MultiRocket
    from aeon.transformations.collection.shapelet_based import RandomDilatedShapeletTransform
    from waveform_analysis.models.kernel import multirocket_common, rdst_common

    spec, model_space = get_model(name), space(name)
    control = DummyDataset(tmp_path / "control", "control", n_events=20)
    development = DummyDataset(tmp_path / "development", "development", n_events=24)
    config = {**_config(tmp_path), "ml_output": {"max_abs_ps": 2000.}}
    calls = []
    for cls in (MultiRocket, RandomDilatedShapeletTransform):
        original = cls._fit
        def fit(self, X, y=None, _original=original, _name=cls.__name__):
            assert y is None
            expected = control.energy_windows
            if name.startswith("shared_"):
                expected = np.concatenate([expected[:, :1], expected[:, 1:]], axis=0)
            np.testing.assert_array_equal(X, expected)
            calls.append(_name)
            return _original(self, X, y)
        monkeypatch.setattr(cls, "_fit", fit)
    frozen = prepare_frozen_transform(spec, model_space, config, control_dataset=control,
                                      development_dataset=development, cache_root=tmp_path)
    assert frozen.fit_role == "control"
    features = prepare_frozen_features(spec, frozen, config, development,
                                       role="development", cache_root=tmp_path)
    artifact = frozen.transform.artifact
    expected = np.concatenate([
        multirocket_common.transform(artifact.multirocket, development.energy_windows),
        rdst_common.transform(artifact.rdst, development.energy_windows),
    ], axis=1)
    np.testing.assert_array_equal(features.features, expected)
    assert isinstance(features.features, np.memmap) and features.features.dtype == np.float32
    candidates = spec.candidates(model_space)
    assert len(candidates) == 9
    for candidate in (candidates[0], candidates[-1]):
        fitted = fit_on_indices(spec, model_space, config, development, np.arange(24),
                                candidate, seed=4, frozen_features=features)
        assert not fitted.artifact.regressor.copy_X
        np.testing.assert_array_equal(features.features, expected)
    changed = deepcopy(model_space)
    changed["parameters"]["rdst_alpha"]["choices"] = [0.1, 10.]
    reused = prepare_frozen_transform(spec, changed, config, control_dataset=control,
                                      development_dataset=development, cache_root=tmp_path)
    assert reused.identity == frozen.identity
    assert calls == ["MultiRocket", "RandomDilatedShapeletTransform"]
    prediction = predict_indices(spec, fitted, development, config["mode"], np.arange(24))
    if artifact.shared:
        reverse = frozen.transform.apply(development.energy_windows[:, ::-1])
        np.testing.assert_allclose(spec.predict(fitted.artifact, reverse), -prediction, atol=1e-5)
        assert fitted.artifact.regressor.intercept_ == 0
    directory = tmp_path / "model"
    save_model(spec, fitted, directory, candidates[-1])
    assert saved_model_complete(spec, directory)
    assert (directory / "feature_transform/multirocket/multirocket.npz").is_file()
    assert (directory / "feature_transform/rdst/shapelets.npz").is_file()
    restored = load_fitted_model(spec, directory, candidates[-1], config)
    np.testing.assert_array_equal(predict_indices(spec, restored, development, config["mode"], np.arange(24)), prediction)
    assert model_xai_method(spec) is None
    with pytest.raises(ValueError, match="XAI is unavailable"):
        model_importance(spec, fitted, development, config["mode"])


@pytest.mark.parametrize("key", ["n_kernels", "max_shapelets"])
def test_transform_settings_must_be_fixed(key, tmp_path):
    model_space = space(NAMES[0])
    model_space["parameters"][key] = {"type": "categorical", "choices": [84, 168]}
    with pytest.raises(ValueError, match="must be fixed"):
        prepare_frozen_transform(get_model(NAMES[0]), model_space, _config(tmp_path),
                                 control_dataset=None, development_dataset=None, cache_root=tmp_path)


def test_real_batch_joint_grid_blind_reporting_resume_and_xai_skip(tmp_path, monkeypatch):
    from waveform_analysis.core.config import load_batch_config
    from waveform_analysis.data.dataset import InputTransform
    from waveform_analysis.data.results import read_run_manifest
    from waveform_analysis.data.storage import RunStore
    from waveform_analysis.engine import study, batch as batch_engine
    from waveform_analysis.engine.xai_replay import PreparedCacheLookup, rerun_xai

    resolved = load_batch_config(ROOT / "config/batches/test_multirocket_rdst.json")
    runs = []
    for original in resolved.runs:
        c = deepcopy(original)
        name = c["model"]["name"]
        c.update(model={"name": name, "space": space(name)}, run_id=name, name=name,
                 output_dir=str(tmp_path / name), batch_output_dir=str(tmp_path))
        c["preprocessing"]["cache_dir"] = str(tmp_path / "cache")
        c["cross_validation"].update(folds=2, minimum_events_per_fold=20)
        c["cross_validation"]["pruning"]["enabled"] = False
        c["bootstrap"]["n_resamples"] = 3
        c["xai"]["enabled"] = True  # Unsupported model must skip even when globally enabled.
        runs.append(c)
    protocol = deepcopy(resolved.protocol)
    protocol["bootstrap"]["n_resamples"] = 3
    resolved = replace(resolved, output_dir=str(tmp_path), protocol=protocol, runs=tuple(runs))
    datasets = {}
    for seed, role in enumerate(("control", "development", "blind"), 314):
        d = DummyDataset(tmp_path / role, role, n_events=512, n_samples=20)
        rng = np.random.default_rng(seed)
        d.energy_windows = rng.normal(size=(512, 2, 20)).astype(np.float32)
        d.timing_windows = d.energy_windows
        d.timing_time_ps = d.energy_time_ps
        d.energy_target_ps = d.timing_target_ps = rng.normal(size=512) * 80
        d.energy_led_time_ps = d.timing_led_time_ps = np.column_stack([d.timing_target_ps, np.zeros(d.n_events)])
        d.energy_transform = d.timing_transform = InputTransform(np.zeros((2, 1)), np.ones((2, 1)))
        d.bias_voltage_V = np.full(d.n_events, 48.)
        d.manifest.update(dataset_role=role, led_control_mean_ps=0., sample_mask_fit_role="control")
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
    monkeypatch.setattr(study, "model_importance", lambda *a, **kw: pytest.fail("XAI must be skipped"))
    outputs = batch_engine.run_batch(resolved)
    assert fits == list(NAMES)
    for output in outputs:
        store = RunStore(output)
        assert len(store.read_candidate_rows()) == 9
        assert (output / "tables/cv.csv").is_file()
        assert not store.xai_path.exists()
        assert read_run_manifest(store.root)["blind_used_in_selection"] is False
    assert batch_engine.run_batch(resolved) == outputs
    assert len(fits) == 2
    monkeypatch.setattr(PreparedCacheLookup, "find", lambda *a, **kw: pytest.fail("Skipped XAI must not load caches"))
    assert rerun_xai(tmp_path) == []
    (outputs[0] / "artifacts/pred.npz").unlink()
    study.run_study(runs[0])
    assert len(fits) == 2
