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


@pytest.mark.parametrize("key", ["n_kernels", "max_shapelets"])
def test_transform_settings_must_be_fixed(key, tmp_path):
    model_space = space(NAMES[0])
    model_space["parameters"][key] = {"type": "categorical", "choices": [84, 168]}
    with pytest.raises(ValueError, match="must be fixed"):
        prepare_frozen_transform(get_model(NAMES[0]), model_space, _config(tmp_path),
                                 control_dataset=None, development_dataset=None, cache_root=tmp_path)


def test_transform_settings_must_be_fixed(key, tmp_path):
    model_space = space(NAMES[0])
    model_space["parameters"][key] = {"type": "categorical", "choices": [84, 168]}
    with pytest.raises(ValueError, match="must be fixed"):
        prepare_frozen_transform(get_model(NAMES[0]), model_space, _config(tmp_path),
                                 control_dataset=None, development_dataset=None, cache_root=tmp_path)
