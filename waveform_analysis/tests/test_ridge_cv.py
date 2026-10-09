"""Sklearn equivalence and batch/stage contracts for linear RidgeCV."""

from copy import deepcopy
import json
import logging
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from sklearn.linear_model import RidgeCV

from waveform_analysis.core.config import ConfigError, load_batch_config
from waveform_analysis.core.ridge import ridge_cv_config, normalize_ridge_space
from waveform_analysis.data.storage import RunStore
from waveform_analysis.engine.batch import _planner_fingerprints
from waveform_analysis.engine import study
from waveform_analysis.models import get_model
from waveform_analysis.tests.test_batch_planner import config as planner_config


@pytest.mark.parametrize(
    "name,intercept", [("direct_linear_ridge", True), ("shared_linear_ridge", False)]
)
@pytest.mark.parametrize("cv", [None, 3])
def test_matches_sklearn_and_persisted_predictions(name, intercept, cv, tmp_path):
    rng = np.random.default_rng(847)
    pair = rng.normal(size=(80, 2, 8)).astype(np.float32)
    y = rng.normal(size=80) * 30
    spec = get_model(name)
    transform, x = spec.feature_transform.fit_transform({}, pair, seed=1, config={})
    config = {
        "ridge_cv": {"alphas": [0.001, 0.1, 10, 100], "cv": cv, "gcv_mode": "auto"}
    }
    artifact = spec.fit({"ridge_alpha": 999}, x, y, seed=1, config=config)
    reference = RidgeCV(
        alphas=(0.001, 0.1, 10, 100),
        cv=cv,
        fit_intercept=intercept,
        scoring="neg_mean_squared_error",
        gcv_mode="auto",
    ).fit(np.asarray(x, dtype=np.float64), y)
    np.testing.assert_array_equal(artifact.regressor.coef_, reference.coef_)
    np.testing.assert_array_equal(artifact.regressor.intercept_, reference.intercept_)
    np.testing.assert_array_equal(
        spec.predict(artifact, x), reference.predict(np.asarray(x, dtype=np.float64))
    )
    assert artifact.metadata["ridge_alpha"] == reference.alpha_
    assert artifact.metadata["internal_cv_mse_ps2"] == -reference.best_score_
    assert artifact.metadata["training_events"] == len(y)
    spec.save(artifact, tmp_path)
    import pickle

    with (tmp_path / "model.pkl").open("rb") as stream:
        restored = pickle.load(stream)
    np.testing.assert_array_equal(spec.predict(restored, x), spec.predict(artifact, x))
    if not intercept:
        swapped = spec.feature_transform.transform(transform, pair[:, ::-1])
        np.testing.assert_array_equal(
            spec.predict(artifact, swapped), -spec.predict(artifact, x)
        )


@pytest.mark.parametrize("values", [[], [0], [-1], [float("nan")], [float("inf")]])
def test_invalid_alpha_grid(values):
    with pytest.raises(ValueError):
        ridge_cv_config({"ridge_cv": {"alphas": values}})


def test_legacy_space_is_migrated_without_outer_optimization():
    space = {
        "parameters": {
            "ridge_alpha": {"type": "float", "low": 0.01, "high": 100, "log": True}
        },
        "optimization": {"method": "optuna", "n_trials": 100},
    }
    effective = normalize_ridge_space(space, "direct_linear_ridge")
    np.testing.assert_array_equal(
        effective["ridge_cv"]["alphas"], np.geomspace(0.01, 100, 23)
    )
    assert "optimization" not in effective


def test_ridge_only_batch_does_not_require_outer_cv(tmp_path):
    source = Path(__file__).resolve().parents[1] / "config/batches/test_ridge.json"
    raw = json.loads(source.read_text())
    # Resolve relative includes before placing the modified configuration elsewhere.
    raw["control_dataset"] = str((source.parent / raw["control_dataset"]).resolve())
    raw["protocol"]["preprocessing_config"] = str(
        (source.parent / raw["protocol"]["preprocessing_config"]).resolve()
    )
    raw["protocol"].pop("cross_validation", None)
    path = tmp_path / "batch.json"
    path.write_text(json.dumps(raw))
    batch = load_batch_config(path)
    assert batch.protocol["cross_validation"] is None
    raw["sweep"]["models"].append("direct_mlp")
    path.write_text(json.dumps(raw))
    with pytest.raises(ConfigError, match="cross_validation"):
        load_batch_config(path)


def test_planner_ignores_unused_outer_cv_but_tracks_alpha_grid(tmp_path):
    c = planner_config(tmp_path)
    c["model"] = {
        "name": "direct_linear_ridge",
        "space": normalize_ridge_space({}, "direct_linear_ridge"),
    }
    before = _planner_fingerprints(c)
    c["cross_validation"] = {"folds": 500, "pruning": {"enabled": True}}
    assert _planner_fingerprints(c) == before
    c["model"]["space"]["ridge_cv"]["alphas"] = [1, 2]
    assert _planner_fingerprints(c)["cv"] != before["cv"]


@pytest.mark.parametrize("name", ["direct_linear_ridge", "shared_linear_ridge"])
def test_full_development_selection_and_resume_skips_outer_search(
    name, tmp_path, monkeypatch
):
    spec = get_model(name)
    rng = np.random.default_rng(90)
    x = rng.normal(size=(100, 8))
    y = rng.normal(size=100)
    store = RunStore(tmp_path)
    config = {"model": {"name": name, "space": {}}, "seed": 1, "save_model": False}
    space = {"ridge_cv": {"alphas": [0.1, 1, 10]}}
    config["model"]["space"] = space
    development = SimpleNamespace(
        n_events=100, manifest={"analysis_protocol_identity": "development-only"}
    )
    calls = []

    def fit_final(spec, space, config, data, best, logger, frozen):
        assert data is development
        calls.append(data.n_events)
        return SimpleNamespace(artifact=spec.fit({}, x, y, seed=1, config=space)), 1

    monkeypatch.setattr(study, "_fit_final", fit_final)
    logger = logging.getLogger("ridge-test")
    first = study._ridge_cv_selection_and_fit(
        store, spec, space, config, development, logger
    )
    second = study._ridge_cv_selection_and_fit(
        store, spec, space, config, development, logger
    )
    assert calls == [100]
    assert first[3] == second[3]
    assert first[3]["n_train"] == 100
    assert first[3]["validation_ctr_mean_ps"] is None
    assert first[3]["folds"] == 0
    assert not (tmp_path / "tables/cv.csv").exists()
    changed = deepcopy(config)
    changed["model"]["space"]["ridge_cv"]["alphas"] = [20, 30]
    study._ridge_cv_selection_and_fit(
        store, spec, changed["model"]["space"], changed, development, logger
    )
    assert calls == [100, 100]


@pytest.mark.parametrize("mixed", [False, True])
def test_actual_ridge_batch_blind_reporting_and_resume(tmp_path, monkeypatch, mixed):
    from dataclasses import replace
    from waveform_analysis.engine import batch as batch_engine
    from waveform_analysis.tests.test_feature_cache import DummyDataset
    from waveform_analysis.data.dataset import InputTransform
    from waveform_analysis.data.shared_artifacts import ExperimentArtifactStore

    source = Path(__file__).resolve().parents[1] / "config/batches/test_ridge.json"
    resolved = load_batch_config(source)
    protocol = deepcopy(resolved.protocol)
    protocol["bootstrap"]["n_resamples"] = 3
    protocol["xai"]["enabled"] = False
    runs = []
    for original in resolved.runs:
        c = deepcopy(original)
        c["output_dir"] = str(tmp_path / c["run_id"])
        c["batch_output_dir"] = str(tmp_path)
        c["preprocessing"]["cache_dir"] = str(tmp_path / "cache")
        c["cross_validation"] = None
        c["bootstrap"] = protocol["bootstrap"]
        c["xai"] = protocol["xai"]
        runs.append(c)
    if mixed:
        c = deepcopy(runs[0])
        c["run_id"] = c["run_id"].replace("direct_linear_ridge", "direct_mlp")
        c["output_dir"] = str(tmp_path / c["run_id"])
        c["cross_validation"] = deepcopy(
            load_batch_config(source.parent / "benchmark_FBK.json").protocol[
                "cross_validation"
            ]
        )
        c["cross_validation"]["folds"] = 2
        c["cross_validation"]["pruning"]["enabled"] = False
        c["model"] = {
            "name": "direct_mlp",
            "space": {
                "model": "direct_mlp",
                "optimization": {"strategy": "fixed"},
                "parameters": {
                    k: {"type": "fixed", "value": v}
                    for k, v in {
                        "architecture": [8],
                        "activation": "relu",
                        "learning_rate": 0.001,
                        "batch_size": 32,
                    }.items()
                },
                "training": {"epochs": 2, "device": "cpu"},
            },
        }
        runs.append(c)
    resolved = replace(
        resolved, output_dir=str(tmp_path), protocol=protocol, runs=tuple(runs)
    )
    datasets = {}
    for role in ("control", "development", "blind"):
        d = DummyDataset(tmp_path / role, role, n_events=256, n_samples=8)
        d.timing_windows = d.energy_windows
        d.timing_time_ps = d.energy_time_ps
        d.timing_target_ps = d.energy_target_ps * 80
        d.timing_led_time_ps = np.column_stack(
            [d.timing_target_ps, np.zeros(d.n_events)]
        )
        d.timing_transform = InputTransform(np.zeros((2, 1)), np.ones((2, 1)))
        d.bias_voltage_V = np.full(d.n_events, 48.0)
        datasets[role] = d
    fit_calls = []
    original_fit = study._fit_final

    def fit(*args, **kwargs):
        fit_calls.append(args[0].name)
        return original_fit(*args, **kwargs)

    monkeypatch.setattr(study, "_fit_final", fit)

    def prepare(config, role, *args, **kwargs):
        if role == "blind":
            assert (
                config["model"]["name"] in fit_calls
            )  # Never select alpha with blind data.
        return None, None, datasets[role]

    monkeypatch.setattr(study, "prepare_role_dataset", prepare)
    for module in (study, batch_engine):
        monkeypatch.setattr(
            module, "fit_control", lambda *a, **kw: ({}, tmp_path / "control")
        )
        monkeypatch.setattr(
            module, "publish_preprocessing_diagnostics", lambda *a, **kw: None
        )

    def forbidden(*a, **kw):
        pytest.fail("Linear Ridge must skip the outer development CV/search pipeline")

    original_search = study._search

    def search(store, spec, *args, **kwargs):
        if spec.selection_method == "ridge_cv":
            forbidden()
        return original_search(store, spec, *args, **kwargs)

    if not mixed:
        monkeypatch.setattr(ExperimentArtifactStore, "prepare_cv", forbidden)
    monkeypatch.setattr(study, "_search", search)
    outputs = batch_engine.run_batch(resolved)
    assert sorted(fit_calls) == sorted(
        ["direct_linear_ridge", "shared_linear_ridge"]
        + (["direct_mlp"] if mixed else [])
    )
    for output in outputs:
        manifest = json.loads((output / "metadata/manifest.json").read_text())
        best = json.loads((output / "metadata/best.json").read_text())
        assert manifest["blind_used_in_selection"] is False
        assert set(manifest["dataset_populations"]) == {
            "control",
            "development",
            "blind",
        }
        assert all(
            row["n_selected"] == 256 for row in manifest["dataset_populations"].values()
        )
        if manifest["selection_method"] == "ridge_cv":
            assert manifest["cv"]["enabled"] is False
            assert best["n_train"] == datasets["development"].n_events
            assert not (output / "tables/cv.csv").exists()
        else:
            assert (output / "tables/cv.csv").exists()
        assert (output / "artifacts/pred.npz").exists()
    table = tmp_path / "report/tables/datasets/FBK_selected_events.tex"
    assert table.is_file()
    assert table.read_text().count(" & 256") == 3  # Once per role, not once per model.
    assert not (table.parent / "UC_selected_events.tex").exists()
    assert (tmp_path / "report/tables/ridge_cv.csv").is_file()
    assert (tmp_path / "report/tables/validation.csv").exists() == mixed
    assert batch_engine.run_batch(resolved) == outputs
    assert len(fit_calls) == (3 if mixed else 2)
    # Missing predictions recover through the persisted model, without fitting.
    (outputs[0] / "artifacts/pred.npz").unlink()
    study.run_study(runs[0])
    assert len(fit_calls) == (3 if mixed else 2)


def test_log_spaced_alpha_range_matches_explicit_grid():
    grid = np.geomspace(1e-8, 1e3, 23)
    ranged = {"ridge_cv": {"alphas": {"low": 1e-8, "high": 1e3, "num": 23}}}
    explicit = {"ridge_cv": {"alphas": grid.tolist()}}
    assert ridge_cv_config(ranged) == ridge_cv_config(explicit)
    assert normalize_ridge_space(
        ranged, "direct_linear_ridge"
    ) == normalize_ridge_space(explicit, "direct_linear_ridge")


@pytest.mark.parametrize(
    "limits",
    [
        {"low": 0, "high": 1, "num": 3},
        {"low": 2, "high": 1, "num": 3},
        {"low": 1, "high": float("inf"), "num": 3},
        {"low": 1, "high": 10, "num": 0},
        {"low": 1, "high": 10, "num": 2.5},
        {"low": 1, "high": 10, "num": True},
        {"low": 1, "high": 10},
    ],
)
def test_invalid_log_spaced_alpha_range(limits):
    with pytest.raises(ValueError):
        ridge_cv_config({"ridge_cv": {"alphas": limits}})


@pytest.mark.parametrize("name", ["direct_linear_ridge", "shared_linear_ridge"])
def test_candidate_inspection_uses_effective_ridgecv_grid(name):
    spec = get_model(name)
    for space in (
        {},
        {"ridge_cv": {"alphas": {"low": 0.001, "high": 100.0, "num": 7}}},
    ):
        assert [
            candidate["ridge_alpha"] for candidate in spec.candidates(space)
        ] == list(ridge_cv_config(space).alphas)
