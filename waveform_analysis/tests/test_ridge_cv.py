"""Standard grid-search and cross-validation contracts for linear Ridge."""

from copy import deepcopy
import json
from pathlib import Path

import numpy as np
import pytest
from sklearn.linear_model import Ridge

from waveform_analysis.core.config import ConfigError, load_batch_config
from waveform_analysis.core.dependencies import planner_fingerprints
from waveform_analysis.data.results import read_run_manifest
from waveform_analysis.data.storage import RunStore
from waveform_analysis.engine import study
from waveform_analysis.models import get_model
from waveform_analysis.tests.test_batch_planner import config as planner_config


@pytest.mark.parametrize(
    "name,intercept", [("direct_linear_ridge", True), ("shared_linear_ridge", False)]
)
def test_matches_sklearn_ridge_and_persisted_predictions(name, intercept, tmp_path):
    rng = np.random.default_rng(847)
    pair = rng.normal(size=(80, 2, 8)).astype(np.float32)
    y = rng.normal(size=80) * 30
    spec = get_model(name)
    transform, x = spec.feature_transform.fit_transform({}, pair, seed=1, config={})
    alpha = 0.1
    artifact = spec.fit({"ridge_alpha": alpha}, x, y, seed=1, config={})
    reference = Ridge(alpha=alpha, fit_intercept=intercept).fit(
        np.asarray(x, dtype=np.float64), y
    )
    np.testing.assert_array_equal(artifact.regressor.coef_, reference.coef_)
    np.testing.assert_array_equal(artifact.regressor.intercept_, reference.intercept_)
    np.testing.assert_array_equal(
        spec.predict(artifact, x), reference.predict(np.asarray(x, dtype=np.float64))
    )
    assert artifact.metadata["ridge_alpha"] == alpha
    assert artifact.metadata["selection_method"] == "development_cv"
    assert "internal_cv_mse_ps2" not in artifact.metadata
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


def test_ridge_models_use_standard_categorical_grid_and_cv():
    source = Path(__file__).resolve().parents[1] / "config/batches/test_ridge.json"
    batch = load_batch_config(source)
    assert batch.protocol["cross_validation"]["folds"] == 3
    assert batch.protocol["cross_validation"]["metric"] == "ctr"
    for run in batch.runs:
        space = run["model"]["space"]
        assert space["optimization"] == {"strategy": "grid"}
        assert "ridge_cv" not in space
        assert space["parameters"]["ridge_alpha"]["type"] == "categorical"
        spec = get_model(run["model"]["name"])
        assert spec.selection_method == "development_cv"
        assert spec.candidates(space) == [
            {"ridge_alpha": float(value)}
            for value in space["parameters"]["ridge_alpha"]["choices"]
        ]


def test_ridge_requires_the_common_outer_cross_validation(tmp_path):
    source = Path(__file__).resolve().parents[1] / "config/batches/test_ridge.json"
    raw = json.loads(source.read_text())
    raw["control_dataset"] = str((source.parent / raw["control_dataset"]).resolve())
    raw["protocol"]["preprocessing_config"] = str(
        (source.parent / raw["protocol"]["preprocessing_config"]).resolve()
    )
    raw["protocol"].pop("cross_validation", None)
    path = tmp_path / "batch.json"
    path.write_text(json.dumps(raw))
    with pytest.raises(ConfigError, match="cross_validation"):
        load_batch_config(path)


def test_planner_tracks_both_outer_cv_and_alpha_grid(tmp_path):
    config = planner_config(tmp_path)
    source = Path(__file__).resolve().parents[1] / "config/batches/test_ridge.json"
    batch = load_batch_config(source)
    ridge_run = next(run for run in batch.runs if run["mode"] == "energy_to_energy")
    config["model"] = deepcopy(ridge_run["model"])
    config["cross_validation"] = deepcopy(ridge_run["cross_validation"])
    before = planner_fingerprints(config)
    changed_cv = deepcopy(config)
    changed_cv["cross_validation"]["folds"] += 1
    assert planner_fingerprints(changed_cv)["cv"] != before["cv"]
    changed_grid = deepcopy(config)
    changed_grid["model"]["space"]["parameters"]["ridge_alpha"]["choices"] = [1.0, 2.0]
    assert planner_fingerprints(changed_grid)["cv"] != before["cv"]


@pytest.mark.parametrize("name", ["direct_linear_ridge", "shared_linear_ridge"])
def test_ridge_candidate_factory_matches_standard_grid(name):
    spec = get_model(name)
    space = {
        "optimization": {"strategy": "grid"},
        "parameters": {
            "ridge_alpha": {"type": "categorical", "choices": [0.01, 0.1, 1.0]}
        },
    }
    assert spec.candidates(space) == [
        {"ridge_alpha": 0.01}, {"ridge_alpha": 0.1}, {"ridge_alpha": 1.0}
    ]


@pytest.mark.parametrize("mixed", [False, True])
def test_actual_ridge_batch_blind_reporting_and_resume(tmp_path, monkeypatch, mixed):
    from dataclasses import replace
    from waveform_analysis.engine import batch as batch_engine
    from waveform_analysis.tests.test_feature_cache import DummyDataset
    from waveform_analysis.data.dataset import InputTransform

    source = Path(__file__).resolve().parents[1] / "config/batches/test_ridge.json"
    resolved = load_batch_config(source)
    protocol = deepcopy(resolved.protocol)
    protocol["bootstrap"]["n_resamples"] = 3
    protocol["xai"]["enabled"] = False
    runs = []
    for original in resolved.runs:
        c = deepcopy(original)
        mode_tag = "energy" if c["mode"] == "energy_to_energy" else "timing"
        c["output_dir"] = str(
            tmp_path / mode_tag / c["window_name"] / c["model"]["name"]
        )
        c["batch_output_dir"] = str(tmp_path)
        c["preprocessing"]["cache_dir"] = str(tmp_path / "cache")
        c["preprocessing"]["artifact_root"] = str(tmp_path / "artifact_store")
        c["bootstrap"] = protocol["bootstrap"]
        c["xai"] = protocol["xai"]
        runs.append(c)
    if mixed:
        c = deepcopy(runs[0])
        c["run_id"] = c["run_id"].replace("direct_linear_ridge", "direct_mlp")
        mode_tag = "energy" if c["mode"] == "energy_to_energy" else "timing"
        c["output_dir"] = str(
            tmp_path / mode_tag / c["window_name"] / c["model"]["name"]
        )
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
        protocol['cross_validation'] = c['cross_validation']
        for run_config in runs:
            run_config['cross_validation'] = c['cross_validation']
    resolved = replace(
        resolved, output_dir=str(tmp_path), protocol=protocol, runs=tuple(runs)
    )
    datasets = {}
    for role in ("control", "development", "blind"):
        d = DummyDataset(tmp_path / role, role, n_events=256, n_samples=8)
        d.timing_windows = d.energy_windows
        d.timing_time_ps = d.energy_time_ps
        d.energy_target_ps = d.energy_target_ps * 80
        d.timing_target_ps = d.energy_target_ps.copy()
        d.timing_led_time_ps = np.column_stack(
            [d.timing_target_ps, np.zeros(d.n_events)]
        )
        d.timing_transform = InputTransform(np.zeros((2, 1)), np.ones((2, 1)))
        d.bias_voltage_V = np.full(d.n_events, 48.0)
        d.manifest["led_control_mean_ps"] = 0.0
        d.sample_mask = np.ones(d.timing_time_ps.size, dtype=bool)
        d.manifest["sample_mask_fit_role"] = "control"
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

    outputs = batch_engine.run_batch(resolved)
    # The fixture config contains direct_linear_ridge in both modes.
    # The optional MLP clone adds one additional energy-mode run.
    expected_fit_calls = ["direct_linear_ridge"] * 2
    if mixed:
        expected_fit_calls.append("direct_mlp")
    assert sorted(fit_calls) == sorted(expected_fit_calls)
    for output, run_config in zip(outputs, runs):
        manifest = read_run_manifest(output)
        best = json.loads((output / "metadata/best.json").read_text())
        run_store = RunStore(
            output,
            artifact_root=run_config["preprocessing"]["artifact_root"],
        )
        assert manifest["blind_used_in_selection"] is False
        assert set(manifest["dataset_populations"]) == {
            "control",
            "development",
            "blind",
        }
        assert all(
            row["n_selected"] == 256 for row in manifest["dataset_populations"].values()
        )
        assert manifest["selection_method"] == "development_cv"
        assert manifest["cv"]["folds"] == run_config["cross_validation"]["folds"]
        assert best["selected_from"] == "development_cv_only"
        assert best["folds"] == run_config["cross_validation"]["folds"]
        assert (output / "tables/cv.csv").is_file()
        assert run_store.predictions_path.is_file()
        development_path = run_store.development_predictions_path
        assert development_path.is_file()
        with np.load(development_path) as diagnostic:
            assert diagnostic["dataset_role"].item() == "development"
            np.testing.assert_array_equal(diagnostic["event_id"], datasets["development"].event_index)
            np.testing.assert_array_equal(diagnostic["led_residual_ps"], datasets["development"].timing_target_ps)
            np.testing.assert_array_equal(diagnostic["corrected_ps"],
                                          diagnostic["led_residual_ps"] - diagnostic["prediction_ps"])
        assert (output / "plots/development.png").exists()
    # The table name follows the board declared by the batch configuration,
    # not the synthetic dataset fixture or the temporary output directory.
    board = next(
        part for part in Path(resolved.results["folder"]).parts if part in {"UC", "FBK"}
    )
    table = tmp_path / "report" / "tables" / "datasets" / f"{board}_selected_events.tex"
    assert table.is_file()
    assert table.read_text().count(" & 256") == 6  # Three roles for each of two modes; not once per model.
    other_board = "FBK" if board == "UC" else "UC"
    assert not (table.parent / f"{other_board}_selected_events.tex").exists()
    assert not (tmp_path / "report/tables/ridge_cv.csv").exists()
    assert (tmp_path / "report/tables/validation.csv").is_file()
    assert batch_engine.run_batch(resolved) == outputs
    assert len(fit_calls) == (3 if mixed else 2)
    # Missing artifact references trigger inference from the persisted model,
    # without retraining or assuming that payloads live inside the run folder.
    reference_path = outputs[0] / "metadata/artifact_references.json"
    references = json.loads(reference_path.read_text())
    references.pop("blind_predictions")
    reference_path.write_text(json.dumps(references))
    study.run_study(runs[0])
    assert len(fit_calls) == (3 if mixed else 2)
    assert RunStore(
        outputs[0], artifact_root=runs[0]["preprocessing"]["artifact_root"]
    ).predictions_path.is_file()
    # Missing development prediction references recover through inference too.
    references = json.loads(reference_path.read_text())
    references.pop("development_predictions")
    reference_path.write_text(json.dumps(references))
    study.run_study(runs[0])
    assert RunStore(
        outputs[0], artifact_root=runs[0]["preprocessing"]["artifact_root"]
    ).development_predictions_path.is_file()
    assert len(fit_calls) == (3 if mixed else 2)

