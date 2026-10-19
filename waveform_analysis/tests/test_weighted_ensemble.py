"""Development-only saved-output selection and frozen blind evaluation."""
from copy import deepcopy
import csv
import json

import numpy as np
import pytest

from waveform_analysis.reporting import weighted_ensemble as module
from waveform_analysis.reporting import report_engine as engine
from waveform_analysis.reporting.stats import metric_values
from waveform_analysis.tests.test_report_model_comparisons import report_inputs


@pytest.fixture
def saved_pair(tmp_path):
    rng = np.random.default_rng(142)
    fit = {"histogram_bin_width_ps": 2.}
    led = rng.normal(0, 120, 1600)
    first = rng.normal(0, 35, 1600)
    second = -2 * first + rng.normal(0, 12, 1600)
    blind_led = rng.normal(0, 120, 1600)
    blind_first = rng.normal(0, 80, 1600)
    blind_second = rng.normal(0, 18, 1600)
    runs, blind, paths = [], [], []
    for i, name in enumerate(("direct_mlp", "shared_cnn1d")):
        directory = tmp_path / name
        (directory / "artifacts").mkdir(parents=True)
        ids = rng.permutation(np.arange(0, 1500) if i == 0 else np.arange(100, 1600))
        residual = first if i == 0 else second
        path = directory / "artifacts/development_pred.npz"
        np.savez(path, event_id=ids, prediction_ps=led[ids] - residual[ids],
                 led_residual_ps=led[ids], corrected_ps=residual[ids],
                 dataset_role=np.asarray("development"), final_fit_fingerprint=np.asarray(f"fit-{i}"),
                 analysis_protocol_identity=np.asarray("development-protocol"))
        paths.append(path)
        blind_ids = rng.permutation(np.arange(1600))
        blind_error = blind_first if i == 0 else blind_second
        blind.append({"event_id": blind_ids, "prediction_ps": blind_led[blind_ids] - blind_error[blind_ids],
                      "led_residual_ps": blind_led[blind_ids]})
        runs.append({"directory": directory,
                     "manifest": {"model": name, "mode": "timing_to_timing", "window_name": "short",
                                  "stage_fingerprints": {"final_fit": f"fit-{i}"},
                                  "dataset_populations": {"development": {"analysis_protocol_identity": "development-protocol"}}},
                     "blind": {"ctr_ps": metric_values(blind_error, fit, seed=1)["ctr_ps"], "n_events": len(blind_ids)}})
    return runs, blind, fit, paths


def test_real_scan_matches_ctr_reference_and_uses_frozen_weight_on_blind(saved_pair):
    runs, blind, fit, paths = saved_pair
    before = [path.read_bytes() for path in paths]
    result = module.weighted_ensemble(runs, blind, fit, seed=4)
    assert not result["unavailable_pairs"]
    assert len(result["scan"]) == 21
    assert sum(row["selected"] for row in result["scan"]) == 1
    assert {row["n_events"] for row in result["scan"]} == {1400}
    assert [row["weight_a"] for row in result["scan"]] == list(module.WEIGHT_GRID)
    _, a, b, led = module.aligned_predictions(*(module.read_development(run) for run in runs))
    reference = [metric_values(led - (w * a + (1 - w) * b), fit, seed=1)["ctr_ps"] for w in module.WEIGHT_GRID]
    np.testing.assert_allclose([row["development_ctr_ps"] for row in result["scan"]], reference)
    weight = result["weights"][0, 1]
    assert 0.55 < weight < 0.8
    assert result["weights"][1, 0] == 1 - weight
    assert result["development_ctr"][0, 1] == min(reference)
    _, a, b, led = module.aligned_predictions(*blind)
    expected = metric_values(led - (weight * a + (1 - weight) * b), fit, seed=1)["ctr_ps"]
    assert result["blind_ctr"][0, 1] == pytest.approx(expected)
    np.testing.assert_array_equal(np.diag(result["blind_ctr"]), [run["blind"]["ctr_ps"] for run in runs])
    assert [path.read_bytes() for path in paths] == before


def test_blind_changes_never_change_selection_and_model_order_is_invariant(saved_pair):
    runs, blind, fit, _ = saved_pair
    original = module.weighted_ensemble(runs, blind, fit, seed=4)
    changed = deepcopy(blind)
    changed[0]["prediction_ps"] *= -10
    changed[1]["prediction_ps"] += 700
    other = module.weighted_ensemble(runs, changed, fit, seed=4)
    np.testing.assert_array_equal(original["weights"], other["weights"])
    np.testing.assert_array_equal(original["development_ctr"], other["development_ctr"])
    assert original["scan"] == other["scan"]
    reversed_result = module.weighted_ensemble(runs[::-1], blind[::-1], fit, seed=4)
    np.testing.assert_array_equal(original["weights"], reversed_result["weights"][::-1, ::-1])
    np.testing.assert_array_equal(original["blind_ctr"], reversed_result["blind_ctr"][::-1, ::-1])


@pytest.mark.parametrize("problem", ["missing", "stale", "protocol", "role", "duplicate", "nonfinite", "baseline", "corrupt"])
def test_invalid_development_is_skipped_without_touching_blind(saved_pair, problem):
    runs, _, fit, paths = saved_pair
    with np.load(paths[0]) as saved:
        data = {key: saved[key] for key in saved.files}
    if problem == "missing":
        paths[0].unlink()
    elif problem == "corrupt":
        paths[0].write_bytes(b"PK\x03\x04broken zip")
    else:
        if problem == "stale":
            data["final_fit_fingerprint"] = np.asarray("old-fit")
        elif problem == "protocol":
            data["analysis_protocol_identity"] = np.asarray("other-protocol")
        elif problem == "role":
            data["dataset_role"] = np.asarray("blind")
        elif problem == "duplicate":
            data["event_id"][:] = 0
        elif problem == "baseline":
            data["led_residual_ps"] += 10
        else:
            data["prediction_ps"][:] = np.nan
        np.savez(paths[0], **data)
    class NoBlind:
        def __getitem__(self, key):
            pytest.fail("An unavailable selection must not read blind predictions")
    result = module.weighted_ensemble(runs, NoBlind(), fit, seed=4)
    assert np.isnan(result["weights"][0, 1])
    assert not result["scan"]
    assert result["unavailable_pairs"][0]["stage"] == "development_selection"
    assert np.isfinite(np.diag(result["blind_ctr"])).all()


def test_exact_ties_prefer_half_and_failed_candidates_do_not_abort_scan(saved_pair, monkeypatch):
    runs, blind, fit, _ = saved_pair
    calls = []
    def ctr(values, *args, **kwargs):
        calls.append(len(values))
        if len(calls) == 3:  # First pair candidate fails after the two diagonals.
            raise ValueError("unestimable candidate")
        return {"ctr_ps": 50.}
    monkeypatch.setattr(module, "metric_values", ctr)
    result = module.weighted_ensemble(runs, blind, fit, seed=4)
    assert result["weights"][0, 1] == 0.5
    assert result["scan"][0]["status"] == "failed"
    assert len(result["scan"]) == 21
    assert sum(row["selected"] for row in result["scan"]) == 1


def test_single_model_endpoint_can_win(saved_pair, monkeypatch):
    runs, blind, fit, paths = saved_pair
    # A and B predict opposite directions around a target which A predicts
    # exactly. This tests selection logic with a simple deterministic score.
    for i, path in enumerate(paths):
        with np.load(path) as saved:
            data = {key: saved[key] for key in saved.files}
        data["prediction_ps"] = data["led_residual_ps"] + (0 if i == 0 else 100)
        np.savez(path, **data)
    monkeypatch.setattr(module, "metric_values", lambda residual, *a, **kw: {"ctr_ps": float(np.mean(residual ** 2))})
    result = module.weighted_ensemble(runs, blind, fit, seed=4)
    assert result["weights"][0, 1] == 1.
    assert result["weights"][1, 0] == 0.


def test_blind_failure_retains_selected_weight(saved_pair):
    runs, blind, fit, _ = saved_pair
    blind[1]["led_residual_ps"] += 20
    result = module.weighted_ensemble(runs, blind, fit, seed=4)
    assert np.isfinite(result["weights"][0, 1])
    assert np.isnan(result["blind_ctr"][0, 1])
    assert result["unavailable_pairs"][0]["stage"] == "blind_evaluation"


def test_unestimable_endpoint_does_not_discard_valid_ensemble(saved_pair, monkeypatch):
    runs, blind, fit, _ = saved_pair
    calls = []
    def ctr(*args, **kwargs):
        calls.append(1)
        if len(calls) <= 2:
            raise ValueError("single-model CTR cannot be estimated")
        return {"ctr_ps": 50.}
    monkeypatch.setattr(module, "metric_values", ctr)
    result = module.weighted_ensemble(runs, blind, fit, seed=4)
    assert result["weights"][0, 1] == .5
    assert np.isnan(np.diag(result["development_ctr"])).all()
    assert not result["unavailable_pairs"]


def test_report_exports_weighted_outputs_and_clears_stale_scans(report_inputs, monkeypatch):
    root, runs, _ = report_inputs
    for i, run in enumerate(runs):
        run["manifest"]["stage_fingerprints"] = {"final_fit": f"fit-{i}"}
        with np.load(run["directory"] / "artifacts/pred.npz") as saved:
            data = {key: saved[key] for key in saved.files}
        np.savez(run["directory"] / "artifacts/development_pred.npz", **data,
                 dataset_role=np.asarray("development"), final_fit_fingerprint=np.asarray(f"fit-{i}"),
                 analysis_protocol_identity=np.asarray("protocol"))
    report = engine.generate_report(root, reuse_numeric=True)
    tables = report / "tables/timing/short"
    labels, weights = engine._read_matrix(tables / "ensemble_weights.csv")
    assert labels == ["direct_mlp", "independent_cnn1d", "antisymmetric_mlp", "shared_cnn1d"]
    np.testing.assert_allclose(weights + weights.T, np.ones((4, 4)) + np.eye(4))
    scan_path = tables / "ensemble_weight_scan.csv"
    with scan_path.open() as stream:
        assert len(list(csv.DictReader(stream))) == 6 * 21
    metadata = json.loads((tables / "ensemble_weighted.json").read_text())
    assert metadata["blind_used_in_selection"] is False
    assert metadata["development_prediction_basis"] == "in_sample_final_model"
    for run in runs:
        (run["directory"] / "artifacts/development_pred.npz").unlink()
    engine.generate_report(root, reuse_numeric=True, exclude_models=["direct_mlp"])
    labels, weights = engine._read_matrix(tables / "ensemble_weights.csv")
    assert "direct_mlp" not in labels
    assert np.isnan(weights[0, 1])
    with scan_path.open() as stream:
        assert list(csv.DictReader(stream)) == []
    assert (tables / "ensemble_ctr.csv").is_file()  # Equal-weight report survives.
