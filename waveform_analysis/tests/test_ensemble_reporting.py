"""Saved-output ensemble metrics, event alignment and absolute-value colours."""
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pytest

from waveform_analysis.reporting import plotting, report_engine as engine
from waveform_analysis.reporting.stats import metric_values
from waveform_analysis.tests.test_report_model_comparisons import report_inputs


def synthetic_pair(monkeypatch):
    rng = np.random.default_rng(24)
    led = rng.normal(0, 100, 1000)
    error = rng.normal(0, 35, 1000)
    other_error = -error + rng.normal(0, 12, 1000)
    fit = {"histogram_bin_width_ps": 5.}
    payloads, runs = [], []
    for name, indices, residual in [
        ("direct_mlp", rng.permutation(np.arange(950)), error),
        ("shared_cnn1d", rng.permutation(np.arange(50, 1000)), other_error),
    ]:
        payloads.append({"event_id": indices, "prediction_ps": led[indices] - residual[indices],
                         "led_residual_ps": led[indices], "corrected_ps": residual[indices]})
        runs.append({"directory": Path(name),
                     "manifest": {"model": name, "mode": "timing_to_timing", "window_name": "short"},
                     "blind": {"ctr_ps": metric_values(residual[indices], fit, seed=1)["ctr_ps"], "n_events": len(indices)}})
    monkeypatch.setattr(engine, "resolve_run_config", lambda _: {"fit": fit})
    return runs, payloads, fit, (error[50:950] + other_error[50:950]) / 2


def test_actual_ctr_of_average_predictions_on_aligned_events(monkeypatch):
    runs, payloads, fit, expected = synthetic_pair(monkeypatch)
    labels, matrix, counts, unavailable = engine._ensemble_ctr(runs, payloads, seed=1)
    assert labels == ["direct_mlp", "shared_cnn1d"]
    assert unavailable == []
    np.testing.assert_array_equal(counts, [[950, 900], [900, 950]])
    assert matrix[0, 1] == matrix[1, 0]
    assert matrix[0, 1] == pytest.approx(metric_values(expected, fit, seed=1)["ctr_ps"])
    np.testing.assert_array_equal(np.diag(matrix), [r["blind"]["ctr_ps"] for r in runs])
    # Cancellation in the averaged predictions improves CTR; averaging the two
    # scalar CTR values would give a very different result.
    assert matrix[0, 1] < min(np.diag(matrix)) / 2
    _, reversed_matrix, _, _ = engine._ensemble_ctr(runs[::-1], payloads[::-1], seed=1)
    np.testing.assert_array_equal(reversed_matrix, matrix[::-1, ::-1])


@pytest.mark.parametrize("case", ["different_baseline", "too_few", "nonfinite", "disjoint"])
def test_unavailable_pairs_preserve_single_model_results(monkeypatch, case):
    runs, payloads, _, _ = synthetic_pair(monkeypatch)
    if case == "different_baseline":
        payloads[1]["led_residual_ps"] += 2
    elif case == "disjoint":
        payloads[1]["event_id"] += 2000
    elif case == "nonfinite":
        payloads[1]["prediction_ps"][:] = np.nan
    else:
        payloads[1] = {k: v[:3] for k, v in payloads[1].items()}
    _, matrix, counts, unavailable = engine._ensemble_ctr(runs, payloads, seed=1)
    assert np.isnan(matrix[0, 1]) and np.isnan(matrix[1, 0])
    assert len(unavailable) == 1 and unavailable[0]["reason"]
    assert counts[0, 1] == counts[1, 0]
    np.testing.assert_array_equal(np.diag(matrix), [r["blind"]["ctr_ps"] for r in runs])


def test_reports_add_ensemble_to_previous_runs_and_filter_models(report_inputs, monkeypatch):
    root, runs, _ = report_inputs
    figures = []
    monkeypatch.setattr(engine, "heatmap", lambda *a, **kw: figures.append((a, kw)))
    report = engine.generate_report(root, reuse_numeric=True)
    tables = report / "tables/timing/short"
    labels, matrix = engine._read_matrix(tables / "ensemble_ctr.csv")
    assert labels == ["direct_mlp", "independent_cnn1d", "antisymmetric_mlp", "shared_cnn1d"]
    np.testing.assert_array_equal(np.diag(matrix), [90., 95., 105., 110.])
    assert engine._read_matrix(tables / "ensemble_ctr_n.csv")[1].shape == (4, 4)
    manifest = json.loads((tables / "ensemble_ctr.json").read_text())
    assert manifest["prediction"] == "arithmetic_mean_of_two_saved_predictions"
    assert any(kwargs.get("absolute") for _, kwargs in figures)
    # Cached files must not conceal changed saved predictions.
    (tables / "ensemble_ctr.csv").write_text("stale matrix")
    engine.generate_report(root, reuse_numeric=True, exclude_models=["direct_mlp"])
    labels, matrix = engine._read_matrix(tables / "ensemble_ctr.csv")
    assert labels == ["independent_cnn1d", "antisymmetric_mlp", "shared_cnn1d"]
    np.testing.assert_array_equal(np.diag(matrix), [95., 105., 110.])


def test_ensemble_heatmap_uses_absolute_scale_and_supports_old_plot_settings(tmp_path, monkeypatch):
    cfg = json.loads((Path(__file__).parents[1] / "config/plots/default.json").read_text())
    cfg["heatmap"].pop("ensemble_cmap")  # Existing results have older plots.json.
    captured = []
    def inspect(fig, path):
        captured.append(fig)
    monkeypatch.setattr(plotting, "_save", inspect)
    plotting.heatmap(np.array([[90., 70.], [70., 110.]]), ["A", "B"], tmp_path / "matrix.png", cfg,
                     title="Ensemble CTR", absolute=True, colorbar_label="CTR [ps]")
    fig = captured.pop()
    try:
        image = fig.axes[0].images[0]
        assert image.get_clim() == (70., 110.)
        assert image.get_cmap().name == "viridis_r"
        assert fig.axes[1].get_ylabel() == "CTR [ps]"
        assert [text.get_text() for text in fig.axes[0].texts] == ["90.0", "70.0", "70.0", "110.0"]
        assert fig.axes[0].texts[1].get_color() == cfg["heatmap"]["text_color"]
        fig.savefig(tmp_path / "ensemble.png")
        assert (tmp_path / "ensemble.png").stat().st_size > 1000
    finally:
        plt.close(fig)
