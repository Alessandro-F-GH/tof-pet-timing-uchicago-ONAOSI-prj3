"""Unit tests for the standalone, no-synthetic-samples Lasso distillation."""
import numpy as np
import pytest
from waveform_analysis.scripts.lasso_surrogate import (
    fit_lasso_path, _positive_unique, _timing_metrics,
)


def test_sparse_path_uses_teacher_outputs_and_reports_blind_generalization():
    rng = np.random.default_rng(23)
    dev = rng.normal(size=(160, 8)).astype(np.float32)
    blind = rng.normal(size=(70, 8)).astype(np.float32)
    weights = np.array([2.5, -1.7, 0, 0, 0, 0, 0, 0])
    y_dev = dev @ weights + 4.0
    y_blind = blind @ weights + 4.0
    dev_snapshot, blind_snapshot = dev.copy(), blind.copy()
    result = fit_lasso_path(dev, y_dev, blind, y_blind, alpha_ratios=[1, .01, .001])
    rows = result["rows"]
    assert len(rows) == 3
    assert rows[0]["n_nonzero"] == 0
    assert rows[-1]["n_nonzero"] <= 2
    assert rows[-1]["blind"]["r2"] > .99
    assert rows[-1]["blind"]["rmse_ps"] < .1
    assert result["coefficients"].shape == (3, 8)
    assert all(row["n_zero"] + row["n_nonzero"] == 8 for row in rows)
    np.testing.assert_array_equal(dev, dev_snapshot)
    np.testing.assert_array_equal(blind, blind_snapshot)


def test_blind_targets_never_affect_model_fit_or_scaling():
    rng = np.random.default_rng(3)
    dev = rng.normal(size=(70, 4)).astype(np.float32)
    blind = rng.normal(size=(20, 4)).astype(np.float32)
    dev_teacher = dev[:, 0] - 3 * dev[:, 1]
    blind_teacher = blind[:, 0] - 3 * blind[:, 1]
    a = fit_lasso_path(dev, dev_teacher, blind, blind_teacher, alpha_ratios=[.1, .01])
    b = fit_lasso_path(dev, dev_teacher, blind, blind_teacher + 500, alpha_ratios=[.1, .01])
    np.testing.assert_allclose(a["coefficients"], b["coefficients"])
    np.testing.assert_allclose(a["predictions_blind"], b["predictions_blind"])
    assert b["rows"][0]["blind"]["rmse_ps"] > a["rows"][0]["blind"]["rmse_ps"]


def test_scaled_coefficients_reconstruct_predictions_without_clipping():
    rng = np.random.default_rng(19)
    dev = rng.normal(size=(90, 3)).astype(np.float32) * [1, 8, .02]
    blind = rng.normal(size=(40, 3)).astype(np.float32) * [1, 8, .02]
    weights = np.array([4., -2., .3])
    result = fit_lasso_path(dev, dev @ weights, blind, blind @ weights, alpha_ratios=[.01])
    inferred = blind @ result["coefficients"][0] + result["intercepts"][0]
    np.testing.assert_allclose(inferred, result["predictions_blind"][0], rtol=1e-5, atol=1e-5)


def test_invalid_inputs_rejected():
    with pytest.raises(ValueError):
        _positive_unique([0], "alpha")
    with pytest.raises(ValueError):
        fit_lasso_path(np.zeros((10, 3)), np.zeros(10), np.zeros((8, 2)), np.zeros(8))
    with pytest.raises(ValueError):
        fit_lasso_path(np.zeros((10, 2)), np.zeros(10), np.zeros((8, 2)), np.zeros(8),
                       alpha_ratios=[.1])
