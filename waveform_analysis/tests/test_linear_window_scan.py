"""Temporal scan geometry and shared-Ridge Development-only selection tests."""
import numpy as np
import pytest

from waveform_analysis.engine.linear_window_scan import (
    temporal_windows, selected_indices, _shared_difference, _choose_alpha,
)
from waveform_analysis.data.splits import make_cv_split


def test_default_fixed_windows_are_nonoverlapping_and_progressive():
    windows = temporal_windows(-2, 3, 1)
    progressive = [row for row in windows if row[0] == "progressive"]
    fixed = [row for row in windows if row[0] == "fixed"]
    assert len(progressive) == len(fixed) == 5
    assert progressive[0] == ("progressive", -2.0, -1.0)
    assert progressive[-1] == ("progressive", -2.0, 3.0)
    assert fixed == [
        ("fixed", -2.0, -1.0), ("fixed", -1.0, 0.0),
        ("fixed", 0.0, 1.0), ("fixed", 1.0, 2.0),
        ("fixed", 2.0, 3.0),
    ]
    for first, second in zip(fixed[:-1], fixed[1:]):
        assert first[2] == second[1]


def test_onishi_progressive_anchor_and_full_wide_fixed_scan():
    windows = temporal_windows(
        -2.0, 5.0, 1.0, progressive_start_ns=-1.5,
        progressive_initial_end_ns=2.0,
    )
    progressive = [row for row in windows if row[0] == "progressive"]
    fixed = [row for row in windows if row[0] == "fixed"]
    assert progressive[0] == ("progressive", -1.5, 2.0)
    assert progressive[-1] == ("progressive", -1.5, 5.0)
    assert fixed[0] == ("fixed", -2.0, -1.0)
    assert fixed[-1] == ("fixed", 4.0, 5.0)
    assert len(fixed) == 7


def test_explicit_width_overrides_default_without_changing_step():
    fixed = [x for x in temporal_windows(0, 5, 1, 2.0) if x[0] == "fixed"]
    assert len(fixed) == 4
    assert fixed[0] == ("fixed", 0.0, 2.0)
    assert fixed[1] == ("fixed", 1.0, 3.0)


def test_frozen_mask_and_right_open_intervals_do_not_overlap():
    t = np.arange(-2000, 3001, 12.5)
    frozen = np.ones(len(t), dtype=bool)
    frozen[(t >= 0) & (t < 1000)] = False
    first = selected_indices(t, frozen, -2, -1)
    second = selected_indices(t, frozen, -1, 0)
    third = selected_indices(t, frozen, 0, 1)
    assert first.size == second.size == 80
    assert third.size == 0
    assert not np.intersect1d(first, second).size
    end = selected_indices(t, frozen, 2, 3, last_edge=True)
    assert t[end[-1]] == 3000.


def test_scan_rejects_bad_geometry():
    with pytest.raises(ValueError):
        temporal_windows(1, 1)
    with pytest.raises(ValueError):
        temporal_windows(-2, 3, 0)
    with pytest.raises(ValueError):
        temporal_windows(-2, 3, 1, 6)


def test_shared_difference_uses_only_requested_columns():
    class Dataset:
        energy_windows = np.array([
            [[1., 4., 5.], [2., 1., 3.]],
            [[4., 3., 8.], [1., 7., 2.]],
        ], dtype=np.float32)
    x = _shared_difference(Dataset(), "energy_to_energy", np.array([0, 2]))
    np.testing.assert_array_equal(x, [[-1., 2.], [3., 6.]])
    assert x.dtype == np.float32


def test_alpha_selection_is_based_on_development_only(monkeypatch):
    # Only replace the metric estimator for a small synthetic population;
    # all CV, model fitting and parameter selection remain real.
    from waveform_analysis.engine import linear_window_scan
    def fake_metrics(residuals, fit_cfg, *, seed):
        values = np.asarray(residuals, dtype=np.float64)
        score = float(np.sqrt(np.mean(values ** 2)))
        return {"rmse_ps": score, "ctr_ps": score}
    monkeypatch.setattr(linear_window_scan, "metric_values", fake_metrics)
    rng = np.random.default_rng(19)
    n = 72
    x = rng.normal(size=(n, 5)).astype(np.float32)
    y = 4 * x[:, 0] - x[:, 1] + rng.normal(scale=.1, size=n)
    cv = {
        "metric": "rmse",
        "pruning": {
            "enabled": False, "startup_complete_candidates": 1,
            "min_folds_before_prune": 1, "max_degradation_ps": 5,
            "prune_if_worse_than_led": False, "led_max_degradation_ps": 0,
        },
    }
    split = make_cv_split(n, population_identity="development", batch_seed=5,
                          n_folds=3, shuffle=True)
    alpha, score = _choose_alpha(x, y, split, [0.01, 10.0], cv, {}, 5, 2000)
    assert alpha in (0.01, 10.0)
    assert np.isfinite(score)
