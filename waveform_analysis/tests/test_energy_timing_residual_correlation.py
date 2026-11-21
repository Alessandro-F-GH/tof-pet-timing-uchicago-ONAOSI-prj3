"""Tests for definitive paired Energy/Timing correlation analysis."""
import numpy as np

from waveform_analysis.scripts.analyze_energy_timing_correlation import (
    tukey, moments, variance_contribution, event_bins, summarize_board,
)


def test_iqr_joint_flag():
    rng = np.random.default_rng(10)
    points = rng.normal(size=(300, 2))
    points[-1] = [20, 0]
    points[-2] = [0, -20]
    _, flags = tukey(points, 3)
    assert flags[-1, 0] and flags[-2, 1]
    assert np.sum(flags.any(axis=1)) == 2


def test_moments_and_variance_share():
    points = np.array([[1., 2.], [2., 4.], [3., 6.], [4., 8.], [5., 10.]])
    assert np.isclose(moments(points)["rho"], 1)
    shares = variance_contribution(points, np.array([False, False, False, False, True]))
    assert all(0 < v < 1 for v in shares)


def test_acquisition_order_bins():
    ids = np.arange(200)
    mask = np.zeros(200, bool)
    mask[::10] = True
    bins, result = event_bins(ids, mask, 10)
    assert len(bins) == 10
    assert sum(row["outliers"] for row in bins) == 20


def test_common_core_selection():
    rng = np.random.default_rng(12)
    led = rng.normal(size=(500, 2))
    ml = led * .5
    led[-1] = [100., 0.]
    ml[-2] = [100., 0.]
    data = {"event_id": np.arange(500), "led": led, "ml": ml,
            "prediction": led-ml, "matched": 500,
            "energy_events": 500, "timing_events": 500}
    summary, masks, core = summarize_board(data, 3, 10, 20, rng)
    assert not core[-1] and not core[-2]
    assert summary["model_outputs_core"]["n"] == np.sum(core)
