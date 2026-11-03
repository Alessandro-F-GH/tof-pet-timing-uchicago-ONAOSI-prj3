"""Tests for standalone paired residual Gaussian mixture analysis."""
import numpy as np
import pytest

from waveform_analysis.scripts.compare_energy_timing_residuals import (
    align_prediction_pairs, fit_mixture, load_predictions, marginal_pdf, analyze,
)


def test_npz_requires_led_ml_and_unique_event_id(tmp_path):
    path = tmp_path / "example.npz"
    np.savez(path, event_id=[1, 2, 3], corrected_ps=[1., 2., 3.])
    with pytest.raises(ValueError, match="led_residual_ps"):
        load_predictions(path)
    np.savez(path, event_id=[1, 1, 2] + list(range(3, 12)),
             led_residual_ps=np.arange(12), corrected_ps=np.arange(12))
    with pytest.raises(ValueError, match="unique"):
        load_predictions(path)


def test_align_both_led_and_ml_on_same_event_ids():
    energy = (np.array([30, 10, 20, 40, 50, 60, 70, 80, 90, 100, 110]),
              np.arange(11, dtype=float) + 100,
              np.arange(11, dtype=float) + 200)
    timing = (np.array([20, 30, 10, 40, 50, 60, 70, 80, 90, 100, 110]),
              np.arange(11, dtype=float) + 300,
              np.arange(11, dtype=float) + 400)
    ids, led, ml = align_prediction_pairs(energy, timing)
    assert ids.tolist() == [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110]
    np.testing.assert_array_equal(led[:3], [[101., 302.], [102., 300.], [100., 301.]])
    np.testing.assert_array_equal(ml[:3], [[201., 402.], [202., 400.], [200., 401.]])


def test_two_component_fit_recovers_correlated_broad_tail():
    rng = np.random.default_rng(104)
    core = rng.multivariate_normal([0, 0], [[1600, 300], [300, 500]], size=9000)
    tail = rng.multivariate_normal([0, 0], [[160000, 145000], [145000, 150000]], size=1000)
    x = np.concatenate((core, tail), axis=0)
    rng.shuffle(x)
    result = fit_mixture(x, seed=3, n_init=3)
    c, t = result["components"]
    assert result["converged"]
    assert result["delta_bic_one_minus_two"] > 0
    assert c["weight"] > t["weight"]
    assert t["correlation"] > c["correlation"]
    assert t["sigma_energy_ps"] > c["sigma_energy_ps"]
    assert t["shared_variance_ps2"] == pytest.approx(
        t["correlation"] * t["sigma_energy_ps"] * t["sigma_timing_ps"],
        rel=1e-9)
    assert c["sigma_common_ps"] == pytest.approx(np.sqrt(c["shared_variance_ps2"]))


def test_gaussian_marginal_mixture_integrates_to_one():
    parts = [
        {"weight": 0.95, "mean_ps": [0., 0.],
         "sigma_energy_ps": 20., "sigma_timing_ps": 12.},
        {"weight": 0.05, "mean_ps": [0., 0.],
         "sigma_energy_ps": 200., "sigma_timing_ps": 120.},
    ]
    grid = np.linspace(-1800, 1800, 10001)
    sub, total = marginal_pdf(grid, parts, 0)
    assert np.trapezoid(total, grid) == pytest.approx(1., abs=1e-4)
    assert np.allclose(sub[0] + sub[1], total)


def test_standalone_full_run_without_cache(tmp_path):
    rng = np.random.default_rng(55)
    n = 1200
    ids = rng.permutation(np.arange(n))
    early = rng.normal(0, 65, n)
    target_e = early + rng.normal(0, 40, n)
    target_t = early + rng.normal(0, 20, n)
    energy_file, timing_file = tmp_path / "energy.npz", tmp_path / "timing.npz"
    np.savez(energy_file, event_id=ids, led_residual_ps=target_e,
             corrected_ps=.8 * target_e)
    np.savez(timing_file, event_id=ids[::-1], led_residual_ps=target_t[::-1],
             corrected_ps=(.8 * target_t)[::-1])
    results = analyze(energy_file, timing_file, tmp_path / "output",
                      seed=4, n_init=2)
    assert results["n_matched_events"] == n
    for filename in ("mixture_summary.json", "gaussian_mixture_distributions.png",
                     "gaussian_mixture_joint.pdf"):
        assert (tmp_path / "output" / filename).is_file()
