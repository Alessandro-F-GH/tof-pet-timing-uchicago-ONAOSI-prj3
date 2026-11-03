"""Standalone Gaussian-core / empirical-tail tests."""
import json

import numpy as np
import pytest

from waveform_analysis.scripts.compare_energy_timing_residuals import (
    align_prediction_pairs,
    analyze,
    analyze_distribution,
    empirical_statistics,
    fit_gaussian_core,
    load_predictions,
    mahalanobis_squared,
)


def test_npz_requires_led_and_unique_ids(tmp_path):
    path = tmp_path / "pred.npz"
    np.savez(path, event_id=np.arange(30), corrected_ps=np.zeros(30))
    with pytest.raises(ValueError, match="led_residual_ps"):
        load_predictions(path)
    np.savez(path, event_id=np.zeros(30, dtype=int),
             led_residual_ps=np.arange(30.), corrected_ps=np.arange(30.))
    with pytest.raises(ValueError, match="unique"):
        load_predictions(path)


def test_join_same_events_for_both_methods():
    ids = np.arange(30)
    energy = (ids[::-1], 100. + ids[::-1], 200. + ids[::-1])
    timing = (ids, 300. + ids, 400. + ids)
    matched, led, ml = align_prediction_pairs(energy, timing)
    np.testing.assert_array_equal(matched, ids)
    np.testing.assert_array_equal(led[:, 0], 100. + ids)
    np.testing.assert_array_equal(led[:, 1], 300. + ids)
    np.testing.assert_array_equal(ml[:, 0], 200. + ids)


def test_gaussian_core_recovers_untruncated_covariance():
    rng = np.random.default_rng(17)
    covariance = np.array([[1600., 250.], [250., 450.]])
    x = rng.multivariate_normal([5., -8.], covariance, size=40000)
    fit = fit_gaussian_core(x, fit_fraction=.95)
    np.testing.assert_allclose(fit["mean_ps"], [5., -8.], atol=2.)
    np.testing.assert_allclose(fit["covariance_matrix_ps2"], covariance, rtol=.13)
    assert fit["truncation_covariance_factor"] < 1.
    assert fit["numerically_converged"]


def test_tails_are_empirical_not_gaussian_fitted():
    rng = np.random.default_rng(23)
    cov = np.array([[1600, 250], [250, 450]])
    core = rng.multivariate_normal([0, 0], cov, 18000)
    outliers = rng.normal(size=(150, 1)) * np.array([[180., 165.]])
    x = np.vstack((core, outliers))
    report, mask, distance = analyze_distribution(
        x, tail_fraction=.999, n_reference=10000)
    assert mask.shape == (len(x),)
    assert distance.shape == (len(x),)
    assert report["tail_definition"]["tail_count"] > 20
    assert report["empirical_outside_core"]["correlation"] > .5
    assert "cov_energy_timing_ps2" in report["empirical_outside_core"]
    assert "components" not in report


def test_mahalanobis_and_empirical_covariance():
    x = np.array([[0., 0.], [2., 2.], [-2., -2.], [1., 1.]])
    distances = mahalanobis_squared(x, np.zeros(2), np.eye(2))
    np.testing.assert_allclose(distances, [0., 8., 8., 2.])
    statistics = empirical_statistics(x)
    assert statistics["correlation"] == pytest.approx(1.)


def test_standalone_end_to_end(tmp_path):
    rng = np.random.default_rng(12)
    n = 6000
    ids = rng.permutation(np.arange(n))
    shared = rng.normal(0, 20, n)
    tails = rng.random(n) < .015
    shared[tails] = rng.normal(0, 430, tails.sum())
    e = shared + rng.normal(0, 35, n)
    t = shared + rng.normal(0, 15, n)
    ep, tp = tmp_path / "energy.npz", tmp_path / "timing.npz"
    np.savez(ep, event_id=ids, led_residual_ps=e, corrected_ps=e*.8)
    np.savez(tp, event_id=ids[::-1], led_residual_ps=t[::-1],
             corrected_ps=(t*.8)[::-1])
    result = analyze(ep, tp, tmp_path / "output", n_reference=5000)
    assert result["n_matched_events"] == n
    assert result["led"]["tail_definition"]["tail_count"] > 5
    summary = json.loads((tmp_path / "output" / "core_tail_summary.json").read_text())
    assert "empirical_outside_core" in summary["ml"]
    for filename in ("gaussian_core_distributions.png", "core_and_empirical_tails.pdf",
                     "radial_tail_exceedance.png"):
        assert (tmp_path / "output" / filename).is_file()
