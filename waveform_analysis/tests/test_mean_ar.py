"""Synthetic tests for standalone AR suffix forecasting (no experimental ROOT data)."""
import numpy as np
import pytest

from waveform_analysis.scripts.evaluate_mean_ar import (
    ARFit, choose_cutoffs, fit_one_ar, forecast_suffix, is_stable,
    select_events, waveform_statistics,
)


def test_forward_only_and_clipping():
    mean = np.linspace(.2, .8, 30)
    sigma = np.ones(30) * .1
    waves = np.tile(mean, (3, 1))
    waves[:, :10] += .05
    fit = ARFit(1, np.array([.8]), 1.0, True)
    original = forecast_suffix(waves, mean, sigma, 10, fit, True)
    altered_future = waves.copy()
    altered_future[:, 10:] = 1.0
    assert np.array_equal(original, forecast_suffix(altered_future, mean, sigma, 10, fit, True))
    assert np.all((original >= 0) & (original <= 1))


def test_unstable_rejected():
    assert is_stable(np.array([.8]))
    assert not is_stable(np.array([1.1]))
    assert not is_stable(np.array([1.0]))


def test_mean_and_standard_deviation():
    wave = np.array([[.2, .3, .4], [.2, .7, .4]])
    mean, sigma = waveform_statistics(wave, std_floor=.01)
    np.testing.assert_allclose(mean, [.2, .5, .4])
    np.testing.assert_allclose(sigma, [.01, .2, .01])


def test_cutoffs_dense_and_valid():
    t = np.linspace(-6, 6, 700)
    mean = .1 + .8 / (1 + np.exp(-6 * (t + 1))) * np.exp(-np.maximum(t + 1, 0) / 2)
    cutoffs, anchors = choose_cutoffs(mean, np.ones_like(mean) * .02, 130, 10)
    assert len(cutoffs) == 10
    assert np.array_equal(cutoffs, np.unique(cutoffs))
    assert cutoffs.min() >= 130
    assert cutoffs.max() < len(mean)
    assert 0 <= anchors["rise"] < len(mean)


def test_fit_and_prediction_on_synthetic_ar():
    rng = np.random.default_rng(47)
    z = np.zeros((32, 160))
    for t in range(1, 160):
        z[:, t] = .85 * z[:, t-1] + rng.normal(scale=.05, size=len(z))
    fit = fit_one_ar(z[:24], z[24:], 1, [.001, .1, 1], 2000, 5)
    assert fit.stable
    assert abs(fit.coefficients[0] - .85) < .15


def test_seeded_sampling():
    np.testing.assert_array_equal(select_events(100, 12, 42), select_events(100, 12, 42))
    assert len(np.unique(select_events(100, 12, 42))) == 12
