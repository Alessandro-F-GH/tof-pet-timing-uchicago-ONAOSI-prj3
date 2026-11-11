"""Analytical regression tests for prepared multi-threshold time pickoff."""
import numpy as np
import pytest

from waveform_analysis.engine.multithreshold_led import (
    validate_threshold_sets, interpolate_upward_crossing,
    linear_pickoff, multithreshold_corrections)


def test_threshold_ceiling_and_minimum_count():
    assert validate_threshold_sets([(10, 20, 50)]) == ((10., 20., 50.),)
    for candidate in ((10, 20), (10, 20, 51), (20, 10, 30),
                      (10, 20, 20), (0, 10, 20)):
        with pytest.raises(ValueError):
            validate_threshold_sets([candidate])


def test_interpolates_crossing_between_samples():
    t = np.array([-100., 0., 100., 200.])
    x = np.array([[[0., .10, .30, .40],
                   [0., .10, .30, .40]]])
    result = interpolate_upward_crossing(
        x, np.array([[.2, .2]]), t)
    assert result.shape == (1, 2)
    assert np.allclose(result, 50.)


def test_ols_recovers_intercepts_and_slopes():
    v = np.array([10., 20., 30., 40.])
    t0 = np.array([[42., -11.], [-17., 38.]])
    slopes = np.array([[1.5, 2.0], [3.0, .75]])
    crossing = t0[..., None] + slopes[..., None] * v
    intercept, fitted_slope, fit_error = linear_pickoff(crossing, v)
    np.testing.assert_allclose(intercept, t0, atol=1e-10)
    np.testing.assert_allclose(fitted_slope, slopes, atol=1e-10)
    np.testing.assert_allclose(fit_error, 0., atol=1e-10)


def test_rise_times_must_increase_and_missing_crossing_invalidates():
    v = np.array([10., 20., 30.])
    times = np.array([[[20., 10., 30.], [15., 20., 30.]]])
    intercept, slope, rms = linear_pickoff(times, v)
    assert np.isnan(intercept[0, 0])
    assert np.isnan(slope[0, 0])
    assert np.isfinite(intercept[0, 1])
    times[0, 1, 0] = np.nan
    assert np.isnan(linear_pickoff(times, v)[0]).all()


def test_multi_threshold_pair_sign_and_amplitude_scaling():
    # Two different linear rising-edge slopes. A zero LED reference
    # crossing is interpolated on the same sample grid.
    t = np.arange(-20, 41, dtype=float) * 10.  # ps
    levels_mV = [10., 20., 30.]
    waves = np.zeros((1, 2, t.size), float)
    # True signal in mV: d1=1mV/ps starting -30ps,
    # d2=0.5mV/ps starting -60ps; local LED crossing at +0ps.
    physical = np.array([
        np.maximum(0, t + 30.),
        np.maximum(0, .5 * (t + 60.)),
    ])
    waves[0] = physical / 300.
    corr, diag = multithreshold_corrections(
        waves, t, threshold_sets=[levels_mV],
        baseline_indices=[0, 1], led_threshold_mV=30.,
        physical_scale_mV=[300., 300.],
        crossing_window_ns=(-.1, .2),
        chunk_size=1)
    assert corr.shape == (1, 1)
    # t0 = (-30,-60) ps and LED crossings = (0,0) ps;
    # prediction=(0+30)-(0+60)=-30 ps.
    assert corr[0, 0] == pytest.approx(-30., abs=1e-8)
    np.testing.assert_allclose(diag["intercept_ps"][0, :, 0],
                               [-30., -60.], atol=1e-8)
    np.testing.assert_allclose(diag["fit_rmse_ps"][0, :, 0],
                               [0., 0.], atol=1e-8)


def test_missing_threshold_nan_does_not_emit_fabricated_prediction():
    t = np.arange(-20, 41, dtype=float) * 10.
    waves = np.zeros((2, 2, t.size), float)
    waves[0, 0] = np.clip((t + 30.) / 300., 0., 1.)
    waves[0, 1] = np.clip((t + 60.) / 600., 0., 1.)
    corr, diagnostics = multithreshold_corrections(
        waves, t, threshold_sets=[(10., 20., 30.)],
        baseline_indices=[0, 1], led_threshold_mV=30.,
        physical_scale_mV=[300., 300.], crossing_window_ns=(-.1, .2))
    assert np.isfinite(corr[0, 0])
    assert np.isnan(corr[1, 0])
