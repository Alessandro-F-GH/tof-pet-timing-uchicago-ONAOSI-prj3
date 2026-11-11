"""Regression tests for prepared relative-LED digital CFD."""
import numpy as np
import pytest

from waveform_analysis.engine.cfd_led_correction import (
    crossing_ps_on_grid, prepared_cfd_corrections)


def test_fractional_crossing_interpolation_and_last_rising():
    t = np.arange(-2, 6, dtype=np.float64) * 100
    x = np.zeros((1, 2, t.size), dtype=float)
    x[0, 0] = [0, 0, .1, .9, 1.1, 2, 3, 2]
    x[0, 1] = [0, 0, .2, .8, 1.2, 2, 3, 2]
    levels = np.array([[1.0, 1.0]])
    result = crossing_ps_on_grid(x, levels, t, np.array([[6, 6]]))
    assert result[0, 0] == pytest.approx(150.)
    assert result[0, 1] == pytest.approx(150.)


def test_relative_led_sign_and_subsample_anchor():
    # Detector1 LED crossing at 0 ps and CFD crossing at +20 ps.
    # Detector2 LED crossing at 0 ps and CFD crossing at +40 ps.
    # prediction=(LED1-CFD1)-(LED2-CFD2)=+20 ps.
    t = np.arange(-5, 6, dtype=float) * 100.
    waves = np.zeros((1, 2, len(t)), dtype=float)
    for d, delay in enumerate((20., 40.)):
        # A shifted linear rise in which both thresholds are crossed.
        waves[0, d] = np.clip((t - delay + 200.) / 400., 0., 1.)
    # LED threshold 0.25 prepared units = 100 mV for 400mV scale.
    corrections, led_crossings, _ = prepared_cfd_corrections(
        waves, t, fractions=[.5], baseline_indices=[0, 1],
        led_threshold_mV=100., physical_scale_mV=[400., 400.])
    assert corrections.shape == (1, 1)
    assert np.isfinite(corrections).all()
    # Equal pulse shapes shifted in time: same LED->CFD delay,
    # hence no spurious detector-difference correction.
    assert corrections[0, 0] == pytest.approx(0., abs=1e-8)


def test_bad_cfd_crossings_are_nan_not_zero():
    t = np.arange(-5, 6, dtype=np.float64) * 100.
    waves = np.zeros((2, 2, len(t)))
    waves[0, :, :] = np.clip((t + 200.) / 400., 0, 1)
    waves[1, :, :] = 0.
    result, _, _ = prepared_cfd_corrections(
        waves, t, fractions=[.1, .2], baseline_indices=[0, 1],
        led_threshold_mV=100, physical_scale_mV=[400, 400])
    assert result.shape == (2, 2)
    assert np.isnan(result[1]).all()


def test_crossing_equal_value_fallback():
    t = np.arange(5, dtype=float)
    wave = np.array([[[0., 1., 2., 3., 2.],
                      [0., 1., 2., 3., 2.]]])
    result = crossing_ps_on_grid(
        wave, np.array([[1., 1.]]), t, np.array([[3, 3]]))
    assert result[0, 0] == pytest.approx(1.)


def test_bad_fractions_rejected():
    waves = np.ones((2, 2, 8), dtype=float)
    with pytest.raises(ValueError, match="fractions"):
        prepared_cfd_corrections(
            waves, np.arange(8), fractions=[0., .3],
            baseline_indices=[0, 1],
            led_threshold_mV=10, physical_scale_mV=[100, 100])


def test_nonzero_led_to_cfd_pair_correction_sign():
    # Both LED crossings anchored near t=0, but unequal pulse slopes produce
    # different delays from fixed-threshold LED to half-maximum CFD.
    t = np.arange(-5, 11, dtype=float) * 100.
    waves = np.zeros((1, 2, len(t)), dtype=float)
    # slope 1: rise starts at -100ps, saturates at 300ps
    # slope 2: rise starts at -50ps, saturates at 750ps
    waves[0, 0] = np.clip((t + 100.) / 400., 0., 1.)
    waves[0, 1] = np.clip((t + 50.) / 800., 0., 1.)
    # LED crossings: d1=0ps, d2=150ps (too far: choose native 0 anchor)
    # Use LED level 0.125: d1=-50ps; d2=50ps, both within one step.
    correction, led, _ = prepared_cfd_corrections(
        waves, t, fractions=[.5], baseline_indices=[0, 1],
        led_threshold_mV=50., physical_scale_mV=[400., 400.])
    assert np.isfinite(correction[0, 0])
    assert correction[0, 0] == pytest.approx(
        (led[0, 0] - 100.) - (led[0, 1] - 350.), abs=1e-8)
    # CFD times d1=100ps and d2=350ps, after LED d1=-50ps and d2=50ps.
    assert correction[0, 0] == pytest.approx(150., abs=1e-8)
