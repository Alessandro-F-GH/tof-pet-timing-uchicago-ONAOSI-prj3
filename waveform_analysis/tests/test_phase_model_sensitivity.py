"""Synthetic phase-rotation controls; no trained model required."""
import numpy as np

from waveform_analysis.scripts.test_phase_model_sensitivity import (
    fit_local_oscillation, phase_rotate_prepared)


def test_fit_known_oscillation_phase_and_amplitude():
    t = np.arange(-3., 35., .0125)
    freq = 2.5
    phase = np.array([.4, -1.2])
    amplitude = np.array([.03, .06])
    trace = (.4 + .0001*t[None, :]
             + amplitude[:, None] * np.cos(
                 2*np.pi*freq*t[None, :] + phase[:, None]))
    fit = (t <= -1.5)
    got_amp, got_phase = fit_local_oscillation(trace, t, freq, fit)
    np.testing.assert_allclose(got_amp, amplitude, atol=1e-9)
    np.testing.assert_allclose(
        np.angle(np.exp(1j*(got_phase-phase))), 0, atol=1e-9)


def test_rotation_preserves_protected_interval_and_other_detector():
    t = np.arange(-3., 35., .0125)
    pair = np.zeros((2, 2, t.size), dtype=np.float32) + .5
    for d in range(2):
        pair[:, d, :] += (.04*np.cos(2*np.pi*2.5*t + .3*d))[None, :]
    zero, _ = phase_rotate_prepared(pair, t, phase_shift_deg=0.,
                                   detectors=(0,), region="baseline_tail")
    assert np.array_equal(zero, pair)
    rotated, _ = phase_rotate_prepared(
        pair, t, phase_shift_deg=90.,
        detectors=(0,), region="baseline_tail")
    protected = (t >= -1) & (t <= 12)
    assert np.array_equal(rotated[:, :, protected], pair[:, :, protected])
    assert np.array_equal(rotated[:, 1], pair[:, 1])
    assert np.max(np.abs(rotated[:, 0, t < -1.5] -
                         pair[:, 0, t < -1.5])) > .02
    assert np.max(np.abs(rotated[:, 0, t > 14] -
                         pair[:, 0, t > 14])) > .02
