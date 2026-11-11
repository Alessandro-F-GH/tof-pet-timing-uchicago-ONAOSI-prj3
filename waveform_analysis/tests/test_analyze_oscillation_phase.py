"""Synthetic checks for fixed-frequency phase analysis."""
import numpy as np
import pytest

from waveform_analysis.scripts.analyze_oscillation_phase import (
    project_fixed_frequency, phase_lock, circular_relation)


def test_phase_and_amplitude_recover_known_sinusoid():
    t_ns = np.arange(-2., 0., .0125)
    f = 2.5
    actual_phase = np.array([0.2, -1.1, 2.4])
    amplitude = np.array([1.2, .6, 2.0])
    signal = amplitude[:, None] * np.cos(
        2 * np.pi * f * t_ns[None, :] + actual_phase[:, None])
    _, recovered, phase = project_fixed_frequency(signal, t_ns, f)
    np.testing.assert_allclose(recovered, amplitude, atol=.07)
    assert np.max(np.abs(np.angle(np.exp(1j*(phase-actual_phase))))) < .07


def test_phase_locking_sensitive_to_relative_phase():
    rng = np.random.default_rng(5)
    pre = rng.uniform(-np.pi, np.pi, size=1000)
    locked = pre + .8
    independent = rng.uniform(-np.pi, np.pi, size=1000)
    assert phase_lock(pre, locked)["resultant_length"] > .99
    assert phase_lock(pre, independent)["resultant_length"] < .1


def test_circular_sinusoid_recovers_association():
    phi = np.linspace(-np.pi, np.pi, 400, endpoint=False)
    correction = 4 + 20 * np.sin(phi) - 6 * np.cos(phi)
    association = circular_relation(phi, correction)
    assert association["r2_in_sample"] == pytest.approx(1., abs=1e-10)
    assert association["sin_coefficient_ps"] == pytest.approx(20.)
    assert association["cos_coefficient_ps"] == pytest.approx(-6.)
