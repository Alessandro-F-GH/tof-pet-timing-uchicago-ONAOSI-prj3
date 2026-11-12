"""Safety tests for phase extraction and event identity."""
import numpy as np
import pytest

from waveform_analysis.scripts.test_tail_phase_ridge import (
    _align, _features, _phase_of_tail,
)


def test_join_uses_event_identity_not_position():
    root_ids = np.array([101, 103, 102, 105])
    prediction_ids = np.array([105, 101, 102])
    rows = _align(root_ids, prediction_ids, role="development")
    np.testing.assert_array_equal(rows, [3, 0, 2])


def test_join_rejects_missing_event():
    with pytest.raises(ValueError, match="missing from ROOT"):
        _align([1, 3], [1, 2], role="blind")


def test_join_rejects_duplicate_event():
    with pytest.raises(ValueError, match="duplicate"):
        _align([1, 1], [1], role="blind")


def test_phase_recovers_relative_shift_on_raw_waveform():
    dt_ns = .025
    f = 2.5
    t = np.arange(2400) * dt_ns
    # Pulse for trigger/peak selection, plus an oscillation in its tail.
    pulse = 250 * np.exp(-.5 * ((t - 10.0) / .6)**2)
    c1 = pulse + 8 * np.cos(2 * np.pi * f * t)
    c2 = pulse + 8 * np.cos(2 * np.pi * f * (t - .075))
    a = _phase_of_tail(
        c1, dt_ns=dt_ns, time0_ns=0., frequency_ghz=f,
        tail_start_ns=5., tail_end_ns=25., threshold_mv=40.,
    )
    b = _phase_of_tail(
        c2, dt_ns=dt_ns, time0_ns=0., frequency_ghz=f,
        tail_start_ns=5., tail_end_ns=25., threshold_mv=40.,
    )
    features = _features(np.array([[a, b]]), "relative")
    expected = np.exp(2j * np.pi * f * .075)
    np.testing.assert_allclose(features[0], [expected.real, expected.imag], atol=.07)
