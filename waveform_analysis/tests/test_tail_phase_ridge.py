"""Event-identity and native-time projection tests for exploratory phase add-on."""
import numpy as np
import pytest

from waveform_analysis.scripts.test_tail_phase_ridge import _align, _tail_phase, _features


def test_event_join_is_by_identity_not_position():
    native = np.array([101, 103, 102, 105])
    filtered = np.array([105, 101, 109, 102])
    row, keep = _align(native, filtered, role="development")
    np.testing.assert_array_equal(keep, [True, True, False, True])
    np.testing.assert_array_equal(row, [3, 0, 2])


def test_event_join_rejects_duplicates():
    with pytest.raises(ValueError, match="duplicate"):
        _align([1, 1], [1], role="blind")


def test_phase_tail_recovers_relative_offset():
    n, dt_ns, frequency = 2048, .025, 2.5
    t_ns = np.arange(n) * dt_ns
    shift = .075
    first = np.cos(2*np.pi*frequency*t_ns)
    second = np.cos(2*np.pi*frequency*(t_ns - shift))
    wave = np.stack([first, second], axis=0)[None, :, :]
    native = {
        "windows_mV": wave,
        "window_start_time_s": np.zeros((1, 2)),
        "sample_interval_s": np.full((1, 2), dt_ns*1e-9),
        "rising_stop": np.full((1, 2), 100, dtype=np.int32),
        "event_id": np.array([12], dtype=np.int64),
    }
    coefficients = _tail_phase(native, np.array([0]), frequency_ghz=frequency,
                               start_ns=5, end_ns=30)
    features = _features(coefficients, "relative")
    expected = np.exp(2j*np.pi*frequency*shift)
    np.testing.assert_allclose(features[0], [expected.real, expected.imag], atol=.025)
