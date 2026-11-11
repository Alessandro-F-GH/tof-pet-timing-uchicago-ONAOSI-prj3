"""Lowpass probe tests: cutoff response and untouched paired detector."""
import numpy as np
import pytest

from waveform_analysis.scripts.test_lowpass_model_sensitivity import lowpass_prepared


def test_lowpass_removes_high_frequency_without_deleting_slow_envelope():
    dt = .0125  # ns = 80 GS/s
    t = np.arange(4096) * dt
    slow = .4 + .1 * np.sin(2*np.pi*.12*t)
    high = .025*np.sin(2*np.pi*2.5*t)
    values = np.zeros((1, 2, len(t)), np.float32)
    values[0, 0] = slow + high
    values[0, 1] = .3 + high
    filtered, excess = lowpass_prepared(
        values, sample_rate_GHz=80., cutoff_GHz=1.,
        detectors=(0,), order=4)
    assert np.array_equal(filtered[0, 1], values[0, 1])
    middle = slice(512, -512)  # avoid endpoints
    assert np.std(filtered[0, 0, middle] - slow[middle]) < .003
    assert np.std(values[0, 0, middle] - slow[middle]) > .015
    assert excess == pytest.approx(0., abs=1e-8)


def test_invalid_lowpass_settings():
    values = np.zeros((2, 2, 256), dtype=np.float32)
    with pytest.raises(ValueError):
        lowpass_prepared(values, sample_rate_GHz=80., cutoff_GHz=40.)
    with pytest.raises(ValueError):
        lowpass_prepared(values, sample_rate_GHz=80., cutoff_GHz=1.,
                         detectors=())
