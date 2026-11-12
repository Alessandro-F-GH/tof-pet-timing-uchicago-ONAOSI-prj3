"""Lowpass probe tests: cutoff response and untouched paired detector."""
import numpy as np
import pytest

from waveform_analysis.scripts.test_lowpass_model_sensitivity import (
    lowpass_prepared, temporal_filter_weights)


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


def test_selective_blending_preserves_entire_rising_edge_exactly():
    t_ns = np.arange(-3., 30., .0125)
    x = np.full((3, 2, t_ns.size), .4, dtype=np.float32)
    x[:, 0, :] += .08 * np.sin(2 * np.pi * 2.5 * t_ns)[None, :]
    x[:, 1, :] += .06 * np.cos(2 * np.pi * 2.5 * t_ns)[None, :]
    y, _ = lowpass_prepared(
        x, sample_rate_GHz=80., cutoff_GHz=1., detectors=(0, 1),
        time_ns=t_ns, region="baseline_tail",
        baseline_end_ns=-1., tail_start_ns=12., transition_ns=.5)
    protected = (t_ns >= -1.) & (t_ns <= 12.)
    assert np.array_equal(y[:, :, protected], x[:, :, protected])
    assert np.max(np.abs(y[:, :, t_ns < -2.] - x[:, :, t_ns < -2.])) > .03
    assert np.max(np.abs(y[:, :, t_ns > 14.] - x[:, :, t_ns > 14.])) > .03


def test_selective_region_mask_and_detector_isolation():
    t = np.arange(-3., 20., .0125)
    weights = temporal_filter_weights(
        t, region="baseline_tail", baseline_end_ns=-1.,
        tail_start_ns=12., transition_ns=.5)
    assert np.all(weights[(t >= -1.) & (t <= 12.)] == 0)
    assert np.all(weights[t < -2] == 1)
    assert np.all(weights[t > 14] == 1)
    x = np.zeros((2, 2, len(t)), dtype=np.float32) + .5
    x[:, 0] += .1 * np.sin(2 * np.pi * 2.5 * t)
    y, _ = lowpass_prepared(
        x, sample_rate_GHz=80., cutoff_GHz=1.,
        detectors=(0,), time_ns=t, region="baseline")
    assert np.array_equal(y[:, 1], x[:, 1])
    assert np.array_equal(y[:, 0, t >= -1.], x[:, 0, t >= -1.])
