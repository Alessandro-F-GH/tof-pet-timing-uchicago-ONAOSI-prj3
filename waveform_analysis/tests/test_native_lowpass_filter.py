"""Focused regression checks for optional full-native-waveform lowpass."""
import numpy as np

from waveform_analysis.signal.filtering import filter_native_waveform


def test_disabled_filter_preserves_raw_waveform():
    x = np.array([1., 5., -4., 3., 2.])
    np.testing.assert_array_equal(filter_native_waveform(
        x, 12.5e-12, {}), x)


def test_2ghz_lowpass_suppresses_2p5ghz_oscillation():
    dt = 12.5e-12
    t = np.arange(4096) * dt
    x = (np.cos(2*np.pi*0.5e9*t)
         + np.cos(2*np.pi*2.5e9*t))
    filtered = filter_native_waveform(
        x, dt, {"lowpass": {"enabled": True, "cutoff_GHz": 2., "order": 4}})
    center = slice(512, -512)
    def amplitude(s, ghz):
        z = np.exp(-2j*np.pi*ghz*1e9*t[center])
        return abs(2*np.mean(s[center]*z))
    assert amplitude(filtered, .5) > .9*amplitude(x, .5)
    assert amplitude(filtered, 2.5) < .35*amplitude(x, 2.5)


def test_invalid_cutoff_rejected():
    from pytest import raises
    with raises(ValueError):
        filter_native_waveform(
            np.ones(512), 12.5e-12,
            {"lowpass": {"enabled": True, "cutoff_GHz": 45., "order": 4}})
