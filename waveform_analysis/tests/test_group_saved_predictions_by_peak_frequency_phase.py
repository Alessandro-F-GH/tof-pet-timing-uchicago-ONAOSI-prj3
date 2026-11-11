"""Synthetic peak-frequency and phase extraction tests."""
import numpy as np

from waveform_analysis.scripts.group_saved_predictions_by_peak_frequency_phase import (
    event_peak_frequency_phase, group_grid)


def test_different_event_frequencies_and_phases():
    t = np.arange(12., 45., .0125)
    freqs = np.array([2.2, 2.5, 2.85])
    phases = np.array([-.5, 1.0, -2.0])
    waves = (3.0 + .002*t[None, :]
             + .05 * np.cos(2*np.pi*freqs[:, None]*t[None, :]
                              + phases[:, None]))
    found_f, found_phi, amplitude, _ = event_peak_frequency_phase(
        waves, t, low_GHz=1.5, high_GHz=3.5, fft_padding=8)
    np.testing.assert_allclose(found_f, freqs, atol=.008)
    assert np.max(np.abs(np.angle(np.exp(1j*(found_phi-phases))))) < .25
    assert np.all(amplitude > 0)


def test_frequency_phase_bins_group_predictions():
    freq = np.array([1.6, 1.65, 3.4])
    phase = np.array([-2., -2.2, 2.])
    predictions = np.array([10., 30., 40.])
    rec, f_bin, p_bin = group_grid(
        freq, phase, predictions, frequency_bins=2, phase_bins=4,
        low=1.5, high=3.5)
    assert f_bin[0] == f_bin[1] != f_bin[2]
    assert p_bin[0] == p_bin[1] != p_bin[2]
    matched = [r for r in rec if r["frequency_bin"] == f_bin[0]
               and r["phase_bin"] == p_bin[0]]
    assert len(matched) == 1
    assert matched[0]["n"] == 2
    assert matched[0]["mean_prediction_ps"] == 20.
