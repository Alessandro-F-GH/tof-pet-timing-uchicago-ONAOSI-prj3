"""Synthetic peak-frequency and phase extraction tests."""
import numpy as np

from waveform_analysis.scripts.group_saved_predictions_by_peak_frequency_phase import (
    event_peak_frequency_phase, event_joint_peak_frequency_phases, group_grid)


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


def test_quantile_frequency_bins_balance_counts_and_preserve_ties():
    rng = np.random.default_rng(12)
    values = np.round(rng.normal(2.4, .18, size=8000), 2)
    phase = rng.uniform(-np.pi, np.pi, size=len(values))
    pred = rng.normal(size=len(values))
    rows, fbin, pbin = group_grid(
        values, phase, pred, frequency_bins=8, phase_bins=12,
        frequency_binning="quantile", phase_binning="uniform")
    assert len(rows) == 96
    counts = np.bincount(fbin, minlength=8)
    assert counts.sum() == len(values)
    assert np.max(np.abs(counts - 1000)) < 200
    for val in np.unique(values):
        assert np.unique(fbin[values == val]).size == 1


def test_optional_quantile_phase_bins():
    freq = np.linspace(2, 3, 240)
    phase = np.linspace(-3, 3, 240)**3 / 9
    pred = np.sin(phase)
    rows, fbin, pbin = group_grid(
        freq, phase, pred, frequency_bins=6, phase_bins=8,
        frequency_binning="quantile", phase_binning="quantile")
    assert np.bincount(fbin, minlength=6).tolist() == [40]*6
    assert np.bincount(pbin, minlength=8).tolist() == [30]*8


def test_joint_peak_frequency_and_relative_phase():
    t = np.arange(12., 45., .0125)
    frequencies = np.array([2.28, 2.52, 2.64])
    phases1 = np.array([.4, -1.2, .8])
    phases2 = np.array([1.1, -.6, -2.])
    x = np.zeros((3, 2, len(t)), dtype=np.float64)
    x[:, 0] = 1. + .07*np.cos(
        2*np.pi*frequencies[:, None]*t[None, :] + phases1[:, None])
    x[:, 1] = 1. + .06*np.cos(
        2*np.pi*frequencies[:, None]*t[None, :] + phases2[:, None])
    f, phases, amps, delta = event_joint_peak_frequency_phases(
        x, t, low_GHz=1.5, high_GHz=3.5, fft_padding=8)
    np.testing.assert_allclose(f, frequencies, atol=.008)
    np.testing.assert_allclose(
        np.angle(np.exp(1j*(phases[:,0]-phases1))), 0, atol=.25)
    np.testing.assert_allclose(
        np.angle(np.exp(1j*(phases[:,1]-phases2))), 0, atol=.25)
    np.testing.assert_allclose(
        np.angle(np.exp(1j*(delta-(phases2-phases1)))), 0, atol=.25)
    assert np.all(amps > 0)
