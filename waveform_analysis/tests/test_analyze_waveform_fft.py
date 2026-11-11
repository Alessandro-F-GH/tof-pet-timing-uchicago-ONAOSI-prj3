"""Checks frequency calibration, coherence, event ordering, detector isolation."""
import numpy as np
import pytest

from waveform_analysis.scripts.analyze_waveform_fft import (
    remove_linear_trend, spectral_accumulators, validate_grid)


def test_grid_sampling_interval():
    t = np.arange(-300, 700, 10., dtype=float)
    grid, dt = validate_grid(t)
    assert dt == 10.
    with pytest.raises(ValueError):
        validate_grid(np.array([0., 10., 20., 31., 40.]))


def test_detrend_removes_dc_and_slope():
    data = np.stack([4 + .2 * np.arange(128),
                     -3 - .8 * np.arange(128)])
    corrected = remove_linear_trend(data)
    assert np.max(np.abs(corrected)) < 1.e-10


def test_common_phase_oscillation_survives_averaging():
    # 4-GHz oscillation; individual and coherent powers should be identical
    # for identical traces, with phase-consistency = 1.
    time = np.arange(-2000, 2000 + 10, 10., dtype=float)
    wave = np.sin(2 * np.pi * 4 * time / 1000.) / 4.
    signals = np.zeros((10, 2, time.size), dtype=np.float32)
    signals[:, 0, :] = wave[None, :]
    signals[:, 1, :] = 0.  # detector 2 must not contribute to detector 1
    out = spectral_accumulators(
        signals, time, selected_rows=np.arange(10), detector=0,
        window=(-1.5, -.5), max_individual=2, chunk_size=3,
        physical_min=0., physical_span=1.)
    freq = out["frequency_GHz"]
    k = int(np.argmin(np.abs(freq - 4.)))
    assert out["phase_consistency"][k] == pytest.approx(1., abs=1e-6)
    assert out["coherent_power"][k] == pytest.approx(
        out["mean_individual_power"][k], rel=1e-6)


def test_random_phase_oscillation_vanishes_in_aggregate():
    time = np.arange(-2000, 2000 + 10, 10., dtype=float)
    rng = np.random.default_rng(7)
    n = 300
    phases = rng.uniform(0, 2 * np.pi, size=n)
    signals = np.zeros((n, 2, time.size), dtype=np.float32)
    signals[:, 0, :] = np.sin(
        2 * np.pi * 4 * time[None, :] / 1000. + phases[:, None])
    out = spectral_accumulators(
        signals, time, selected_rows=np.arange(n), detector=0,
        window=(-1.5, -.5), max_individual=2, chunk_size=41,
        physical_min=0., physical_span=1.)
    k = int(np.argmin(np.abs(out["frequency_GHz"] - 4.)))
    assert out["mean_individual_power"][k] > 0
    assert out["phase_consistency"][k] < .05


def test_mean_subtracted_power_removes_common_component():
    time = np.arange(-2000, 4000, 12.5)
    rng = np.random.default_rng(123)
    phases = rng.uniform(-np.pi, np.pi, 250)
    common = 3.0*np.cos(2*np.pi*1.5*time/1000.)
    random_component = (.4*np.cos(2*np.pi*2.5*time[None,:]/1000.
                                + phases[:, None]))
    waves = np.zeros((len(phases), 2, len(time)), np.float32)
    waves[:,0,:] = common[None,:] + random_component
    result = spectral_accumulators(
        waves, time, selected_rows=np.arange(len(phases)), detector=0,
        window=(-1.5, 3.5), max_individual=3, chunk_size=37,
        physical_min=0., physical_span=1.)
    freq = result["frequency_GHz"]
    common_bin = np.argmin(abs(freq-1.5))
    variable_bin = np.argmin(abs(freq-2.5))
    assert result["variable_power"][common_bin] < result["coherent_power"][common_bin]*.01
    assert result["variable_power"][variable_bin] > result["coherent_power"][variable_bin]*10
    np.testing.assert_allclose(
        result["variable_power"],
        result["mean_individual_power"]-result["coherent_power"],
        atol=1e-5, rtol=1e-7)
    centered = result["individual_centered_waveforms_mV"]
    assert centered.shape == (3, sum((time >= -1500) & (time <= 3500)))
