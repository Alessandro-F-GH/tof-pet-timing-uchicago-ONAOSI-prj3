"""Tests for full-window event-averaged FFT spectra."""
from types import SimpleNamespace

import numpy as np
import pytest

from waveform_analysis.scripts import global_fft


def test_fft_peak_for_known_sinusoid_and_difference(monkeypatch):
    n, samples, dt_ps = 8, 128, 100.0
    t = np.arange(samples) * dt_ps * 1e-12
    frequency = 1 / (16 * dt_ps * 1e-12)
    pair = np.zeros((n, 2, samples), np.float32)
    for i in range(n):
        pair[i, 0] = np.sin(2*np.pi*frequency*t + 2*np.pi*i/n)
        pair[i, 1] = .5*np.sin(2*np.pi*frequency*t + 2*np.pi*i/n)
    ds = SimpleNamespace(n_events=n, waves=pair)
    monkeypatch.setattr(global_fft, "waveform_view",
                        lambda data, mode, indices: SimpleNamespace(
                            materialize=lambda: data.waves[np.asarray(indices)]))
    frequencies, powers, count = global_fft.average_power(
        ds, "energy_to_energy", np.ones(samples, bool), np.arange(samples)*dt_ps,
        chunk_size=3)
    assert count == n
    np.testing.assert_allclose(powers.shape, (3, len(frequencies)))
    assert np.argmax(powers[2, 1:]) + 1 == 8
    np.testing.assert_allclose(powers[0, 8] / powers[2, 8], 4., rtol=.01)
    np.testing.assert_allclose(powers[1, 8] / powers[2, 8], 1., rtol=.01)


def test_fft_rejects_nonuniform_time_grid():
    with pytest.raises(ValueError, match="uniformly"):
        global_fft.validate_grid([0, 100, 201, 300])
    with pytest.raises(ValueError, match="at least four"):
        global_fft.validate_grid([0, 100, 200])


def test_fft_saves_figure(tmp_path):
    pytest.importorskip("matplotlib")
    frequencies = np.linspace(0, 5e9, 25)
    global_fft.plot_power(frequencies, np.ones((3, 25))*1e-13,
                          tmp_path / "spectra.png")
    assert (tmp_path / "spectra.png").stat().st_size > 0
