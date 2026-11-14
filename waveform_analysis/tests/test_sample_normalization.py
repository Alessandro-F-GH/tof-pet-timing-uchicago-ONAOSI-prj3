"""Regression checks for the streaming normalization utility."""
import numpy as np
import pytest

torch = pytest.importorskip("torch")

from waveform_analysis.engine.sample_normalization import fit_time_sample_stats, tensor_stats


def test_training_only_statistics_and_std_floor():
    rng = np.random.default_rng(9)
    waves = rng.normal(size=(15, 2, 32)).astype("float32")
    waves[:, :, 1] = 5.0
    waves[12:] += 1000.0
    idx = np.arange(12)
    mean, scale, diag = fit_time_sample_stats(waves, idx, chunk_size=5,
                                              scale_floor_ratio=0.1)
    np.testing.assert_allclose(mean, waves[idx].mean(axis=0), atol=2e-6)
    train_std = waves[idx].astype("float64").std(axis=0)
    np.testing.assert_allclose(
        scale,
        np.maximum(train_std, np.sqrt((train_std ** 2).mean(axis=1))[:, None] * .1),
        rtol=1e-5, atol=1e-6,
    )
    assert np.all(scale > 0)
    assert diag["fraction_floored"][0] > 0


def test_tensor_stats_normalization_matches_numpy():
    rng = np.random.default_rng(5)
    waves = rng.normal(size=(18, 2, 64)).astype("float32")
    mean, scale, _ = fit_time_sample_stats(waves, np.arange(16), chunk_size=4)
    tm, ts = tensor_stats(mean, scale, device="cpu")
    expected = (waves[:3] - mean[None, :, :]) / scale[None, :, :]
    actual = (torch.from_numpy(waves[:3]) - tm.unsqueeze(0)) / ts.unsqueeze(0)
    np.testing.assert_allclose(actual.numpy(), expected, rtol=1e-5, atol=1e-5)
