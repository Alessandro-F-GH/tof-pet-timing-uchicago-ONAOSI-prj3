"""Regression tests for waveform profiles stratified by saved teacher output."""
import numpy as np

from waveform_analysis.scripts.plot_normalized_waveforms_by_prediction import (
    group_moments, prediction_groups,
)


def test_prediction_groups_are_training_only():
    predictions = np.arange(200, dtype=np.float64)
    training = np.arange(100, dtype=np.int64)
    groups = prediction_groups(predictions, training)
    assert len(groups) == 3
    assert all(np.all(ids < 100) for _, ids, _, _ in groups)
    assert all(len(ids) == 10 for _, ids, _, _ in groups)


def test_group_moments_matches_numpy_z_scores():
    rng = np.random.default_rng(5)
    waves = rng.normal(size=(12, 2, 20)).astype(np.float32)
    mean = waves[:9].mean(axis=0).astype(np.float64)
    scale = np.maximum(waves[:9].std(axis=0), 0.1).astype(np.float64)
    ids = np.array([1, 4, 6, 8])
    mu, sd = group_moments(waves, ids, mean, scale, chunk_size=2)
    expected = (waves[ids].astype(np.float64) - mean) / scale
    np.testing.assert_allclose(mu, expected.mean(axis=0), atol=1e-12)
    np.testing.assert_allclose(sd, expected.std(axis=0), atol=1e-12)
