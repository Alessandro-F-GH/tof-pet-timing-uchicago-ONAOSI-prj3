"""Normalized full-waveform kNN against an exhaustive Euclidean reference."""
import numpy as np
from waveform_analysis.scripts.evaluate_full_signal_knn import (
    _standardized_flat, _reference_search, _predict_knn,
)


class DummyLog:
    def info(self, *_args, **_kwargs):
        pass


def test_weighted_aggregation_exact_matches():
    dist = np.array([[0.0, 0.0, 4.0], [1.0, 4.0, 9.0]])
    target = np.array([[3.0, 7.0, 9.0], [1.0, 3.0, 5.0]])
    predicted = _predict_knn(dist, target, np.array([1, 2, 3]), True)
    np.testing.assert_allclose(predicted[0, 1, 1], 5.0)
    np.testing.assert_allclose(predicted[0, 2, 1], 5.0)
    np.testing.assert_allclose(predicted[1, 1, 0], 2.0)
    np.testing.assert_allclose(predicted[1, 1, 1], 5.0 / 3.0)


def test_streamed_knn_matches_brute_force_reference():
    rng = np.random.default_rng(19)
    waves = rng.normal(size=(20, 2, 16)).astype(np.float32)
    labels = rng.normal(size=20)
    train = np.arange(13)
    queries = np.arange(13, 20)
    mean = waves[train].mean(axis=0)
    scale = np.maximum(waves[train].std(axis=0), 1e-3)
    ks = np.array([1, 3, 5])
    estimate, nearest = _reference_search(
        waves, labels, train, queries, mean, scale, ks,
        True, reference_batch=4, query_batch=3, log=DummyLog(), log_every=1)
    x = _standardized_flat(waves, queries, mean, scale).astype(np.float64)
    y = _standardized_flat(waves, train, mean, scale).astype(np.float64)
    distances = ((x[:, None, :] - y[None, :, :]) ** 2).sum(axis=2)
    order = np.argsort(distances, axis=1)
    for row in range(len(queries)):
        np.testing.assert_allclose(nearest[row], np.sqrt(distances[row].min()),
                                   rtol=1e-5, atol=1e-5)
        for idx, k in enumerate(ks):
            ids = order[row, :k]
            np.testing.assert_allclose(estimate[row, idx, 0],
                                       labels[train[ids]].mean(),
                                       rtol=1e-5, atol=1e-5)
            weights = 1 / np.sqrt(distances[row, ids])
            expected = np.average(labels[train[ids]], weights=weights)
            np.testing.assert_allclose(estimate[row, idx, 1], expected,
                                       rtol=1e-5, atol=1e-5)
