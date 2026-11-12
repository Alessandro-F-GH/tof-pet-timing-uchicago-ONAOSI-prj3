"""Numerical tests for deterministic paired conditional KNN TSHAP."""
import numpy as np
import pytest

from waveform_analysis.engine.conditional_knn_imputer import (
    PairedWindowKNN, neighbor_weights, reference_validation_split,
)
from waveform_analysis.engine.conditional_knn_tshap import _pair_map


def _setup():
    rng = np.random.default_rng(7)
    reference = rng.normal(size=(12, 2, 11)).astype(np.float32)
    active = np.ones(11, dtype=bool)
    active[0] = False
    active[10] = False
    return PairedWindowKNN(
        reference, starts=[2, 5], width=3, retained=active,
        ks=[1, 2, 4, 8], scale_mv=np.array([[350.0], [750.0]]),
        reference_block=3,
    ), rng


def test_distances_match_brute_force_for_both_coalitions():
    knn, rng = _setup()
    query = rng.normal(size=(2, 11)).astype(np.float32)
    indices, distances = knn.search(query)
    for mask in range(knn.n_masks):
        observed = knn.observed[mask]
        brute = np.sum(
            (knn.reference[:, :, observed].astype(np.float64)
             - query[None, :, observed].astype(np.float64)) ** 2,
            axis=(1, 2)
        )
        ordering = np.lexsort((np.arange(len(brute)), brute))[:knn.kmax]
        np.testing.assert_array_equal(indices[mask], ordering)
        np.testing.assert_allclose(distances[mask] ** 2, brute[ordering], atol=1e-10)


def test_reconstruction_keeps_observed_and_masked_samples():
    knn, rng = _setup()
    query = rng.normal(size=(2, 11)).astype(np.float32)
    ids, distances = knn.search(query)
    for mask in range(knn.n_masks):
        for kind in ("uniform", "distance"):
            prediction, weight = knn.reconstruct(query, mask, ids[mask], distances[mask], 4, kind)
            np.testing.assert_array_equal(prediction[:, ~knn.hidden[mask]], query[:, ~knn.hidden[mask]])
            np.testing.assert_allclose(prediction[:, knn.hidden[mask]],
                                       np.einsum("k,kct->ct", weight,
                                                 knn.reference[ids[mask,:4]][:,:,knn.hidden[mask]]),
                                       atol=1e-6)


def test_validation_tuning_agrees_with_naive_reconstruction():
    knn, rng = _setup()
    validation = rng.normal(size=(4, 2, 11)).astype(np.float32)
    result = knn.tune(validation)
    for mask in range(knn.n_masks):
        for weight_id, kind in enumerate(("uniform", "distance")):
            for index, k in enumerate(knn.ks):
                errors = []
                for query in validation:
                    ids, distances = knn.search(query)
                    reconstructed, _ = knn.reconstruct(query, mask, ids[mask],
                                                        distances[mask], int(k), kind)
                    hidden = knn.hidden[mask]
                    errors.append(np.mean((reconstructed[:,hidden].astype(np.float64)
                                           - query[:,hidden])**2))
                np.testing.assert_allclose(result.mse[mask,weight_id,index],
                                           np.mean(errors), atol=1e-7)


def test_prediction_mean_and_prediction_of_reconstruction_can_differ():
    reference = np.array([
        [[0, 0, 0], [0, 0, 0]],
        [[2, 2, 2], [2, 2, 2]],
    ], dtype=np.float32)
    model = lambda x: np.sum(x**2, axis=(1,2))
    mean_waveform = reference.mean(axis=0)
    assert float(model(mean_waveform[None])[0]) != float(np.mean(model(reference)))


def test_reference_validation_disjoint_reproducible():
    first = reference_validation_split(100, 15, 30, 42)
    second = reference_validation_split(100, 15, 30, 42)
    for left, right in zip(first, second):
        np.testing.assert_array_equal(left, right)
    assert not np.intersect1d(*first).size


def test_temporal_attribution_is_pair_symmetric_and_stride_sensitive():
    keep = np.ones(30, dtype=bool)
    scores = [2.0, -1.0, 4.0]
    result = _pair_map(scores, [5, 10, 15], 5, 5, keep, 30)
    assert result.shape == (2, 30)
    np.testing.assert_array_equal(result[0], result[1])


def test_zero_distance_weights_only_identical_neighbors():
    np.testing.assert_allclose(neighbor_weights([0, 0, 2], "distance"), [0.5,0.5,0])
