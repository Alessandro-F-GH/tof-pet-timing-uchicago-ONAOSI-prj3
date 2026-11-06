"""Tests for conditional KNN: never require a model or a fitted feature transform."""
from __future__ import annotations

import numpy as np
import pytest

from waveform_analysis.engine.conditional_knn import ConditionalKNN, tshap_windows
from waveform_analysis.scripts.evaluate_conditional_knn import _prediction_payload, _choose


def test_aggregated_distances_and_loo_against_bruteforce():
    rng = np.random.default_rng(7)
    waves = rng.uniform(0.0, 1.0, (31, 2, 14)).astype(np.float32)
    labels = rng.normal(0, 10, 31)
    keep = np.ones(14, dtype=bool)
    starts, mask = tshap_windows(14, 3, 3, keep)
    k = np.array([1, 2, 5, 10], dtype=int)
    engine = ConditionalKNN(waves, labels, starts=starts, width=3,
                            keep=mask, k_values=k, weighted=True, reference_batch=7)
    for query_id in (0, 17, 30):
        estimated, nearest, radii = engine.estimate(query_id)
        for wi, start in enumerate(starts):
            for side in range(2):
                observed = np.zeros(14, dtype=bool)
                observed[start:start + 3] = True
                if side:
                    observed = ~observed
                dist = np.sqrt(np.mean(
                    (waves.astype(np.float64)[:, :, observed]
                     - waves[query_id].astype(np.float64)[None, :, observed]) ** 2,
                    axis=(1, 2),
                ))
                dist[query_id] = np.inf
                ordered = np.lexsort((np.arange(len(dist)), dist))[:10]
                np.testing.assert_allclose(nearest[wi, side], dist[ordered[0]], atol=1e-10)
                np.testing.assert_allclose(radii[wi, side], dist[ordered[-1]], atol=1e-10)
                for ki, count in enumerate(k):
                    np.testing.assert_allclose(
                        estimated[wi, side, ki, 0], np.mean(labels[ordered[:count]]),
                        atol=1e-10,
                    )
                    weights = 1 / dist[ordered[:count]]
                    np.testing.assert_allclose(
                        estimated[wi, side, ki, 1],
                        np.sum(labels[ordered[:count]] * weights) / weights.sum(),
                        atol=1e-10,
                    )


def test_masked_coordinates_do_not_affect_neighbors():
    rng = np.random.default_rng(1)
    waves = rng.normal(size=(21, 2, 12)).astype(np.float32)
    keep = np.ones(12, bool)
    keep[[0, 11]] = False
    starts, keep = tshap_windows(12, 2, 2, keep)
    values = np.arange(len(waves), dtype=float)
    before = ConditionalKNN(waves, values, starts=starts, width=2, keep=keep,
                            k_values=[1, 5], reference_batch=6).estimate(3)[0]
    changed = waves.copy()
    changed[:, :, ~keep] += 1e5
    after = ConditionalKNN(changed, values, starts=starts, width=2, keep=keep,
                           k_values=[1, 5], reference_batch=4).estimate(3)[0]
    np.testing.assert_allclose(before, after)


def test_zero_distance_weighted_neighbors_and_leave_one_out():
    wave = np.zeros((7, 2, 10), dtype=np.float32)
    # Query 0 and two other events are duplicates; query itself must be omitted.
    for j in range(3, 7):
        wave[j, :, :] = float(j)
    labels = np.array([100., 5., 7., 30., 40., 50., 60.])
    starts, keep = tshap_windows(10, 2, 2)
    knn = ConditionalKNN(wave, labels, starts=starts, width=2, keep=keep,
                         k_values=[1, 2, 4], weighted=True)
    values, nearest, _ = knn.estimate(0)
    np.testing.assert_allclose(values[:, :, 1, 1], 6.)
    np.testing.assert_allclose(nearest, 0.)


def test_invalid_inputs_and_k_bounds():
    waves = np.zeros((3, 2, 10), np.float32)
    starts, keep = tshap_windows(10, 2, 2)
    with pytest.raises(ValueError):
        ConditionalKNN(waves, np.zeros(3), starts=starts, width=2, keep=keep, k_values=[3])
    with pytest.raises(ValueError):
        ConditionalKNN(waves, np.zeros(2), starts=starts, width=2, keep=keep, k_values=[1])
    with pytest.raises(IndexError):
        ConditionalKNN(waves, np.zeros(3), starts=starts, width=2, keep=keep,
                       k_values=[1]).estimate(3)


def test_saved_prediction_provenance_and_id_alignment(tmp_path):
    class Dataset:
        event_index = np.array([8, 3, 12])
        n_events = 3
        manifest = {"analysis_protocol_identity": "p1"}
    manifest = {"stage_fingerprints": {"final_fit": "f1"}}
    file = tmp_path / "saved.npz"
    np.savez(file, event_id=[12, 8, 3], prediction_ps=[120., 80., 30.],
             final_fit_fingerprint="f1", analysis_protocol_identity="p1",
             dataset_role="development")
    np.testing.assert_array_equal(_prediction_payload(file, Dataset(), manifest),
                                  [80., 30., 120.])
    np.savez(file, event_id=[12, 8, 3], prediction_ps=[120., 80., 30.],
             final_fit_fingerprint="OTHER", analysis_protocol_identity="p1",
             dataset_role="development")
    with pytest.raises(ValueError, match="fingerprint"):
        _prediction_payload(file, Dataset(), manifest)


def test_choose_can_select_single_or_distinct_k():
    # [events, windows, {W,R}, k, weight]
    e = np.ones((4, 3, 2, 2, 1))
    e[:, :, 0, 0, 0] = 0.1
    e[:, :, 0, 1, 0] = 1.0
    e[:, :, 1, 0, 0] = 1.0
    e[:, :, 1, 1, 0] = 0.1
    decision = _choose(e, np.array([5, 50]), False)
    assert decision["separate"]["window"]["k"] == 5
    assert decision["separate"]["complement"]["k"] == 50
    assert decision["single"]["k"] in (5, 50)
