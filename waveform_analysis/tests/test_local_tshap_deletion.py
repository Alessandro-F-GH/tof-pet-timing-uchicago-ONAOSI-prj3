"""Synthetic checks for the three-imputer paired local TSHAP/deletion protocol."""
from __future__ import annotations

import numpy as np
import pytest

from waveform_analysis.engine.local_tshap_deletion import (
    FrozenImputationBank, validate_windows, paired_shapley,
    deletion_curve, signed_window_order,
)


def bank_and_query():
    rng = np.random.default_rng(19)
    n, t = 80, 24
    latent = rng.normal(size=(n, 1, 1)) * .05
    reference = np.clip(
        .5 + latent + .04 * rng.standard_normal((n, 2, t)), 0, 1).astype(np.float32)
    retained = np.ones(t, bool)
    bank = FrozenImputationBank(reference, retained, knn_k=5,
                                ppca_components=4, knn_block=16)
    query = reference[7].copy()
    starts = np.array([4, 8, 12, 16])
    return bank, query, starts


def predict(x):
    values = np.asarray(x)
    return values[:, 0, 4:20].sum(axis=1) - .8 * values[:, 1, 4:20].sum(axis=1)


@pytest.mark.parametrize("method", ["knn", "ppca", "mean"])
def test_preserves_original_observed_coordinates(method):
    bank, query, starts = bank_and_query()
    context = bank.knn_context(query, starts, 4)
    observed = bank.retained.copy()
    observed[8:12] = False
    costs = context.full_cost - context.window_cost[:, 1]
    result = bank.impute(method, query, observed, knn_cost=costs)
    np.testing.assert_array_equal(result[:, observed], query[:, observed])
    assert np.isfinite(result).all()


@pytest.mark.parametrize("method", ["knn", "ppca", "mean"])
def test_two_player_shapley_and_signed_deletion(method):
    bank, query, starts = bank_and_query()
    context = bank.knn_context(query, starts, 4)
    full = float(predict(query[None])[0])
    result = paired_shapley(bank, query, starts, 4, predict, full,
                            method=method, knn_context=context)
    assert result["scores"].shape == (len(starts),)
    assert np.isfinite(result["scores"]).all()
    expected = .5 * (
        result["v_window"] - result["v_empty"]
        + full - result["v_complement"]
    )
    np.testing.assert_allclose(result["scores"], expected)
    for direction in ("positive", "negative"):
        curve = deletion_curve(
            bank, query, starts, 4, result["scores"], predict, full,
            method=method, direction=direction, knn_context=context)
        assert len(curve["prediction_ps"]) == len(
            curve["removed_window_indices"]) + 1
        np.testing.assert_allclose(curve["prediction_ps"][0], full)
        np.testing.assert_allclose(curve["signed_prediction_change_ps"],
                                   curve["prediction_ps"] - full)
        np.testing.assert_allclose(
            curve["absolute_prediction_deviation_ps"],
            np.abs(curve["prediction_ps"] - full))
        assert curve["absolute_prediction_deviation_ps"][0] == 0
        assert "absolute_error_change_ps" not in curve


def test_deletion_uses_original_query_and_union_not_previous_imputations():
    bank, query, starts = bank_and_query()
    score = np.array([4., 3., -2., -1.])
    context = bank.knn_context(query, starts, 4)
    full = float(predict(query[None])[0])
    for method in ("knn", "ppca", "mean"):
        curve = deletion_curve(bank, query, starts, 4, score, predict, full,
                               method=method, direction="positive",
                               knn_context=context)
        remaining = bank.retained.copy()
        remaining[starts[0]:starts[0]+4] = False
        remaining[starts[1]:starts[1]+4] = False
        costs = (context.full_cost -
                 context.window_cost[:, 0] - context.window_cost[:, 1])
        direct = bank.impute(method, query, remaining, knn_cost=costs)
        np.testing.assert_allclose(curve["prediction_ps"][2],
                                   predict(direct[None])[0], atol=1e-5)


def test_order_signs_and_zero_exclusion():
    scores = np.array([-2., 3., 0., 1., -4., 3.])
    np.testing.assert_array_equal(signed_window_order(scores, "positive"),
                                  [1, 5, 3])
    np.testing.assert_array_equal(signed_window_order(scores, "negative"),
                                  [4, 0])


def test_reject_overlapping_deletion_windows():
    with pytest.raises(ValueError, match="disjoint"):
        validate_windows(np.ones(30, bool), [3, 5], 8)


def test_cache_ppca_reuse_exactly_preserves_reconstruction():
    bank, query, starts = bank_and_query()
    reference = bank.reference
    restored = FrozenImputationBank(
        reference, bank.retained, knn_k=5, ppca_components=4,
        ppca_state=bank.ppca_state)
    mask = np.ones(len(query[0]), bool)
    mask[4:8] = False
    np.testing.assert_allclose(
        bank.impute("ppca", query, mask),
        restored.impute("ppca", query, mask), atol=1e-7)
