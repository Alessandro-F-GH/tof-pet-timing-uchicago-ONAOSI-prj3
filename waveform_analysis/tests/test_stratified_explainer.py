"""Deterministic rank groups and original linear-head explanation invariants."""
import numpy as np
import pytest

from waveform_analysis.engine.stratified_explainer import (
    select_quantile_groups, aggregate_contributions,
)


def test_rank_groups_equal_population_and_representatives():
    n = 1000
    ids = np.arange(n, dtype=np.int64) + 500
    predictions = np.linspace(-80, 80, n)
    labels, groups = select_quantile_groups(ids, predictions)
    assert [groups[g]["n_events"] for g in ("low", "central", "high")] == [80, 80, 80]
    assert sum(labels >= 0) == 240
    assert groups["low"]["representative_index"] == 60
    assert groups["central"]["representative_index"] == 500
    assert groups["high"]["representative_index"] == 940
    assert len(np.unique(np.concatenate([g["indices"] for g in groups.values()]))) == 240


def test_ties_are_deterministic_and_nonoverlapping():
    ids = np.arange(500)
    predictions = np.zeros(500)
    labels, groups = select_quantile_groups(ids, predictions)
    labels2, groups2 = select_quantile_groups(ids, predictions)
    np.testing.assert_array_equal(labels, labels2)
    assert all(np.array_equal(groups[k]["indices"], groups2[k]["indices"]) for k in groups)
    assert all(groups[k]["n_events"] == 40 for k in groups)


def test_invalid_quantiles_and_id_alignment():
    with pytest.raises(ValueError):
        select_quantile_groups([1, 1, 2], [1., 2., 3.])
    with pytest.raises(ValueError):
        select_quantile_groups([1, 2], [1., np.nan])
    with pytest.raises(ValueError, match="overlap"):
        select_quantile_groups(np.arange(100), np.arange(100),
                              {"a": (.1, .4), "b": (.3, .5)})


def test_aggregate_preserves_net_rdst_cancellation_and_centering():
    n = 1000
    features = np.zeros((n, 6), dtype=np.float32)
    features[:, 0] = np.linspace(-2, 2, n)
    features[:, 1] = -features[:, 0]
    features[:, 3] = np.linspace(-1, 1, n)
    weights = np.array([3., 3., 0., 4., 0., 0.])
    mu = np.array([1., -1., 0., 0., 0., 0.])
    y = 2. + features @ weights
    labels, _ = select_quantile_groups(np.arange(n), y)
    result = aggregate_contributions(features, weights, mu, y, labels,
                                     group_size=3, intercept=2., clip=None, chunk_size=117)
    assert result["clipped_events"] == 0
    assert result["max_reconstruction_error_ps"] < 1e-5
    # First shapelet's +3 and -3 cancel *before* absolute value.
    assert result["mean_absolute_ps"][0, 0] == pytest.approx(0., abs=1e-12)
    assert result["mean_signed_ps"][0, 1] == pytest.approx(0., abs=1e-6)
    assert result["reference_prediction_ps"] == pytest.approx(2.)


def test_shared_feature_difference_flips_contribution_sign():
    n = 1000
    rng = np.random.default_rng(3)
    one = rng.normal(size=(n, 6)).astype("float32")
    two = rng.normal(size=(n, 6)).astype("float32")
    weights = np.arange(1, 7, dtype=float)
    delta = one - two
    swapped = two - one
    y = delta @ weights
    labels, _ = select_quantile_groups(np.arange(n), y)
    reference = np.zeros(6)
    first = aggregate_contributions(delta, weights, reference, y, labels,
                                    group_size=3, intercept=0., clip=None, chunk_size=113)
    reverse = aggregate_contributions(swapped, weights, reference, -y, labels,
                                      group_size=3, intercept=0., clip=None, chunk_size=113)
    np.testing.assert_allclose(first["mean_signed_ps"], -reverse["mean_signed_ps"], atol=1e-6)
    np.testing.assert_allclose(first["mean_absolute_ps"], reverse["mean_absolute_ps"], atol=1e-6)


def test_clipping_checked_against_teacher_and_raises_on_mismatch():
    n = 500
    features = np.arange(n, dtype=np.float32).reshape(-1, 1)
    predictions = np.clip(features[:, 0], -10, 10)
    labels, _ = select_quantile_groups(np.arange(n), predictions)
    result = aggregate_contributions(features, [1.], [0.], predictions, labels,
                                     group_size=1, intercept=0, clip=10, chunk_size=70)
    assert result["clipped_events"] == n - 11
    with pytest.raises(ValueError, match="reconstruction"):
        aggregate_contributions(features, [1.], [0.], predictions+1, labels,
                                group_size=1, intercept=0, clip=10, chunk_size=70)
