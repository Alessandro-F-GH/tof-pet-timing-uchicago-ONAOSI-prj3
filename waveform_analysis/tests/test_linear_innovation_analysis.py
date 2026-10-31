import numpy as np
import pytest

from waveform_analysis.scripts.analyze_linear_innovations import (
    _build_temporal_groups,
    _contribution_batch,
    _compute_profiles,
    _normalized_profile,
    _parse_board_paths,
    _safe_correlation,
)


def test_parse_board_paths_accepts_windows_paths():
    parsed = _parse_board_paths(
        [r"UC=C:\\results\\UC", r"FBK=D:\\results\\FBK"], "--results"
    )
    assert set(parsed) == {"UC", "FBK"}
    assert parsed["UC"].name == "UC"


@pytest.mark.parametrize("value", ["UC", "UC=", "OTHER=path"])
def test_parse_board_paths_rejects_invalid_or_duplicate_entries(value):
    values = [value]
    with pytest.raises(ValueError):
        _parse_board_paths(values, "--results")


def test_temporal_groups_cover_samples_without_crossing_masked_gaps():
    indices = np.asarray([0, 1, 2, 5, 6, 7, 8])
    time = np.asarray([0, 1, 2, 5, 6, 7, 8], dtype=float)
    full = np.arange(10, dtype=float)
    groups = _build_temporal_groups(indices, time, full, 2, 2)
    covered = np.concatenate([group["positions"] for group in groups])
    np.testing.assert_array_equal(np.sort(covered), np.arange(len(indices)))
    for group in groups:
        original = indices[group["positions"]]
        assert np.all(np.diff(original) == 1)
    assert len(groups) == 4


def test_temporal_groups_require_gap_free_stride_coverage():
    with pytest.raises(ValueError, match="stride_samples"):
        _build_temporal_groups(np.arange(8), np.arange(8), np.arange(8), 4, 5)


def test_direct_and_shared_contributions_sum_to_ridge_prediction():
    rng = np.random.default_rng(4)
    n, p = 12, 7
    direct_x = rng.normal(size=(n, 2 * p))
    direct_w = rng.normal(size=2 * p)
    direct, detectors = _contribution_batch(direct_x, direct_w, "direct", p)
    assert direct.shape == (n, p)
    assert detectors.shape == (n, 2, p)
    np.testing.assert_allclose(direct.sum(axis=1), direct_x @ direct_w)

    shared_x = rng.normal(size=(n, p))
    shared_w = rng.normal(size=p)
    shared, by_detector = _contribution_batch(shared_x, shared_w, "shared", p)
    assert by_detector is None
    np.testing.assert_allclose(shared.sum(axis=1), shared_x @ shared_w)


def test_contribution_profiles_and_grouped_contributions_are_consistent():
    rng = np.random.default_rng(5)
    features = rng.normal(size=(20, 8))
    weights = rng.normal(size=8)
    groups = [
        {"index": 0, "positions": np.asarray([0, 1])},
        {"index": 1, "positions": np.asarray([2, 3])},
        {"index": 2, "positions": np.asarray([4, 5, 6, 7])},
    ]
    profile, grouped, prediction = _compute_profiles(
        features, np.arange(20), weights, 0.3, "shared", 8, groups, 6
    )
    np.testing.assert_allclose(
        grouped.sum(axis=1) + 0.3, prediction, atol=2e-6
    )
    np.testing.assert_allclose(profile["mean_signed_pair"].sum() + 0.3,
                               np.mean(prediction), atol=2e-6)
    assert profile["rms_pair"].shape == (8,)


def test_normalized_profile_has_unit_area_and_safe_correlation():
    values = _normalized_profile([1, 2, 1], [0, 1000, 2000])
    np.testing.assert_allclose(np.trapz(values, [0, 1, 2]), 1.0)
    assert _safe_correlation([1, 2, 3], [2, 4, 6]) == pytest.approx(1.0)
    assert np.isnan(_safe_correlation([1, 1], [2, 3]))
