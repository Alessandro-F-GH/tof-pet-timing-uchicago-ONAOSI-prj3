"""Tests of RDST best-match temporal attribution without requiring aeon."""
import numpy as np
import pytest
from types import SimpleNamespace

from waveform_analysis.engine.rdst_temporal import recover_argmin, temporal_match_maps


def test_decode_scaled_argmin_and_reject_invalid_positions():
    scales = np.array([2., 4., 3., 2., 0.5, 2.])
    scaled = np.array([
        [0.1, 2/4, 0.4, 0.2, 5/0.5, 0.3],
        [0.2, 4/4, 0.5, 0.1, 1/0.5, 0.4],
    ], dtype=np.float32)
    decoded = recover_argmin(scaled, scales, [3, 4], [2, 1], 10)
    np.testing.assert_array_equal(decoded, [[2, 5], [4, 1]])
    bad = scaled.copy()
    bad[0, 1] = 0.375
    with pytest.raises(ValueError, match="argmin"):
        recover_argmin(bad, scales, [3, 4], [2, 1], 10)


def _dummy_fitted(shared):
    scaler = SimpleNamespace(scale_=np.ones(6, dtype=np.float64))
    transformer = SimpleNamespace(shapelets_=(
        np.zeros((2, 1, 3)), None, np.array([3, 3]),
        np.array([1, 1]), None, None, None, None, None,
    ))
    transform = SimpleNamespace(transformer=transformer, scaler=scaler, shared=shared)
    return SimpleNamespace(feature_transform=SimpleNamespace(artifact=transform),
                           sample_mask=np.ones(8, dtype=bool))


def test_direct_temporal_map_conserves_net_shapelet_contribution():
    n = 30
    features = np.zeros((n, 6), dtype=np.float32)
    features[:, 1] = np.arange(n) % 5
    features[:, 4] = 3
    features[:, 0] = 10.
    features[:, 2] = -10.
    weights = np.array([1., 0., 1., 2., 0., 1.], dtype=float)
    labels = np.full(n, -1, dtype=np.int8)
    labels[0:4] = 0
    labels[4:8] = 1
    labels[8:12] = 2
    result = temporal_match_maps(
        spec=None, fitted=_dummy_fitted(False), blind=None, mode="energy_to_energy",
        features=features, reference_mean=np.zeros(6),
        coefficients=weights, group_labels=labels,
        time_ps=np.arange(8)*100., chunk_size=7,
    )
    assert result["n_events"].tolist() == [30, 4, 4, 4]
    assert result["detector_names"].tolist() == ["joint_pair"]
    # First shapelet's terms +10 and -10 cancel *before* abs.
    # Second shapelet has a +3 contribution.
    assert result["net_shapelet_abs_ps"][0] == pytest.approx(3.)
    assert result["combined_absolute_ps"][0, 3] == pytest.approx(3.)
    assert result["combined_fraction"][0].sum() == pytest.approx(1.)
    assert result["combined_match_count_per_event"][0].sum() == pytest.approx(2.)
