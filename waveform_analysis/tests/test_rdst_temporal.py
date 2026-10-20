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
        fitted=_dummy_fitted(False), blind=None, mode="energy_to_energy",
        features=features, reference_mean=np.zeros(6),
        coefficients=weights,
        time_ps=np.arange(8)*100., chunk_size=7,
    )
    assert int(result["n_events"]) == 30
    assert result["detector_names"].tolist() == ["joint_pair"]
    # First shapelet's terms +10 and -10 cancel *before* abs.
    # Second shapelet has a +3 contribution.
    assert float(result["net_shapelet_abs_ps"]) == pytest.approx(3.)
    assert result["combined_absolute_ps"][3] == pytest.approx(3.)
    assert result["combined_fraction"].sum() == pytest.approx(1.)
    assert result["combined_match_count_per_event"].sum() == pytest.approx(2.)


def test_shared_position_cache_reuses_events_and_recovers_after_interruption(tmp_path, monkeypatch):
    from waveform_analysis.engine import rdst_position_cache as cache

    class FakeTransformer:
        def __init__(self):
            self.calls = 0
            self.shapelets_ = (
                np.zeros((2, 1, 3)), np.zeros(2),
                np.array([3, 3]), np.array([1, 1]),
                np.zeros(2), np.zeros(2), np.zeros(2), np.zeros(2), np.zeros(2),
            )
            self.n_shapelets_ = 2

        def transform(self, x):
            self.calls += 1
            z = np.zeros((len(x), 6), dtype=np.float32)
            # Original pair has detector-specific argmin at 2 and 3.
            z[:, 1] = np.asarray(x[:, 0, 0], dtype=np.float32)
            z[:, 4] = np.asarray(x[:, 0, 0], dtype=np.float32)
            return z

    class FakeScaler:
        scale_ = np.ones(6, dtype=np.float64)

        def transform(self, x):
            return x

    transform = FakeTransformer()
    artifact = SimpleNamespace(shared=True, transformer=transform, scaler=FakeScaler())
    fitted = SimpleNamespace(feature_transform=SimpleNamespace(
        artifact=artifact, identity="fixture_frozen_transform"),
        sample_mask=np.ones(8, dtype=bool))
    values = np.zeros((5, 2, 8), dtype=np.float32)
    values[:, 0, 0] = 2
    values[:, 1, 0] = 3
    dataset = SimpleNamespace(
        n_events=5, event_index=np.arange(5),
        manifest={"analysis_protocol_identity": "fixture_population"})
    frozen = np.zeros((5, 6), dtype=np.float32)
    frozen[:, 1] = -1
    frozen[:, 4] = -1
    monkeypatch.setattr(cache, "waveform_view", lambda data, mode, ids: SimpleNamespace(
        materialize=lambda: values[ids]))
    indices = np.array([0, 2, 4])
    args = dict(cache_root=tmp_path, fitted=fitted, dataset=dataset,
                mode="energy_to_energy", frozen_features=frozen,
                time_ps=np.arange(8, dtype=np.float64))
    a = cache.shared_positions_for_events(event_indices=indices, **args)
    assert transform.calls == 1
    np.testing.assert_array_equal(a[:, 0, :], 2)
    np.testing.assert_array_equal(a[:, 1, :], 3)

    b = cache.shared_positions_for_events(event_indices=indices, **args)
    np.testing.assert_array_equal(b, a)
    assert transform.calls == 1

    # Only unseen event index 1 requires another frozen transform.
    cache.shared_positions_for_events(event_indices=np.array([1, 2]), **args)
    assert transform.calls == 2

    # Changing the control mask must not reuse the old position cache.
    fitted.sample_mask[7] = False
    with pytest.raises(ValueError, match="grid"):
        cache.shared_positions_for_events(event_indices=indices, **args)

def test_distribute_dilated_matches_uses_only_actual_taps():
    from waveform_analysis.engine.rdst_temporal import distribute_dilated_matches
    signed, absolute, frequency = distribute_dilated_matches(
        np.array([[2]]), np.array([[-12.]]),
        np.array([3]), np.array([2]), 10)
    np.testing.assert_allclose(signed[[2, 4, 6]], [-4, -4, -4])
    np.testing.assert_allclose(absolute[[2, 4, 6]], [4, 4, 4])
    assert signed.sum() == pytest.approx(-12.)
    assert absolute.sum() == pytest.approx(12.)
    assert frequency.sum() == pytest.approx(1.)
    assert np.count_nonzero(absolute) == 3
    assert absolute[3] == 0 and absolute[5] == 0

def test_distribute_dilated_matches_rejects_outside_grid():
    from waveform_analysis.engine.rdst_temporal import distribute_dilated_matches
    with pytest.raises(ValueError, match="Invalid dilated"):
        distribute_dilated_matches(
            np.array([[5]]), np.array([[1.]]),
            np.array([3]), np.array([2]), 9)
