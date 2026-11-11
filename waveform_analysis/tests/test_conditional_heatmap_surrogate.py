"""Tests for pointwise heatmap association and additive Ridge fidelity."""
import numpy as np
import pytest

from waveform_analysis.engine.conditional_heatmap_surrogate import (
    ConditionalMap, split_development, heatmap_pointwise_features,
    calibrate_ridge, surrogate_components, fidelity_metrics,
    load_conditional_map,
)


def _map():
    return ConditionalMap(
        positions=np.array([0, 1, 2]),
        time_ps=np.array([-2000., -1000., 0.]),
        edges=np.array([-2., 0., 2.]),
        mean=np.zeros((2, 3)), scale=np.ones((2, 3)),
        target_mean=np.array([
            [[-10., -10., -10.], [10., 10., 10.]],
            [[-20., -20., -20.], [20., 20., 20.]],
        ]),
        counts=np.full((2, 2, 3), 30, dtype=np.int64),
        baseline_ps=0., representation="original",
        prediction_type="saved_reference",
    )


def test_split_is_disjoint_and_reproduces_reference_rng():
    reference, cal, val = split_development(100, seed=42, calibration_count=10,
                                           validation_count=12)
    expected = np.random.default_rng(42).choice(100, 22, replace=False)
    np.testing.assert_array_equal(cal, expected[:10])
    np.testing.assert_array_equal(val, expected[10:])
    assert not (set(reference) & set(cal))
    assert not (set(reference) & set(val))
    assert not (set(cal) & set(val))
    assert len(reference) + len(cal) + len(val) == 100


def test_lookup_two_detectors_and_out_of_range_mask():
    m = _map()
    wave = np.array([
        [[-1., 1., 0.], [1., -1., 0.]],
        [[3., 1., -1.], [-1., -1., -1.]],
    ])
    result, valid = heatmap_pointwise_features(
        wave, np.array([0, 1]), m,
        full_time_ps=m.time_ps, chunk_size=1)
    np.testing.assert_allclose(result[0, 0], [-10., 10., 10.])
    np.testing.assert_allclose(result[0, 1], [20., -20., 20.])
    assert not valid[1, 0, 0]
    assert result[1, 0, 0] == 0.


def test_sparse_cells_are_missing_not_signed_zero():
    m = _map()
    m.counts[0, 0, 1] = 0
    wave = np.zeros((2, 2, 3))
    result, valid = heatmap_pointwise_features(
        wave, np.arange(2), m, full_time_ps=m.time_ps)
    assert not valid[0, 0, 1]
    assert result[0, 0, 1] == 0.


def test_timing_mask_disables_cropped_samples():
    m = _map()
    wave = np.zeros((2, 2, 3))
    result, valid = heatmap_pointwise_features(
        wave, np.arange(2), m, full_time_ps=m.time_ps,
        retained=np.array([True, False, True]))
    assert not np.any(valid[:, :, 1])
    assert np.all(result[:, :, 1] == 0.)


def test_ridge_contributions_reconstruct_its_prediction():
    rng = np.random.default_rng(8)
    a = rng.normal(size=(140, 2, 3))
    target = 5. + 3. * a[:, 0, 0] - a[:, 1, 2]
    model, grid = calibrate_ridge(
        a.reshape(len(a), -1), target, alphas=[.1, 1., 10.], folds=4)
    contribution, reconstructed = surrogate_components(model, a)
    np.testing.assert_allclose(reconstructed,
                               model.predict(a.reshape(len(a), -1)),
                               atol=1e-8)
    assert contribution.shape == a.shape
    assert grid["alpha"] in (.1, 1., 10.)
    assert fidelity_metrics(reconstructed, target, baseline=5.)["rmse_ps"] < .2


def test_heatmap_loader_rejects_comparison_outputs(tmp_path):
    p = tmp_path / "test.npz"
    m = _map()
    np.savez_compressed(
        p, positions=m.positions, time_ps=m.time_ps, z_edges=m.edges,
        target_mean_ps=m.target_mean[None],
        support_counts=m.counts[None],
        normalization_mean=m.mean, normalization_scale=m.scale,
        training_prediction_mean_ps=m.baseline_ps,
        representation=m.representation, prediction_type="reference_minus_comparison",
        feature_names=np.array(["waveform"]))
    with pytest.raises(ValueError, match="saved_reference"):
        load_conditional_map(p)
