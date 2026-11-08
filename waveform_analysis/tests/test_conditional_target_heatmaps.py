"""Binned conditional teacher-output heatmaps: support and derivatives."""
import numpy as np

from waveform_analysis.scripts.plot_conditional_target_heatmaps import accumulate_heatmaps


def test_heatmap_weighted_means_and_paired_derivative():
    waves = np.zeros((4, 2, 5), dtype=np.float32)
    waves[:, 0, :] = np.array([
        [0, 1, 2, 3, 4],
        [0, 1, 2, 3, 4],
        [0, 2, 4, 6, 8],
        [0, 2, 4, 6, 8],
    ])
    waves[:, 1, :] = 0
    labels = np.array([10., 30., 50., 70.])
    mean = np.zeros((2, 5), dtype=np.float32)
    scale = np.ones((2, 5), dtype=np.float32)
    edges, sums, counts = accumulate_heatmaps(
        waves, labels, np.arange(4), np.array([0, 1, 3, 4]),
        mean, scale, mean, scale, bins=16, z_limit=10, chunk_size=2)
    # At t=1, detector 1 value 1 has labels 10 and 30.
    bin_at_one = int(np.searchsorted(edges, 1, side="right") - 1)
    assert counts[0, 0, bin_at_one, 1] == 2
    assert sums[0, 0, bin_at_one, 1] == 40.
    # Same binned values for derivative at t=1 (dx=1 and dx=2).
    assert counts[1, 0, bin_at_one, 1] == 2
    assert sums[1, 0, bin_at_one, 1] == 40.
    # Difference detector2 - detector1 equals -1 at t=1.
    bin_minus_one = int(np.searchsorted(edges, -1, side="right") - 1)
    assert counts[0, 2, bin_minus_one, 1] == 2
    assert sums[0, 2, bin_minus_one, 1] == 40.
    # Absolute train-only targets: no target binning or percentiles.
    assert int(counts[0, 0].sum()) == 4 * 4


def test_heatmap_excludes_out_of_range_values():
    waves = np.zeros((2, 2, 3), dtype=np.float32)
    waves[0, 0, 2] = 10
    labels = np.array([-1., 1.])
    mean = np.zeros((2, 3), dtype=np.float32)
    scale = np.ones((2, 3), dtype=np.float32)
    _, _, counts = accumulate_heatmaps(
        waves, labels, np.arange(2), np.arange(3),
        mean, scale, mean, scale, bins=8, z_limit=3, chunk_size=2)
    assert int(counts[0, 0].sum()) == 5


def test_prediction_comparison_metrics_agreement_and_quantiles():
    from waveform_analysis.scripts.plot_conditional_target_heatmaps import (
        prediction_comparison_metrics,
    )
    ref = np.linspace(-100., 100., 201)
    other = ref - 3.
    metrics = prediction_comparison_metrics(ref, other)
    assert metrics["n_events"] == 201
    assert metrics["pearson_correlation"] == 1.0
    assert np.isclose(metrics["bias_ps"], 3.)
    assert np.isclose(metrics["rmse_difference_ps"], 3.)
    assert np.isclose(metrics["mae_difference_ps"], 3.)
    assert np.isclose(metrics["p99_abs_difference_ps"], 3.)
    assert sum(x["n"] for x in metrics["by_reference_prediction_quantile"]) == 201


def test_baseline_indices_are_led_relative_and_inclusive():
    from waveform_analysis.scripts.plot_conditional_target_heatmaps import baseline_indices
    import pytest
    time_ps = np.arange(-2500, 501, 100)
    np.testing.assert_array_equal(
        baseline_indices(time_ps), np.arange(5, 16))
    with pytest.raises(ValueError, match="does not cover baseline"):
        baseline_indices(np.arange(-1500, 501, 100))


def test_representations_subtract_baseline_before_peak_normalization():
    from waveform_analysis.scripts.plot_conditional_target_heatmaps import (
        WaveformRepresentation,
    )
    # Each detector has a different event baseline and amplitude.
    s = np.array([0., 0., 0., 1., 2., 3.], dtype=np.float32)
    waves = np.empty((2, 2, 6), np.float32)
    for i in range(2):
        for d in range(2):
            waves[i, d] = (i + 1) * 0.05 + d * 0.1 + s * (i + d + 1) * 0.1
    baseline = WaveformRepresentation(waves, "baseline_subtracted", [0, 1, 2])
    peak = WaveformRepresentation(waves, "amplitude_normalized", [0, 1, 2])
    np.testing.assert_allclose(baseline[np.arange(2)][:, :, :3], 0., atol=1e-7)
    expected = np.broadcast_to(s / 3, (2, 2, 6))
    np.testing.assert_allclose(peak[np.arange(2)], expected, atol=1e-6)
    # Dataset is untouched and a single-event index still yields [2,T].
    assert waves[1, 0, 0] != 0
    np.testing.assert_allclose(peak[0], expected[0], atol=1e-6)


def test_amplitude_normalization_fails_for_nonpositive_peak_without_dropping_rows():
    import pytest
    from waveform_analysis.scripts.plot_conditional_target_heatmaps import (
        WaveformRepresentation,
    )
    waves = np.ones((2, 2, 6), np.float32) * 0.3
    v = WaveformRepresentation(waves, "amplitude_normalized", [0, 1, 2])
    with pytest.raises(ValueError, match="Cannot peak-normalize"):
        v[np.arange(2)]


def test_each_representation_has_its_own_train_statistics():
    from waveform_analysis.engine.sample_normalization import fit_time_sample_stats
    from waveform_analysis.scripts.plot_conditional_target_heatmaps import (
        WaveformRepresentation,
    )
    signals = np.array([
        [0, 0, 0, 1, 2, 3],
        [0, 0, 0, 2, 4, 6],
        [0, 0, 0, 3, 6, 9],
        [0, 0, 0, 4, 8, 12],
    ], dtype=np.float32)
    waves = np.stack([signals + .5, signals * 1.2 + .1], axis=1)
    rows = np.arange(4)
    original_mu, original_std, _ = fit_time_sample_stats(
        waves, rows, chunk_size=2)
    bview = WaveformRepresentation(waves, "baseline_subtracted", [0, 1, 2])
    pview = WaveformRepresentation(waves, "amplitude_normalized", [0, 1, 2])
    baseline_mu, baseline_std, _ = fit_time_sample_stats(
        bview, rows, chunk_size=2)
    peak_mu, peak_std, _ = fit_time_sample_stats(
        pview, rows, chunk_size=2)
    assert not np.allclose(original_mu, baseline_mu)
    assert not np.allclose(baseline_mu, peak_mu)
    assert np.all(original_std > 0)
    assert np.all(baseline_std > 0)
    assert np.all(peak_std > 0)
    np.testing.assert_allclose(peak_mu[:, -1], 1., atol=1e-6)
    # After independent z-scoring, the conditional bins use transformed data.
    edges, _, counts = accumulate_heatmaps(
        pview, np.array([1., 2., 3., 4.]), rows, np.arange(6),
        peak_mu, peak_std, None, None,
        bins=20, z_limit=4, chunk_size=2, signal_only=True)
    assert len(edges) == 21
    assert counts.shape == (1, 3, 20, 6)
    assert int(counts[0, 0].sum()) == 4 * 6


def test_heatmap_physical_time_edges():
    from waveform_analysis.scripts.plot_conditional_target_heatmaps import (
        sample_centers_to_edges,
    )
    import pytest
    # 12.5 ps native sampling and plotting stride 8 => 0.1 ns bins.
    centers_ns = np.array([-2.0, -1.9, -1.8, -1.7])
    expected = np.array([-2.05, -1.95, -1.85, -1.75, -1.65])
    np.testing.assert_allclose(sample_centers_to_edges(centers_ns), expected)
    # Retained timing windows need not be uniformly spaced.
    np.testing.assert_allclose(
        sample_centers_to_edges(np.array([0., .1, .4])),
        np.array([-.05, .05, .25, .55]))
    with pytest.raises(ValueError, match="strictly increasing"):
        sample_centers_to_edges(np.array([0., 0., .1]))
