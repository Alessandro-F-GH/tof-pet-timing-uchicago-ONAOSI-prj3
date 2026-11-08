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


def test_grouped_physical_waveforms_use_original_mv_and_exact_percentiles():
    from waveform_analysis.scripts.plot_conditional_target_heatmaps import (
        grouped_physical_waveforms,
    )
    n = 100
    # Each detector has its own fixed physical scale and offset.
    wave = np.zeros((n, 2, 4), np.float32)
    wave[:, 0] = np.arange(n, dtype=np.float32)[:, None] / 100.
    wave[:, 1] = (100 - np.arange(n, dtype=np.float32)[:, None]) / 100.
    predictions = np.arange(n, dtype=np.float64) - 50
    minimum = np.array([[-50.], [-20.]])
    maximum = np.array([[350.], [140.]])
    grouped, meta = grouped_physical_waveforms(
        wave, predictions, np.arange(n), np.arange(4),
        minimum, maximum, chunk_size=7)
    assert grouped.shape == (3, 3, 4)
    assert [m["n_events"] for m in meta] == [10, 10, 10]
    low_rows = np.arange(2, 12)
    expected_1 = ((low_rows / 100) * 400 - 50).mean()
    expected_2 = (((100 - low_rows) / 100) * 160 - 20).mean()
    np.testing.assert_allclose(grouped[0, 0], expected_1, atol=1e-5)
    np.testing.assert_allclose(grouped[0, 1], expected_2, atol=1e-5)
    np.testing.assert_allclose(grouped[:, 2], grouped[:, 1] - grouped[:, 0])
    assert np.isclose(meta[0]["mean_prediction_ps"], predictions[low_rows].mean())


def test_grouped_waveforms_unchanged_by_heatmap_representation():
    from waveform_analysis.scripts.plot_conditional_target_heatmaps import (
        grouped_physical_waveforms, WaveformRepresentation,
    )
    rng = np.random.default_rng(2)
    data = rng.uniform(.15, .9, (100, 2, 6)).astype(np.float32)
    data[:, :, :2] = .15
    predictions = np.arange(100, dtype=float)
    args = (data, predictions, np.arange(100), np.arange(6),
            np.array([[-50.], [-20.]]), np.array([[350.], [140.]]))
    originals, info = grouped_physical_waveforms(*args)
    # The overlay must always use *data*, not transformed views.
    transformed = WaveformRepresentation(data, "amplitude_normalized", [0, 1])
    assert not np.allclose(transformed[np.arange(3)], data[:3])
    again, again_info = grouped_physical_waveforms(*args)
    np.testing.assert_array_equal(originals, again)
    assert info == again_info


def test_grouped_amplitude_normalized_overlays_are_not_physical_mv():
    from waveform_analysis.scripts.plot_conditional_target_heatmaps import (
        grouped_amplitude_normalized_waveforms, grouped_physical_waveforms,
    )
    # Same physical shape with distinct event/detector offsets and amplitudes.
    n, t = 100, 6
    shape = np.array([0., 0., .1, .5, 1., .7], np.float32)
    waves = np.zeros((n, 2, t), dtype=np.float32)
    for i in range(n):
        waves[i, 0] = .1 + (.15 + i * .001) * shape
        waves[i, 1] = .2 + (.25 + i * .001) * shape
    preds = np.arange(n, dtype=np.float64)
    original, groups = grouped_physical_waveforms(
        waves, preds, np.arange(n), np.arange(t),
        np.array([[-50.], [-20.]]), np.array([[350.], [140.]]))
    amplitude = grouped_amplitude_normalized_waveforms(
        waves, preds, np.arange(n), np.arange(t), [0, 1], groups,
        chunk_size=13)
    assert amplitude.shape == (3, 3, t)
    np.testing.assert_allclose(
        amplitude[:, 0], np.broadcast_to(shape, (3, t)), atol=1e-6)
    np.testing.assert_allclose(
        amplitude[:, 1], np.broadcast_to(shape, (3, t)), atol=1e-6)
    np.testing.assert_allclose(amplitude[:, 2], 0., atol=1e-6)
    assert not np.allclose(original[:, 0], amplitude[:, 0])


def test_cross_window_intersection_preserves_reference_order_and_targets():
    from waveform_analysis.scripts.plot_conditional_target_heatmaps import (
        match_event_populations, EventSubsetView,
    )
    reference_ids = np.array([12, 10, 14, 13, 11], dtype=np.int64)
    comparison_ids = np.array([11, 10, 15, 13], dtype=np.int64)
    reference_targets = np.array([1., 2., 3., 4., 5.])
    comparison_targets = np.array([5., 2., 9., 4.])
    ref_rows, cmp_rows, event_ids, details = match_event_populations(
        reference_ids, comparison_ids, reference_targets, comparison_targets)
    np.testing.assert_array_equal(event_ids, [10, 13, 11])
    np.testing.assert_array_equal(ref_rows, [1, 3, 4])
    np.testing.assert_array_equal(cmp_rows, [1, 3, 0])
    assert details["common_events"] == 3
    assert details["reference_only_events"] == 2
    assert details["comparison_only_events"] == 1
    waves = np.arange(5 * 2 * 6).reshape(5, 2, 6).astype(np.float32)
    view = EventSubsetView(waves, ref_rows)
    assert view.shape == (3, 2, 6)
    np.testing.assert_array_equal(view[np.array([2, 0])],
                                  waves[[4, 1]])
    np.testing.assert_array_equal(view[1], waves[3])
    # A comparison window can have another waveform length; event
    # intersection does not require identical time dimensions.


def test_cross_window_intersection_rejects_duplicate_ids_and_target_mismatch():
    import pytest
    from waveform_analysis.scripts.plot_conditional_target_heatmaps import (
        match_event_populations,
    )
    with pytest.raises(ValueError, match="duplicate"):
        match_event_populations(
            [3, 3, 4], [3, 4], [1., 1., 2.], [1., 2.])
    with pytest.raises(ValueError, match="Calibrated targets disagree"):
        match_event_populations(
            [2, 3, 4], [4, 2, 3], [0., 1., 2.], [2., 0., 5.])
    with pytest.raises(ValueError, match="Fewer than two"):
        match_event_populations(
            [2, 3], [4, 3], [0., 1.], [2., 1.])


def test_timing_representations_skip_amplitude_normalization():
    import pytest
    from waveform_analysis.scripts.plot_conditional_target_heatmaps import (
        effective_representations,
    )
    requested = ["original", "baseline_subtracted", "amplitude_normalized"]
    assert effective_representations("energy_to_energy", requested) == tuple(requested)
    assert effective_representations("timing_to_timing", requested) == (
        "original", "baseline_subtracted")
    with pytest.raises(ValueError, match="No applicable"):
        effective_representations("timing_to_timing", ["amplitude_normalized"])


def test_cropped_timing_spans_on_full_physical_grid():
    from waveform_analysis.scripts.plot_conditional_target_heatmaps import (
        mask_excluded_spans,
    )
    # 0.1 ns steps; consecutive masked samples become contiguous intervals.
    centers_ps = np.arange(6) * 100.
    kept = np.array([True, False, False, True, False, True])
    np.testing.assert_allclose(
        mask_excluded_spans(kept, centers_ps),
        [(0.05, 0.25), (0.35, 0.45)])
    assert mask_excluded_spans(np.ones(6, dtype=bool), centers_ps) == []


def test_onishi_window_bounds_use_sibling_context_and_respect_override(tmp_path):
    import json
    import pytest
    from waveform_analysis.scripts.plot_conditional_target_heatmaps import (
        onishi_window_bounds,
    )
    base = tmp_path / "results" / "energy"
    (base / "wide" / "rocket").mkdir(parents=True)
    (base / "onishi").mkdir()
    (base / "onishi" / "context.json").write_text(
        json.dumps({"window_ns": {"start": -1.25, "end": 2.75}}),
        encoding="utf-8")
    model_dir = base / "wide" / "rocket"
    assert onishi_window_bounds(model_dir) == (-1.25, 2.75)
    assert onishi_window_bounds(model_dir, start_ns=-1.5, end_ns=2.) == (-1.5, 2.)
    assert onishi_window_bounds(model_dir, disabled=True) is None
    with pytest.raises(ValueError, match="Specify both"):
        onishi_window_bounds(model_dir, start_ns=-1.5)
    with pytest.raises(ValueError, match="Invalid"):
        onishi_window_bounds(model_dir, start_ns=2., end_ns=-2.)
    (base / "onishi" / "context.json").unlink()
    assert onishi_window_bounds(model_dir) is None
