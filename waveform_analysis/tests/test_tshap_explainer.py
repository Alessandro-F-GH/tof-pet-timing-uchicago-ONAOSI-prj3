"""Numerical and persistence checks for the bounded upstream TSHAP adapter."""
import csv
import json

import numpy as np
import pytest

from waveform_analysis.engine.tshap_explainer import (
    _choose_indices, _cleanup_legacy_result_tshap, _complete, _explanation_destination,
    _center_detector_difference, _complete_local, _dataset_mean_difference_mv,
    _detector_difference, _led_line_extent, _led_reference_levels,
    _native_led_baselines,
    _is_excluded_tshap_window, _new_explainer, _plot_global, _plot_local_event,
    _publish_directory, _refresh_presentation, _select_local_events,
    _selection_seed, _write_global_importance_csv,
)


class _ZeroImputer:
    def impute_segment_batch(self, waveforms, detector, start, width):
        return np.zeros((len(waveforms), width), dtype=np.float64)


@pytest.mark.parametrize("roi", [False, True])
def test_conditional_window_occlusion_is_signed_and_baseline_independent(roi):
    rng = np.random.default_rng(47)
    sample = rng.uniform(0.1, 1, size=(2, 32)).astype(np.float32)
    explainer = _new_explainer(8, 4, roi, predict_batch_size=3,
                              imputer=_ZeroImputer())
    positions = [(0, 4, 8), (1, 8, 8), (0, 16, 8)]
    def predict(values):
        return np.asarray(values)[:, 0].sum(axis=1) - 2 * np.asarray(values)[:, 1].sum(axis=1)
    expected = np.asarray([sample[0, 4:12].sum(),
                           -2 * sample[1, 8:16].sum(),
                           sample[0, 16:24].sum()])
    observed = explainer._calculate_window_attrib_scores(
        predict, sample, np.zeros((5, 2, 32)), positions)
    np.testing.assert_allclose(observed, expected, rtol=1e-6)
    other = explainer._calculate_window_attrib_scores(
        predict, sample, rng.normal(size=(2, 2, 32)), positions)
    np.testing.assert_allclose(other, observed, rtol=0, atol=0)


def test_tshap_constant_prediction_is_zero():
    model = _new_explainer(6, 3, False, 4, imputer=_ZeroImputer())
    sample = np.ones((2, 24), dtype=np.float32)
    values = model._calculate_window_attrib_scores(
        lambda values: np.full(len(values), 7.0), sample, sample[None],
        [(0, 0, 6), (1, 8, 6)])
    np.testing.assert_allclose(values, 0)


def test_conditional_imputer_cache_reuses_fitted_parameters(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from waveform_analysis.engine import tshap_explainer as t
    rng = np.random.default_rng(3)
    waves = rng.normal(0.5, .15, (70, 2, 60)).astype(np.float32)
    control = SimpleNamespace(
        n_events=len(waves), event_index=np.arange(len(waves)),
        energy_windows=waves, energy_time_ps=np.arange(60)*12.5,
        manifest={"analysis_protocol_identity": "control-v1",
                  "event_population_identity": "population-v1"})
    config = {"_board": "UC", "window_ns": {"start": -2, "end": 50},
              "output_dir": str(tmp_path / "UC" / "test_ridge")}
    options = dict(window_samples=20, stride_samples=20,
                   components=4, fit_events=30, tuning_events=12,
                   slope_points=5, batch_size=8, seed=42)
    first, first_meta = t._cached_hermite_ppca_imputer(
        control, "energy_to_energy", config, tmp_path, **options)
    assert first_meta["cache_hit"] is False
    def forbidden(*args, **kwargs):
        raise AssertionError("Imputer was refitted on cache hit")
    monkeypatch.setattr(t, "_fit_hermite_ppca_imputer", forbidden)
    second, second_meta = t._cached_hermite_ppca_imputer(
        control, "energy_to_energy", config, tmp_path, **options)
    assert second_meta["cache_hit"] is True
    assert second_meta["cache_identity"] == first_meta["cache_identity"]
    np.testing.assert_allclose(
        first.impute_segment_batch(waves[:3], 0, 20, 20),
        second.impute_segment_batch(waves[:3], 0, 20, 20),
        rtol=1e-12, atol=1e-12)


def test_event_selection_is_reproducible():
    np.testing.assert_array_equal(_choose_indices(75, 10, 14), _choose_indices(75, 10, 14))
    assert len(_choose_indices(75, 10, 14)) == 10
    np.testing.assert_array_equal(_choose_indices(3, 10, 14), [0, 1, 2])


def test_complete_identity_and_replace(tmp_path):
    target = tmp_path / "explanation"
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    (scratch / "summary.json").write_text(json.dumps({"identity_sha256": "123"}))
    for file in ("attributions.npz", "signed_window_importance.csv", "global_importance.png"):
        (scratch / file).touch()
    _publish_directory(scratch, target, overwrite=False)
    assert _complete(target, "123")
    assert not _complete(target, "456")
    newer = tmp_path / "replacement"
    newer.mkdir()
    (newer / "summary.json").write_text(json.dumps({"identity_sha256": "456"}))
    for file in ("attributions.npz", "signed_window_importance.csv", "global_importance.png"):
        (newer / file).touch()
    _publish_directory(newer, target, overwrite=True)
    assert _complete(target, "456")


def test_explanation_destination_is_separate_and_collision_safe(tmp_path, monkeypatch):
    from waveform_analysis.engine import tshap_explainer

    batch = tmp_path / "results" / "batch-A"
    run = batch / "energy" / "wide" / "direct_linear_ridge"
    run.mkdir(parents=True)
    monkeypatch.setattr(tshap_explainer, "batch_root", lambda _: batch)
    output_root = tmp_path / "xai"
    path = _explanation_destination(output_root, run)
    assert path == output_root / "batch-A" / "energy" / "wide" / "direct_linear_ridge" / "tshap"
    assert not path.is_relative_to(batch)
    assert path.name == "tshap"  # Identity fingerprints stay in summary.json, not in path names.


def test_cleanup_removes_only_legacy_tshap_outputs(tmp_path):
    run = tmp_path / "run"
    legacy = run / "explanations" / "tshap" / "old-hash"
    legacy.mkdir(parents=True)
    (legacy / "summary.json").write_text("{}")
    unrelated = run / "explanations" / "other-xai"
    unrelated.mkdir(parents=True)
    (unrelated / "keep.txt").write_text("keep")

    _cleanup_legacy_result_tshap(run)

    assert not (run / "explanations" / "tshap").exists()
    assert (unrelated / "keep.txt").read_text() == "keep"


def test_sampling_seed_is_fixed_and_model_independent():
    seed = 2026
    blind_seed_1 = _selection_seed(seed, "blind_events")
    blind_seed_2 = _selection_seed(seed, "blind_events")
    baseline_seed_1 = _selection_seed(seed, "control_baselines")
    baseline_seed_2 = _selection_seed(seed, "control_baselines")

    assert blind_seed_1 == blind_seed_2
    assert baseline_seed_1 == baseline_seed_2
    assert blind_seed_1 != baseline_seed_1
    np.testing.assert_array_equal(
        _choose_indices(1000, 100, blind_seed_1),
        _choose_indices(1000, 100, blind_seed_2),
    )
    np.testing.assert_array_equal(
        _choose_indices(500, 8, baseline_seed_1),
        _choose_indices(500, 8, baseline_seed_2),
    )
    assert not np.array_equal(
        _choose_indices(1000, 100, _selection_seed(seed + 1, "blind_events")),
        _choose_indices(1000, 100, blind_seed_1),
    )


def test_local_selection_returns_two_nonoverlapping_events_per_group_and_trims_extremes():
    # Build a controlled population with enough examples in each correction/
    # improvement category, plus predictions near the population median.
    pred_parts = [
        np.full(40, 10.0),    # good positive
        np.full(40, -10.0),   # good negative
        np.full(40, 10.0),    # bad positive
        np.full(40, -10.0),   # bad negative
        np.linspace(-2.0, 2.0, 20),  # typical corrections around the median
    ]
    target_parts = [
        np.linspace(6.0, 9.9, 40),
        -np.linspace(6.0, 9.9, 40),
        np.linspace(0.1, 4.9, 40),
        -np.linspace(0.1, 4.9, 40),
        np.linspace(-2.0, 2.0, 20),
    ]
    predictions = np.concatenate(pred_parts)
    targets = np.concatenate(target_parts)

    selected, improvement, median_prediction = _select_local_events(
        predictions, targets, events_per_group=2, extreme_margin=0.05
    )
    assert len(selected) == 10
    assert len({item["index"] for item in selected}) == 10
    counts = {group: sum(item["group"] == group for item in selected) for group in (
        "good_positive", "good_negative", "bad_positive", "bad_negative", "typical"
    )}
    assert all(count == 2 for count in counts.values())

    for item in selected:
        index = item["index"]
        group = item["group"]
        if group.endswith("positive"):
            assert predictions[index] > 0
        elif group.endswith("negative"):
            assert predictions[index] < 0
        if group.startswith("good_"):
            assert improvement[index] > 0
        elif group.startswith("bad_"):
            assert improvement[index] < 0

    # Good selections should not be the maximum improvement; bad selections
    # should not be the minimum improvement in their respective eligible pools.
    good_positive = np.flatnonzero((predictions > 0) & (improvement > 0))
    bad_positive = np.flatnonzero((predictions > 0) & (improvement < 0))
    selected_good_values = [improvement[item["index"]] for item in selected if item["group"] == "good_positive"]
    selected_bad_values = [improvement[item["index"]] for item in selected if item["group"] == "bad_positive"]
    assert max(selected_good_values) < np.max(improvement[good_positive])
    assert min(selected_bad_values) > np.min(improvement[bad_positive])

    # Typical events are the closest unused predictions to the blind median.
    typical_indices = [item["index"] for item in selected if item["group"] == "typical"]
    used_before_typical = {item["index"] for item in selected if item["group"] != "typical"}
    remaining = np.asarray([i for i in range(len(predictions)) if i not in used_before_typical])
    distances = np.abs(predictions[remaining] - median_prediction)
    typical_distances = np.abs(predictions[typical_indices] - median_prediction)
    assert np.max(typical_distances) <= np.sort(distances)[1]


def test_local_selection_rejects_insufficient_categories():
    with pytest.raises(ValueError, match="Cannot select"):
        _select_local_events(
            np.asarray([1.0, -1.0, 1.0]),
            np.asarray([2.0, -2.0, 0.0]),
            events_per_group=2,
            extreme_margin=0.05,
        )


def test_local_cache_requires_all_expected_event_plots(tmp_path):
    destination = tmp_path / "tshap_local"
    destination.mkdir()
    (destination / "events.csv").write_text("event_id\n")
    (destination / "local_attributions.npz").touch()
    plot_names = ["good.png", "bad.png"]
    for name in plot_names:
        (destination / name).touch()
    (destination / "summary.json").write_text(json.dumps({
        "identity_sha256": "fingerprint",
        "n_explained_events": 2,
        "plots": plot_names,
    }))

    assert _complete_local(destination, "fingerprint", expected_events=2)
    assert not _complete_local(destination, "fingerprint", expected_events=3)
    (destination / "bad.png").unlink()
    assert not _complete_local(destination, "fingerprint", expected_events=2)


def test_local_explanations_use_a_separate_stable_folder(tmp_path, monkeypatch):
    from waveform_analysis.engine import tshap_explainer

    batch = tmp_path / "results" / "batch-A"
    run = batch / "energy" / "wide" / "direct_linear_ridge"
    run.mkdir(parents=True)
    monkeypatch.setattr(tshap_explainer, "batch_root", lambda _: batch)
    path = tshap_explainer._local_explanation_destination(tmp_path / "xai", run)
    assert path == tmp_path / "xai" / "batch-A" / "energy" / "wide" / "direct_linear_ridge" / "tshap_local"
    assert path.name == "tshap_local"
    assert not path.is_relative_to(batch)


def test_local_signal_difference_subtracts_dataset_mean_per_sample():
    values = np.asarray([
        [0.10, 0.40, 0.80, 0.70],
        [0.06, 0.30, 0.74, 0.76],
    ])
    dataset_mean = np.asarray([0.02, 0.04, 0.01, -0.03])
    raw_difference = _detector_difference(values)
    centered_difference = _center_detector_difference(values, dataset_mean)

    np.testing.assert_allclose(raw_difference, values[0] - values[1])
    np.testing.assert_allclose(centered_difference, raw_difference - dataset_mean)
    assert not np.isclose(np.mean(centered_difference), 0.0)


def test_signal_difference_helpers_reject_invalid_shapes():
    with pytest.raises(ValueError, match="2 detectors"):
        _detector_difference(np.ones((3, 10)))
    with pytest.raises(ValueError, match="2 detectors"):
        _center_detector_difference(np.ones((10, 2)), np.zeros(2))


def test_dataset_mean_difference_is_per_sample_and_seeded():
    from types import SimpleNamespace

    waveforms = np.asarray([
        [[1.0, 3.0], [0.0, 1.0]],
        [[2.0, 5.0], [1.0, 2.0]],
        [[4.0, 7.0], [1.0, 4.0]],
        [[8.0, 9.0], [3.0, 5.0]],
    ], dtype=np.float32)

    class IdentityTransform:
        @staticmethod
        def inverse(values):
            return np.asarray(values, dtype=np.float32)

    dataset = SimpleNamespace(
        n_events=waveforms.shape[0],
        energy_windows=waveforms,
        timing_windows=None,
        energy_time_ps=np.asarray([0.0, 1.0]),
        timing_time_ps=None,
    )
    mean, indices = _dataset_mean_difference_mv(
        dataset, "energy_to_energy", IdentityTransform(), seed=42, max_events=4
    )
    all_differences = waveforms[:, 0, :] - waveforms[:, 1, :]
    np.testing.assert_allclose(mean, np.mean(all_differences, axis=0))
    np.testing.assert_array_equal(
        indices,
        _dataset_mean_difference_mv(
            dataset, "energy_to_energy", IdentityTransform(), seed=42, max_events=4
        )[1],
    )



def test_local_plot_accepts_millivolt_waveforms_and_led_threshold(tmp_path):
    time_ps = np.asarray([-1000, -750, -500, -250, 0, 250, 500, 750], dtype=np.float64)
    waveform_mv = np.asarray([
        [-8.0, -7.0, -3.0, 12.0, 25.0, 15.0, 2.0, -5.0],
        [-7.0, -6.0, -2.0, 10.0, 22.0, 14.0, 1.0, -4.0],
    ])
    attribution = np.asarray([
        [0.1, 0.2, 0.5, 1.0, 0.7, 0.3, 0.1, 0.0],
        [0.0, 0.1, 0.4, 0.8, 0.9, 0.4, 0.2, 0.1],
    ])
    output = tmp_path / "local_event.png"

    _plot_local_event(
        output,
        time_ps=time_ps,
        waveform_mv=waveform_mv,
        dataset_mean_difference_mv=np.zeros(waveform_mv.shape[1], dtype=np.float64),
        led_baseline_mv=np.asarray([5.0, 2.0]),
        attribution=attribution,
        window_starts=np.asarray([0, 2]), window_scores=np.asarray([[2.0, -1.0], [-2.0, 1.0]]),
        window_samples=2,
        group="typical",
        event_id=123,
        target_ps=10.0,
        prediction_ps=2.0,
        improvement_ps=2.0,
        led_threshold_mv=20.0,
    )

    assert output.is_file()
    assert output.stat().st_size > 0
    pdf_output = tmp_path / "global_importance.pdf"
    assert pdf_output.is_file()
    assert pdf_output.stat().st_size > 0



def test_led_reference_levels_add_calibrated_threshold_to_native_baselines():
    levels = _led_reference_levels(np.asarray([-6.0, -5.0]), threshold_mv=20.0)
    np.testing.assert_allclose(levels, [14.0, 15.0])


def test_led_reference_line_stops_at_time_zero():
    np.testing.assert_allclose(
        _led_line_extent(np.asarray([-2.0, -1.0, 0.0, 1.0, 2.0])),
        (-2.0, 0.0),
    )


def test_native_led_baselines_use_original_trigger_relative_window(tmp_path):
    from types import SimpleNamespace

    native = tmp_path / "blind_native" / "native-a"
    native.mkdir(parents=True)
    (native / "manifest.json").write_text(json.dumps({
        "source": "source.root",
        "mode": "energy_to_energy",
        "family": "energy",
        "materialized_window_ns": {"before": 5.0, "after": 5.0},
    }))
    event_ids = np.asarray([101, 102, 103], dtype=np.int64)
    waves = np.zeros((3, 2, 10), dtype=np.float32)
    waves[0, 0, 3:5] = [1.0, 3.0]
    waves[0, 1, 3:5] = [5.0, 7.0]
    waves[1, 0, 3:5] = [10.0, 12.0]
    waves[1, 1, 3:5] = [-2.0, 0.0]
    waves[2] = 999.0
    intervals = np.full((3, 2), 1e-9, dtype=np.float64)
    np.save(native / "event_index.npy", event_ids)
    np.save(native / "energy_windows_mV.npy", waves)
    np.save(native / "energy_sample_interval_s.npy", intervals)

    dataset = SimpleNamespace(
        manifest={"analysis_source": "source.root"},
        event_index=np.asarray([101, 102], dtype=np.int64),
        n_events=2,
    )
    baselines = _native_led_baselines(
        tmp_path, dataset, "energy_to_energy", np.asarray([1, 0]),
        baseline_window_ns=(-2.0, -1.0),
    )
    np.testing.assert_allclose(baselines, [[11.0, -1.0], [2.0, 6.0]])



@pytest.mark.parametrize("window_name", ["onishi", "Onishi", "ONISHI"])
def test_tshap_excludes_onishi_window(window_name):
    assert _is_excluded_tshap_window({"window_name": window_name})


@pytest.mark.parametrize("window_name", ["wide", "default", ""])
def test_tshap_keeps_non_onishi_windows(window_name):
    assert not _is_excluded_tshap_window({"window_name": window_name})



@pytest.mark.parametrize("formulation", ["direct", "shared"])
def test_global_saliency_plot_uses_physical_waveforms_and_onishi_interval(tmp_path, formulation):
    pytest.importorskip("matplotlib")
    time_ps = np.linspace(-2000.0, 60000.0, 125, dtype=np.float64)
    waveform_mv = np.vstack([
        20.0 * np.exp(-((time_ps / 1000.0 - 2.0) / 3.0) ** 2),
        18.0 * np.exp(-((time_ps / 1000.0 - 2.2) / 3.5) ** 2),
    ])
    importance = np.vstack([
        np.linspace(0.0, 2.0, time_ps.size),
        np.linspace(1.0, 3.0, time_ps.size),
    ])
    _plot_global(
        tmp_path, time_ps, importance, waveform_mv,
        model_name="direct_multirocket" if formulation == "direct" else "shared_linear_ridge",
        formulation=formulation,
        mode="timing_to_timing",
        window_name="wide",
        board_name="FBK",
        n_events=64,
        window_samples=80,
        window_starts=np.asarray([0, 80]),
        signed_positive=np.ones((2, 2)),
        signed_negative=-np.ones((2, 2)),
    )
    output = tmp_path / "global_importance.png"
    assert output.is_file()
    assert output.stat().st_size > 0


def test_shared_global_importance_csv_pools_detector_curves(tmp_path):
    time_ps = np.asarray([-1000.0, 0.0, 1000.0])
    importance = np.asarray([
        [1.0, 2.0, 3.0],
        [5.0, 6.0, 7.0],
    ])
    _write_global_importance_csv(tmp_path, time_ps, importance, "shared")
    import csv
    with (tmp_path / "global_importance.csv").open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 3
    assert {row["detector_or_shared"] for row in rows} == {"shared"}
    np.testing.assert_allclose(
        [float(row["mean_absolute_tshap_importance_ps_per_sample"]) for row in rows],
        [3.0, 4.0, 5.0],
    )


def test_global_presentation_refresh_uses_cached_tshap_arrays_only(tmp_path):
    pytest.importorskip("matplotlib")
    time_ps = np.asarray([-2000.0, -1000.0, 0.0, 1000.0, 2000.0])
    maps = np.asarray([
        [[1.0, -2.0, 3.0, -4.0, 5.0], [2.0, -3.0, 4.0, -5.0, 6.0]],
        [[-3.0, 4.0, -5.0, 6.0, -7.0], [-4.0, 5.0, -6.0, 7.0, -8.0]],
    ])
    waveform_mv = np.asarray([
        [-2.0, 1.0, 8.0, 3.0, -1.0],
        [-1.0, 2.0, 7.0, 2.0, -2.0],
    ])
    np.savez_compressed(
        tmp_path / "attributions.npz",
        time_ps=time_ps,
        window_attribution_ps=maps,
        raw_window_scores_ps=np.asarray([[[2.0, -1.0], [-2.0, 1.0]],
                                        [[-1.0, 2.0], [1.0, -2.0]]]),
        window_starts=np.asarray([0, 2]),
        mean_waveform_mv=waveform_mv,
    )
    (tmp_path / "summary.json").write_text(json.dumps({
        "plot_version": 3,
        "model": "shared_linear_ridge",
        "mode": "timing_to_timing",
        "window_name": "wide",
        "board": "UC",
        "formulation": "shared",
        "n_explained_events": 2,
    }), encoding="utf-8")
    _refresh_presentation(tmp_path)
    summary = json.loads((tmp_path / "summary.json").read_text(encoding="utf-8"))
    assert summary["plot_version"] == 16
    assert summary["global_plot"]["layout"] == "three vertically stacked panels (detector 1 waveform, detector 2 waveform, shared importance histogram), with a common time axis"
    assert summary["global_plot"]["importance_colour_scale"].startswith("shared PowerNorm(gamma=1.6)")
    assert summary["global_plot"]["histogram_colour"] == "#4c78a8"
    assert summary["global_plot"]["output_formats"] == ["PNG (240 dpi)", "PDF (vector)"]
    assert (tmp_path / "global_importance.png").is_file()
    assert (tmp_path / "signed_window_importance.csv").is_file()
    with (tmp_path / "signed_window_importance.csv").open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 2
    assert {row["detector_or_shared"] for row in rows} == {"shared"}
    assert all(float(row["mean_positive_contribution_ps"]) >= 0 for row in rows)
    assert all(float(row["mean_negative_contribution_ps"]) <= 0 for row in rows)



def test_direct_global_plot_includes_aggregated_signed_histogram_and_csv(tmp_path):
    pytest.importorskip("matplotlib")
    import csv
    from waveform_analysis.engine.tshap_explainer import (
        _plot_global, _write_signed_csv, _signed_means,
    )

    time_ps = np.arange(8, dtype=np.float64) * 1000.0
    starts = np.asarray([0, 2, 4, 6])
    raw_scores = np.asarray([
        [[2.0, -1.0, 3.0, -2.0], [-1.0, 2.0, -1.0, 1.0]],
        [[-3.0, 1.0, 2.0, -1.0], [1.0, -2.0, -3.0, 2.0]],
    ])
    detector_positive, detector_negative = _signed_means(raw_scores)
    aggregate_positive, aggregate_negative = _signed_means(raw_scores.sum(axis=1))
    _write_signed_csv(
        tmp_path, time_ps, starts, detector_positive, detector_negative, 2, "direct",
        aggregated_positive=aggregate_positive, aggregated_negative=aggregate_negative,
    )
    rows = list(csv.DictReader((tmp_path / "signed_window_importance.csv").open(newline="", encoding="utf-8")))
    assert {row["detector_or_shared"] for row in rows} == {"detector_1", "detector_2", "aggregated"}
    grouped = [row for row in rows if row["detector_or_shared"] == "aggregated"]
    assert len(grouped) == len(starts)
    np.testing.assert_allclose(
        [float(row["mean_positive_contribution_ps"]) for row in grouped],
        aggregate_positive,
    )
    np.testing.assert_allclose(
        [float(row["mean_negative_contribution_ps"]) for row in grouped],
        aggregate_negative,
    )

    _plot_global(
        tmp_path, time_ps, np.ones((2, 8)), np.ones((2, 8)),
        model_name="direct_multirocket", formulation="direct", mode="energy_to_energy",
        window_name="wide", board_name="UC", n_events=2, window_samples=2,
        window_starts=starts, signed_positive=detector_positive,
        signed_negative=detector_negative, aggregated_positive=aggregate_positive,
        aggregated_negative=aggregate_negative,
    )
    assert (tmp_path / "global_importance.png").is_file()
    assert (tmp_path / "global_importance.pdf").is_file()

def test_cli_exposes_only_tshap_interpretability_commands():
    from waveform_analysis.cli import _parser

    commands = _parser()._subparsers._group_actions[0].choices
    assert "explain-tshap" in commands
    assert "explain-tshap-local" in commands
    assert "xai" not in commands
    assert "explain-regression" not in commands
