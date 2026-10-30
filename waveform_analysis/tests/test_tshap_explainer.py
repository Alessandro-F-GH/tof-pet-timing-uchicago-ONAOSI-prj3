"""Numerical and persistence checks for the bounded upstream TSHAP adapter."""
import csv
import json

import numpy as np
import pytest

from waveform_analysis.engine.tshap_explainer import (
    _choose_indices, _complete, _explanation_destination,
    _center_detector_difference, _complete_local, _dataset_mean_difference_mv,
    _detector_difference, _led_line_extent, _led_reference_levels,
    _native_led_baselines,
    _matches_tshap_window, _resolve_model_run_directory, _apply_tshap_cache,
    _new_explainer, _plot_global, _plot_local_event,
    _publish_directory, _refresh_presentation, _select_local_events,
    _selection_seed, _write_global_importance_csv,
)


class _ZeroImputer:
    def impute_segment_batch(self, waveforms, detector, start, width):
        return np.zeros((len(waveforms), width), dtype=np.float64)


@pytest.mark.parametrize("roi", [False, True])
def test_conditional_window_occlusion_is_symmetric_and_uses_baselines(roi):
    rng = np.random.default_rng(47)
    sample = rng.uniform(0.1, 1, size=(2, 32)).astype(np.float32)
    explainer = _new_explainer(8, 4, roi, predict_batch_size=3,
                              imputer=_ZeroImputer())
    positions = [(0, 4, 8), (1, 8, 8), (0, 16, 8)]
    def predict(values):
        return np.asarray(values)[:, 0].sum(axis=1) - 2 * np.asarray(values)[:, 1].sum(axis=1)
    expected = 0.5 * np.asarray([sample[0, 4:12].sum(),
                                 -2 * sample[1, 8:16].sum(),
                                 sample[0, 16:24].sum()])
    observed = explainer._calculate_window_attrib_scores(
        predict, sample, np.zeros((5, 2, 32)), positions)
    np.testing.assert_allclose(observed, expected, rtol=1e-6)
    other_baselines = rng.normal(size=(2, 2, 32)).astype(np.float32)
    other = explainer._calculate_window_attrib_scores(
        predict, sample, other_baselines, positions)
    assert not np.allclose(other, observed)


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
        directory=tmp_path / "control",
        n_events=len(waves), event_index=np.arange(len(waves)),
        energy_windows=waves, energy_time_ps=np.arange(60)*12.5,
        manifest={"analysis_protocol_identity": "control-v1",
                  "event_population_identity": "population-v1"})
    config = {
        "_board": "UC",
        "window_ns": {"start": -2, "end": 50},
        "output_dir": str(tmp_path / "UC" / "test_ridge"),
        "preprocessing": {"artifact_root": str(tmp_path / "artifact_store")},
    }
    options = dict(window_samples=20, stride_samples=20,
                   components=4, fit_events=30, tuning_events=12,
                   slope_points=5, batch_size=8, seed=42)
    first, first_meta = t._cached_hermite_ppca_imputer(
        control, "energy_to_energy", config, **options)
    assert first_meta["cache_hit"] is False
    def forbidden(*args, **kwargs):
        raise AssertionError("Imputer was refitted on cache hit")
    monkeypatch.setattr(t, "_fit_hermite_ppca_imputer", forbidden)
    second, second_meta = t._cached_hermite_ppca_imputer(
        control, "energy_to_energy", config, **options)
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
    for file in ("attributions.npz", "global_importance.csv", "global_importance.png"):
        (scratch / file).touch()
    _publish_directory(scratch, target, overwrite=False)
    assert _complete(target, "123")
    assert not _complete(target, "456")
    newer = tmp_path / "replacement"
    newer.mkdir()
    (newer / "summary.json").write_text(json.dumps({"identity_sha256": "456"}))
    for file in ("attributions.npz", "global_importance.csv", "global_importance.png"):
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


def test_blind_event_sampling_seed_is_fixed():
    seed = 2026
    first = _selection_seed(seed)
    second = _selection_seed(seed)

    assert first == second
    np.testing.assert_array_equal(
        _choose_indices(1000, 100, first),
        _choose_indices(1000, 100, second),
    )
    assert not np.array_equal(
        _choose_indices(1000, 100, _selection_seed(seed + 1)),
        _choose_indices(1000, 100, first),
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
        group="typical",
        event_id=123,
        target_ps=10.0,
        prediction_ps=2.0,
        improvement_ps=2.0,
        led_threshold_mv=20.0,
    )

    assert output.is_file()
    assert output.stat().st_size > 0



def test_led_reference_levels_add_calibrated_threshold_to_native_baselines():
    levels = _led_reference_levels(np.asarray([-6.0, -5.0]), threshold_mv=20.0)
    np.testing.assert_allclose(levels, [14.0, 15.0])


def test_led_reference_line_stops_at_time_zero():
    np.testing.assert_allclose(
        _led_line_extent(np.asarray([-2.0, -1.0, 0.0, 1.0, 2.0])),
        (-2.0, 0.0),
    )


@pytest.mark.parametrize("window_name", ["onishi", "Onishi", "ONISHI"])
def test_tshap_window_selector_accepts_onishi(window_name):
    assert _matches_tshap_window({"window_name": window_name}, "onishi")
    assert not _matches_tshap_window({"window_name": window_name}, "wide")


@pytest.mark.parametrize("window_name", ["wide", "WIDE"])
def test_tshap_window_selector_accepts_wide(window_name):
    assert _matches_tshap_window({"window_name": window_name}, "wide")
    assert not _matches_tshap_window({"window_name": window_name}, "onishi")


def test_tshap_window_selector_all_and_invalid():
    assert _matches_tshap_window({"window_name": "onishi"}, "all")
    with pytest.raises(ValueError, match="ml_window"):
        _matches_tshap_window({"window_name": "wide"}, "invalid")


def test_tshap_model_directory_and_shared_cache_resolution(tmp_path):
    from waveform_analysis.engine.tshap_explainer import _effective_tshap_window

    run = tmp_path / "results" / "energy" / "wide" / "direct_multirocket"
    (run / "metadata").mkdir(parents=True)
    (run / "metadata" / "manifest.json").write_text("{}", encoding="utf-8")
    (run / "model").mkdir()
    assert _resolve_model_run_directory(run) == run.resolve()
    assert _resolve_model_run_directory(run / "model") == run.resolve()
    assert _effective_tshap_window(None, model_dir=run) == "all"
    assert _effective_tshap_window(None, model_dir=None) == "wide"

    cache = tmp_path / "cache"
    (cache / "artifact_store").mkdir(parents=True)
    config = {"preprocessing": {"cache_dir": "old-cache", "artifact_root": "old-store"}}
    _apply_tshap_cache(config, cache_root=cache)
    assert config["preprocessing"]["cache_dir"] == str((cache / "preprocessing").resolve())
    assert config["preprocessing"]["artifact_root"] == str((cache / "artifact_store").resolve())
    with pytest.raises(ValueError, match="cache_root or artifact_root"):
        _apply_tshap_cache(config, cache_root=cache, artifact_root=cache / "artifact_store")


def test_tshap_model_directory_requires_run_metadata(tmp_path):
    with pytest.raises(ValueError, match="metadata/manifest.json"):
        _resolve_model_run_directory(tmp_path / "raw_model_artifact")



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
    assert summary["plot_version"] == 13
    assert summary["global_plot"]["layout"] == "three vertically stacked panels (detector 1 waveform, detector 2 waveform, shared importance histogram), with a common time axis"
    assert summary["global_plot"]["importance_colour_scale"].startswith("shared PowerNorm(gamma=1.6)")
    assert summary["global_plot"]["histogram_colour"] == "#4c78a8"
    assert summary["global_plot"]["output_formats"] == ["PNG (240 dpi)", "PDF (vector)"]
    assert (tmp_path / "global_importance.png").is_file()
    with (tmp_path / "global_importance.csv").open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == len(time_ps)
    np.testing.assert_allclose(
        [float(row["mean_absolute_tshap_importance_ps_per_sample"]) for row in rows],
        np.mean(np.mean(np.abs(maps), axis=0), axis=0),
    )


def test_cli_exposes_only_tshap_interpretability_commands():
    from waveform_analysis.cli import _parser

    commands = _parser()._subparsers._group_actions[0].choices
    assert "explain-tshap" in commands
    assert "explain-tshap-local" in commands
    assert "xai" not in commands
    global_args = commands["explain-tshap"]._mutually_exclusive_groups
    local_args = commands["explain-tshap-local"]._mutually_exclusive_groups
    assert any({"--results", "--model-dir"} <= {option for action in group._group_actions for option in action.option_strings}
               for group in global_args)
    assert any({"--cache-root", "--artifact-root"} <= {option for action in group._group_actions for option in action.option_strings}
               for group in global_args)
    assert any({"--results", "--model-dir"} <= {option for action in group._group_actions for option in action.option_strings}
               for group in local_args)
    assert any({"--cache-root", "--artifact-root"} <= {option for action in group._group_actions for option in action.option_strings}
               for group in local_args)
    assert "explain-regression" not in commands
    assert "--n-baselines" not in {
        option
        for command in ("explain-tshap", "explain-tshap-local")
        for action in commands[command]._actions
        for option in action.option_strings
    }



def test_tshap_recomputes_fitted_features_for_each_waveform():
    from types import SimpleNamespace
    from waveform_analysis.engine.tshap_explainer import _predict_bounded

    transformed_inputs = []

    class Transform:
        def apply(self, values):
            transformed_inputs.append(np.array(values, copy=True))
            return np.asarray(values).reshape(len(values), -1)

    class Spec:
        @staticmethod
        def predict(artifact, values):
            return np.asarray(values).sum(axis=1)

    fitted = SimpleNamespace(
        sample_mask=np.ones(8, dtype=bool),
        feature_transform=Transform(),
        output_max_abs_ps=None,
        artifact=object(),
    )
    original = np.ones((1, 2, 8), dtype=np.float32)
    occluded = original.copy()
    occluded[:, 0, 2:5] = 0

    first = _predict_bounded(Spec, fitted, original, batch_size=8)
    second = _predict_bounded(Spec, fitted, occluded, batch_size=8)

    assert len(transformed_inputs) == 2
    np.testing.assert_array_equal(transformed_inputs[0], original)
    np.testing.assert_array_equal(transformed_inputs[1], occluded)
    assert first[0] != second[0]


def test_reference_substitution_uses_symmetric_multi_baseline_score():
    from waveform_analysis.engine.tshap_explainer import _new_explainer

    sample = np.ones((2, 16), dtype=np.float32)
    baselines = np.stack([
        np.zeros((2, 16), dtype=np.float32),
        np.full((2, 16), 1.0, dtype=np.float32),
    ])
    explainer = _new_explainer(4, 4, False, predict_batch_size=4, imputer=None)

    def predict(values):
        return np.asarray(values, dtype=np.float64)[:, 0].sum(axis=1)

    observed = explainer._calculate_window_attrib_scores(
        predict, sample, baselines, [(0, 4, 4)]
    )
    # The two independent references are averaged after the symmetric
    # two-player swap; this is not the single-event conditional score.
    np.testing.assert_allclose(observed, [2.0])


def test_selection_seed_separates_blind_and_control_roles():
    from waveform_analysis.engine.tshap_explainer import _selection_seed

    assert _selection_seed(42, "blind_events") == _selection_seed(42, "blind_events")
    assert _selection_seed(42, "control_baselines") == _selection_seed(42, "control_baselines")
    assert _selection_seed(42, "blind_events") != _selection_seed(42, "control_baselines")


def test_hermite_ppca_fit_excludes_reference_baseline_events(tmp_path):
    from types import SimpleNamespace
    from waveform_analysis.engine.tshap_explainer import _fit_hermite_ppca_imputer

    rng = np.random.default_rng(91)
    waves = rng.normal(0.5, 0.1, size=(80, 2, 40)).astype(np.float32)
    control = SimpleNamespace(
        n_events=len(waves),
        event_index=np.arange(1000, 1000 + len(waves), dtype=np.int64),
        energy_windows=waves,
        energy_time_ps=np.arange(waves.shape[-1], dtype=np.float64),
        directory=tmp_path / "control",
    )
    imputer, metadata = _fit_hermite_ppca_imputer(
        control, "energy_to_energy", np.asarray([1, 5, 9, 13]),
        components=3, fit_events=30, tuning_events=10,
        slope_points=3, batch_size=8, seed=42,
    )
    excluded = {1, 5, 9, 13}
    assert excluded.isdisjoint(metadata["training_indices"].tolist())
    assert excluded.isdisjoint(metadata["tuning_indices"].tolist())
    assert metadata["n_excluded_reference_baselines"] == len(excluded)
    assert metadata["training_and_tuning_exclude_reference_baselines"]
    assert imputer is not None
