"""Numerical and persistence checks for the bounded upstream TSHAP adapter."""
import json

import numpy as np
import pytest

from waveform_analysis.engine.tshap_explainer import (
    _choose_indices, _cleanup_legacy_result_tshap, _complete, _explanation_destination,
    _complete_local, _new_explainer, _publish_directory, _select_local_events, _selection_seed,
)


@pytest.mark.parametrize("roi", [False, True])
def test_tshap_matches_upstream_reference(roi):
    from tshap.tshap import TSHAPExplainer

    rng = np.random.default_rng(47)
    waveforms = rng.normal(size=(2, 2, 32)).astype(np.float32)
    baselines = rng.normal(size=(3, 2, 32)).astype(np.float32)
    def predict(x):
        x = np.asarray(x)
        return (x[:, 0, 3:13].sum(axis=1)
                - 0.4 * x[:, 1, 9:22].sum(axis=1)
                + 0.07 * x[:, 0, 4] * x[:, 1, 6])
    reference = TSHAPExplainer(window_length=8, stride=4, interpolation=True, roi=roi)
    bounded = _new_explainer(8, 4, roi, predict_batch_size=5)
    expected = reference.explain(waveforms, baselines, predict)
    actual = bounded.explain(waveforms, baselines, predict)
    for observed, original in zip(actual, expected):
        np.testing.assert_allclose(observed, original, rtol=1e-5, atol=1e-5)


def test_tshap_constant_prediction_is_zero():
    model = _new_explainer(6, 3, True, 4)
    x = np.ones((1, 2, 24), dtype=np.float32)
    refs = np.zeros((2, 2, 24), dtype=np.float32)
    maps = model.explain(x, refs, lambda v: np.full(len(v), 7.0))
    for value in maps:
        np.testing.assert_allclose(value, 0)


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
    (destination / "events.csv").write_text("event_id\\n")
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
    path = _local_explanation_destination(tmp_path / "xai", run)
    assert path == tmp_path / "xai" / "batch-A" / "energy" / "wide" / "direct_linear_ridge" / "tshap_local"
    assert path.name == "tshap_local"
    assert not path.is_relative_to(batch)
