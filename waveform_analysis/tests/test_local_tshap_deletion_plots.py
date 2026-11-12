"""Plotting and inference-free replot tests for local TSHAP/deletion."""
from __future__ import annotations

import json

import numpy as np
import pytest

from waveform_analysis.scripts.local_tshap_deletion import (
    METHODS, PLOT_FILES, _window_plot_geometry, _write_event_plots,
    _reference_pointwise_stats, _event_pointwise_zscores,
    parser, replot_saved, run,
)


def _example():
    time_ps = np.arange(32, dtype=float) * 12.5 - 200.
    starts = np.array([4, 8, 12, 16], dtype=int)
    scores = {
        "knn": np.array([3., -1., 2., -2.]),
        "ppca": np.array([2., -3., 1., -1.]),
        "mean": np.array([-2., 4., 2., -1.]),
    }
    deletion = {}
    for method in METHODS:
        deletion[method] = {}
        for sequence in ("positive_negative", "negative_positive"):
            values = scores[method]
            positive = np.flatnonzero(values > 0)
            positive = positive[np.argsort(-values[positive])]
            negative = np.flatnonzero(values < 0)
            negative = negative[np.argsort(values[negative])]
            order = np.r_[positive, negative] if sequence == "positive_negative" else np.r_[negative, positive]
            steps = np.arange(len(order) + 2)
            changes = .5 * steps.astype(float)
            changes[-1] = 2.0
            deletion[method][sequence] = {
                "step": steps,
                "removed_window_indices": order,
                "first_phase_count": len(positive) if sequence == "positive_negative" else len(negative),
                "second_phase_count": len(negative) if sequence == "positive_negative" else len(positive),
                "signed_prediction_change_ps": changes,
                "fraction_deleted": np.r_[0., .125, .25, .375, .5, 1.],
                "prediction_ps": 5.5 + changes,
            }
    return time_ps, starts, scores, deletion


def test_plot_geometry_uses_prepared_time_grid_not_fixed_sample_duration():
    times = np.arange(20, dtype=float) * 25.
    center, width = _window_plot_geometry(times, np.array([2, 8]), 4)
    np.testing.assert_allclose(width, [.1, .1])
    np.testing.assert_allclose(center, [.1, .25])


def test_split_plots_saved_and_old_plots_removed(tmp_path):
    times, starts, scores, deletion = _example()
    for obsolete in ("tshap_deletion.png", "deletion_error.png",
                     "deletion_positive.png", "deletion_negative.png"):
        (tmp_path / obsolete).write_text("obsolete")
    _write_event_plots(tmp_path, 172, times, starts, 4,
                       scores, deletion, original=5.5,
                       waveform_mv=np.stack((np.arange(len(times)), -np.arange(len(times)))),
                       waveform_zscore=np.stack((np.linspace(-2, 2, len(times)),
                                                  np.linspace(2, -2, len(times)))),
                       reference_count=80)
    assert all((tmp_path / name).is_file() for name in PLOT_FILES)
    assert all((tmp_path / name).stat().st_size > 1000 for name in PLOT_FILES)
    assert not (tmp_path / "tshap_deletion.png").exists()
    assert not (tmp_path / "deletion_error.png").exists()
    assert not (tmp_path / "deletion_positive.png").exists()
    assert not (tmp_path / "deletion_negative.png").exists()
    assert (tmp_path / "deletion.png").exists()


def test_replot_from_saved_artifacts_does_not_require_model(tmp_path):
    times, starts, scores, deletion = _example()
    np.savez_compressed(tmp_path / "reference_details.npz",
                        starts=starts, time_ps=times)
    (tmp_path / "summary.json").write_text(
        json.dumps({"window_samples": 4}), encoding="utf-8")
    folder = tmp_path / "event_172"
    folder.mkdir()
    (folder / "results.json").write_text(json.dumps({
        "event_id": 172, "original_prediction_ps": 5.5, "target_ps": 6.,
        "waveform_mv": np.stack((np.arange(len(times)), -np.arange(len(times)))).tolist(),
        "waveform_zscore": np.stack((np.linspace(-2, 2, len(times)),
                                     np.linspace(2, -2, len(times)))).tolist(),
        "zscore_reference_count": 80,
        "signed_attributions_ps": {k: v.tolist() for k, v in scores.items()},
        "deletion": {
            method: {
                sign: {key: value.tolist() for key, value in result.items()}
                for sign, result in deletion[method].items()
            } for method in METHODS
        },
    }), encoding="utf-8")
    args = parser().parse_args([
        "--replot-only", "--output-root", str(tmp_path)])
    assert args.model_dir is None
    assert run(args) == tmp_path.resolve()
    assert all((folder / name).is_file() for name in PLOT_FILES)
    saved = json.loads((tmp_path / "summary.json").read_text(encoding="utf-8"))
    assert saved["event_plot_files"] == list(PLOT_FILES)


def test_replot_reports_missing_numeric_data(tmp_path):
    with pytest.raises(FileNotFoundError, match="reference_details"):
        replot_saved(tmp_path)

def test_replot_migrates_legacy_target_error_metrics_without_inference(tmp_path):
    times, starts, scores, deletion = _example()
    np.savez_compressed(tmp_path / "reference_details.npz", starts=starts, time_ps=times)
    (tmp_path / "summary.json").write_text(
        json.dumps({"window_samples": 4}), encoding="utf-8")
    folder = tmp_path / "event_173"
    folder.mkdir()
    for method in METHODS:
        for direction in ("positive_negative", "negative_positive"):
            deletion[method][direction]["absolute_error_change_ps"] = [99] * (
                len(deletion[method][direction]["step"]))
    (folder / "results.json").write_text(json.dumps({
        "event_id": 173, "original_prediction_ps": 5.5, "target_ps": 7.,
        "signed_attributions_ps": {k: v.tolist() for k, v in scores.items()},
        "deletion": {
            method: {direction: {k: v.tolist() if isinstance(v, np.ndarray) else v
                                  for k, v in result.items()}
                     for direction, result in deletion[method].items()}
            for method in METHODS},
    }), encoding="utf-8")
    replot_saved(tmp_path)
    payload = json.loads((folder / "results.json").read_text(encoding="utf-8"))
    for method in METHODS:
        for direction in ("positive_negative", "negative_positive"):
            curve = payload["deletion"][method][direction]
            assert "absolute_error_change_ps" not in curve
            assert curve["signed_prediction_change_ps"][0] == 0
    assert not (folder / "event_waveforms.png").exists()

def test_waveforms_only_requires_saved_model_run(tmp_path):
    args = parser().parse_args([
        "--waveforms-only", "--output-root", str(tmp_path)])
    with pytest.raises(ValueError, match="model-dir"):
        run(args)


def test_pointwise_reference_zscores_per_detector_and_time():
    # Different means across time/detectors and finite across-event variance.
    values = np.array([
        [[0., 10., 20.], [100., 200., 300.]],
        [[2., 12., 22.], [102., 202., 302.]],
        [[4., 14., 24.], [104., 204., 304.]],
    ], dtype=np.float64)
    mean, std, floor = _reference_pointwise_stats(values)
    np.testing.assert_allclose(mean, [[2., 12., 22.], [102., 202., 302.]])
    np.testing.assert_allclose(std, 2.)
    assert floor.shape == (2,)
    example = np.array([[4., 12., 20.], [102., 204., 304.]])
    z = _event_pointwise_zscores(example, mean, std)
    np.testing.assert_allclose(z, [[1., 0., -1.], [0., 1., 1.]])


def test_pointwise_reference_std_floor_avoids_division_by_zero():
    values = np.array([
        [[1., 2., 3.], [3., 2., 1.]],
        [[1., 4., 3.], [3., 2., 3.]],
        [[1., 6., 3.], [3., 2., 5.]],
    ], dtype=float)
    mean, std, floor = _reference_pointwise_stats(values)
    assert np.all(std > 0)
    assert np.all(floor > 0)
    assert np.isfinite(_event_pointwise_zscores(values[0], mean, std)).all()
