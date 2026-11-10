"""Plotting and inference-free replot tests for local TSHAP/deletion."""
from __future__ import annotations

import json

import numpy as np
import pytest

from waveform_analysis.scripts.local_tshap_deletion import (
    METHODS, PLOT_FILES, _window_plot_geometry, _write_event_plots,
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
        for direction in ("positive", "negative"):
            order = np.flatnonzero(scores[method] > 0 if direction == "positive"
                                   else scores[method] < 0)
            steps = np.arange(len(order) + 1)
            deletion[method][direction] = {
                "step": steps,
                "removed_window_indices": order,
                "signed_prediction_change_ps": .5 * steps,
                "absolute_prediction_deviation_ps": .5 * steps,
                "prediction_ps": 5.5 + .5 * steps,
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
                       waveform_mv=np.stack((np.arange(len(times)), -np.arange(len(times)))))
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
        for direction in ("positive", "negative"):
            deletion[method][direction].pop("absolute_prediction_deviation_ps")
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
        for direction in ("positive", "negative"):
            curve = payload["deletion"][method][direction]
            assert curve["absolute_prediction_deviation_ps"][0] == 0
            assert "absolute_error_change_ps" not in curve
    assert not (folder / "event_waveforms.png").exists()

def test_waveforms_only_requires_saved_model_run(tmp_path):
    args = parser().parse_args([
        "--waveforms-only", "--output-root", str(tmp_path)])
    with pytest.raises(ValueError, match="model-dir"):
        run(args)
