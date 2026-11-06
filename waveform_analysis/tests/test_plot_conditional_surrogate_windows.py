"""Offline per-window plots only read prediction artifacts."""
import csv
import json
import numpy as np
import pytest
pytest.importorskip("matplotlib")
from waveform_analysis.scripts.plot_conditional_surrogate_windows import generate


def _write_case(root, *, offset=0.0, shuffle=False):
    root.mkdir()
    starts = np.array([0, 20, 40])
    target = np.array([1., 2., 3., 4.])
    if shuffle:
        target = target[::-1]
    est = np.broadcast_to(target[:, None, None], (4, 3, 2)).copy()
    est += offset
    np.savez_compressed(root / "diagnostics.npz",
                        window_starts=starts,
                        tuning_query_rows=np.array([10, 11, 12, 13]),
                        holdout_query_rows=np.array([20, 21, 22, 23]),
                        tuning_original_prediction_ps=target,
                        holdout_original_prediction_ps=target,
                        tuning_conditional_prediction_ps=est,
                        holdout_conditional_prediction_ps=est)
    (root / "summary.json").write_text(json.dumps({"method": "test"}))


def test_plots_csv_and_all_metrics_without_waveform_sources(tmp_path):
    mlp, cnn, output = (tmp_path / n for n in ("mlp", "cnn", "out"))
    _write_case(mlp, offset=1.)
    _write_case(cnn, offset=2.)
    paths = generate(mlp, cnn, output, sample_period_ps=12.5)
    assert len(paths) == 6
    assert all(p.is_file() and p.stat().st_size > 0 for p in paths)
    with (output / "per_window_metrics.csv").open(newline="") as file:
        rows = list(csv.DictReader(file))
    assert len(rows) == 2 * 2 * 3 * 2
    assert {r["window_start_ns"] for r in rows} == {"0.0", "0.25", "0.5"}
    assert all(float(r["rmse_ps"]) == (1. if r["model"] == "MLP" else 2.)
               for r in rows)


def test_different_events_are_rejected(tmp_path):
    mlp, cnn = (tmp_path / n for n in ("mlp", "cnn"))
    _write_case(mlp)
    _write_case(cnn, shuffle=True)
    with pytest.raises(AssertionError, match="original targets differ"):
        generate(mlp, cnn, tmp_path / "out")
