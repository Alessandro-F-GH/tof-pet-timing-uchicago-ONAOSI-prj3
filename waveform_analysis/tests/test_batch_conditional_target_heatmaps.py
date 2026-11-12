"""Board-wide conditional heatmap job planning and resume contracts."""
from __future__ import annotations

from argparse import Namespace
from pathlib import Path
import json

import pytest

from waveform_analysis.scripts import batch_conditional_target_heatmaps as batch


def make_results(tmp_path, monkeypatch):
    root = tmp_path / "results"
    for mode, models in (("energy", ("ridge", "rocket", "cnn")),
                         ("timing", ("ridge", "rocket"))):
        for name in models:
            (root / mode / "wide" / name).mkdir(parents=True)
    monkeypatch.setattr(
        batch, "resolve_run_config",
        lambda p: {
            "mode": batch.MODES[Path(p).parts[-3]],
            "window_name": Path(p).parts[-2],
        },
    )
    monkeypatch.setattr(
        batch, "read_run",
        lambda p: {"manifest": {"status": "complete"}},
    )
    return root


def test_plan_separates_modes_and_creates_unique_unordered_pairs(tmp_path, monkeypatch):
    root = make_results(tmp_path, monkeypatch)
    jobs = batch.plan(root, tmp_path / "plots", "wide",
                      ["cnn", "rocket", "ridge"], ["rocket", "ridge"])
    assert len(jobs) == 3 + 3 + 2 + 1
    pairs = [(j["mode"], j["reference"], j["comparison"]) for j in jobs
             if j["kind"] == "pairwise"]
    assert pairs == [
        ("energy", "cnn", "rocket"),
        ("energy", "cnn", "ridge"),
        ("energy", "rocket", "ridge"),
        ("timing", "rocket", "ridge"),
    ]
    assert all(j["output"].parts[-4:-1] == (
        j["mode"], "wide", j["kind"]) for j in jobs)
    assert all(
        j["model_dir"].parts[-3] == j["mode"] and (
            j["comparison_dir"] is None or
            j["comparison_dir"].parts[-3] == j["mode"])
        for j in jobs)


def test_plan_rejects_missing_run_duplicate_and_wrong_mode(tmp_path, monkeypatch):
    root = make_results(tmp_path, monkeypatch)
    with pytest.raises(ValueError, match="Duplicate"):
        batch.plan(root, tmp_path / "out", "wide", ["ridge", "ridge"], [])
    with pytest.raises(FileNotFoundError, match="Model not found"):
        batch.plan(root, tmp_path / "out", "wide", ["missing"], [])
    with pytest.raises(ValueError, match="simple directory"):
        batch.plan(root, tmp_path / "out", "wide", ["../other"], [])
    monkeypatch.setattr(
        batch, "resolve_run_config",
        lambda p: {"mode": "timing_to_timing", "window_name": "wide"},
    )
    with pytest.raises(ValueError, match="mismatch"):
        batch.plan(root, tmp_path / "out", "wide", ["ridge"], [])


def args(root, out, **overrides):
    defaults = dict(
        results_root=root, output_root=out, window="wide",
        energy_models=["cnn", "ridge"], timing_models=["rocket"],
        cache_root=None, artifact_root=None,
        seed=42, tuning_queries=500, holdout_queries=500, chunk_size=128,
        normalization_scale_floor_ratio=.05, time_stride=8,
        z_bins=80, z_limit=3., min_count=25,
        baseline_start_ns=-2., baseline_end_ns=-1., peak_floor=1e-8,
        representations=list(batch.REPRESENTATIONS),
        include_original_derivative=False, derivative_end_sample=1200,
        retained_only=False,
        onishi_start_ns=None, onishi_end_ns=None,
        disable_onishi_highlight=False,
        overwrite=False, continue_on_error=False, dry_run=False,
    )
    defaults.update(overrides)
    return Namespace(**defaults)


def fake_runner(options):
    path = options.output_root
    path.mkdir(parents=True, exist_ok=True)
    reps = batch.effective_representations(
        batch.MODES[options.model_dir.parts[-3]], options.representations)
    for rep in reps:
        (path / rep).mkdir(exist_ok=True)
        (path / rep / "conditional_target_heatmaps.npz").write_bytes(b"test")
        (path / rep / "figure.png").write_bytes(b"test")
    (path / "summary.json").write_text(
        json.dumps({
            "representations": list(reps),
            "figures": [f"{r}/figure.png" for r in reps],
        }), encoding="utf-8",
    )


def test_batch_runs_resumes_and_detects_changed_settings(tmp_path, monkeypatch):
    root = make_results(tmp_path, monkeypatch)
    out = tmp_path / "out"
    a = args(root, out)
    calls = []
    def runner(o):
        calls.append((o.model_dir, o.comparison_model_dir))
        fake_runner(o)

    first = batch.run(a, runner=runner)
    assert len(calls) == 4   # 2 energy singles + pair + 1 timing single
    assert first["failed"] == 0
    second = batch.run(a, runner=runner)
    assert len(calls) == 4
    assert second["skipped"] == 4
    third = batch.run(args(root, out, time_stride=4), runner=runner)
    assert len(calls) == 8
    assert third["complete"] == 4


def test_dry_run_does_not_write(tmp_path, monkeypatch):
    root = make_results(tmp_path, monkeypatch)
    out = tmp_path / "out"
    result = batch.run(args(root, out, dry_run=True))
    assert result["planned"] == 4
    assert not out.exists()
