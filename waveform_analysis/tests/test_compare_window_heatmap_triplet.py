"""Contracts for three matched prediction targets on the same wide waveform grid."""
from __future__ import annotations

from argparse import Namespace
import json
from pathlib import Path

import pytest

from waveform_analysis.scripts import compare_window_heatmap_triplet as triplet


def options(tmp_path, **extras):
    data = {
        "wide_model_dir": tmp_path / "energy" / "wide" / "rocket",
        "onishi_model_dir": tmp_path / "energy" / "onishi" / "rocket",
        "cache_root": tmp_path / "cache",
        "artifact_root": None,
        "output_root": tmp_path / "results",
        "seed": 42, "tuning_queries": 500, "holdout_queries": 500,
        "chunk_size": 128, "normalization_scale_floor_ratio": .05,
        "time_stride": 8, "z_bins": 80, "z_limit": 3.,
        "min_count": 25,
        "representations": ["original", "amplitude_normalized"],
        "baseline_start_ns": -2., "baseline_end_ns": -1.,
        "peak_floor": 1e-8, "retained_only": False,
        "include_original_derivative": False, "derivative_end_sample": 1200,
        "onishi_start_ns": None, "onishi_end_ns": None,
        "disable_onishi_highlight": False,
    }
    data.update(extras)
    return Namespace(**data)


def test_all_three_jobs_use_wide_reference_and_identical_settings(tmp_path):
    jobs = triplet.construct_jobs(options(tmp_path))
    assert [name for name, _ in jobs] == [
        "wide", "onishi_on_wide", "wide_minus_onishi"]
    assert [job.comparison_target for _, job in jobs] == [
        "reference", "comparison", "difference"]
    for _, job in jobs:
        assert job.model_dir == tmp_path / "energy" / "wide" / "rocket"
        assert job.comparison_model_dir == tmp_path / "energy" / "onishi" / "rocket"
        assert job.seed == 42
        assert job.time_stride == 8
        assert job.representations == ["original", "amplitude_normalized"]
        assert job.output_root.parent == tmp_path / "results"


def test_triplet_creates_manifest_and_checks_common_population(tmp_path):
    args = options(tmp_path)
    calls = []
    def fake_run(single):
        calls.append(single.comparison_target)
        single.output_root.mkdir(parents=True)
        (single.output_root / "summary.json").write_text(
            json.dumps({
                "matched_population": {"common_events": 2000},
                "events": 1000,
                "figures": ["original/foo.png"],
            }), encoding="utf-8")
    result = triplet.run(args, runner=fake_run)
    assert calls == ["reference", "comparison", "difference"]
    assert len(result["plots"]) == 3
    assert (args.output_root / "triplet_summary.json").is_file()


def test_triplet_rejects_same_model_dir(tmp_path):
    args = options(tmp_path, onishi_model_dir=tmp_path / "energy" / "wide" / "rocket")
    with pytest.raises(ValueError, match="must differ"):
        triplet.construct_jobs(args)
