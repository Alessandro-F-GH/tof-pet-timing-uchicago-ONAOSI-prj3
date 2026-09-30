from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from waveform_analysis.ml_pipeline.config import ConfigError, _save_models, load_batch_config
from waveform_analysis.ml_pipeline.report import model_output_correlations
from waveform_analysis.ml_pipeline.study import _should_save_model


def test_model_save_policy_semantics():
    assert _should_save_model("all", 1)
    assert _should_save_model("all", 3)
    assert _should_save_model("first", 1)
    assert not _should_save_model("first", 2)
    assert not _should_save_model("none", 1)
    with pytest.raises(ConfigError):
        _save_models("sometimes")


def test_benchmark_batch_uses_first_replica_model_persistence():
    package_root = Path(__file__).resolve().parents[1]
    config_path = package_root / "config" / "batches" / "benchmark_other_models_49V.json"
    batch = load_batch_config(config_path, project_root=package_root)
    assert {run["save_models"] for run in batch.runs} == {"first"}


def _record(tmp_path, *, model, mode, replica, output, led):
    seed = 1000 + replica
    run_dir = tmp_path / f"{model}_{mode}"
    residual_dir = run_dir / "blind_residuals"
    residual_dir.mkdir(parents=True, exist_ok=True)
    candidate_id = "selected"
    np.savez_compressed(
        residual_dir / f"seed_{seed}_{candidate_id}.npz",
        corrected_ps=np.asarray(led, float) - np.asarray(output, float),
    )
    shared_dir = tmp_path / f"shared_{mode}_{replica}"
    shared_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        shared_dir / "blind_reference.npz",
        event_index=np.arange(len(led), dtype=np.int64),
        led_ps=np.asarray(led, float),
    )
    return {
        "study": "study",
        "source_run": str(run_dir),
        "dataset_key": "dataset-A",
        "dataset": "data.root",
        "population_identity": f"protocol-{mode}",
        "sampling_identity": f"sampling-{mode}",
        "model": model,
        "mode": mode,
        "window": "onishi",
        "window_start_ns": -1.5,
        "window_end_ns": 2.0,
        "replica_index": replica,
        "seed": seed,
        "candidate_id": candidate_id,
        "shared_replica": str(shared_dir),
    }


def test_model_output_correlation_is_replica_paired_and_mode_separated(tmp_path):
    led = np.asarray([10.0, 20.0, 30.0, 40.0, 50.0])
    records = []
    for mode in ("energy_to_energy", "timing_to_timing"):
        for replica in (1, 2):
            records.append(_record(tmp_path, model="model_a", mode=mode, replica=replica, output=[1, 2, 3, 4, 5], led=led))
            records.append(_record(tmp_path, model="model_b", mode=mode, replica=replica, output=[2, 4, 6, 8, 10], led=led))
    outputs, rows = model_output_correlations(records, tmp_path / "report")
    assert len(rows) == 2
    assert {row["mode"] for row in rows} == {"energy_to_energy", "timing_to_timing"}
    assert all(row["n_paired_replicas"] == 2 for row in rows)
    assert all(row["pearson_r_fisher_mean"] > 0.999 for row in rows)
    assert any(path.suffix == ".png" for path in outputs)
