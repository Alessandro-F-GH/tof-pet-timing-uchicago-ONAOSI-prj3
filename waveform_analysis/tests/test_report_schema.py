from __future__ import annotations

import csv
import json

import pytest

from waveform_analysis.ml_pipeline.report import CURRENT_RESULT_SCHEMA, collect_results


def _write_run(tmp_path, *, schema, model, formulation):
    run_dir = tmp_path / model
    run_dir.mkdir(parents=True)
    manifest = {
        "schema_version": schema,
        "name": f"study_{model}",
        "study_name": "report_schema",
        "analysis": {"root_file": "data.root", "true_tof_ps": 0.0},
        "model": model,
        "estimator_formulation": formulation,
        "mode": "energy_to_energy",
        "window_ns": {"start": -1.5, "end": 2.0},
        "window_name": "onishi",
        "analysis_protocol_identity": "protocol",
        "sampling_identity": "sampling",
        "shared_replicas": {"1": "replica_1"},
    }
    (run_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    fields = [
        "phase",
        "replica_index",
        "seed",
        "candidate_id",
        "ctr_ps",
        "uncorrected_ctr_ps",
        "improvement_ps",
        "improvement_percent",
        "rmse_ps",
        "uncorrected_rmse_ps",
        "rmse_improvement_ps",
        "rmse_improvement_percent",
    ]
    with (run_dir / "results.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerow(
            {
                "phase": "replica",
                "replica_index": 1,
                "seed": 123,
                "candidate_id": "candidate",
                "ctr_ps": 60.0,
                "uncorrected_ctr_ps": 80.0,
                "improvement_ps": 20.0,
                "improvement_percent": 25.0,
                "rmse_ps": 50.0,
                "uncorrected_rmse_ps": 70.0,
                "rmse_improvement_ps": 20.0,
                "rmse_improvement_percent": 28.57,
            }
        )
    return run_dir


@pytest.mark.parametrize(
    ("model", "expected_formulation"),
    [
        ("direct_minirocket", "direct"),
        ("shared_minirocket", "shared"),
        ("direct_linear_ridge", "direct"),
        ("shared_linear_ridge", "shared"),
    ],
)
def test_collect_results_uses_current_schema_and_model_registry(
    tmp_path, model, expected_formulation
):
    run_dir = _write_run(
        tmp_path,
        schema=CURRENT_RESULT_SCHEMA,
        model=model,
        formulation="stale-value-intentionally-ignored",
    )
    records = collect_results([run_dir])
    assert len(records) == 1
    assert records[0]["model"] == model
    assert records[0]["estimator_formulation"] == expected_formulation
    assert records[0]["ctr_ps"] == pytest.approx(60.0)


def test_collect_results_rejects_noncurrent_schema(tmp_path):
    run_dir = _write_run(
        tmp_path,
        schema=CURRENT_RESULT_SCHEMA - 1,
        model="shared_linear_ridge",
        formulation="shared",
    )
    with pytest.raises(RuntimeError, match="current code requires schema"):
        collect_results([run_dir])


def test_collect_results_rejects_removed_model_name(tmp_path):
    run_dir = _write_run(
        tmp_path,
        schema=CURRENT_RESULT_SCHEMA,
        model="linear_ridge",
        formulation="shared",
    )
    with pytest.raises(RuntimeError, match="Unregistered model"):
        collect_results([run_dir])
