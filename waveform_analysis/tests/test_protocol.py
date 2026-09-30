from __future__ import annotations

import inspect
import json
from pathlib import Path

import numpy as np
import pytest

from waveform_analysis.ml_pipeline import prepared_data, study
from waveform_analysis.ml_pipeline.config import ConfigError, load_batch_config, load_config
from waveform_analysis.ml_pipeline.hyperparameter_plot import plot_hyperparameter_validation
from waveform_analysis.ml_pipeline.search import candidate_id, candidate_manifest
from waveform_analysis.ml_pipeline.splits import make_fixed_validation_split, make_replica_split
from waveform_analysis.ml_pipeline.storage import RESULT_FIELDS, RunStore


def test_fixed_validation_is_deterministic_from_batch_seed():
    a = make_fixed_validation_split(100, analysis_identity="population-A", batch_seed=1001, validation_fraction=0.10)
    b = make_fixed_validation_split(100, analysis_identity="population-A", batch_seed=1001, validation_fraction=0.10)
    np.testing.assert_array_equal(a.tuning_train, b.tuning_train)
    np.testing.assert_array_equal(a.validation, b.validation)
    assert a.seed == b.seed
    assert len(a.validation) == 10
    assert len(a.tuning_train) == 90


def test_replica_partitions_nonvalidation_pool_and_never_reuses_validation():
    fixed = make_fixed_validation_split(100, analysis_identity="population-A", batch_seed=1001, validation_fraction=0.10)
    first = make_replica_split(100, fixed, analysis_identity="population-A", batch_seed=1001, replica_index=1, blind_fraction=0.50)
    second = make_replica_split(100, fixed, analysis_identity="population-A", batch_seed=1001, replica_index=2, blind_fraction=0.50)
    assert len(first.test) == 50
    assert len(first.train) == 40
    assert set(first.test).isdisjoint(set(fixed.validation))
    assert set(first.train).isdisjoint(set(fixed.validation))
    assert set(first.train) | set(first.test) == set(fixed.tuning_train)
    assert not np.array_equal(first.test, second.test)
    assert first.seed != second.seed


def test_replica_fraction_cannot_consume_entire_nonvalidation_pool():
    fixed = make_fixed_validation_split(100, analysis_identity="population-A", batch_seed=1001, validation_fraction=0.10)
    with pytest.raises(ValueError, match="after excluding fixed validation"):
        make_replica_split(100, fixed, analysis_identity="population-A", batch_seed=1001, replica_index=1, blind_fraction=0.90)


def test_batch_seed_controls_both_fixed_validation_and_replica_sampling():
    fixed_a = make_fixed_validation_split(100, analysis_identity="population-A", batch_seed=1001, validation_fraction=0.20)
    fixed_b = make_fixed_validation_split(100, analysis_identity="population-A", batch_seed=2002, validation_fraction=0.20)
    replica_a = make_replica_split(100, fixed_a, analysis_identity="population-A", batch_seed=1001, replica_index=1, blind_fraction=0.50)
    replica_b = make_replica_split(100, fixed_b, analysis_identity="population-A", batch_seed=2002, replica_index=1, blind_fraction=0.50)
    assert fixed_a.seed != fixed_b.seed
    assert replica_a.seed != replica_b.seed
    assert not np.array_equal(fixed_a.validation, fixed_b.validation)
    assert not np.array_equal(replica_a.test, replica_b.test)


def test_candidate_ids_stable_under_grid_reordering():
    a = {"learning_rate": 0.01, "batch_size": 16}
    b = {"learning_rate": 0.001, "batch_size": 16}
    first = candidate_manifest([a, b])
    second = candidate_manifest([{"x": 1}, b, a])
    assert candidate_id(a) in first and candidate_id(a) in second
    assert candidate_id(a) == candidate_id(dict(reversed(list(a.items()))))


def test_results_schema_separates_fixed_tuning_from_replicas():
    assert "stage" not in RESULT_FIELDS
    assert "phase" in RESULT_FIELDS
    assert "replica_index" in RESULT_FIELDS
    assert "sampling_identity" in RESULT_FIELDS


def test_result_upsert_keys_validation_and_replica_separately(tmp_path):
    store = RunStore(tmp_path / "run")
    validation = {"phase": "hyperparameter_validation", "replica_index": "", "seed": 1, "candidate_id": "abc", "selected": False, "rmse_ps": 60.0}
    replica = {"phase": "replica", "replica_index": 1, "seed": 2, "candidate_id": "abc", "selected": True, "ctr_ps": 55.0}
    store.upsert_result(validation)
    store.upsert_result(replica)
    store.upsert_result(dict(replica, ctr_ps=54.0))
    rows = store.read_results()
    assert len(rows) == 2
    assert [row["phase"] for row in rows] == ["hyperparameter_validation", "replica"]
    assert float(rows[0]["rmse_ps"]) == 60.0
    assert float(rows[1]["ctr_ps"]) == 54.0


def test_hyperparameter_plot_uses_fixed_validation_rmse(tmp_path):
    candidates = {"a": {"learning_rate": 1e-3}, "b": {"learning_rate": 1e-2}}
    rows = [
        {"phase": "hyperparameter_validation", "replica_index": "", "candidate_id": "a", "rmse_ps": 60.0},
        {"phase": "hyperparameter_validation", "replica_index": "", "candidate_id": "b", "rmse_ps": 55.0},
    ]
    path = plot_hyperparameter_validation(rows, candidates, tmp_path / "validation_rmse.png")
    assert path is not None and path.is_file()


def test_unsupported_study_schema_is_rejected(tmp_path):
    path = tmp_path / "old.json"
    path.write_text(json.dumps({
        "reference_dataset": {"root_file": "a.root", "true_tof_ps": 0, "channels": {}},
        "analysis_dataset": {"root_file": "b.root", "true_tof_ps": 0, "channels": {}},
        "preprocessing_config": {},
        "model": "antisymmetric_mlp",
        "mode": "energy_to_energy",
        "window": {"start": -1, "end": 1},
        "seed": 1,
        "model_selection": {"validation_fraction": 0.1},
        "evaluation": {"n_replicas": 2, "blind_fraction": 0.5, "minimum_events_per_split": 1},
        "fit": {"histogram_bin_width_ps": 20},
        "ml_output": {"max_abs_ps": 100},
        "output_dir": "x",
        "resampling": {"seed": 1}
    }), encoding="utf-8")
    with pytest.raises(ConfigError, match="Unsupported study fields"):
        load_config(path, project_root=tmp_path)


def test_compact_benchmark_uses_fixed_validation_and_blind_fraction():
    package_root = Path(study.__file__).resolve().parents[1]
    config_path = package_root / "config" / "batches" / "benchmark_other_models_49V.json"
    resolved = load_batch_config(config_path, project_root=package_root)
    assert len(resolved.runs) == 6 * 2 * 2
    assert resolved.protocol["seed"] == 1001
    assert resolved.protocol["model_selection"] == {"validation_fraction": 0.20}
    assert resolved.protocol["evaluation"] == {"n_replicas": 5, "blind_fraction": 0.50, "minimum_events_per_split": 50}
    assert {config["save_models"] for config in resolved.runs} == {"first"}
    assert all("resampling" not in config for config in resolved.runs)


def test_prepared_data_uses_control_led_and_window_scoped_population():
    source = inspect.getsource(prepared_data.prepare_ml_dataset)
    assert 'selected_led_threshold_mV' in source
    assert 'coincidence & window_valid' in source


def test_study_flow_uses_rmse_tuning_and_excludes_validation_from_replicas():
    source = inspect.getsource(study.run_study)
    assert '"hyperparameter_validation"' in source
    assert 'for replica_index in range(1, n_replicas + 1)' in source
    assert 'fixed.split.tuning_train' in source
    assert 'fixed.split.validation' in source
    assert 'CandidateScore(candidate_id, float(row["rmse_ps"]))' in source
    assert '"selection_metric": "fixed_validation_rmse_ps"' in source
    assert 'fixed validation excluded' in source.lower()
    assert 'fixed validation included in train' not in source.lower()
    assert 'hyperparameter_validation_ctr.png' not in source
    assert 'final refit' not in source.lower()
