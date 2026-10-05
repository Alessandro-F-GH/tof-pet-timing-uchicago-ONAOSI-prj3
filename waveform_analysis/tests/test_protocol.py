from __future__ import annotations

import inspect
import json
from pathlib import Path

import numpy as np
import pytest

from waveform_analysis.ml_pipeline import batch, event_selection, prepared_data, preprocessing_plots, study
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
    validation = {"phase": "hyperparameter_validation", "replica_index": "", "seed": 1, "candidate_id": "abc", "selected": False, "rmse_ps": 60.0, "ctr_ps": 70.0}
    replica = {"phase": "replica", "replica_index": 1, "seed": 2, "candidate_id": "abc", "selected": True, "ctr_ps": 55.0}
    store.upsert_result(validation)
    store.upsert_result(replica)
    store.upsert_result(dict(replica, ctr_ps=54.0))
    rows = store.read_results()
    assert len(rows) == 2
    assert [row["phase"] for row in rows] == ["hyperparameter_validation", "replica"]
    assert float(rows[0]["rmse_ps"]) == 60.0
    assert float(rows[0]["ctr_ps"]) == 70.0
    assert float(rows[1]["ctr_ps"]) == 54.0


def test_hyperparameter_plot_accepts_ctr_or_rmse(tmp_path):
    candidates = {"a": {"learning_rate": 1e-3}, "b": {"learning_rate": 1e-2}}
    rows = [
        {"phase": "hyperparameter_validation", "replica_index": "", "candidate_id": "a", "rmse_ps": 60.0, "ctr_ps": 70.0},
        {"phase": "hyperparameter_validation", "replica_index": "", "candidate_id": "b", "rmse_ps": 55.0, "ctr_ps": 65.0},
    ]
    rmse = plot_hyperparameter_validation(rows, candidates, tmp_path / "validation_rmse.png")
    ctr = plot_hyperparameter_validation(
        rows,
        candidates,
        tmp_path / "validation_ctr.png",
        metric="ctr_ps",
        metric_label="CTR",
    )
    assert rmse is not None and rmse.is_file()
    assert ctr is not None and ctr.is_file()


def test_model_selection_metric_is_required_and_validated(tmp_path):
    base = {
        "reference_dataset": {"root_file": "a.root", "true_tof_ps": 0, "channels": {"energy": [1, 2], "polarities": [1, 1]}},
        "analysis_dataset": {"root_file": "b.root", "true_tof_ps": 0, "channels": {"energy": [1, 2], "polarities": [1, 1]}},
        "preprocessing_config": {
            "materialized_window_ns": {"before": 7, "after": 40},
            "energy": {}, "timing": {},
            "selection": {"baseline_window_ns": [-2, -1], "baseline_noise": {"lambda_mad": 5}, "baseline_clipping": {"margin_mV": 1}},
            "photopeak": {}, "tot_peak": {},
            "led_selection": {"thresholds_mV": [15], "minimum_crossing_efficiency": 0.95, "coincidence_window_ns": 2},
            "io": {},
        },
        "model": "shared_linear_ridge",
        "mode": "energy_to_energy",
        "window": {"start": -1, "end": 1},
        "seed": 1,
        "evaluation": {"n_replicas": 2, "blind_fraction": 0.5, "minimum_events_per_split": 1},
        "fit": {"histogram_bin_width_ps": 20},
        "ml_output": {"max_abs_ps": 100},
        "output_dir": "x",
    }
    missing = tmp_path / "missing.json"
    missing.write_text(json.dumps({**base, "model_selection": {"validation_fraction": 0.1}}), encoding="utf-8")
    with pytest.raises(ConfigError, match="validation_fraction and metric"):
        load_config(missing, project_root=tmp_path)

    invalid = tmp_path / "invalid.json"
    invalid.write_text(json.dumps({**base, "model_selection": {"validation_fraction": 0.1, "metric": "mae"}}), encoding="utf-8")
    with pytest.raises(ConfigError, match="model_selection.metric"):
        load_config(invalid, project_root=tmp_path)


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
        "model_selection": {"validation_fraction": 0.1, "metric": "ctr"},
        "evaluation": {"n_replicas": 2, "blind_fraction": 0.5, "minimum_events_per_split": 1},
        "fit": {"histogram_bin_width_ps": 20},
        "ml_output": {"max_abs_ps": 100},
        "output_dir": "x",
        "resampling": {"seed": 1}
    }), encoding="utf-8")
    with pytest.raises(ConfigError, match="Unsupported study fields"):
        load_config(path, project_root=tmp_path)


def test_compact_benchmark_uses_configured_validation_metric_and_blind_fraction():
    package_root = Path(study.__file__).resolve().parents[1]
    config_path = package_root / "config" / "batches" / "benchmark_other_models_49V.json"
    resolved = load_batch_config(config_path, project_root=package_root)
    assert len(resolved.runs) == 9 * 2 * 2
    assert resolved.protocol["seed"] == 1001
    assert resolved.protocol["model_selection"] == {"validation_fraction": 0.20, "metric": "ctr"}
    assert resolved.protocol["evaluation"] == {"n_replicas": 5, "blind_fraction": 0.50, "minimum_events_per_split": 50}
    assert {config["save_models"] for config in resolved.runs} == {"first"}
    assert all("resampling" not in config for config in resolved.runs)


def test_default_preprocessing_uses_one_shared_baseline_window():
    package_root = Path(study.__file__).resolve().parents[1]
    config_path = package_root / "config" / "batches" / "benchmark_other_models_49V.json"
    resolved = load_batch_config(config_path, project_root=package_root)
    selection = resolved.runs[0]["preprocessing"]["selection"]
    assert selection == {
        "baseline_window_ns": [-4.0, -1.0],
        "baseline_noise": {"lambda_mad": 5.0},
        "baseline_clipping": {"margin_mV": 1.0},
    }
    assert "window_ns" not in selection["baseline_noise"]
    source = inspect.getsource(event_selection._scan_baseline)
    assert 'selection["baseline_window_ns"]' in source
    assert 'noise["window_ns"]' not in source


def test_baseline_clipping_clearance_preserves_original_rule():
    signal = np.asarray([0.0, 0.5, 1.0, 8.5, 9.0], float)
    rms, clipped, clearance = event_selection._baseline_quality_metrics(
        signal,
        trigger_index=4,
        sample_interval_s=1e-9,
        window_ns=[-4.0, 0.0],
        vertical_limits_mV=[-1.0, 10.0],
        clipping_margin_mV=1.0,
    )
    assert np.isfinite(rms)
    assert clipped is True
    assert clearance == pytest.approx(1.0)
    public_rms, public_clipped = event_selection.baseline_quality(
        signal,
        trigger_index=4,
        sample_interval_s=1e-9,
        window_ns=[-4.0, 0.0],
        vertical_limits_mV=[-1.0, 10.0],
        clipping_margin_mV=1.0,
    )
    assert public_rms == pytest.approx(rms)
    assert public_clipped is clipped


def test_preprocessing_histograms_fill_selected_region_without_margin_lines():
    for function in (
        preprocessing_plots.plot_photopeak,
        preprocessing_plots.plot_baseline_noise,
        preprocessing_plots.plot_tot,
    ):
        source = inspect.getsource(function)
        assert "_hist_with_selection" in source
        assert "axvline" not in source
    clipping_source = inspect.getsource(preprocessing_plots.plot_baseline_clipping)
    assert "_hist_with_selection" in clipping_source
    assert "ax.bar" not in clipping_source


def test_batch_publishes_materialized_waveform_example_for_each_mode():
    source = inspect.getsource(batch._publish_batch_selection_diagnostics)
    assert "preprocess_selected" in source
    assert "plot_materialized_event" in source
    assert 'f"{family}_selected_event_waveform.png"' in source


def test_prepared_data_uses_control_led_and_window_scoped_population():
    source = inspect.getsource(prepared_data.prepare_ml_dataset)
    assert 'selected_led_threshold_mV' in source
    assert 'coincidence & window_valid' in source


def test_study_flow_supports_ctr_or_rmse_tuning_and_excludes_validation_from_replicas():
    source = inspect.getsource(study.run_study)
    assert 'selection_field = "ctr_ps" if selection_name == "ctr" else "rmse_ps"' in source
    assert 'row.get(selection_field' in source
    assert '"selection_metric": selection_metric' in source
    assert 'fixed.split.validation' in source
    assert 'fixed validation excluded' in source.lower()
    assert 'fixed validation included in train' not in source.lower()
    assert 'final refit' not in source.lower()


def test_validation_ctr_uses_central_fit_without_bootstrap(monkeypatch):
    calls = []

    class Result:
        ctr_ps = 42.0

    def fake_ctr(values, config, *, seed, bootstrap):
        calls.append(bootstrap)
        return Result()

    monkeypatch.setattr(study, "ctr_estimate", fake_ctr)
    row = study._validation_row(
        seed=1,
        candidate_id="a",
        selected=False,
        corrected=np.asarray([1.0, -1.0, 0.5]),
        raw_validation_rmse=2.0,
        raw_validation_ctr=50.0,
        spec=type("Spec", (), {"name": "shared_linear_ridge", "estimator_formulation": "shared"})(),
        config={"fit": {"histogram_bin_width_ps": 10.0}, "mode": "energy_to_energy", "window_ns": {"start": -1.0, "end": 1.0}},
        event_identity="events",
        protocol_identity="protocol",
        sampling_identity="sampling",
        train_n=3,
    )
    assert calls == [False]
    assert row["ctr_ps"] == pytest.approx(42.0)
    assert row["uncorrected_ctr_ps"] == pytest.approx(50.0)
