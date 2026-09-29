from __future__ import annotations
import inspect
import json
import numpy as np
import pytest
from waveform_analysis.ml_pipeline import batch,prepared_data,study
from waveform_analysis.ml_pipeline.config import ConfigError,load_config
from waveform_analysis.ml_pipeline.event_selection import baseline_quality
from waveform_analysis.ml_pipeline.hyperparameter_plot import plot_hyperparameter_validation
from waveform_analysis.ml_pipeline.search import candidate_id,candidate_manifest
from waveform_analysis.ml_pipeline.splits import make_resampling_split
from waveform_analysis.ml_pipeline.storage import RESULT_FIELDS,RunStore


def test_split_pairing_invariant():
    a=make_resampling_split(100,analysis_identity="population-A",resampling_seed=7,validation_fraction=.2,test_fraction=.2)
    b=make_resampling_split(100,analysis_identity="population-A",resampling_seed=7,validation_fraction=.2,test_fraction=.2)
    np.testing.assert_array_equal(a.train,b.train);np.testing.assert_array_equal(a.validation,b.validation);np.testing.assert_array_equal(a.test,b.test)


def test_seed_changes_split_reproducibly():
    a=make_resampling_split(100,analysis_identity="population-A",resampling_seed=7,validation_fraction=.2,test_fraction=.2)
    b=make_resampling_split(100,analysis_identity="population-A",resampling_seed=8,validation_fraction=.2,test_fraction=.2)
    c=make_resampling_split(100,analysis_identity="population-A",resampling_seed=8,validation_fraction=.2,test_fraction=.2)
    assert not np.array_equal(a.test,b.test);np.testing.assert_array_equal(b.test,c.test)


def test_bootstrap_seeds_are_deterministic_from_one_base_seed():
    cfg={"seed":1001,"n_bootstrap":10}
    first=study._resampling_seeds(cfg);second=study._resampling_seeds(cfg)
    assert first==second and len(first)==10 and len(set(first))==10
    assert first!=study._resampling_seeds({"seed":1002,"n_bootstrap":10})


def test_candidate_ids_stable_under_grid_reordering():
    a={"learning_rate":0.01,"batch_size":16};b={"learning_rate":0.001,"batch_size":16}
    first=candidate_manifest([a,b]);second=candidate_manifest([{"x":1},b,a])
    assert candidate_id(a) in first and candidate_id(a) in second
    assert candidate_id(a)==candidate_id(dict(reversed(list(a.items()))))


@pytest.mark.parametrize("waveform,expected",[(np.asarray([0.,0.1,-0.1,0.]),False),(np.asarray([-9.5,-9.3,-9.2,-9.4]),True),(np.asarray([9.2,9.4,9.3,9.5]),True)])
def test_baseline_clipping_checks_both_boundaries(waveform,expected):
    _,clipped=baseline_quality(waveform,trigger_index=4,sample_interval_s=1e-9,window_ns=(-4.,-1.),vertical_limits_mV=(-10.,10.),clipping_margin_mV=1.0)
    assert clipped is expected


def test_partial_results_resume_without_duplicates(tmp_path):
    store=RunStore(tmp_path/"run")
    row={"seed":1,"stage":"validation","candidate_id":"abc","selected":False,"ctr_ps":60.0,"ctr_uncertainty_ps":float("nan"),"uncorrected_ctr_ps":90.0,"n":50,"rmse_ps":25.0}
    store.upsert_result(row);store.upsert_result(dict(row,ctr_ps=59.0));rows=RunStore(tmp_path/"run",resume=True).read_results()
    assert len(rows)==1 and float(rows[0]["ctr_ps"])==59.0


def test_results_schema_has_no_legacy_scan_columns():
    assert "voltage" not in RESULT_FIELDS
    assert "threshold" not in RESULT_FIELDS
    assert "hyperparameters" not in RESULT_FIELDS
    required=(
        "seed","stage","model","estimator_formulation","mode","population_identity","candidate_id",
        "ctr_ps","ctr_uncertainty_ps","uncorrected_ctr_ps",
        "rmse_ps","rmse_uncertainty_ps","uncorrected_rmse_ps","rmse_improvement_ps",
        "n","swap_rmse_ps",
    )
    assert set(required)<=set(RESULT_FIELDS)


def test_hyperparameter_plot_preserves_full_combinations(tmp_path):
    candidates={"a":{"learning_rate":1e-3,"batch_size":16},"b":{"learning_rate":1e-3,"batch_size":32},"c":{"learning_rate":1e-2,"batch_size":16},"d":{"learning_rate":1e-2,"batch_size":32}}
    rows=[{"seed":seed,"stage":"validation","candidate_id":cid,"ctr_ps":50+i} for seed in (1,2) for i,cid in enumerate(candidates)]
    path=plot_hyperparameter_validation(rows,candidates,tmp_path/"grid.png");assert path is not None and path.is_file()


def test_hyperparameter_plot_supports_rmse(tmp_path):
    candidates={"a":{"learning_rate":1e-3},"b":{"learning_rate":1e-2}}
    rows=[{"seed":1,"stage":"validation","candidate_id":"a","rmse_ps":30.0},{"seed":1,"stage":"validation","candidate_id":"b","rmse_ps":25.0}]
    path=plot_hyperparameter_validation(rows,candidates,tmp_path/"rmse.png",metric="rmse_ps",metric_label="RMSE")
    assert path is not None and path.is_file()


def test_one_candidate_has_no_validation_plot(tmp_path):
    assert plot_hyperparameter_validation([{"seed":1,"stage":"validation","candidate_id":"a","ctr_ps":50.0}],{"a":{"learning_rate":1e-3}},tmp_path/"single.png") is None


def test_old_experiment_schema_is_rejected_before_resolution(tmp_path):
    path=tmp_path/"legacy.json";path.write_text(json.dumps({"experiment":{"type":"model_study"}}),encoding="utf-8")
    with pytest.raises(ConfigError,match="Old experiment schema"):
        load_config(path,project_root=tmp_path)


def test_batch_execution_is_sequential(monkeypatch,tmp_path):
    calls=[]
    def fake_run(cfg,**kwargs):
        calls.append(cfg["name"]);return tmp_path/cfg["name"]
    monkeypatch.setattr(batch,"run_study",fake_run)
    configs=[{"name":"first"},{"name":"second"},{"name":"third"}]
    outputs=batch.run_batch(configs)
    assert calls==["first","second","third"]
    assert [p.name for p in outputs]==calls


def test_prepared_data_uses_control_led_and_window_scoped_population():
    source=inspect.getsource(prepared_data.prepare_ml_dataset)
    assert 'selected_led_threshold_mV' in source
    assert 'led_selection"]["thresholds_mV' not in source
    assert 'coincidence & window_valid' in source
    assert '"population_scope":"dataset+mode+window"' in source


def test_orchestration_has_generic_candidate_policy_and_no_concrete_model_branching():
    source=inspect.getsource(study.run_study)
    assert 'if len(candidates)==1' in source
    assert 'np.concatenate([split.train,split.validation])' in source
    assert 'dataset,split.train,params' in source
    assert 'dataset,config["mode"],split.validation' in source
    assert "onishi_cnn" not in source
    assert "locally_connected_mlp" not in source
    assert "model_name ==" not in source


def test_study_manifest_contains_resolved_preprocessing_configuration():
    source=inspect.getsource(study.run_study)
    assert '"preprocessing":config["preprocessing"]' in source
    assert '"preprocessing_fingerprint"' in source
