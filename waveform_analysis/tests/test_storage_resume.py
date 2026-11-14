import numpy as np

from waveform_analysis.core.dependencies import INVALIDATES, STAGE_DEPENDENCIES
from waveform_analysis.data.storage import RunStore
def test_fold_upsert_and_scoped_invalidation(tmp_path):
    s=RunStore(tmp_path);s.upsert_fold({"candidate_id":"a","fold_id":1,"ctr_ps":100});s.upsert_fold({"candidate_id":"a","fold_id":2,"ctr_ps":101});s.upsert_fold({"candidate_id":"a","fold_id":2,"ctr_ps":99});assert len(s.read_fold_rows("a"))==2 and float(s.read_fold_rows("a")[1]["ctr_ps"])==99
    s.save_predictions(event_id=[1,2],prediction_ps=[0,0],corrected_ps=[1,2],led_residual_ps=[2,3]);s.save_xai(time_ps=np.arange(2),importance_ps=np.arange(2));s.write_bootstrap({"n_resamples":10});s.save_bootstrap_draws({"ctr_ps":np.arange(3)});s.invalidate_from("bootstrap");assert s.predictions_path.is_file() and s.xai_path.is_file() and not s.bootstrap_path.exists()

def test_reading_does_not_implicitly_migrate(tmp_path):
    (tmp_path/"manifest.json").write_text('{"schema_version":51,"status":"complete"}')
    store=RunStore(tmp_path)
    assert store.manifest_path==tmp_path/"metadata"/"manifest.json"
    assert not store.manifest_path.is_file()
    assert (tmp_path/"manifest.json").is_file()
    store.write_best({"candidate_id":"example"})
    assert store.best_path.is_file()
    assert not (tmp_path/"best.json").exists()



def test_numeric_products_live_in_run_and_invalidation_removes_only_downstream(tmp_path):
    store = RunStore(tmp_path / "results" / "run")
    predictions = store.save_predictions(
        event_id=np.arange(5),
        prediction_ps=np.arange(5, dtype=float),
        corrected_ps=np.arange(5, dtype=float) + 1,
        led_residual_ps=np.arange(5, dtype=float) + 2,
        metadata={
            "final_fit_fingerprint": np.asarray("example"),
            "analysis_protocol_identity": np.asarray("development"),
        },
    )
    assert predictions == store.root / "artifacts" / "predictions.npz"
    assert predictions.is_file()
    np.testing.assert_array_equal(store.load_predictions()["event_id"], np.arange(5))
    draws = store.save_bootstrap_draws({"ctr_ps": np.arange(4, dtype=float)})
    assert draws == store.root / "artifacts" / "bootstrap.npz"
    assert draws.is_file()

    store.invalidate_from("blind")
    assert not predictions.exists()
    assert not draws.exists()

def test_dependency_graph_defines_transitive_invalidation():
    assert STAGE_DEPENDENCIES["selection"] == ("cv",)
    assert STAGE_DEPENDENCIES["final_fit"] == ("selection",)
    assert STAGE_DEPENDENCIES["blind"] == ("final_fit",)
    assert STAGE_DEPENDENCIES["bootstrap"] == ("blind",)
    assert STAGE_DEPENDENCIES["xai"] == ("final_fit", "blind")
    assert INVALIDATES["cv"] == (
        "cv", "selection", "final_fit", "blind", "bootstrap", "xai", "plots"
    )
    assert INVALIDATES["bootstrap"] == ("bootstrap", "plots")


def test_stage_local_invalidation_preserves_upstream_state(tmp_path):
    store = RunStore(tmp_path)
    for stage in ("cv", "selection", "final_fit", "blind", "bootstrap"):
        store.mark_stage(stage, f"{stage}-fingerprint")
    store.invalidate_stage("blind")
    assert store.stage_status("cv", "cv-fingerprint") == "complete"
    assert store.stage_status("selection", "selection-fingerprint") == "complete"
    assert store.stage_status("final_fit", "final_fit-fingerprint") == "complete"
    assert store.stage_status("blind", "blind-fingerprint") == "missing"
    assert store.stage_status("bootstrap", "bootstrap-fingerprint") == "complete"


def test_stage_resume_uses_explicit_completion_without_fingerprint(tmp_path):
    store = RunStore(tmp_path)
    assert store.stage_status("cv", "any") == "missing"
    store.mark_stage("cv", "old-fingerprint")
    assert store.stage_status("cv", "changed-fingerprint") == "complete"
    assert "fingerprint" not in store.read_state()["stages"]["cv"]
