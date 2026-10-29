import numpy as np
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



def test_numeric_products_are_shared_and_invalidation_drops_only_references(tmp_path):
    artifact_root = tmp_path / "artifact_store"
    store = RunStore(tmp_path / "results" / "run", artifact_root=artifact_root)
    predictions = store.save_predictions(
        event_id=np.arange(5),
        prediction_ps=np.arange(5, dtype=float),
        corrected_ps=np.arange(5, dtype=float) + 1,
        led_residual_ps=np.arange(5, dtype=float) + 2,
        dataset_role="blind",
        metadata={
            "final_fit_fingerprint": np.asarray("a" * 64),
            "analysis_protocol_identity": np.asarray("b" * 64),
        },
    )
    assert predictions.is_file()
    assert predictions.is_relative_to(artifact_root)
    assert not (store.artifacts_dir / "pred.npz").exists()
    np.testing.assert_array_equal(store.load_predictions()["event_id"], np.arange(5))

    draws = store.save_bootstrap_draws(
        {"ctr_ps": np.arange(4, dtype=float)},
        fingerprint="c" * 64,
        blind_fingerprint="d" * 64,
    )
    assert draws.is_file()
    assert draws.is_relative_to(artifact_root)
    assert not (store.artifacts_dir / "bootstrap.npz").exists()

    store.invalidate_from("blind")
    assert not store.predictions_path.exists()
    assert not store.bootstrap_draws_path.exists()
    assert predictions.is_file()
    assert draws.is_file()


def test_external_store_does_not_fall_back_to_local_predictions(tmp_path):
    store = RunStore(tmp_path / "run", artifact_root=tmp_path / "artifact_store")
    store.artifacts_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        store.artifacts_dir / "pred.npz",
        event_id=np.arange(2),
        prediction_ps=np.zeros(2),
        corrected_ps=np.ones(2),
        led_residual_ps=np.ones(2),
    )
    assert not store.predictions_path.is_file()
    assert store.predictions_path != store.artifacts_dir / "pred.npz"

