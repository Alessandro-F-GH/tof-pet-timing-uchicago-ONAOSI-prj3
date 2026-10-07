import numpy as np
from waveform_analysis.ml_pipeline.storage import RunStore
def test_fold_upsert_and_scoped_invalidation(tmp_path):
    s=RunStore(tmp_path);s.upsert_fold({"candidate_id":"a","fold_id":1,"ctr_ps":100});s.upsert_fold({"candidate_id":"a","fold_id":2,"ctr_ps":101});s.upsert_fold({"candidate_id":"a","fold_id":2,"ctr_ps":99});assert len(s.read_fold_rows("a"))==2 and float(s.read_fold_rows("a")[1]["ctr_ps"])==99
    s.save_predictions(event_id=[1,2],prediction_ps=[0,0],corrected_ps=[1,2],led_residual_ps=[2,3]);s.save_xai(time_ps=np.arange(2),importance_ps=np.arange(2));s.write_bootstrap({"n_resamples":10});s.save_bootstrap_draws({"ctr_ps":np.arange(3)});s.invalidate_from("bootstrap");assert s.predictions_path.is_file() and s.xai_path.is_file() and not s.bootstrap_path.exists()
