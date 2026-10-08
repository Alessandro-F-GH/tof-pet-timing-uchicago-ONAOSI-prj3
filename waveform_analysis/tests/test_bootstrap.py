from types import SimpleNamespace
import numpy as np
from waveform_analysis.reporting import stats
def fake(values,fit_cfg,seed=None,bootstrap=False):return SimpleNamespace(ctr_ps=float(np.std(np.asarray(values,float))))
def test_blind_central_independent_of_bootstrap_count(monkeypatch):
    monkeypatch.setattr(stats,"ctr_estimate",fake);corrected=np.linspace(-20,20,80);led=corrected*1.4+2;a,_=stats.blind_event_bootstrap(corrected,led,{},n_resamples=20,seed=11);b,_=stats.blind_event_bootstrap(corrected,led,{},n_resamples=200,seed=11);assert a["ctr_ps"]==b["ctr_ps"] and a["rmse_ps"]==b["rmse_ps"];assert a["bootstrap_retrains_model"] is False
def test_bootstrap_reproducible_paired_and_event_aligned(monkeypatch):
    monkeypatch.setattr(stats,"ctr_estimate",fake);corrected=np.arange(50,dtype=float)-25;led=corrected+np.linspace(-3,3,50);a,da=stats.blind_event_bootstrap(corrected,led,{},n_resamples=30,seed=99);b,db=stats.blind_event_bootstrap(corrected,led,{},n_resamples=30,seed=99)
    for key in da:np.testing.assert_allclose(da[key],db[key])
    np.testing.assert_allclose(da["rmse_improvement_ps"],da["led_rmse_ps"]-da["rmse_ps"]);assert a==b
    r=stats.paired_model_bootstrap([1,2,3,4,5],[1,2,3,4,5],[5,4,3,2,9],[1,1,1,1,100],{},metric="rmse",n_resamples=20,seed=7);assert r["n_matched"]==4
