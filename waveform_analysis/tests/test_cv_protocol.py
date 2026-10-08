from types import SimpleNamespace
import numpy as np
from waveform_analysis.data.splits import make_cv_split
from waveform_analysis.engine.validation import best_complete_candidate,evaluate_candidate,pruning_decision
def row(fold,value,led=100.):return {"candidate_id":"c","fold_id":fold,"ctr_ps":float(value),"rmse_ps":float(value+10),"led_ctr_ps":float(led),"led_rmse_ps":float(led+10)}
def pruning(**kw):
    v={"enabled":True,"startup_complete_candidates":2,"min_folds_before_prune":1,"max_degradation_ps":5.0,"prune_if_worse_than_led":False,"led_max_degradation_ps":0.0};v.update(kw);return v
def test_folds_are_deterministic_disjoint_and_cover():
    a=make_cv_split(103,population_identity="dev",batch_seed=1001,n_folds=5,shuffle=True);b=make_cv_split(103,population_identity="dev",batch_seed=1001,n_folds=5,shuffle=True);c=make_cv_split(103,population_identity="dev",batch_seed=1002,n_folds=5,shuffle=True)
    for x,y in zip(a.folds,b.folds):np.testing.assert_array_equal(x.validation,y.validation);assert not set(x.train)&set(x.validation)
    assert set(np.concatenate([x.validation for x in a.folds]))==set(range(103));assert any(not np.array_equal(x.validation,y.validation) for x,y in zip(a.folds,c.folds))
def test_common_fold_pruning_and_exact_threshold():
    candidate=[row(1,111),row(2,109)];inc=[row(1,100),row(2,100),row(3,1),row(4,1),row(5,1)];d=pruning_decision(candidate,metric="ctr",pruning=pruning(),total_folds=5,incumbent_candidate_id="best",incumbent_rows=inc);assert d.pruned and d.incumbent_degradation_ps==10
    d=pruning_decision([row(1,105)],metric="ctr",pruning=pruning(),total_folds=5,incumbent_candidate_id="best",incumbent_rows=inc);assert not d.pruned
def test_led_first_startup_std_and_selection():
    d=pruning_decision([row(1,101,100)],metric="ctr",pruning=pruning(prune_if_worse_than_led=True),total_folds=5,incumbent_candidate_id="best",incumbent_rows=[row(1,50)]);assert d.reason=="led"
    folds=[SimpleNamespace(fold_id=i) for i in range(1,6)];summary,_=evaluate_candidate(candidate_id="x",parameters={},folds=folds,evaluate_fold=lambda p,f:row(f.fold_id,150,100),metric="ctr",pruning=pruning(prune_if_worse_than_led=True));assert summary["pruned"] and summary["completed_folds"]==1 and summary["ctr_std_ps"]==0
    rows=[{"candidate_id":"partial","pruned":True,"completed_folds":1,"total_folds":5,"ctr_mean_ps":1.0},{"candidate_id":"full","pruned":False,"completed_folds":5,"total_folds":5,"ctr_mean_ps":100.0}];assert best_complete_candidate(rows,"ctr")["candidate_id"]=="full"
