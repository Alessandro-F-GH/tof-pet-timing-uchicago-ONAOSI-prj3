from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np

from .common import canonical_hash
from .control_preprocessing import fit_control_artifact
from .data import preprocess_selected
from .event_selection import apply_selection_rules
from .hyperparameter_plot import plot_hyperparameter_validation
from .models import get_model
from .prepared_data import prepare_ml_dataset
from .progress import ProgressTracker
from .result_plots import blind_rmse_ctr_correlation, make_study_result_plots
from .search import CandidateScore, candidate_manifest, choose_best
from .shared_artifacts import ExperimentArtifactStore
from .splits import make_resampling_split, semantic_seed
from .stats import ctr_estimate, paired_ctr_improvement, rmse_ps
from .storage import RunStore
from .train import FeatureTransformCache, FitInputCache, detector_swap_rmse, fit_on_indices, predict_indices, save_model
from .view import model_target


def _logger(run_dir):
    logger=logging.getLogger(f"waveform-study:{run_dir}");logger.setLevel(logging.INFO);logger.handlers.clear();logger.propagate=False;fmt=logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    for handler in (logging.StreamHandler(),logging.FileHandler(Path(run_dir)/"study.log",encoding="utf-8")):handler.setFormatter(fmt);logger.addHandler(handler)
    return logger

def _resampling_seeds(resampling):
    base_seed=int(resampling["seed"]);n=int(resampling["n_replicas"]);return [semantic_seed(base_seed,"resampling",i) for i in range(n)]

def _metric(values,fit_cfg,seed):
    values=np.asarray(values,float);finite=values[np.isfinite(values)]
    if finite.size!=values.size or not finite.size:raise RuntimeError("CTR requires one finite residual per evaluated event")
    result=ctr_estimate(finite,fit_cfg,seed=int(seed),bootstrap=False);return float(result.ctr_ps),int(result.n_valid)

def _baseline(values,fit_cfg,seed):
    ctr,n=_metric(values,fit_cfg,seed);return ctr,rmse_ps(values),n

def _row(seed,stage,cid,selected,corrected,uncorrected,fit_cfg,*,spec,config,event_population_identity,analysis_protocol_identity,swap_rmse_ps=float("nan"),paired=None,baseline=None):
    if stage=="blind" and paired is not None:
        ctr=float(paired.corrected_ctr_ps);raw_ctr=float(paired.led_ctr_ps);n=int(np.isfinite(corrected).sum());improvement=float(paired.improvement_ps);improvement_percent=float(paired.improvement_percent);rmse=float(paired.corrected_rmse_ps);raw_rmse=float(paired.led_rmse_ps);rmse_improvement=float(paired.rmse_improvement_ps);rmse_improvement_percent=float(paired.rmse_improvement_percent)
    else:
        ctr,n=_metric(corrected,fit_cfg,semantic_seed(seed,stage,cid))
        if baseline is None:raw_ctr,raw_rmse,_=_baseline(uncorrected,fit_cfg,semantic_seed(seed,stage,"baseline"))
        else:raw_ctr,raw_rmse,_=baseline
        improvement=improvement_percent=float("nan");rmse=rmse_ps(corrected);rmse_improvement=rmse_improvement_percent=float("nan")
    return {"seed":int(seed),"stage":stage,"model":spec.name,"estimator_formulation":spec.estimator_formulation,"mode":config["mode"],"window_start_ns":float(config["window_ns"]["start"]),"window_end_ns":float(config["window_ns"]["end"]),"population_identity":analysis_protocol_identity,"event_population_identity":event_population_identity,"analysis_protocol_identity":analysis_protocol_identity,"candidate_id":cid,"selected":bool(selected),"ctr_ps":ctr,"uncorrected_ctr_ps":raw_ctr,"improvement_ps":improvement,"improvement_percent":improvement_percent,"rmse_ps":rmse,"uncorrected_rmse_ps":raw_rmse,"rmse_improvement_ps":rmse_improvement,"rmse_improvement_percent":rmse_improvement_percent,"n":n,"swap_rmse_ps":float(swap_rmse_ps) if stage=="blind" else float("nan")}


def _blind_evaluation(store,spec,fitted,dataset,config,split,target,seed,cid,event_identity,protocol_identity,shared_replica=None):
    pred=predict_indices(spec,fitted,dataset,config["mode"],split.test);corrected=np.asarray(target[split.test],float)-pred;led=np.asarray(shared_replica.led_ps if shared_replica is not None else target[split.test],float);swap=detector_swap_rmse(spec,fitted,dataset,config["mode"],split.test)
    paired=paired_ctr_improvement(corrected,led,config["fit"],seed=semantic_seed(seed,"paired_led_ml",cid),led_ctr_ps=None if shared_replica is None else shared_replica.led_ctr_ps,led_rmse_ps=None if shared_replica is None else shared_replica.led_rmse_ps)
    store.save_blind_residuals(seed,cid,corrected)
    return _row(seed,"blind",cid,True,corrected,led,config["fit"],spec=spec,config=config,event_population_identity=event_identity,analysis_protocol_identity=protocol_identity,swap_rmse_ps=swap,paired=paired)


def run_study(config,*,overwrite=False,resume=False,rebuild_preprocessing=False):
    run_dir=Path(config["output_dir"]).resolve();store=RunStore(run_dir,overwrite=overwrite,resume=resume);logger=_logger(run_dir);store.write_resolved_config({k:v for k,v in config.items() if not str(k).startswith("_")})
    spec=get_model(config["model"]["name"]);candidates=candidate_manifest(list(spec.candidates(config["model"]["space"])));seeds=_resampling_seeds(config["resampling"])
    manifest={"schema_version":37,"status":"running","name":config["name"],"study_name":config.get("study_name",config["name"]),"run_id":config.get("run_id",config["name"]),"window_name":config.get("window_name"),"config_fingerprint":config["_config_fingerprint"],"reference":config["reference"],"analysis":config["analysis"],"mode":config["mode"],"model":config["model"]["name"],"estimator_formulation":spec.estimator_formulation,"window_ns":config["window_ns"],"feature_transform":None if spec.feature_transform is None else spec.feature_transform.name,"resampling":config["resampling"],"generated_replica_seeds":seeds,"preprocessing": config["preprocessing"],"preprocessing_fingerprint":canonical_hash(config["preprocessing"]),"statistical_unit": "repeated_holdout_replica","fit_bootstrap": False,"event_level_bootstrap": False}
    if resume and (run_dir/"manifest.json").is_file():
        old=json.loads((run_dir/"manifest.json").read_text())
        if old.get("config_fingerprint")!=config["_config_fingerprint"]:raise RuntimeError("Cannot resume with a different resolved configuration")
        if config.get("_batch_artifact_root") and int(old.get("schema_version",0))<37:raise RuntimeError("Cannot resume a pre-shared-artifact batch run; rerun it cleanly")
    store.write_manifest(manifest);store.write_candidates(candidates)
    logger.info("Study | reference=%s | analysis=%s | mode=%s | model=%s | formulation=%s | window=%s | replicas=%d | base_seed=%d | candidates=%d",config["reference"]["root_file"],config["analysis"]["root_file"],config["mode"],config["model"]["name"],spec.estimator_formulation,config["window_ns"],len(seeds),int(config["resampling"]["seed"]),len(candidates))

    rebuild_control=bool(config.get("_rebuild_control",rebuild_preprocessing));rebuild_analysis=bool(config.get("_rebuild_analysis",rebuild_preprocessing));rebuild_prepared=bool(config.get("_rebuild_prepared",rebuild_preprocessing))
    control,control_dir=fit_control_artifact(config["reference"]["root_file"],config["reference"],config["preprocessing"],config["fit"],fit_by_mode=config.get("_control_fit_by_mode"),modes=config.get("_control_modes") or [config["mode"]],cache_root=config["preprocessing"]["cache_dir"],rebuild=rebuild_control,logger=logger)
    selection=apply_selection_rules(config["analysis"]["root_file"],config["analysis"],config["preprocessing"],control["selection_rules"],config["mode"],cache_dir=Path(config["preprocessing"]["cache_dir"])/"analysis_selection",rebuild=rebuild_analysis,logger=logger)
    native=preprocess_selected(config["analysis"]["root_file"],selection,config["analysis"],config["preprocessing"],config["mode"],cache_dir=Path(config["preprocessing"]["cache_dir"])/"analysis_native",rebuild=rebuild_analysis,logger=logger)
    dataset=prepare_ml_dataset(native,control,config,cache_dir=config["preprocessing"]["cache_dir"],rebuild=rebuild_prepared,logger=logger)
    event_identity=str(dataset.manifest.get("event_population_identity") or dataset.manifest["analysis_population_identity"]);protocol_identity=str(dataset.manifest.get("analysis_protocol_identity") or dataset.manifest["analysis_population_identity"]);shared_store=ExperimentArtifactStore(config["_batch_artifact_root"]) if config.get("_batch_artifact_root") else None
    manifest.update({"control_artifact":str(control_dir),"control_fingerprint":control["fingerprint"],"control_mode_fingerprint":(control.get("mode_fingerprints") or {}).get(config["mode"],control["fingerprint"]),"fixed_led_threshold_mV":control["selected_led_threshold_mV"][config["mode"]],"event_population_identity":event_identity,"analysis_protocol_identity":protocol_identity,"analysis_population_identity":protocol_identity,"prepared_dataset":str(dataset.directory),"shared_artifact_root":str(shared_store.root) if shared_store is not None else None,"shared_replicas":{}});store.write_manifest(manifest)

    target=model_target(dataset,config["mode"]);progress=ProgressTracker(logger,{"replica":len(seeds)})
    for pos,seed in enumerate(seeds,1):
        shared_replica=None
        if shared_store is not None:
            shared_replica=shared_store.prepare_replica(dataset,config,seed,target);split=shared_replica.split;manifest["shared_replicas"][str(seed)]=str(shared_replica.directory)
        else:
            split=make_resampling_split(dataset.n_events,analysis_identity=event_identity,resampling_seed=int(seed),validation_fraction=config["resampling"]["validation_fraction"],test_fraction=config["resampling"]["test_fraction"]);minimum=int(config["resampling"]["minimum_events_per_split"])
            if min(len(split.train),len(split.validation),len(split.test))<minimum:raise RuntimeError(f"Resampling seed {seed} violates minimum_events_per_split={minimum}")
            store.save_split(seed,dataset.event_index,split)
        logger.info("Replica %d/%d | seed=%s | train=%d validation=%d test=%d | shared=%s",pos,len(seeds),seed,len(split.train),len(split.validation),len(split.test),shared_store is not None)
        transform_cache=FeatureTransformCache();fit_input_cache=FitInputCache();transform_seed_base=semantic_seed(seed,config["model"]["name"],"transform")

        if len(candidates) == 1:
            cid,params=next(iter(candidates.items()))
            if not store.has_result(seed,"blind",cid):
                fit_idx=np.concatenate([split.train, split.validation]);fitted=fit_on_indices(spec,config["model"]["space"],config,dataset,fit_idx,params,seed=semantic_seed(seed,config["model"]["name"],cid,"final"),transform_seed_base=transform_seed_base,feature_transform_cache=transform_cache,fit_input_cache=fit_input_cache,logger=logger)
                row=_blind_evaluation(store,spec,fitted,dataset,config,split,target,seed,cid,event_identity,protocol_identity,shared_replica);store.upsert_result(row);save_model(spec,fitted,store.model_dir(seed,cid),params)
                logger.info("Blind | seed=%s | candidate=%s | CTR=%.3f ps | LED CTR=%.3f ps | improvement=%.3f ps (%.2f%%) | RMSE=%.3f ps | LED RMSE=%.3f ps",seed,cid,row["ctr_ps"],row["uncorrected_ctr_ps"],row["improvement_ps"],row["improvement_percent"],row["rmse_ps"],row["uncorrected_rmse_ps"])
            progress.complete("replica",f"seed {seed}",announce=False);continue

        validation_led=np.asarray(target[split.validation],float);validation_baseline=_baseline(validation_led,config["fit"],semantic_seed(seed,"validation_baseline"));scores=[]
        for i,(cid,params) in enumerate(candidates.items(),1):
            existing=next((r for r in store.read_results() if str(r.get("seed"))==str(seed) and r.get("stage")=="validation" and r.get("candidate_id")==cid),None)
            if existing:scores.append(CandidateScore(cid,float(existing["ctr_ps"])));continue
            logger.info("Candidate %d/%d | %s",i,len(candidates),cid)
            fitted=fit_on_indices(spec, config["model"]["space"], config, dataset, split.train, params,seed=semantic_seed(seed,config["model"]["name"],cid,"candidate"),transform_seed_base=transform_seed_base,feature_transform_cache=transform_cache,fit_input_cache=fit_input_cache,logger=logger)
            pred=predict_indices(spec, fitted, dataset, config["mode"], split.validation);row=_row(seed,"validation",cid,False,validation_led-pred,validation_led,config["fit"],spec=spec,config=config,event_population_identity=event_identity,analysis_protocol_identity=protocol_identity,baseline=validation_baseline);store.upsert_result(row);scores.append(CandidateScore(cid,float(row["ctr_ps"])));logger.info("Validation | candidate=%s | CTR=%.3f ps | RMSE=%.3f ps",cid,row["ctr_ps"],row["rmse_ps"])

        best=choose_best(scores);cid=best.candidate_id;params=candidates[cid];logger.info("Selected candidate | seed=%s | %s | criterion=validation CTR",seed,cid);rows=store.read_results()
        for row in rows:
            if str(row.get("seed"))==str(seed) and row.get("stage")=="validation" and row.get("candidate_id")==cid:row["selected"]=True
        store._atomic_rows(rows)
        if not store.has_result(seed,"blind",cid):
            fit_idx=np.concatenate([split.train, split.validation]);logger.info("Final refit | candidate=%s | n=%d",cid,len(fit_idx));fitted=fit_on_indices(spec,config["model"]["space"],config,dataset,fit_idx,params,seed=semantic_seed(seed,config["model"]["name"],cid,"final"),transform_seed_base=transform_seed_base,feature_transform_cache=transform_cache,fit_input_cache=fit_input_cache,logger=logger)
            row=_blind_evaluation(store,spec,fitted,dataset,config,split,target,seed,cid,event_identity,protocol_identity,shared_replica);store.upsert_result(row);save_model(spec,fitted,store.model_dir(seed,cid),params);logger.info("Blind | seed=%s | candidate=%s | CTR=%.3f ps | LED CTR=%.3f ps | improvement=%.3f ps (%.2f%%) | RMSE=%.3f ps | LED RMSE=%.3f ps",seed,cid,row["ctr_ps"],row["uncorrected_ctr_ps"],row["improvement_ps"],row["improvement_percent"],row["rmse_ps"],row["uncorrected_rmse_ps"])
        progress.complete("replica",f"seed {seed}",note=f"candidate={cid}");store.write_manifest(manifest)

    rows=store.read_results()
    if len(candidates)>1:
        plot_hyperparameter_validation(rows,candidates,run_dir/"hyperparameter_validation_ctr.png",logger,metric="ctr_ps",metric_label="CTR");plot_hyperparameter_validation(rows,candidates,run_dir/"hyperparameter_validation_rmse.png",logger,metric="rmse_ps",metric_label="RMSE")
    plots=make_study_result_plots(rows,run_dir,model=spec.name,mode=config["mode"],window_ns=config["window_ns"]);blind_ctr=[float(r["ctr_ps"]) for r in rows if r.get("stage")=="blind"];blind_rmse=[float(r["rmse_ps"]) for r in rows if r.get("stage")=="blind"];ctr_improvements=[float(r["improvement_ps"]) for r in rows if r.get("stage")=="blind" and str(r.get("improvement_ps","")).strip()];rmse_improvements=[float(r["rmse_improvement_ps"]) for r in rows if r.get("stage")=="blind" and str(r.get("rmse_improvement_ps","")).strip()];correlation,n_correlation=blind_rmse_ctr_correlation(rows)
    manifest.update({"status":"complete","blind_ctr_mean_ps":float(np.mean(blind_ctr)) if blind_ctr else None,"blind_ctr_std_ps":float(np.std(blind_ctr,ddof=1)) if len(blind_ctr)>1 else 0.0,"blind_rmse_mean_ps":float(np.mean(blind_rmse)) if blind_rmse else None,"blind_rmse_std_ps":float(np.std(blind_rmse,ddof=1)) if len(blind_rmse)>1 else 0.0,"paired_ctr_improvement_mean_ps":float(np.mean(ctr_improvements)) if ctr_improvements else None,"paired_ctr_improvement_std_ps":float(np.std(ctr_improvements,ddof=1)) if len(ctr_improvements)>1 else 0.0,"paired_rmse_improvement_mean_ps":float(np.mean(rmse_improvements)) if rmse_improvements else None,"paired_rmse_improvement_std_ps":float(np.std(rmse_improvements,ddof=1)) if len(rmse_improvements)>1 else 0.0,"replica_count":len(blind_ctr),"blind_rmse_ctr_pearson_r":correlation if np.isfinite(correlation) else None,"blind_rmse_ctr_correlation_n":n_correlation,"result_plots":{key:(str(value) if value is not None else None) for key,value in plots.items()}});store.write_manifest(manifest)
    logger.info("Study complete | %s | blind CTR mean=%.3f ± %.3f ps across %d replicas | paired LED improvement mean=%.3f ± %.3f ps",run_dir,manifest["blind_ctr_mean_ps"],manifest["blind_ctr_std_ps"],manifest["replica_count"],manifest["paired_ctr_improvement_mean_ps"],manifest["paired_ctr_improvement_std_ps"]);return run_dir
