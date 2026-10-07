from __future__ import annotations
import json,logging,pickle,shutil
from pathlib import Path
import numpy as np
from .common import canonical_hash
from .feature_cache import prepare_frozen_features,prepare_frozen_transform,uses_frozen_model_input
from .models import get_model
from .preprocessing import fit_control,prepare_role_dataset,publish_preprocessing_diagnostics
from .search import candidate_id,candidate_manifest,fixed_parameters,grid_candidates,optimization_config,suggest_parameters
from .shared_artifacts import ExperimentArtifactStore
from .splits import semantic_seed
from .stats import blind_event_bootstrap,metric_values,paired_central_metrics
from .storage import RUN_SCHEMA_VERSION,RunStore
from .train import FeatureTransformCache,FitInputCache,fit_on_indices,load_fitted_model,predict_indices,release_training_memory,save_model,saved_model_complete
from .validation import best_complete_candidate,evaluate_candidate
from .view import model_target
from .xai import temporal_occlusion_importance

def _logger(run_dir):
    logger=logging.getLogger(f"waveform-study:{run_dir}");logger.setLevel(logging.INFO);logger.handlers.clear();logger.propagate=False;fmt=logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    for h in (logging.StreamHandler(),logging.FileHandler(Path(run_dir)/"study.log",encoding="utf-8")):h.setFormatter(fmt);logger.addHandler(h)
    return logger
def _bool(v):return str(v).lower() in {"true","1","yes"} if isinstance(v,str) else bool(v)
def _stage(stage,payload):return canonical_hash({"schema_version":RUN_SCHEMA_VERSION,"stage":stage,"payload":payload})
def _sync(store,stage,fp):
    status=store.stage_status(stage,fp)
    if status=="stale":store.invalidate_from(stage);return "missing"
    return status
def _candidate_row(summary):
    row=dict(summary);row["parameters"]=json.dumps(row.get("parameters",{}),sort_keys=True,separators=(",",":"));return row
def _n_complete(rows):return sum(not _bool(r.get("pruned",False)) and int(r.get("completed_folds",0))==int(r.get("total_folds",-1)) for r in rows)

def _model_label(name):
    return str(name).replace("_"," ").title()

def _optimized_parameter_names(space):
    return [str(name) for name,spec in (space.get("parameters") or {}).items() if isinstance(spec,dict) and str(spec.get("type","")).lower()!="fixed"]

def _format_value(value):
    if isinstance(value,float):
        if value!=0.0 and (abs(value)<1e-3 or abs(value)>=1e4):return f"{value:.3g}"
        return f"{value:g}"
    if isinstance(value,(list,tuple)):return "["+", ".join(_format_value(v) for v in value)+"]"
    return str(value)

def _format_optimized_params(space,params):
    names=_optimized_parameter_names(space)
    if not names:return "fixed parameters"
    return ", ".join(f"{name}={_format_value(params[name])}" for name in names if name in params)

def _candidate_log(logger,label,space,summary):
    params=_format_optimized_params(space,summary.get("parameters") or {})
    folds=f"{int(summary['completed_folds'])}/{int(summary['total_folds'])}"
    metrics=f"CTR={float(summary['ctr_mean_ps']):.1f} ps | RMSE={float(summary['rmse_mean_ps']):.1f} ps"
    if not _bool(summary.get("pruned",False)):
        logger.info("%s | %s | %s",label,params,metrics)
        return
    reason=str(summary.get("pruning_reason") or "criterion")
    tolerance=float(summary.get("pruning_tolerance_ps") or 0.0)
    if reason=="led":
        degradation=float(summary.get("candidate_vs_led_degradation_ps") or 0.0)
        detail=f"LED +{degradation:.1f} ps > {tolerance:.1f} ps"
    else:
        degradation=float(summary.get("candidate_vs_incumbent_degradation_ps") or 0.0)
        detail=f"incumbent +{degradation:.1f} ps > {tolerance:.1f} ps"
    logger.info("%s | %s | PRUNED after %s folds | %s | %s",label,params,folds,detail,metrics)

def _fold_evaluator(spec,space,config,development,shared,candidate,logger,frozen_development=None):
    target=np.asarray(model_target(development,config["mode"]),dtype=np.float64);chunk=int(config["runtime"]["prediction_chunk_size"])
    def evaluate(parameters,fold):
        fold_id=int(fold.fold_id);seed=semantic_seed(config["seed"],config["mode"],config.get("window_name"),spec.name,candidate,fold_id,"fit");input_cache=FitInputCache();transform_cache=FeatureTransformCache();fitted=prediction=corrected=None
        try:
            fitted=fit_on_indices(spec,space,config,development,fold.train,parameters,seed=seed,transform_seed_base=semantic_seed(seed,"transform"),feature_transform_cache=transform_cache,fit_input_cache=input_cache,logger=logger,frozen_features=frozen_development);prediction=predict_indices(spec,fitted,development,config["mode"],fold.validation,chunk_size=chunk,frozen_features=frozen_development);corrected=target[np.asarray(fold.validation,dtype=np.int64)]-prediction;score=metric_values(corrected,config["fit"],seed=semantic_seed(config["seed"],candidate,fold_id,"metric"));led=shared.fold_metrics[fold_id]
            return {"seed":int(seed),"n_train":int(len(fold.train)),"n_validation":int(len(fold.validation)),"ctr_ps":score["ctr_ps"],"rmse_ps":score["rmse_ps"],"led_ctr_ps":led["led_ctr_ps"],"led_rmse_ps":led["led_rmse_ps"]}
        finally:
            input_cache.clear();transform_cache.clear();del corrected,prediction,fitted;release_training_memory()
    return evaluate

def _evaluate_candidate(store,spec,space,config,development,shared,identifier,params,logger,*,log_label,frozen_development=None):
    summary=store.candidate_row(identifier);folds=store.read_fold_rows(identifier)
    if summary is not None and (_bool(summary.get("pruned",False)) or int(summary.get("completed_folds",0))==int(summary.get("total_folds",-1))):return summary
    rows=store.read_candidate_rows();startup=int(config["cross_validation"]["pruning"]["startup_complete_candidates"]);force=_n_complete(rows)<startup;inc=None if force else best_complete_candidate(rows,config["cross_validation"]["metric"]);inc_id=None if inc is None else str(inc["candidate_id"]);inc_folds=None if inc_id is None else store.read_fold_rows(inc_id)
    summary,_=evaluate_candidate(candidate_id=identifier,parameters=params,folds=shared.split.folds,evaluate_fold=_fold_evaluator(spec,space,config,development,shared,identifier,logger,frozen_development),metric=config["cross_validation"]["metric"],pruning=config["cross_validation"]["pruning"],existing_fold_rows=folds,persisted_candidate=summary,incumbent_candidate_id=inc_id,incumbent_rows=inc_folds,force_complete=force,persist_fold=store.upsert_fold);store.upsert_candidate(_candidate_row(summary));_candidate_log(logger,log_label,space,summary);return summary

def _fixed_grid(store,spec,space,config,development,shared,logger,frozen_development=None):
    opt=optimization_config(space);raw=[fixed_parameters(space)] if opt.strategy=="fixed" else grid_candidates(space);candidates=candidate_manifest(raw);store.write_candidates({i:{"parameters":p} for i,p in candidates.items()})
    total=len(candidates)
    for index,(i,p) in enumerate(candidates.items(),1):
        label="Fixed configuration" if opt.strategy=="fixed" else f"Candidate {index}/{total}"
        _evaluate_candidate(store,spec,space,config,development,shared,i,p,logger,log_label=label,frozen_development=frozen_development)
    return candidates
def _sampler(path,seed,startup):
    if path.is_file():
        with path.open("rb") as stream:return pickle.load(stream)
    import optuna
    return optuna.samplers.TPESampler(seed=int(seed),n_startup_trials=int(startup))
def _save_sampler(path,sampler):
    tmp=path.with_name(f".{path.name}.tmp")
    with tmp.open("wb") as stream:pickle.dump(sampler,stream,pickle.HIGHEST_PROTOCOL);stream.flush()
    tmp.replace(path)
def _optuna(store,spec,space,config,development,shared,logger,frozen_development=None):
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING);opt=optimization_config(space);sampler_path=store.root/"optuna_sampler.pkl";sampler=_sampler(sampler_path,semantic_seed(config["seed"],config["mode"],config.get("window_name"),spec.name,"optuna"),opt.n_startup_trials);study=optuna.create_study(study_name="development_cv",direction="minimize",sampler=sampler,storage=f"sqlite:///{(store.root/'optuna.db').as_posix()}",load_if_exists=True);registry=store.read_json(store.candidates_path,{}) or {}
    def finish(number,params,identifier):
        summary=_evaluate_candidate(store,spec,space,config,development,shared,identifier,params,logger,log_label=f"Trial {int(number)+1}/{int(opt.n_trials)}",frozen_development=frozen_development)
        if _bool(summary.get("pruned",False)):study.tell(int(number),state=optuna.trial.TrialState.PRUNED)
        else:study.tell(int(number),float(summary[f"{config['cross_validation']['metric']}_mean_ps"]))
        _save_sampler(sampler_path,sampler)
    for frozen in list(study.trials):
        if frozen.state!=optuna.trial.TrialState.RUNNING:continue
        params=frozen.user_attrs.get("resolved_parameters");identifier=frozen.user_attrs.get("candidate_id")
        if not isinstance(params,dict) or not identifier:raise RuntimeError("Cannot deterministically resume Optuna trial")
        registry[str(identifier)]={"parameters":params,"trial_number":int(frozen.number)};store.write_candidates(registry);finish(frozen.number,params,str(identifier))
    while len(study.trials)<int(opt.n_trials):
        trial=study.ask();params=suggest_parameters(trial,space);identifier=candidate_id(params);trial.set_user_attr("candidate_id",identifier);trial.set_user_attr("resolved_parameters",params);registry[identifier]={"parameters":params,"trial_number":int(trial.number)};store.write_candidates(registry);_save_sampler(sampler_path,sampler);finish(trial.number,params,identifier)
    return {i:dict(v["parameters"]) for i,v in registry.items()}
def _search(store,spec,space,config,development,shared,logger,frozen_development=None):
    opt=optimization_config(space);optimized=_optimized_parameter_names(space);target=", ".join(optimized) if optimized else "none"
    if opt.strategy=="optuna":search=f"Optuna/TPE, {int(opt.n_trials)} trials"
    elif opt.strategy=="grid":search=f"grid, {len(grid_candidates(space))} candidates"
    else:search="fixed configuration"
    logger.info("CV search | %s | optimize=%s | %s | %d folds",_model_label(spec.name),target,search,int(config["cross_validation"]["folds"]))
    return _fixed_grid(store,spec,space,config,development,shared,logger,frozen_development) if opt.strategy in {"fixed","grid"} else _optuna(store,spec,space,config,development,shared,logger,frozen_development)
def _select(store,candidates,config):
    best=best_complete_candidate(store.read_candidate_rows(),config["cross_validation"]["metric"])
    if best is None:raise RuntimeError("No fully evaluated CV candidate is eligible")
    i=str(best["candidate_id"]);out={"candidate_id":i,"parameters":dict(candidates[i]),"selection_metric":config["cross_validation"]["metric"],"folds":int(config["cross_validation"]["folds"]),"validation_ctr_mean_ps":float(best["ctr_mean_ps"]),"validation_ctr_std_ps":float(best["ctr_std_ps"]),"validation_rmse_mean_ps":float(best["rmse_mean_ps"]),"validation_rmse_std_ps":float(best["rmse_std_ps"]),"selected_from":"development_cv_only"};store.write_best(out);return out
def _fit_final(spec,space,config,development,best,logger,frozen_development=None):
    seed=semantic_seed(config["seed"],config["mode"],config.get("window_name"),spec.name,best["candidate_id"],"final_fit");fitted=fit_on_indices(spec,space,config,development,np.arange(development.n_events,dtype=np.int64),best["parameters"],seed=seed,transform_seed_base=semantic_seed(seed,"transform"),logger=logger,frozen_features=frozen_development);return fitted,seed
def _model(store,spec,space,config,development,best,fp,logger,frozen_development=None):
    status=_sync(store,"final_fit",fp)
    if status=="complete" and config["save_model"]:
        if saved_model_complete(spec,store.model_dir):
            return load_fitted_model(spec,store.model_dir,best["parameters"],config)
        logger.warning("Saved final model is incomplete or incompatible | retraining on full development set")
        store.invalidate_from("final_fit")
    fitted,seed=_fit_final(spec,space,config,development,best,logger,frozen_development)
    if config["save_model"]:
        if store.model_dir.exists():shutil.rmtree(store.model_dir)
        save_model(spec,fitted,store.model_dir,best["parameters"])
    store.write_final_fit({"candidate_id":best["candidate_id"],"parameters":best["parameters"],"training_dataset_role":"development","n_train":int(development.n_events),"seed":int(seed),"model_saved":bool(config["save_model"])});store.mark_stage("final_fit",fp,metadata={"model_saved":bool(config["save_model"])});return fitted

def run_study(config,*,logger=None):
    run_dir=Path(config["output_dir"]).resolve();store=RunStore(run_dir);logger=logger or _logger(run_dir);spec=get_model(config["model"]["name"]);space=config["model"]["space"];store.write_resolved_config({k:v for k,v in config.items() if not str(k).startswith("_")})
    if store.manifest_path.is_file() and not store.compatible_schema():raise RuntimeError("Incompatible result schema must be rebuilt by batch planner")
    control,control_dir=fit_control(config,rebuild=False,logger=logger);_,_,development=prepare_role_dataset(config,"development",control,rebuild=False,logger=logger)
    publish_preprocessing_diagnostics(config,"control",control,config["batch_output_dir"],logger=logger)
    publish_preprocessing_diagnostics(config,"development",control,config["batch_output_dir"],logger=logger)
    frozen_transform=None;frozen_development=None
    if uses_frozen_model_input(spec):
        _,_,control_prepared=prepare_role_dataset(config,"control",control,rebuild=False,logger=None)
        frozen_transform=prepare_frozen_transform(spec,space,config,control_dataset=control_prepared,development_dataset=development,cache_root=config["preprocessing"]["cache_dir"],logger=logger)
        frozen_development=prepare_frozen_features(spec,frozen_transform,config,development,role="development",cache_root=config["preprocessing"]["cache_dir"],logger=logger)
    shared=ExperimentArtifactStore(Path(config["batch_output_dir"])/"artifacts").prepare_cv(development,config)
    cv_fp=_stage("cv",{"development":development.manifest["analysis_protocol_identity"],"shared_cv":shared.fingerprint,"model":config["model"],"cross_validation":config["cross_validation"],"fit":config["fit"],"seed":config["seed"],"frozen_transform":None if frozen_transform is None else frozen_transform.identity})
    if _sync(store,"cv",cv_fp)!="complete":candidates=_search(store,spec,space,config,development,shared,logger,frozen_development);store.mark_stage("cv",cv_fp,metadata={"candidates":len(store.read_candidate_rows())})
    else:candidates={i:dict(v.get("parameters",v)) for i,v in (store.read_json(store.candidates_path,{}) or {}).items()}
    selection_fp=_stage("selection",{"cv":cv_fp,"metric":config["cross_validation"]["metric"]});best=store.read_json(store.best_path) if _sync(store,"selection",selection_fp)=="complete" else None
    if not isinstance(best,dict):
        best=_select(store,candidates,config);store.mark_stage("selection",selection_fp,metadata={"candidate_id":best["candidate_id"]});logger.info("CV selected | %s | %s | CTR=%.1f ± %.1f ps | RMSE=%.1f ± %.1f ps",_model_label(spec.name),_format_optimized_params(space,best["parameters"]),float(best["validation_ctr_mean_ps"]),float(best["validation_ctr_std_ps"]),float(best["validation_rmse_mean_ps"]),float(best["validation_rmse_std_ps"]))
    final_fp=_stage("final_fit",{"development":development.manifest["analysis_protocol_identity"],"model":config["model"],"candidate_id":best["candidate_id"],"parameters":best["parameters"],"seed":config["seed"],"frozen_transform":None if frozen_transform is None else frozen_transform.identity});fitted=None
    if _sync(store,"final_fit",final_fp)!="complete":
        logger.info("Final fit | %s | training on full development set | n=%d",_model_label(spec.name),int(development.n_events))
        fitted=_model(store,spec,space,config,development,best,final_fp,logger,frozen_development)
        logger.info("Final fit complete | %s",_model_label(spec.name))
    else:
        logger.info("Final fit | reusing saved model")
    logger.info("Blind preparation | applying frozen preprocessing")
    _,_,blind=prepare_role_dataset(config,"blind",control,rebuild=False,logger=logger)
    publish_preprocessing_diagnostics(config,"blind",control,config["batch_output_dir"],logger=logger)
    frozen_blind=prepare_frozen_features(spec,frozen_transform,config,blind,role="blind",cache_root=config["preprocessing"]["cache_dir"],logger=logger) if frozen_transform is not None else None
    blind_fp=_stage("blind",{"final_fit":final_fp,"blind":blind.manifest["analysis_protocol_identity"],"fit":config["fit"],"frozen_transform":None if frozen_transform is None else frozen_transform.identity})
    if _sync(store,"blind",blind_fp)!="complete" or not store.predictions_path.is_file():
        logger.info("Blind evaluation | %s | n=%d",_model_label(spec.name),int(blind.n_events))
        if fitted is None:fitted=_model(store,spec,space,config,development,best,final_fp,logger,frozen_development)
        indices=np.arange(blind.n_events,dtype=np.int64);prediction=predict_indices(spec,fitted,blind,config["mode"],indices,chunk_size=int(config["runtime"]["prediction_chunk_size"]),frozen_features=frozen_blind);led=np.asarray(model_target(blind,config["mode"]),dtype=np.float64);corrected=led-prediction;central=paired_central_metrics(corrected,led,config["fit"],seed=semantic_seed(config["seed"],spec.name,"blind_central"));store.save_predictions(event_id=np.asarray(blind.event_index,dtype=np.int64),prediction_ps=prediction,corrected_ps=corrected,led_residual_ps=led);store.write_blind({**central,"dataset_role":"blind","dataset_source":blind.manifest["dataset_source"],"event_population_identity":blind.manifest["event_population_identity"],"candidate_id":best["candidate_id"],"parameters":best["parameters"],"evaluation_count":1,"central_value_source":"original_non_resampled_blind_distribution"});store.mark_stage("blind",blind_fp,metadata={"n_events":int(blind.n_events)})
        logger.info("Blind result | CTR=%.1f ps | RMSE=%.1f ps | LED CTR=%.1f ps | ΔCTR=%.1f ps",float(central["ctr_ps"]),float(central["rmse_ps"]),float(central["led_ctr_ps"]),float(central["ctr_improvement_ps"]))
    else:
        logger.info("Blind evaluation | reusing saved predictions")
    pred=store.load_predictions();bootstrap_fp=_stage("bootstrap",{"blind":blind_fp,"bootstrap":config["bootstrap"],"seed":config["seed"]})
    if _sync(store,"bootstrap",bootstrap_fp)!="complete" or not store.bootstrap_path.is_file():
        logger.info("Blind bootstrap | %d event resamples | no retraining",int(config["bootstrap"]["n_resamples"]))
        summary,draws=blind_event_bootstrap(pred["corrected_ps"],pred["led_residual_ps"],config["fit"],n_resamples=int(config["bootstrap"]["n_resamples"]),seed=semantic_seed(config["seed"],spec.name,"bootstrap"));store.write_bootstrap(summary);store.save_bootstrap_draws(draws);store.mark_stage("bootstrap",bootstrap_fp)
        logger.info("Blind uncertainty | CTR ± %.1f ps | RMSE ± %.1f ps",float(summary["ctr_bootstrap_std_ps"]),float(summary["rmse_bootstrap_std_ps"]))
    else:
        logger.info("Blind bootstrap | reusing saved uncertainty")
    xai_fp=_stage("xai",{"final_fit":final_fp,"blind":blind.manifest["analysis_protocol_identity"],"xai":config["xai"],"seed":config["seed"]})
    if config["xai"]["enabled"] and (_sync(store,"xai",xai_fp)!="complete" or not store.xai_path.is_file()):
        logger.info("XAI | grouped temporal occlusion | max_events=%d | group=%d samples",int(config["xai"]["max_events"]),int(config["xai"]["group_size_samples"]))
        if fitted is None:fitted=_model(store,spec,space,config,development,best,final_fp,logger)
        store.save_xai(**temporal_occlusion_importance(spec,fitted,blind,config["mode"],group_size_samples=int(config["xai"]["group_size_samples"]),max_events=int(config["xai"]["max_events"]),seed=semantic_seed(config["seed"],spec.name,"xai")));store.mark_stage("xai",xai_fp,metadata={"method":"grouped_temporal_occlusion"})
        logger.info("XAI complete")
    elif not config["xai"]["enabled"]:
        store.mark_stage("xai",xai_fp,metadata={"enabled":False})
    else:
        logger.info("XAI | reusing saved importance")
    manifest={"schema_version":RUN_SCHEMA_VERSION,"status":"complete","name":config["name"],"study_name":config.get("study_name"),"run_id":config.get("run_id"),"mode":config["mode"],"window_name":config.get("window_name"),"window_ns":config["window_ns"],"model":spec.name,"estimator_formulation":spec.estimator_formulation,"control_dataset":config["control"],"development_dataset":config["development"],"blind_dataset":config["blind"],"control_artifact":str(Path(control_dir).resolve()),"preprocessing_fit_role":"control","model_selection_role":"development","blind_role":"final_one_time_evaluation_only","blind_used_in_selection":False,"cv":config["cross_validation"],"selected_candidate_id":best["candidate_id"],"selected_hyperparameters":best["parameters"],"validation_ctr_mean_ps":best["validation_ctr_mean_ps"],"validation_ctr_std_ps":best["validation_ctr_std_ps"],"validation_rmse_mean_ps":best["validation_rmse_mean_ps"],"validation_rmse_std_ps":best["validation_rmse_std_ps"],"validation_std_interpretation":"fold-to-fold development validation variability","blind_bootstrap_interpretation":"event-level uncertainty conditional on final fitted model; no retraining","bootstrap_unit":"blind_event","single_configured_seed":int(config["seed"]),"stage_fingerprints":{"cv":cv_fp,"selection":selection_fp,"final_fit":final_fp,"blind":blind_fp,"bootstrap":bootstrap_fp,"xai":xai_fp}}
    store.write_manifest(manifest);release_training_memory();logger.info("Study complete | %s | %s | %s",_model_label(spec.name),config["mode"],config.get("window_name"));return run_dir
