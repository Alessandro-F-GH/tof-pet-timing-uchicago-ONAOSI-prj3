from __future__ import annotations
import json,logging
from pathlib import Path
import numpy as np
from .common import canonical_hash
from .models import get_model
from .search import candidate_manifest,CandidateScore,choose_best
from .splits import make_resampling_split,semantic_seed
from .storage import RunStore
from .train import FeatureTransformCache,detector_swap_rmse,fit_on_indices,predict_indices,save_model
from .stats import ctr_estimate,paired_ctr_improvement
from .view import model_target
from .control_preprocessing import fit_control_artifact
from .event_selection import apply_selection_rules
from .data import preprocess_selected
from .prepared_data import prepare_ml_dataset
from .hyperparameter_plot import plot_hyperparameter_validation
from .result_plots import make_study_result_plots
from .progress import ProgressTracker

def _logger(run_dir):
    logger=logging.getLogger(f"waveform-study:{run_dir}");logger.setLevel(logging.INFO);logger.handlers.clear();logger.propagate=False
    fmt=logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    for h in (logging.StreamHandler(),logging.FileHandler(Path(run_dir)/"study.log",encoding="utf-8")):h.setFormatter(fmt);logger.addHandler(h)
    return logger

def _resampling_seeds(resampling):
    base_seed=int(resampling["seed"]);n=int(resampling["n_bootstrap"])
    return [semantic_seed(base_seed,"resampling",i) for i in range(n)]

def _metric(values,fit_cfg,seed,bootstrap=False):
    v=np.asarray(values,float);finite=v[np.isfinite(v)]
    if finite.size!=v.size or not finite.size:raise RuntimeError("CTR requires one finite residual per evaluated event")
    r=ctr_estimate(finite,fit_cfg,seed=int(seed),bootstrap=bootstrap)
    return float(r.ctr_ps),float(r.ctr_error_ps),int(r.n_valid)

def _row(seed,stage,cid,selected,corrected,uncorrected,fit_cfg,*,spec,config,population_identity,swap_rmse_ps=float("nan"),paired=None):
    if stage=="blind" and paired is not None:
        ctr=float(paired.corrected_ctr_ps);err=float(paired.corrected_ctr_error_ps);rawctr=float(paired.led_ctr_ps);rawerr=float(paired.led_ctr_error_ps);n=int(np.isfinite(corrected).sum())
        improvement=float(paired.improvement_ps);improvement_err=float(paired.improvement_error_ps);improvement_percent=float(paired.improvement_percent)
        ci_low=float(paired.improvement_ci_low_ps);ci_high=float(paired.improvement_ci_high_ps);paired_success=int(paired.bootstrap_successful)
    else:
        ctr,err,n=_metric(corrected,fit_cfg,semantic_seed(seed,stage,cid),bootstrap=False)
        rawctr,rawerr,_=_metric(uncorrected,fit_cfg,semantic_seed(seed,"uncorrected"),bootstrap=False)
        improvement=improvement_err=improvement_percent=ci_low=ci_high=float("nan");paired_success=0
    rmse=float(np.sqrt(np.mean(np.square(np.asarray(corrected,float)))))
    return {"seed":int(seed),"stage":stage,"model":spec.name,"estimator_formulation":spec.estimator_formulation,
        "mode":config["mode"],"window_start_ns":float(config["window_ns"]["start"]),"window_end_ns":float(config["window_ns"]["end"]),
        "population_identity":population_identity,"candidate_id":cid,"selected":bool(selected),"ctr_ps":ctr,
        "ctr_uncertainty_ps":err if stage=="blind" else float("nan"),"uncorrected_ctr_ps":rawctr,
        "uncorrected_ctr_uncertainty_ps":rawerr if stage=="blind" else float("nan"),"improvement_ps":improvement,
        "improvement_uncertainty_ps":improvement_err,"improvement_percent":improvement_percent,
        "improvement_ci_low_ps":ci_low,"improvement_ci_high_ps":ci_high,"paired_bootstrap_successful":paired_success,
        "n":n,"rmse_ps":rmse,"swap_rmse_ps":float(swap_rmse_ps) if stage=="blind" else float("nan")}

def _blind_evaluation(store,spec,fitted,dataset,config,split,target,seed,cid,population_identity):
    pred=predict_indices(spec,fitted,dataset,config["mode"],split.test)
    corrected=target[split.test]-pred;led=target[split.test]
    swap=detector_swap_rmse(spec,fitted,dataset,config["mode"],split.test)
    paired=paired_ctr_improvement(corrected,led,config["fit"],seed=semantic_seed(seed,"paired_led_ml",cid))
    store.save_paired_bootstrap(seed,cid,paired)
    return _row(seed,"blind",cid,True,corrected,led,config["fit"],spec=spec,config=config,
        population_identity=population_identity,swap_rmse_ps=swap,paired=paired)

def run_study(config,*,overwrite=False,resume=False,rebuild_preprocessing=False):
    run_dir=Path(config["output_dir"]).resolve();store=RunStore(run_dir,overwrite=overwrite,resume=resume);logger=_logger(run_dir)
    spec=get_model(config["model"]["name"]);candidates=candidate_manifest(list(spec.candidates(config["model"]["space"])))
    seeds=_resampling_seeds(config["resampling"])
    manifest={"schema_version":33,"status":"running","config_fingerprint":config["_config_fingerprint"],"reference":config["reference"],
        "analysis":config["analysis"],"mode":config["mode"],"model":config["model"]["name"],"estimator_formulation":spec.estimator_formulation,"window_ns":config["window_ns"],
        "feature_transform":None if spec.feature_transform is None else spec.feature_transform.name,
        "resampling":config["resampling"],"generated_resampling_seeds":seeds,"preprocessing":config["preprocessing"],"preprocessing_fingerprint":canonical_hash(config["preprocessing"])}
    if resume and (run_dir/"manifest.json").is_file():
        old=json.loads((run_dir/"manifest.json").read_text())
        if old.get("config_fingerprint")!=config["_config_fingerprint"]:raise RuntimeError("Cannot resume with a different resolved configuration")
    store.write_manifest(manifest);store.write_candidates(candidates)
    logger.info("Study | reference=%s | analysis=%s | mode=%s | model=%s | formulation=%s | window=%s | bootstrap=%d | base_seed=%d | candidates=%d",
        config["reference"]["root_file"],config["analysis"]["root_file"],config["mode"],config["model"]["name"],spec.estimator_formulation,config["window_ns"],
        len(seeds),int(config["resampling"]["seed"]),len(candidates))
    control,control_dir=fit_control_artifact(config["reference"]["root_file"],config["reference"],config["preprocessing"],config["fit"],
        cache_root=config["preprocessing"]["cache_dir"],rebuild=rebuild_preprocessing,logger=logger)
    selection=apply_selection_rules(config["analysis"]["root_file"],config["analysis"],config["preprocessing"],control["selection_rules"],config["mode"],
        cache_dir=Path(config["preprocessing"]["cache_dir"])/"analysis_selection",rebuild=rebuild_preprocessing,logger=logger)
    native=preprocess_selected(config["analysis"]["root_file"],selection,config["analysis"],config["preprocessing"],config["mode"],
        cache_dir=Path(config["preprocessing"]["cache_dir"])/"analysis_native",rebuild=rebuild_preprocessing,logger=logger)
    dataset=prepare_ml_dataset(native,control,config,cache_dir=config["preprocessing"]["cache_dir"],rebuild=rebuild_preprocessing,logger=logger)
    population_identity=dataset.manifest["analysis_population_identity"]
    manifest.update({"control_artifact":str(control_dir),"control_fingerprint":control["fingerprint"],
        "fixed_led_threshold_mV":control["selected_led_threshold_mV"][config["mode"]],
        "analysis_population_identity":population_identity,"prepared_dataset":str(dataset.directory)})
    store.write_manifest(manifest)
    target=model_target(dataset,config["mode"])
    progress=ProgressTracker(logger,{"resampling_seed":len(seeds)})
    for pos,seed in enumerate(seeds,1):
        split=make_resampling_split(dataset.n_events,analysis_identity=population_identity,resampling_seed=int(seed),
            validation_fraction=config["resampling"]["validation_fraction"],test_fraction=config["resampling"]["test_fraction"])
        minimum=int(config["resampling"]["minimum_events_per_split"])
        if min(len(split.train),len(split.validation),len(split.test))<minimum:
            raise RuntimeError(f"Resampling seed {seed} violates minimum_events_per_split={minimum}")
        store.save_split(seed,dataset.event_index,split)
        logger.info("Resampling %d/%d | seed=%s | train=%d validation=%d test=%d",pos,len(seeds),seed,len(split.train),len(split.validation),len(split.test))
        transform_cache=FeatureTransformCache();transform_seed_base=semantic_seed(seed,config["model"]["name"],"transform")
        if len(candidates)==1:
            cid,params=next(iter(candidates.items()))
            if not store.has_result(seed,"blind",cid):
                fit_idx=np.concatenate([split.train,split.validation]);model_seed=semantic_seed(seed,config["model"]["name"],cid,"final")
                fitted=fit_on_indices(spec,config["model"]["space"],config,dataset,fit_idx,params,seed=model_seed,
                    transform_seed_base=transform_seed_base,feature_transform_cache=transform_cache,logger=logger)
                row=_blind_evaluation(store,spec,fitted,dataset,config,split,target,seed,cid,population_identity);store.upsert_result(row)
                save_model(spec,fitted,store.model_dir(seed,cid),params)
                logger.info("Blind | seed=%s | candidate=%s | CTR=%.3f ± %.3f ps | LED=%.3f ± %.3f ps | improvement=%.3f ± %.3f ps (%.2f%%) | swap RMSE=%.3f ps",
                    seed,cid,row["ctr_ps"],row["ctr_uncertainty_ps"],row["uncorrected_ctr_ps"],row["uncorrected_ctr_uncertainty_ps"],
                    row["improvement_ps"],row["improvement_uncertainty_ps"],row["improvement_percent"],row["swap_rmse_ps"])
            progress.complete("resampling_seed",f"seed {seed}",announce=False);continue
        scores=[]
        for i,(cid,params) in enumerate(candidates.items(),1):
            existing=next((r for r in store.read_results() if str(r.get("seed"))==str(seed) and r.get("stage")=="validation" and r.get("candidate_id")==cid),None)
            if existing:scores.append(CandidateScore(cid,float(existing["ctr_ps"])));continue
            logger.info("Candidate %d/%d | %s",i,len(candidates),cid)
            fitted=fit_on_indices(spec,config["model"]["space"],config,dataset,split.train,params,
                seed=semantic_seed(seed,config["model"]["name"],cid,"candidate"),transform_seed_base=transform_seed_base,
                feature_transform_cache=transform_cache,logger=logger)
            pred=predict_indices(spec,fitted,dataset,config["mode"],split.validation)
            row=_row(seed,"validation",cid,False,target[split.validation]-pred,target[split.validation],config["fit"],spec=spec,config=config,population_identity=population_identity);store.upsert_result(row)
            scores.append(CandidateScore(cid,float(row["ctr_ps"])));logger.info("Validation | candidate=%s | CTR=%.3f ps",cid,row["ctr_ps"])
        best=choose_best(scores);cid=best.candidate_id;params=candidates[cid];logger.info("Selected candidate | seed=%s | %s",seed,cid)
        rows=store.read_results()
        for r in rows:
            if str(r.get("seed"))==str(seed) and r.get("stage")=="validation" and r.get("candidate_id")==cid:r["selected"]=True
        store._atomic_rows(rows)
        if not store.has_result(seed,"blind",cid):
            fit_idx=np.concatenate([split.train,split.validation]);logger.info("Final refit | candidate=%s | n=%d",cid,len(fit_idx))
            fitted=fit_on_indices(spec,config["model"]["space"],config,dataset,fit_idx,params,
                seed=semantic_seed(seed,config["model"]["name"],cid,"final"),transform_seed_base=transform_seed_base,
                feature_transform_cache=transform_cache,logger=logger)
            row=_blind_evaluation(store,spec,fitted,dataset,config,split,target,seed,cid,population_identity);store.upsert_result(row)
            save_model(spec,fitted,store.model_dir(seed,cid),params)
            logger.info("Blind | seed=%s | candidate=%s | CTR=%.3f ± %.3f ps | LED=%.3f ± %.3f ps | improvement=%.3f ± %.3f ps (%.2f%%) | swap RMSE=%.3f ps",
                seed,cid,row["ctr_ps"],row["ctr_uncertainty_ps"],row["uncorrected_ctr_ps"],row["uncorrected_ctr_uncertainty_ps"],
                row["improvement_ps"],row["improvement_uncertainty_ps"],row["improvement_percent"],row["swap_rmse_ps"])
        progress.complete("resampling_seed",f"seed {seed}",note=f"candidate={cid}")
    rows=store.read_results()
    if len(candidates)>1:plot_hyperparameter_validation(rows,candidates,run_dir/"hyperparameter_validation.png",logger)
    plots=make_study_result_plots(rows,run_dir,model=spec.name,mode=config["mode"],window_ns=config["window_ns"])
    blind=[float(r["ctr_ps"]) for r in rows if r.get("stage")=="blind"]
    improvements=[float(r["improvement_ps"]) for r in rows if r.get("stage")=="blind" and str(r.get("improvement_ps","")).strip()]
    manifest["status"]="complete";manifest["blind_ctr_mean_ps"]=float(np.mean(blind)) if blind else None;manifest["blind_ctr_std_ps"]=float(np.std(blind,ddof=1)) if len(blind)>1 else 0.0
    manifest["paired_improvement_mean_ps"]=float(np.mean(improvements)) if improvements else None;manifest["paired_improvement_std_ps"]=float(np.std(improvements,ddof=1)) if len(improvements)>1 else 0.0
    manifest["result_plots"]={k:(str(v) if v is not None else None) for k,v in plots.items()}
    store.write_manifest(manifest);logger.info("Study complete | %s | blind CTR mean=%.3f ps | paired improvement mean=%.3f ps",run_dir,manifest["blind_ctr_mean_ps"],manifest["paired_improvement_mean_ps"])
    return run_dir
