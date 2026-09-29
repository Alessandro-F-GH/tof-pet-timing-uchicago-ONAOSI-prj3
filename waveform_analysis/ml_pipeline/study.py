from __future__ import annotations
import json,logging
from pathlib import Path
import numpy as np
from .common import canonical_hash
from .models import get_model
from .search import candidate_manifest,CandidateScore,choose_best
from .splits import make_resampling_split,semantic_seed
from .storage import RunStore
from .train import fit_on_indices,predict_indices,save_model
from .stats import ctr_estimate
from .view import model_target
from .control_preprocessing import fit_control_artifact
from .event_selection import apply_selection_rules
from .data import preprocess_selected
from .prepared_data import prepare_ml_dataset
from .hyperparameter_plot import plot_hyperparameter_validation

def _logger(run_dir):
    logger=logging.getLogger(f"waveform-study:{run_dir}");logger.setLevel(logging.INFO);logger.handlers.clear();logger.propagate=False
    fmt=logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    for h in (logging.StreamHandler(),logging.FileHandler(Path(run_dir)/"study.log",encoding="utf-8")):h.setFormatter(fmt);logger.addHandler(h)
    return logger

def _metric(values,fit_cfg,seed,bootstrap=True):
    v=np.asarray(values,float);finite=v[np.isfinite(v)]
    if finite.size!=v.size or not finite.size:raise RuntimeError("CTR requires one finite residual per evaluated event")
    r=ctr_estimate(finite,fit_cfg,seed=int(seed),bootstrap=bootstrap)
    return float(r.ctr_ps),float(r.ctr_error_ps),int(r.n_valid)

def _row(seed,stage,cid,selected,corrected,uncorrected,fit_cfg):
    ctr,err,n=_metric(corrected,fit_cfg,semantic_seed(seed,stage,cid),bootstrap=(stage=="blind"))
    rawctr,_,_=_metric(uncorrected,fit_cfg,semantic_seed(seed,"uncorrected"),bootstrap=False)
    rmse=float(np.sqrt(np.mean(np.square(np.asarray(corrected,float)))))
    return {"seed":int(seed),"stage":stage,"candidate_id":cid,"selected":bool(selected),"ctr_ps":ctr,
        "ctr_uncertainty_ps":err if stage=="blind" else float("nan"),"uncorrected_ctr_ps":rawctr,"n":n,"rmse_ps":rmse}

def run_study(config,*,overwrite=False,resume=False,rebuild_preprocessing=False):
    run_dir=Path(config["output_dir"]).resolve();store=RunStore(run_dir,overwrite=overwrite,resume=resume);logger=_logger(run_dir)
    spec=get_model(config["model"]["name"]);candidates=candidate_manifest(list(spec.candidates(config["model"]["space"])))
    manifest={"schema_version":30,"status":"running","config_fingerprint":config["_config_fingerprint"],"reference":config["reference"],
        "analysis":config["analysis"],"mode":config["mode"],"model":config["model"]["name"],"window_ns":config["window_ns"],
        "resampling":config["resampling"],"preprocessing":config["preprocessing"],"preprocessing_fingerprint":canonical_hash(config["preprocessing"])}
    if resume and (run_dir/"manifest.json").is_file():
        old=json.loads((run_dir/"manifest.json").read_text())
        if old.get("config_fingerprint")!=config["_config_fingerprint"]:raise RuntimeError("Cannot resume with a different resolved configuration")
    store.write_manifest(manifest);store.write_candidates(candidates)
    logger.info("Study | reference=%s | analysis=%s | mode=%s | model=%s | window=%s | seeds=%d | candidates=%d",
        config["reference"]["root_file"],config["analysis"]["root_file"],config["mode"],config["model"]["name"],config["window_ns"],
        len(config["resampling"]["seeds"]),len(candidates))
    control,control_dir=fit_control_artifact(config["reference"]["root_file"],config["reference"],config["preprocessing"],config["fit"],
        cache_root=config["preprocessing"]["cache_dir"],rebuild=rebuild_preprocessing,logger=logger)
    selection=apply_selection_rules(config["analysis"]["root_file"],config["analysis"],config["preprocessing"],control["selection_rules"],config["mode"],
        cache_dir=Path(config["preprocessing"]["cache_dir"])/"analysis_selection",rebuild=rebuild_preprocessing,logger=logger)
    native=preprocess_selected(config["analysis"]["root_file"],selection,config["analysis"],config["preprocessing"],config["mode"],
        cache_dir=Path(config["preprocessing"]["cache_dir"])/"analysis_native",rebuild=rebuild_preprocessing,logger=logger)
    dataset=prepare_ml_dataset(native,control,config,cache_dir=config["preprocessing"]["cache_dir"],rebuild=rebuild_preprocessing,logger=logger)
    manifest.update({"control_artifact":str(control_dir),"control_fingerprint":control["fingerprint"],
        "fixed_led_threshold_mV":control["selected_led_threshold_mV"][config["mode"]],
        "analysis_population_identity":dataset.manifest["analysis_population_identity"],"prepared_dataset":str(dataset.directory)})
    store.write_manifest(manifest)
    target=model_target(dataset,config["mode"])
    for pos,seed in enumerate(config["resampling"]["seeds"],1):
        split=make_resampling_split(dataset.n_events,analysis_identity=dataset.manifest["analysis_population_identity"],resampling_seed=int(seed),
            validation_fraction=config["resampling"]["validation_fraction"],test_fraction=config["resampling"]["test_fraction"])
        minimum=int(config["resampling"]["minimum_events_per_split"])
        if min(len(split.train),len(split.validation),len(split.test))<minimum:
            raise RuntimeError(f"Resampling seed {seed} violates minimum_events_per_split={minimum}")
        store.save_split(seed,dataset.event_index,split)
        logger.info("Resampling %d/%d | seed=%s | train=%d validation=%d test=%d",pos,len(config["resampling"]["seeds"]),seed,len(split.train),len(split.validation),len(split.test))
        if len(candidates)==1:
            cid,params=next(iter(candidates.items()))
            if not store.has_result(seed,"blind",cid):
                fit_idx=np.concatenate([split.train,split.validation]);model_seed=semantic_seed(seed,config["model"]["name"],cid,"final")
                fitted=fit_on_indices(spec,config["model"]["space"],config,dataset,fit_idx,params,seed=model_seed,logger=logger)
                pred=predict_indices(spec,fitted,dataset,config["mode"],split.test)
                row=_row(seed,"blind",cid,True,target[split.test]-pred,target[split.test],config["fit"]);store.upsert_result(row)
                save_model(spec,fitted,store.model_dir(seed,cid),params)
                logger.info("Blind | seed=%s | candidate=%s | CTR=%.3f ± %.3f ps",seed,cid,row["ctr_ps"],row["ctr_uncertainty_ps"])
            continue
        scores=[]
        for i,(cid,params) in enumerate(candidates.items(),1):
            existing=next((r for r in store.read_results() if str(r.get("seed"))==str(seed) and r.get("stage")=="validation" and r.get("candidate_id")==cid),None)
            if existing:scores.append(CandidateScore(cid,float(existing["ctr_ps"])));continue
            logger.info("Candidate %d/%d | %s",i,len(candidates),cid)
            fitted=fit_on_indices(spec,config["model"]["space"],config,dataset,split.train,params,
                seed=semantic_seed(seed,config["model"]["name"],cid,"candidate"),logger=logger)
            pred=predict_indices(spec,fitted,dataset,config["mode"],split.validation)
            row=_row(seed,"validation",cid,False,target[split.validation]-pred,target[split.validation],config["fit"]);store.upsert_result(row)
            scores.append(CandidateScore(cid,float(row["ctr_ps"])));logger.info("Validation | candidate=%s | CTR=%.3f ps",cid,row["ctr_ps"])
        best=choose_best(scores);cid=best.candidate_id;params=candidates[cid];logger.info("Selected candidate | seed=%s | %s",seed,cid)
        rows=store.read_results()
        for r in rows:
            if str(r.get("seed"))==str(seed) and r.get("stage")=="validation" and r.get("candidate_id")==cid:r["selected"]=True
        store._atomic_rows(rows)
        if not store.has_result(seed,"blind",cid):
            fit_idx=np.concatenate([split.train,split.validation]);logger.info("Final refit | candidate=%s | n=%d",cid,len(fit_idx))
            fitted=fit_on_indices(spec,config["model"]["space"],config,dataset,fit_idx,params,
                seed=semantic_seed(seed,config["model"]["name"],cid,"final"),logger=logger)
            pred=predict_indices(spec,fitted,dataset,config["mode"],split.test)
            row=_row(seed,"blind",cid,True,target[split.test]-pred,target[split.test],config["fit"]);store.upsert_result(row)
            save_model(spec,fitted,store.model_dir(seed,cid),params)
            logger.info("Blind | seed=%s | candidate=%s | CTR=%.3f ± %.3f ps",seed,cid,row["ctr_ps"],row["ctr_uncertainty_ps"])
    if len(candidates)>1:plot_hyperparameter_validation(store.read_results(),candidates,run_dir/"hyperparameter_validation.png",logger)
    blind=[float(r["ctr_ps"]) for r in store.read_results() if r.get("stage")=="blind"]
    manifest["status"]="complete";manifest["blind_ctr_mean_ps"]=float(np.mean(blind)) if blind else None;manifest["blind_ctr_std_ps"]=float(np.std(blind,ddof=1)) if len(blind)>1 else 0.0
    store.write_manifest(manifest);logger.info("Study complete | %s | blind CTR mean=%.3f ps",run_dir,manifest["blind_ctr_mean_ps"])
    return run_dir
