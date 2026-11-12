"""Conditional-mean KNN TSHAP, with shared local/global inference and diagnostics."""
from __future__ import annotations

import csv
import json
import hashlib
import logging
from pathlib import Path
import tempfile
import shutil

import numpy as np

from waveform_analysis.core.io import canonical_hash, atomic_json
from waveform_analysis.data.results import resolve_run_config
from waveform_analysis.data.prepared_cache import PreparedCacheLookup
from waveform_analysis.data.view import mode_family, waveform_view
from waveform_analysis.data.storage import RunStore
from waveform_analysis.engine.train import load_fitted_model, predict_indices, release_training_memory
from waveform_analysis.models import get_model
from waveform_analysis.engine import tshap_explainer as base

METHOD_VERSION = 1
WEIGHTINGS = ("uniform", "distance")


def _weights(distances, kind):
    if kind == "uniform":
        return np.full(len(distances), 1.0 / len(distances))
    if kind != "distance":
        raise ValueError("Unknown weighting")
    zeros = distances <= 1e-12
    w = zeros.astype(np.float64) if zeros.any() else 1.0 / np.maximum(distances, 1e-12)
    return w / w.sum()


class ConditionalNeighborEngine:
    """Paired KNN using one per-reference squared-difference/prefix pass per query.

    Mask indexes 0..n-1 observe W; n..2n-1 observe the complement.
    Only model-retained samples participate in distances and imputation.
    """

    def __init__(self, reference, starts, window_samples, sample_mask, ks,
                 scale_mv, reference_batch=512):
        x = np.asarray(reference, dtype=np.float32)
        self.starts = np.asarray(starts, dtype=np.int64)
        self.ends = self.starts + int(window_samples)
        self.active = np.asarray(sample_mask, dtype=bool)
        self.ks = np.asarray(sorted(set(int(k) for k in ks)), dtype=np.int64)
        self.reference_batch = int(reference_batch)
        if (x.ndim != 3 or x.shape[1] != 2 or not np.isfinite(x).all()
                or self.active.shape != (x.shape[2],) or not self.active.any()):
            raise ValueError("Expected finite [events,2,samples] reference with valid mask")
        if (not len(self.starts) or (self.starts < 0).any()
                or (self.ends > x.shape[2]).any() or window_samples < 1
                or not all(self.active[a:b].all() for a,b in zip(self.starts,self.ends))):
            raise ValueError("Invalid temporal KNN windows")
        if not len(self.ks) or self.ks[0] < 1 or self.ks[-1] > len(x):
            raise ValueError("KNN k exceeds reference population")
        if self.reference_batch < 1:
            raise ValueError("KNN reference batch must be positive")
        self.reference = np.ascontiguousarray(x)
        self.nwin = len(self.starts)
        self.nmask = 2*self.nwin
        self.kmax = int(self.ks[-1])
        self.scale_mv = np.broadcast_to(np.asarray(scale_mv,dtype=np.float64), (2,x.shape[2])).copy()
        if not np.isfinite(self.scale_mv).all() or (self.scale_mv <= 0).any():
            raise ValueError("Invalid inverse-normalization scale")
        observed = []
        for side in range(2):
            for a,b in zip(self.starts,self.ends):
                mask = np.zeros(x.shape[2],dtype=bool)
                mask[a:b] = True
                if side: mask = ~mask
                mask &= self.active
                if not mask.any() or not (self.active & ~mask).any():
                    raise ValueError("KNN requires nonempty observed/hidden regions")
                observed.append(mask)
        self.observed = np.asarray(observed)
        self.hidden = self.active[None,:] & ~self.observed

    def neighbors(self, query):
        query = np.asarray(query,dtype=np.float32)
        if query.shape != self.reference.shape[1:] or not np.isfinite(query).all():
            raise ValueError("Invalid KNN query")
        d2 = np.full((self.nmask,self.kmax),np.inf)
        ids = np.full((self.nmask,self.kmax),-1,dtype=np.int64)
        for first in range(0,len(self.reference),self.reference_batch):
            end = min(first+self.reference_batch,len(self.reference))
            diff = self.reference[first:end].astype(np.float64) - query.astype(np.float64)
            sq = np.sum(diff*diff,axis=1)
            sq[:,~self.active] = 0
            prefix = np.pad(np.cumsum(sq,axis=1),((0,0),(1,0)))
            win = (prefix[:,self.ends]-prefix[:,self.starts]).T
            comp = np.maximum(prefix[:,-1][None,:]-win,0)
            current = np.concatenate((win,comp),axis=0)
            current_ids = np.broadcast_to(np.arange(first,end)[None,:],current.shape)
            merged = np.concatenate((d2,current),axis=1)
            merged_ids = np.concatenate((ids,current_ids),axis=1)
            # Stable ties: select by distance then reference index.
            for mask in range(self.nmask):
                order = np.lexsort((merged_ids[mask],merged[mask]))[:self.kmax]
                d2[mask] = merged[mask,order]
                ids[mask] = merged_ids[mask,order]
        if (ids < 0).any() or not np.isfinite(d2).all():
            raise ValueError("KNN search did not produce enough neighbors")
        return ids,np.sqrt(d2)

    def reconstruct(self, query, mask, ids, distances, k, kind):
        selected = np.asarray(ids[:k],dtype=np.int64)
        weights = _weights(np.asarray(distances[:k]),kind)
        missing = self.hidden[mask]
        reconstruction = np.asarray(query,dtype=np.float32).copy()
        reconstruction[:,missing] = np.tensordot(
            weights,self.reference[selected][:,:,missing],axes=(0,0)
        ).astype(np.float32)
        return reconstruction,selected,weights

    def tune(self, validation, logger=None):
        val = np.asarray(validation,dtype=np.float32)
        if val.ndim != 3 or val.shape[1:] != self.reference.shape[1:] or not len(val):
            raise ValueError("Invalid KNN validation set")
        metrics = np.zeros((4,self.nmask,2,len(self.ks)),dtype=np.float64)
        for i,query in enumerate(val,1):
            ids,d = self.neighbors(query)
            for mask in range(self.nmask):
                hidden = self.hidden[mask]
                residuals = (self.reference[ids[mask]][:,:,hidden].astype(np.float64)
                             - query[None,:,hidden].astype(np.float64)).reshape(self.kmax,-1)
                scale = self.scale_mv[:,hidden].reshape(-1)
                cum = np.cumsum(residuals,axis=0)
                zeros = d[mask] <= 1e-12
                raw_weights = zeros.astype(np.float64) if zeros.any() else 1/np.maximum(d[mask],1e-12)
                wcum = np.cumsum(residuals*raw_weights[:,None],axis=0)
                wsum = np.cumsum(raw_weights)
                for ki,k in enumerate(self.ks):
                    errors = (cum[k-1]/k,wcum[k-1]/wsum[k-1])
                    for wi,error in enumerate(errors):
                        physical = error*scale
                        metrics[:,mask,wi,ki] += (
                            np.mean(error**2),np.mean(np.abs(error)),
                            np.mean(physical**2),np.mean(np.abs(physical))
                        )
            if logger and (i==1 or i%25==0 or i==len(val)):
                logger.info("KNN TUNE | validation=%d/%d | reference=%d | masks=%d",
                            i,len(val),len(self.reference),self.nmask)
        metrics /= len(val)
        selected = np.argmin(metrics[0].reshape(self.nmask,-1),axis=1)
        chosen_w,chosen_i = np.unravel_index(selected,(2,len(self.ks)))
        chosen_k = self.ks[chosen_i]
        if logger:
            for mask in range(self.nmask):
                w,i = chosen_w[mask],chosen_i[mask]
                logger.info(
                    "KNN BEST | region=%s | window=%d | k=%d | weighting=%s | MSE=%.9g | "
                    "RMSE=%.6g | MAE=%.6g | MSE_mV2=%.9g | RMSE_mV=%.6g | MAE_mV=%.6g",
                    "W" if mask<self.nwin else "complement",mask%self.nwin,
                    chosen_k[mask],WEIGHTINGS[w],metrics[0,mask,w,i],
                    np.sqrt(metrics[0,mask,w,i]),metrics[1,mask,w,i],
                    metrics[2,mask,w,i],np.sqrt(metrics[2,mask,w,i]),metrics[3,mask,w,i]
                )
        return chosen_k,chosen_w,metrics


def _tune_cache(cache_root, identity, engine, validation, logger):
    cache = Path(cache_root)/"conditional_knn_tuning"/canonical_hash(identity)
    npz = cache/"tuning.npz"
    summary_path = cache/"summary.json"
    if npz.is_file() and summary_path.is_file():
        with np.load(npz) as data:
            chosen_k=np.asarray(data["k"],dtype=np.int64)
            chosen_w=np.asarray(data["weighting"],dtype=np.int64)
            metrics=np.asarray(data["metrics"],dtype=np.float64)
        if chosen_k.shape==(engine.nmask,) and metrics.shape==(4,engine.nmask,2,len(engine.ks)):
            if logger: logger.info("KNN TUNE CACHE HIT | %s",cache)
            return chosen_k,chosen_w,metrics,cache
    chosen_k,chosen_w,metrics=engine.tune(validation,logger=logger)
    cache.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(npz,k=chosen_k,weighting=chosen_w,metrics=metrics,
                        candidate_k=engine.ks,starts=engine.starts)
    atomic_json(summary_path,{"identity":identity,"n_validation":len(validation),
             "n_reference":len(engine.reference),"mse":metrics[0].tolist(),
             "mae":metrics[1].tolist(),"mse_mv2":metrics[2].tolist(),
             "mae_mv":metrics[3].tolist()})
    if logger: logger.info("KNN TUNE CACHE SAVED | %s",cache)
    return chosen_k,chosen_w,metrics,cache


def _score_events(engine, waves, predict, reference_predictions, baseline_prediction,
                  chosen_k, chosen_w, logger):
    n,_,t=waves.shape
    nw=engine.nwin
    score=np.empty((n,nw),dtype=np.float64)
    f_keep=np.empty_like(score)
    f_remove=np.empty_like(score)
    knn_keep=np.empty_like(score)
    knn_remove=np.empty_like(score)
    maps=np.zeros((n,2,t),dtype=np.float64)
    original=np.asarray(predict(waves),dtype=np.float64)
    for event_i,x in enumerate(waves):
        neighbor_ids,distances=engine.neighbors(x)
        reconstructions=[]
        expected=[]
        for mask in range(engine.nmask):
            k=int(chosen_k[mask])
            kind=WEIGHTINGS[int(chosen_w[mask])]
            restored,ids,weights=engine.reconstruct(
                x,mask,neighbor_ids[mask],distances[mask],k,kind
            )
            reconstructions.append(restored)
            expected.append(float(weights@reference_predictions[ids]))
        predictions=np.asarray(predict(np.asarray(reconstructions,dtype=np.float32)),dtype=np.float64)
        f_keep[event_i]=predictions[:nw]
        f_remove[event_i]=predictions[nw:]
        knn_keep[event_i]=expected[:nw]
        knn_remove[event_i]=expected[nw:]
        score[event_i]=0.5*(original[event_i]-f_remove[event_i]
                            +f_keep[event_i]-baseline_prediction)
        # Paired per-sample aggregation is a display interpolation, not exact sample Shapley.
        for win,(start,end) in enumerate(zip(engine.starts,engine.ends)):
            maps[event_i,:,start:end] += score[event_i,win]/(end-start)
        if logger and (event_i==0 or (event_i+1)%10==0 or event_i+1==n):
            dk=f_keep[event_i]-knn_keep[event_i]
            dr=f_remove[event_i]-knn_remove[event_i]
            logger.info(
                "KNN TSHAP | event=%d/%d | pred=%.6f ps | mean_phi=%.6f ps | "
                "mean_abs_delta_W=%.6f ps | mean_abs_delta_complement=%.6f ps",
                event_i+1,n,original[event_i],np.mean(score[event_i]),
                np.mean(np.abs(dk)),np.mean(np.abs(dr))
            )
    arrays=dict(
        raw_window_scores_ps=score,window_attribution_ps=maps,
        original_prediction_ps=original,conditional_prediction_window_ps=f_keep,
        conditional_prediction_complement_ps=f_remove,
        neighbor_mean_prediction_window_ps=knn_keep,
        neighbor_mean_prediction_complement_ps=knn_remove,
        prediction_difference_window_ps=f_keep-knn_keep,
        prediction_difference_complement_ps=f_remove-knn_remove,
        baseline_prediction_ps=np.asarray(baseline_prediction,dtype=np.float64),
    )
    if logger:
        for side,delta in (("W",arrays["prediction_difference_window_ps"]),
                           ("complement",arrays["prediction_difference_complement_ps"])):
            logger.info(
                "KNN OUTPUT DIAGNOSTIC | region=%s | samples=%d | bias_ps=%.6f | "
                "MAE_ps=%.6f | RMSE_ps=%.6f | STD_ps=%.6f | P95_ABS_ps=%.6f",
                side,delta.size,np.mean(delta),np.mean(np.abs(delta)),
                np.sqrt(np.mean(delta**2)),np.std(delta),
                np.percentile(np.abs(delta),95)
            )
    return arrays


def _write_diagnostics(destination, engine, chosen_k, chosen_w, metrics, scores):
    with (destination/"knn_tuning.csv").open("w",newline="",encoding="utf-8") as stream:
        writer=csv.writer(stream)
        writer.writerow(("observed","window_index","window_start_sample","k","weighting",
                         "MSE","RMSE","MAE","MSE_mV2","RMSE_mV","MAE_mV","selected"))
        for mask in range(engine.nmask):
            for wi,kind in enumerate(WEIGHTINGS):
                for ki,k in enumerate(engine.ks):
                    writer.writerow((
                        "W" if mask<engine.nwin else "complement",mask%engine.nwin,
                        engine.starts[mask%engine.nwin],int(k),kind,
                        metrics[0,mask,wi,ki],np.sqrt(metrics[0,mask,wi,ki]),
                        metrics[1,mask,wi,ki],metrics[2,mask,wi,ki],
                        np.sqrt(metrics[2,mask,wi,ki]),metrics[3,mask,wi,ki],
                        int(chosen_k[mask]==k and chosen_w[mask]==wi)))
    with (destination/"prediction_diagnostics.csv").open("w",newline="",encoding="utf-8") as stream:
        writer=csv.writer(stream)
        writer.writerow(("event_index","window_index","window_start_sample","original_prediction_ps",
                         "conditional_prediction_W_ps","neighbor_mean_prediction_W_ps","delta_W_ps",
                         "conditional_prediction_complement_ps",
                         "neighbor_mean_prediction_complement_ps","delta_complement_ps","shapley_ps"))
        for event_i in range(len(scores["original_prediction_ps"])):
            for wi,start in enumerate(engine.starts):
                writer.writerow((event_i,wi,int(start),scores["original_prediction_ps"][event_i],
                   scores["conditional_prediction_window_ps"][event_i,wi],
                   scores["neighbor_mean_prediction_window_ps"][event_i,wi],
                   scores["prediction_difference_window_ps"][event_i,wi],
                   scores["conditional_prediction_complement_ps"][event_i,wi],
                   scores["neighbor_mean_prediction_complement_ps"][event_i,wi],
                   scores["prediction_difference_complement_ps"][event_i,wi],
                   scores["raw_window_scores_ps"][event_i,wi]))


def explain_conditional_knn(results=None, *, model_dir=None, models=None, ml_window=None,
       mode=None, cache_root=None, artifact_root=None, output_root=None, local=False,
       max_events=100, events_per_group=2, extreme_margin=0.05,
       window_samples=20, stride_samples=10, predict_batch_size=64, seed=42,
       knn_k=(1,3,5,10,20,50,100), knn_tuning_events=128,
       knn_reference_events=5000, knn_reference_batch=512,
       overwrite=False,logger=None):
    """Run KNN conditional-mean TSHAP on saved models without refitting."""
    runs,source=base._tshap_run_inputs(results,model_dir)
    selected_window=base._effective_tshap_window(ml_window,model_dir=model_dir)
    source=Path(source).resolve()
    output_root=Path(output_root or Path.cwd()/"xai").expanduser().resolve()
    if output_root==source or output_root.is_relative_to(source):
        raise ValueError("XAI output must be outside result tree")
    lookup=PreparedCacheLookup()
    outputs=[]
    failures=[]
    shared_knn={}
    selected=0
    for run in runs:
        directory=Path(run["directory"])
        manifest=run["manifest"]
        if models and manifest["model"] not in models: continue
        try:
            config=resolve_run_config(directory)
            base._apply_tshap_cache(config,cache_root=cache_root,artifact_root=artifact_root)
            if not base._matches_tshap_window(config,selected_window): continue
            if not base._matches_tshap_mode(config,mode): continue
            selected+=1
            spec=get_model(manifest["model"])
            store=RunStore(directory,artifact_root=config["preprocessing"]["artifact_root"])
            fitted_path=base._resolve_tshap_model_directory(
                spec,store,directory,explicit_model_dir=model_dir is not None
            )
            development=lookup.find("development",config,manifest)
            blind=lookup.find("blind",config,manifest)
            family=mode_family(config["mode"])
            time_ps=np.asarray(getattr(blind,f"{family}_time_ps"),dtype=np.float64)
            if not np.array_equal(time_ps,np.asarray(getattr(development,f"{family}_time_ps"))):
                raise ValueError("KNN development/blind time grids differ")
            transform=getattr(development,f"{family}_transform")
            blind_transform=getattr(blind,f"{family}_transform")
            if transform is None or blind_transform is None or any(
                    not np.array_equal(getattr(transform,k),getattr(blind_transform,k))
                    for k in ("minimum","maximum")):
                raise ValueError("KNN development/blind normalization differs")
            fitted=load_fitted_model(spec,fitted_path,run["best"]["parameters"],config)
            try:
                mask=np.ones(len(time_ps),dtype=bool) if fitted.sample_mask is None else np.asarray(fitted.sample_mask,dtype=bool)
                starts,_,_=base._valid_window_starts(
                    len(time_ps),window_samples,stride_samples,mask)
                starts=np.asarray(starts,dtype=np.int64)
                if not len(starts):
                    raise ValueError("No KNN TSHAP windows after sample masking")
                ndev=development.n_events
                if not 1<=knn_tuning_events<ndev:
                    raise ValueError("KNN tuning events must be between 1 and development size-1")
                order=np.random.default_rng(seed).permutation(ndev)
                validation_ids=np.sort(order[:knn_tuning_events])
                reference_ids=np.sort(order[knn_tuning_events:][:knn_reference_events])
                if len(reference_ids)<max(knn_k):
                    raise ValueError("KNN reference count smaller than maximum k")
                key=canonical_hash({
                    "development":development.manifest,
                    "family":family,"mask":mask.tolist(),
                    "starts":starts.tolist(),"width":window_samples,
                    "reference_ids":reference_ids.tolist(),
                    "validation_ids":validation_ids.tolist(),
                    "k":list(knn_k),"version":METHOD_VERSION,
                })
                if key not in shared_knn:
                    reference=waveform_view(development,config["mode"],reference_ids).materialize()
                    validation=waveform_view(development,config["mode"],validation_ids).materialize()
                    eng=ConditionalNeighborEngine(
                        reference,starts,window_samples,mask,knn_k,
                        scale_mv=np.asarray(transform.maximum)-np.asarray(transform.minimum),
                        reference_batch=knn_reference_batch,
                    )
                    root=cache_root or Path(config["preprocessing"]["artifact_root"]).parent
                    identity={"key":key,"seed":seed,"k":list(knn_k),
                              "validation_events":len(validation_ids),"reference_events":len(reference_ids),
                              "method_version":METHOD_VERSION}
                    chosen_k,chosen_w,metrics,tune_path=_tune_cache(root,identity,eng,validation,logger)
                    shared_knn[key]=(eng,chosen_k,chosen_w,metrics,reference_ids,tune_path)
                eng,chosen_k,chosen_w,metrics,reference_ids,tune_path=shared_knn[key]
                destination=(base._local_explanation_destination(output_root,directory)
                             if local else base._explanation_destination(output_root,directory))
                destination=destination.with_name(
                    "tshap_conditional_knn_local" if local else "tshap_conditional_knn")
                identity={"method":"paired_conditional_knn_mean_shapley",
                    "version":METHOD_VERSION,"final_fit":manifest["stage_fingerprints"]["final_fit"],
                    "dataset":key,"blind":blind.manifest,
                    "local":local,"seed":seed,"max_events":max_events,
                    "events_per_group":events_per_group,"extreme_margin":extreme_margin,
                    "batch_size":predict_batch_size}
                fingerprint=canonical_hash(identity)
                if (destination/"summary.json").is_file() and not overwrite:
                    saved=json.loads((destination/"summary.json").read_text())
                    if saved.get("identity_sha256")==fingerprint:
                        if logger:logger.info("KNN TSHAP | output cache hit | %s",destination)
                        outputs.append(destination)
                        continue
                predict=lambda arr:base._predict_bounded(spec,fitted,arr,predict_batch_size)
                reference_predictions=predict(eng.reference)
                baseline_prediction=float(np.mean(reference_predictions))
                if logger:logger.info("KNN BASELINE | model=%s | reference=%d | mean_prediction_ps=%.6f | std_prediction_ps=%.6f",
                                      spec.name,len(reference_predictions),baseline_prediction,np.std(reference_predictions))
                if local:
                    predictions,targets=base._load_materialized_local_predictions(
                        store,blind,manifest,config["mode"])
                    cases,improvements,_=base._select_local_events(
                        predictions,targets,events_per_group=events_per_group,
                        extreme_margin=extreme_margin)
                    indices=np.asarray([item["index"] for item in cases],dtype=np.int64)
                else:
                    indices=base._choose_indices(
                        blind.n_events,max_events,base._selection_seed(seed))
                    cases=None
                waves=waveform_view(blind,config["mode"],indices).materialize()
                original=predict_indices(spec,fitted,blind,config["mode"],indices,
                                         chunk_size=predict_batch_size)
                replay=predict(waves)
                if not np.allclose(original,replay,atol=0.1,rtol=1e-3):
                    raise ValueError("KNN TSHAP original inference differs from saved model")
                scored=_score_events(eng,waves,predict,reference_predictions,
                                     baseline_prediction,chosen_k,chosen_w,logger)
                parent=destination.parent
                parent.mkdir(parents=True,exist_ok=True)
                tmp=Path(tempfile.mkdtemp(prefix=".knn-tshap-",dir=parent))
                try:
                    _write_diagnostics(tmp,eng,chosen_k,chosen_w,metrics,scored)
                    raw=scored["raw_window_scores_ps"]
                    maps=scored["window_attribution_ps"]
                    event_ids=np.asarray(blind.event_index[indices],dtype=np.int64)
                    np.savez_compressed(tmp/("local_attributions.npz" if local else "attributions.npz"),
                        **scored,event_id=event_ids,dataset_index=indices,
                        time_ps=time_ps,window_starts=starts,
                        window_samples=np.asarray(window_samples))
                    plots=[]
                    wave_mv=np.asarray(blind_transform.inverse(waves),dtype=np.float64)
                    if local:
                        diff_mean,_=base._dataset_mean_difference_mv(
                            blind,config["mode"],blind_transform,seed,max_events=1000)
                        led_threshold=float(blind.manifest["fixed_led_threshold_mV"])
                        led_baselines=base._native_led_baselines(
                            config["preprocessing"]["artifact_root"],blind,config["mode"],indices,
                            config["preprocessing"]["selection"]["baseline_window_ns"])
                        with (tmp/"events.csv").open("w",newline="",encoding="utf-8") as stream:
                            writer=csv.writer(stream)
                            writer.writerow(("group","event_id","target_correction_ps","predicted_correction_ps",
                                             "signed_improvement_ps"))
                            for i,case in enumerate(cases):
                                name=f"{case['group']}__event_{int(event_ids[i])}.png"
                                base._plot_local_event(
                                    tmp/name,time_ps=time_ps,waveform_mv=wave_mv[i],
                                    dataset_mean_difference_mv=diff_mean,led_baseline_mv=led_baselines[i],
                                    attribution=maps[i],group=case["group"],event_id=event_ids[i],
                                    target_ps=targets[indices[i]],prediction_ps=original[i],
                                    improvement_ps=improvements[indices[i]],led_threshold_mv=led_threshold,
                                    window_starts=starts,window_scores=raw[i:i+1],
                                    window_samples=window_samples)
                                plots.append(name)
                                writer.writerow((case["group"],int(event_ids[i]),
                                    float(targets[indices[i]]),float(original[i]),
                                    float(improvements[indices[i]])))
                    else:
                        pos,neg=base._signed_means(raw[:,None,:])
                        importance=np.mean(np.abs(maps),axis=(0,1))
                        mean=np.mean(wave_mv,axis=0)
                        std=np.std(wave_mv,axis=0)
                        base._write_global_importance_csv(tmp,time_ps,importance)
                        base._write_signed_csv(tmp,time_ps,starts,pos,neg,window_samples)
                        base._plot_global(tmp,time_ps,importance,mean,std_waveform_mv=std,
                            model_name=spec.name,mode=config["mode"],
                            window_name=str(config.get("window_name")),
                            board_name=base._find_board_label(directory,output_root),
                            n_events=len(waves),window_samples=window_samples,
                            window_starts=starts,signed_positive=pos,signed_negative=neg)
                    diagnostics={}
                    for name in ("window","complement"):
                        d=scored[f"prediction_difference_{name}_ps"]
                        diagnostics[name]={"bias_ps":float(np.mean(d)),
                            "rmse_ps":float(np.sqrt(np.mean(d*d))),
                            "mae_ps":float(np.mean(np.abs(d))),
                            "std_ps":float(np.std(d)),
                            "p95_absolute_ps":float(np.percentile(np.abs(d),95))}
                    atomic_json(tmp/"summary.json",{
                        **identity,"identity_sha256":fingerprint,
                        "n_explained_events":len(waves),
                        "n_reference":len(eng.reference),"n_validation":knn_tuning_events,
                        "mean_reference_prediction_ps":baseline_prediction,
                        "reference_prediction_std_ps":float(np.std(reference_predictions)),
                        "diagnostics":diagnostics,"plots":plots,
                        "tuning_cache":str(tune_path),
                        "selected_k":chosen_k.tolist(),
                        "selected_weightings":[WEIGHTINGS[int(i)] for i in chosen_w],
                        "formula":"0.5*(f(x)-f(imputed_W)+f(imputed_complement)-mean_reference_f)",
                        "imputation":"deterministic conditional KNN mean, frozen model inputs",
                        "prediction_expectations":"neighbor weighted mean of saved model outputs, diagnostic only",
                    })
                    base._publish_directory(tmp,destination,overwrite=True)
                finally:
                    if tmp.exists():shutil.rmtree(tmp)
                if logger:logger.info("KNN TSHAP COMPLETE | %s | events=%d",destination,len(waves))
                outputs.append(destination)
            finally:
                del fitted
                release_training_memory()
        except Exception as exc:
            failures.append(f"{directory}: {exc}")
            if logger:logger.exception("KNN TSHAP FAILED | run=%s",directory)
    if not selected:
        raise ValueError("No saved runs match the selected KNN TSHAP filters")
    if failures:
        raise RuntimeError("Some KNN TSHAP runs failed:\n"+"\n".join(failures))
    return outputs
