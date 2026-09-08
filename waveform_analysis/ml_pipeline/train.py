from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
import numpy as np
from .models.spec import ModelSpec
from .search import SearchResult, select_candidate
from .stats import ctr_fwhm
from .storage import atomic_json
from .view import target_correction, waveform_view

@dataclass
class FittedModel:
    artifact: Any
    metadata: dict[str, Any]

def _fit_once(spec, model_config, parameters, train_x, train_target, *, seed, validation_x=None, validation_target=None, final_epochs=None):
    artifact = spec.fit(parameters, np.asarray(train_x,dtype=np.float32), np.asarray(train_target,dtype=np.float64), seed=seed, config=model_config, validation_x=None if validation_x is None else np.asarray(validation_x,dtype=np.float32), validation_target=None if validation_target is None else np.asarray(validation_target,dtype=np.float64), final_epochs=final_epochs)
    return FittedModel(artifact, dict(getattr(artifact,"metadata",{}) or {}))

def predict_model(spec: ModelSpec, fitted: FittedModel, pair: np.ndarray) -> np.ndarray:
    return np.asarray(spec.predict(fitted.artifact, np.asarray(pair,dtype=np.float32)), dtype=np.float64)

def _selection_score(spec,model_config,config,residual):
    metric=str((model_config.get("training",{}) or {}).get("selection_metric","validation_ctr")).lower()
    values=np.asarray(residual,dtype=np.float64)
    if spec.name=="cnn" and metric=="validation_mse": return float(np.mean(values**2))
    if spec.name=="cnn" and metric=="validation_rmse": return float(np.sqrt(np.mean(values**2)))
    return float(ctr_fwhm(values,config.get("fit")).ctr_ps)

def _score_label(spec,model_config):
    metric=str((model_config.get("training",{}) or {}).get("selection_metric","validation_ctr")).lower()
    if spec.name=="cnn" and metric=="validation_mse": return "MSE [ps^2]"
    if spec.name=="cnn" and metric=="validation_rmse": return "RMSE [ps]"
    return "CTR [ps]"

def search_model(spec, model_config, config, dataset, mode: str, *, seed: int, logger=None) -> SearchResult:
    training=np.asarray(dataset.training,dtype=np.int64); validation=np.asarray(dataset.validation,dtype=np.int64)
    train_x=waveform_view(dataset,mode,training).materialize(); validation_x=waveform_view(dataset,mode,validation).materialize(); target=target_correction(dataset,mode); train_target=target[training]; validation_target=target[validation]
    def fit_candidate(parameters,_train_data,candidate_seed):
        return _fit_once(spec,model_config,parameters,train_x,train_target,seed=candidate_seed,validation_x=validation_x,validation_target=validation_target)
    def predict_candidate(_parameters,fitted,_validation_data): return predict_model(spec,fitted,validation_x)-validation_target
    def on_start(number,total,candidate):
        if logger is not None: logger.info('Training %s/%s | candidate %d/%d | %s',mode,spec.name,number,total,candidate)
    def on_result(number,total,result):
        if logger is None:return
        if result.error is None: logger.info('Validation %s/%s | candidate %d/%d | %s %.6g | %s',mode,spec.name,number,total,_score_label(spec,model_config),result.score,result.candidate)
        else: logger.warning('Candidate failed %s/%s | candidate %d/%d | %s | %s',mode,spec.name,number,total,result.candidate,result.error)
    return select_candidate(spec.candidates(model_config),training,validation,fit_candidate=fit_candidate,predict_candidate=predict_candidate,score_candidate=lambda residual:_selection_score(spec,model_config,config,residual),seed=seed,on_candidate_start=on_start,on_candidate_result=on_result)

def refit_selected(spec, model_config, dataset, mode: str, selected, *, seed: int) -> FittedModel:
    development=np.asarray(dataset.development,dtype=np.int64); x=waveform_view(dataset,mode,development).materialize(); target=target_correction(dataset,mode)[development]; epochs=selected.metadata.get("best_epoch") if selected.metadata else None
    return _fit_once(spec,model_config,selected.candidate,x,target,seed=seed,final_epochs=None if epochs is None else int(epochs))

def predict_indices(spec,fitted,dataset,mode,indices):
    view=waveform_view(dataset,mode,np.asarray(indices,dtype=np.int64)); pair=view.materialize(); return predict_model(spec,fitted,pair),view.time_ps,pair

def save_model(spec,fitted,directory:Path,parameters):
    directory.mkdir(parents=True,exist_ok=True); spec.save(fitted.artifact,directory); atomic_json(directory/"metadata.json",{"model":spec.name,"parameters":parameters,"training":fitted.metadata})
