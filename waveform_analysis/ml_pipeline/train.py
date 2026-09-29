from __future__ import annotations
import copy
from dataclasses import dataclass
from pathlib import Path
from typing import Any
import numpy as np
from .sample_mask import apply_sample_mask,apply_sample_mask_to_time,training_sample_mask
from .view import waveform_view,model_target

@dataclass
class FittedModel:
    artifact:Any;metadata:dict[str,Any];output_max_abs_ps:float|None=None;sample_mask:np.ndarray|None=None

def _fit_once(spec,model_config,parameters,x,y,*,seed,output_limit,input_time_ps,sample_mask,logger=None):
    cfg=copy.deepcopy(model_config);cfg["_prediction_max_abs_ps"]=float(output_limit);cfg["_input_time_ps"]=np.asarray(input_time_ps,np.float64)
    if logger is not None:cfg["_logger"]=logger
    artifact=spec.fit(parameters,np.asarray(x,np.float32),np.asarray(y,np.float64),seed=int(seed),config=cfg)
    metadata=dict(getattr(artifact,"metadata",{}) or {})
    metadata["estimator_formulation"]=spec.estimator_formulation
    return FittedModel(artifact,metadata,float(output_limit),np.asarray(sample_mask,bool))

def _mask(spec,train_x):
    return np.ones(train_x.shape[-1],bool) if spec.preserve_temporal_grid else training_sample_mask(train_x)

def fit_on_indices(spec,model_config,config,dataset,indices,parameters,*,seed,logger=None):
    idx=np.asarray(indices,np.int64);view=waveform_view(dataset,config["mode"],idx);xfull=view.materialize()
    mask=_mask(spec,xfull);x=apply_sample_mask(xfull,mask);time=apply_sample_mask_to_time(view.time_ps,mask);y=model_target(dataset,config["mode"])[idx]
    cfg=copy.deepcopy(model_config);cfg["_early_stopping_seed"]=int(seed)
    fitted=_fit_once(spec,cfg,dict(parameters or {}),x,y,seed=seed,output_limit=config["ml_output"]["max_abs_ps"],
        input_time_ps=time,sample_mask=mask,logger=logger)
    fitted.metadata.update({"training_events":int(idx.size),"sample_mask_training_events":int(idx.size),
        "input_samples_before_mask":int(mask.size),"input_samples_after_mask":int(mask.sum())})
    return fitted

def _prediction_from_pair(spec,fitted,pair):
    values=np.asarray(spec.predict(fitted.artifact,pair),np.float64)
    if fitted.output_max_abs_ps is not None:values=np.clip(values,-fitted.output_max_abs_ps,fitted.output_max_abs_ps)
    return values

def predict_indices(spec,fitted,dataset,mode,indices):
    view=waveform_view(dataset,mode,np.asarray(indices,np.int64));pair=apply_sample_mask(view.materialize(),fitted.sample_mask)
    return _prediction_from_pair(spec,fitted,pair)

def detector_swap_rmse(spec,fitted,dataset,mode,indices):
    if spec.estimator_formulation=="shared":return 0.0
    view=waveform_view(dataset,mode,np.asarray(indices,np.int64));pair=apply_sample_mask(view.materialize(),fitted.sample_mask)
    forward=_prediction_from_pair(spec,fitted,pair);reverse=_prediction_from_pair(spec,fitted,pair[:,::-1,:])
    epsilon=forward+reverse
    if not np.all(np.isfinite(epsilon)):raise RuntimeError("Detector-swap diagnostic produced non-finite predictions")
    return float(np.sqrt(np.mean(np.square(epsilon))))

def save_model(spec,fitted,directory,parameters):
    directory=Path(directory);directory.mkdir(parents=True,exist_ok=True);spec.save(fitted.artifact,directory)
    if fitted.sample_mask is not None:np.save(directory/"sample_mask.npy",np.asarray(fitted.sample_mask,bool))
