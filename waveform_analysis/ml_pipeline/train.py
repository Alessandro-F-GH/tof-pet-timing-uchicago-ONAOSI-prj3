from __future__ import annotations
import copy,pickle
from dataclasses import dataclass,field
from pathlib import Path
from typing import Any
import numpy as np
from .common import canonical_hash
from .sample_mask import apply_sample_mask,apply_sample_mask_to_time,training_sample_mask
from .splits import semantic_seed
from .view import waveform_view,model_target

@dataclass
class FittedFeatureTransform:
    spec:Any
    artifact:Any
    identity:str
    parameters:dict[str,Any]
    seed:int
    cache:dict[str,np.ndarray]=field(default_factory=dict)
    def apply(self,values,cache_key):
        key=str(cache_key)
        if key in self.cache:return self.cache[key]
        transformed=np.asarray(self.spec.transform(self.artifact,np.asarray(values,np.float32)))
        if transformed.ndim<2 or transformed.shape[0]!=np.asarray(values).shape[0]:raise RuntimeError(f"Feature transform {self.spec.name} changed the event axis")
        if not np.all(np.isfinite(transformed)):raise RuntimeError(f"Feature transform {self.spec.name} produced non-finite values")
        self.cache[key]=transformed;return transformed

class FeatureTransformCache:
    def __init__(self):self._items={}
    def prepare(self,spec,parameters,values,*,seed_base,scope_key,config):
        transform_parameters=dict(spec.parameters(dict(parameters or {}),config) or {});parameter_hash=canonical_hash(transform_parameters)
        transform_seed=semantic_seed(int(seed_base),"feature_transform",spec.name,parameter_hash)
        identity=canonical_hash({"name":spec.name,"parameters":transform_parameters,"seed":transform_seed,"scope":str(scope_key)})
        cached=self._items.get(identity);fit_key=f"fit:{scope_key}"
        if cached is not None:return cached,cached.cache[fit_key]
        artifact,transformed=spec.fit_transform(transform_parameters,np.asarray(values,np.float32),seed=transform_seed,config=config);transformed=np.asarray(transformed)
        if transformed.ndim<2 or transformed.shape[0]!=np.asarray(values).shape[0]:raise RuntimeError(f"Feature transform {spec.name} changed the event axis")
        if not np.all(np.isfinite(transformed)):raise RuntimeError(f"Feature transform {spec.name} produced non-finite values")
        wrapper=FittedFeatureTransform(spec,artifact,identity,transform_parameters,transform_seed,{fit_key:transformed});self._items[identity]=wrapper
        logger=config.get("_logger")
        if logger is not None:logger.info("Feature transform fitted | %s | id=%s | params=%s | events=%d | features=%s",spec.name,identity[:12],transform_parameters,transformed.shape[0],tuple(transformed.shape[1:]))
        return wrapper,transformed

@dataclass
class FittedModel:
    artifact:Any
    metadata:dict[str,Any]
    output_max_abs_ps:float|None=None
    sample_mask:np.ndarray|None=None
    feature_transform:FittedFeatureTransform|None=None

def _fit_once(spec,model_config,parameters,x,y,*,seed,output_limit,input_time_ps,sample_mask,logger=None):
    cfg=copy.deepcopy(model_config);cfg["_prediction_max_abs_ps"]=float(output_limit);cfg["_input_time_ps"]=np.asarray(input_time_ps,np.float64)
    if logger is not None:cfg["_logger"]=logger
    artifact=spec.fit(parameters,np.asarray(x,np.float32),np.asarray(y,np.float64),seed=int(seed),config=cfg);metadata=dict(getattr(artifact,"metadata",{}) or {});metadata["estimator_formulation"]=spec.estimator_formulation
    return FittedModel(artifact,metadata,float(output_limit),np.asarray(sample_mask,bool))

def _mask(spec,train_x):return np.ones(train_x.shape[-1],bool) if spec.preserve_temporal_grid else training_sample_mask(train_x)

def fit_on_indices(spec,model_config,config,dataset,indices,parameters,*,seed,transform_seed_base=None,feature_transform_cache=None,logger=None):
    idx=np.asarray(indices,np.int64);view=waveform_view(dataset,config["mode"],idx);xfull=view.materialize();mask=_mask(spec,xfull);x=apply_sample_mask(xfull,mask);time=apply_sample_mask_to_time(view.time_ps,mask);y=model_target(dataset,config["mode"])[idx]
    cfg=copy.deepcopy(model_config);cfg["_early_stopping_seed"]=int(seed);cfg["_input_time_ps"]=np.asarray(time,np.float64)
    if logger is not None:cfg["_logger"]=logger
    feature_transform=None
    if spec.feature_transform is not None:
        cache=feature_transform_cache if feature_transform_cache is not None else FeatureTransformCache();scope_key=canonical_hash({"population":dataset.manifest["analysis_population_identity"],"mode":config["mode"],"indices":idx.tolist(),"sample_mask":np.asarray(mask,bool).tolist()})
        feature_transform,x=cache.prepare(spec.feature_transform,parameters,x,seed_base=int(seed if transform_seed_base is None else transform_seed_base),scope_key=scope_key,config=cfg)
    fitted=_fit_once(spec,cfg,dict(parameters or {}),x,y,seed=seed,output_limit=config["ml_output"]["max_abs_ps"],input_time_ps=time,sample_mask=mask,logger=logger);fitted.feature_transform=feature_transform
    fitted.metadata.update({"training_events":int(idx.size),"sample_mask_training_events":int(idx.size),"input_samples_before_mask":int(mask.size),"input_samples_after_mask":int(mask.sum())})
    if feature_transform is not None:fitted.metadata.update({"feature_transform":feature_transform.spec.name,"feature_transform_identity":feature_transform.identity,"feature_transform_parameters":feature_transform.parameters,"feature_transform_seed":int(feature_transform.seed)})
    return fitted

def _model_input(fitted,dataset,mode,indices,*,swapped=False):
    idx=np.asarray(indices,np.int64);view=waveform_view(dataset,mode,idx);pair=apply_sample_mask(view.materialize(),fitted.sample_mask)
    if swapped:pair=pair[:,::-1,:]
    pair=np.ascontiguousarray(pair)
    if fitted.feature_transform is None:return pair
    key=canonical_hash({"population":dataset.manifest["analysis_population_identity"],"mode":mode,"indices":idx.tolist(),"swapped":bool(swapped)})
    return fitted.feature_transform.apply(pair,key)

def _prediction_from_input(spec,fitted,values):
    prediction=np.asarray(spec.predict(fitted.artifact,values),np.float64)
    if fitted.output_max_abs_ps is not None:prediction=np.clip(prediction,-fitted.output_max_abs_ps,fitted.output_max_abs_ps)
    return prediction

def predict_indices(spec,fitted,dataset,mode,indices):return _prediction_from_input(spec,fitted,_model_input(fitted,dataset,mode,indices))

def detector_swap_rmse(spec,fitted,dataset,mode,indices):
    if spec.estimator_formulation=="shared":return 0.0
    forward=_prediction_from_input(spec,fitted,_model_input(fitted,dataset,mode,indices));reverse=_prediction_from_input(spec,fitted,_model_input(fitted,dataset,mode,indices,swapped=True));epsilon=forward+reverse
    if not np.all(np.isfinite(epsilon)):raise RuntimeError("Detector-swap diagnostic produced non-finite predictions")
    return float(np.sqrt(np.mean(np.square(epsilon))))

def save_model(spec,fitted,directory,parameters):
    directory=Path(directory);directory.mkdir(parents=True,exist_ok=True);spec.save(fitted.artifact,directory)
    if fitted.feature_transform is not None and fitted.feature_transform.spec.save is not None:fitted.feature_transform.spec.save(fitted.feature_transform.artifact,directory/"feature_transform")
    if fitted.sample_mask is not None:np.save(directory/"sample_mask.npy",np.asarray(fitted.sample_mask,bool))

def load_fitted_model(spec,directory,parameters,config):
    """Reload a selected saved model for inference-only postprocessing."""
    directory=Path(directory);mask=np.load(directory/"sample_mask.npy").astype(bool);output_limit=float(config["ml_output"]["max_abs_ps"])
    if spec.name=="minirocket":
        with (directory/"model.pkl").open("rb") as s:artifact=pickle.load(s)
        feature_transform=None
        tpath=directory/"feature_transform"/"transform.pkl"
        if tpath.is_file():
            with tpath.open("rb") as s:tartifact=pickle.load(s)
            feature_transform=FittedFeatureTransform(spec.feature_transform,tartifact,"reloaded",{},0,{})
        return FittedModel(artifact,dict(getattr(artifact,"metadata",{}) or {}),output_limit,mask,feature_transform)
    import torch
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu");payload=torch.load(directory/"model.pt",map_location=device,weights_only=False);metadata=dict(payload.get("metadata",{}) or {});n=int(mask.sum())
    p=dict(parameters or {})
    if spec.name=="mlp":
        from .models.mlp import SharedScorerMLP
        from .models._mlp_common import MLPArtifact
        model=SharedScorerMLP(n,p["architecture"],p["activation"]);artifact_type=MLPArtifact
    elif spec.name=="direct_mlp":
        from .models.direct_mlp import DirectPairMLP
        from .models._mlp_common import MLPArtifact
        model=DirectPairMLP(n,p["architecture"],p["activation"]);artifact_type=MLPArtifact
    elif spec.name=="locally_connected_mlp":
        from .models.locally_connected_mlp import SharedLocallyConnectedScorer
        from .models._mlp_common import MLPArtifact
        model=SharedLocallyConnectedScorer(n,p["architecture"],p["activation"],layer1_kernel_samples=p["layer1_kernel_samples"],layer1_stride_samples=p["layer1_stride_samples"],layer2_kernel_positions=p["layer2_kernel_positions"],layer2_stride_positions=p["layer2_stride_positions"],max_correction_ps=p["max_correction_ps"]);artifact_type=MLPArtifact
    elif spec.name in {"shared_cnn1d","independent_cnn1d"}:
        from .models._cnn1d_common import IndependentCNN1D,SharedCNN1D
        from .models._mlp_common import MLPArtifact
        cls=SharedCNN1D if spec.name=="shared_cnn1d" else IndependentCNN1D
        model=cls(n,p["architecture"],p["activation"],conv_channels=p["conv_channels"],kernel_samples=p["kernel_samples"]);artifact_type=MLPArtifact
    elif spec.name=="onishi_cnn":
        from .models.onishi_cnn import OnishiPairedCNN,OnishiCNNArtifact
        model=OnishiPairedCNN(config["model"]["space"].get("architecture",{}));
        with torch.no_grad():model(torch.zeros((1,2,n),dtype=torch.float32))
        artifact_type=OnishiCNNArtifact
    else:raise ValueError(f"Inference reload is not implemented for model {spec.name}")
    model.load_state_dict(payload["state_dict"]);model.to(device);model.eval();artifact=artifact_type(model,str(device),metadata)
    return FittedModel(artifact,metadata,output_limit,mask,None)
