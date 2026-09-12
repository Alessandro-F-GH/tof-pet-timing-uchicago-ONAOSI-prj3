from __future__ import annotations
import copy,itertools,math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np, torch

from utils_fit import fit_ctr_ps
from torch import nn
from torch.utils.data import DataLoader,TensorDataset

from .cnn import SharedScorerCNN,_configure_reproducibility,_device,_loader,_rmse
from .spec import ModelSpec

_LOG_SIGMA_MIN=math.log(1e-3)
_LOG_SIGMA_MAX=math.log(1e4)

class HeteroscedasticSharedScorerCNN(SharedScorerCNN):
    def __init__(self,architecture:dict[str,Any]):
        super().__init__(architecture)
        final=self.head[-1]
        if not isinstance(final,nn.Linear): raise TypeError("CNN head must end with Linear")
        self.head[-1]=nn.Linear(final.in_features,2)

    def initialize_sigma(self,sigma_ps:float):
        sigma=float(sigma_ps)
        if not np.isfinite(sigma) or sigma<=0: raise ValueError("initial sigma must be positive")
        final=self.head[-1]
        with torch.no_grad():
            final.weight[1].zero_()
            final.bias[1].fill_(math.log(sigma))

    def signal_distribution(self,waveform):
        values=self.score(waveform)
        mean=values[:,0]
        log_sigma=torch.clamp(values[:,1],_LOG_SIGMA_MIN,_LOG_SIGMA_MAX)
        return mean,log_sigma

    def pair_distribution(self,pair):
        if pair.ndim!=3 or pair.shape[1]!=2:
            raise ValueError(f"cnn_heteroscedastic expects [event,2,time], got {tuple(pair.shape)}")
        mean1,log_sigma1=self.signal_distribution(pair[:,0,:])
        mean2,log_sigma2=self.signal_distribution(pair[:,1,:])
        mean=mean1-mean2
        log_variance=torch.logaddexp(2*log_sigma1,2*log_sigma2)
        sigma=torch.exp(0.5*log_variance)
        return mean,sigma,log_variance

    def forward(self,pair):
        mean,_,_=self.pair_distribution(pair)
        return mean

@dataclass
class HeteroscedasticCNNArtifact:
    model:HeteroscedasticSharedScorerCNN
    device:str
    sigma_max_ps:float
    metadata:dict[str,Any]

def candidates(config):
    p=config.get("parameters",{}); training=config.get("training",{})
    sigma_values=[float(v) for v in p.get("sigma_max_ps",[20.0])]
    if not sigma_values or any((not np.isfinite(v) or v<=0) for v in sigma_values):
        raise ValueError("sigma_max_ps values must be finite and positive")
    return [
        {"learning_rate":float(lr),"weight_decay":float(wd),"batch_size":int(batch)}
        for lr,wd,batch in itertools.product(
            p.get("learning_rate",[1e-3]),
            p.get("weight_decay",[1e-5]),
            p.get("batch_size",[training.get("batch_size",64)])
        )
    ]

def gaussian_nll(mean,log_variance,target):
    residual=target-mean
    return 0.5*torch.mean(log_variance+residual.square()*torch.exp(-log_variance))

def _predict_distribution(model,x,device,batch):
    loader=DataLoader(torch.from_numpy(np.ascontiguousarray(x,dtype=np.float32)),batch_size=int(batch),shuffle=False)
    means=[]; sigmas=[]; model.eval()
    with torch.no_grad():
        for pair in loader:
            mean,sigma,_=model.pair_distribution(pair.to(device))
            means.append(mean.cpu().numpy()); sigmas.append(sigma.cpu().numpy())
    return np.concatenate(means).astype(np.float64),np.concatenate(sigmas).astype(np.float64)

def _gate(mean,sigma,sigma_max_ps):
    mean=np.asarray(mean,dtype=np.float64); sigma=np.asarray(sigma,dtype=np.float64)
    return np.where(sigma<=float(sigma_max_ps),mean,0.0)

def fit(params,train_x,train_target,*,seed,config,validation_x=None,validation_target=None):
    if validation_x is None or validation_target is None:
        raise ValueError("cnn_heteroscedastic requires validation data")
    training_seed=_configure_reproducibility(seed)
    training=config.get("training",{}); device=_device(config)
    model=HeteroscedasticSharedScorerCNN(config.get("architecture",{})).to(device)
    train_target=np.asarray(train_target,dtype=np.float64)
    validation_target=np.asarray(validation_target,dtype=np.float64)
    initial_pair_sigma=float(np.std(train_target))
    if not np.isfinite(initial_pair_sigma) or initial_pair_sigma<=0: initial_pair_sigma=1.0
    model.initialize_sigma(max(initial_pair_sigma/math.sqrt(2),1.0))
    optimizer=torch.optim.AdamW(model.parameters(),lr=float(params["learning_rate"]),weight_decay=float(params["weight_decay"]))
    batch=int(params.get("batch_size",training.get("batch_size",64)))
    sigma_candidates=[float(v) for v in config.get("parameters",{}).get("sigma_max_ps",[20.0])]
    max_epochs=int(training.get("epochs",300)); patience=int(training.get("patience",10))
    min_delta=float(training.get("min_delta",0.01)); clip=float(training.get("gradient_clip_norm",10.0))
    loader=_loader(train_x,train_target,batch,shuffle=True,seed=training_seed)
    best_score=float("inf"); best_epoch=0; best_state=None; stale=0

    for epoch in range(1,max_epochs+1):
        model.train()
        for pair,target in loader:
            pair=pair.to(device); target=target.to(device); optimizer.zero_grad(set_to_none=True)
            mean,_,log_variance=model.pair_distribution(pair)
            loss=gaussian_nll(mean,log_variance,target)
            if not torch.isfinite(loss): raise RuntimeError("non-finite heteroscedastic NLL")
            loss.backward()
            if clip>0: nn.utils.clip_grad_norm_(model.parameters(),clip)
            optimizer.step()
        validation_mean,_=_predict_distribution(model,validation_x,device,batch)
        score=_rmse(validation_mean-validation_target)
        if score<best_score-min_delta:
            best_score=score; best_epoch=epoch; best_state=copy.deepcopy(model.state_dict()); stale=0
        else:
            stale+=1
            if stale>=patience: break

    if best_state is None: raise RuntimeError("cnn_heteroscedastic produced no valid checkpoint")
    model.load_state_dict(best_state)
    validation_mean,validation_sigma=_predict_distribution(model,validation_x,device,batch)
    fit_config=dict(config.get("_fit_config") or {})
    output_limit=config.get("_prediction_max_abs_ps")
    threshold_rows=[]
    for sigma_max in sigma_candidates:
        gated=_gate(validation_mean,validation_sigma,sigma_max)
        if output_limit is not None:
            gated=np.clip(gated,-float(output_limit),float(output_limit))
        residual=np.asarray(validation_target,dtype=np.float64)-gated
        ctr=float(fit_ctr_ps(residual,fit_config,seed=training_seed,bootstrap=False).ctr_ps)
        threshold_rows.append({
            "sigma_max_ps":float(sigma_max),
            "validation_ctr_ps":ctr,
            "validation_rmse_ps":_rmse(gated-validation_target),
            "suppressed_fraction":float(np.mean(validation_sigma>sigma_max)),
        })
    best_threshold=min(threshold_rows,key=lambda row:(row["validation_ctr_ps"],row["sigma_max_ps"]))
    sigma_max=float(best_threshold["sigma_max_ps"])
    gated=_gate(validation_mean,validation_sigma,sigma_max)
    return HeteroscedasticCNNArtifact(model,str(device),sigma_max,{
        "best_epoch":int(best_epoch),
        "best_validation_rmse_ps":float(best_score),
        "best_validation_gated_rmse_ps":float(best_threshold["validation_rmse_ps"]),
        "best_validation_gated_ctr_ps":float(best_threshold["validation_ctr_ps"]),
        "sigma_threshold_search":threshold_rows,
        "early_stopping_metric":"validation_raw_mean_rmse",
        "loss":"heteroscedastic_gaussian_negative_log_likelihood",
        "sigma_max_ps":sigma_max,
        "validation_sigma_suppressed_fraction":float(best_threshold["suppressed_fraction"]),
        "pair_sigma_definition":"sqrt(sigma(s1)^2 + sigma(s2)^2)",
        "pair_variance_assumption":"conditional independence of single-signal noise contributions",
        "prediction_definition":"mu(s1)-mu(s2) if pair sigma <= sigma_max_ps, else 0 ps",
        "raw_mean_definition":"mu(s1)-mu(s2)",
        "detector_swap_antisymmetry_enforced":True,
        "uncertainty_swap_symmetry_enforced":True,
        "batch_size":batch,
        "output_max_abs_ps":None if output_limit is None else float(output_limit),
        "training_seed":training_seed,
        "deterministic_algorithms":True,
        "parameter_count":int(sum(p.numel() for p in model.parameters()))
    })

def predict_distribution(artifact,normalized_pair):
    return _predict_distribution(artifact.model,normalized_pair,torch.device(artifact.device),512)

def predict(artifact,normalized_pair):
    mean,sigma=predict_distribution(artifact,normalized_pair)
    return _gate(mean,sigma,artifact.sigma_max_ps)

def save(artifact,path:Path):
    path.mkdir(parents=True,exist_ok=True)
    torch.save({"state_dict":artifact.model.state_dict(),"metadata":artifact.metadata,"sigma_max_ps":artifact.sigma_max_ps},path/"model.pt")

def explain(artifact,normalized_pair):
    device=torch.device(artifact.device)
    pair=torch.tensor(np.asarray(normalized_pair,dtype=np.float32),device=device,requires_grad=True)
    artifact.model.eval(); artifact.model.zero_grad(set_to_none=True)
    mean,_,_=artifact.model.pair_distribution(pair); mean.sum().backward()
    return pair.grad.detach().abs().mean(dim=(0,1)).cpu().numpy().astype(np.float64)

MODEL_SPEC=ModelSpec(name="cnn_heteroscedastic",candidates=candidates,fit=fit,predict=predict,save=save,explain=explain)
