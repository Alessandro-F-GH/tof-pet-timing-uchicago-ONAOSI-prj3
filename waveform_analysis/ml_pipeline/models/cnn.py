from __future__ import annotations
import copy,itertools
from dataclasses import dataclass
from pathlib import Path
from typing import Any
import numpy as np, torch
from torch import nn
from torch.utils.data import DataLoader,TensorDataset
from ..stats import ctr_fwhm
from .spec import ModelSpec

class SharedScorerCNN(nn.Module):
    def __init__(self,architecture:dict[str,Any]):
        super().__init__(); channels=[int(v) for v in architecture.get("channels",[16,32,64])]; kernels=[int(v) for v in architecture.get("kernels",[9,7,5])]; strides=[int(v) for v in architecture.get("strides",[2,2,2])]; dilations=[int(v) for v in architecture.get("dilations",[1,1,1])]
        if not(len(channels)==len(kernels)==len(strides)==len(dilations)): raise ValueError("CNN channels/kernels/strides/dilations must have equal length")
        pool_length=int(architecture.get("adaptive_pool_length",128)); pooling=str(architecture.get("pooling","avg_max")).lower()
        if pool_length<1: raise ValueError("adaptive_pool_length must be >= 1")
        if pooling not in {"avg","max","avg_max"}: raise ValueError("CNN pooling must be avg, max, or avg_max")
        layers=[]; incoming=1
        for outgoing,kernel,stride,dilation in zip(channels,kernels,strides,dilations):
            padding=dilation*(kernel-1)//2; layers.extend([nn.Conv1d(incoming,outgoing,kernel,stride=stride,dilation=dilation,padding=padding),nn.BatchNorm1d(outgoing),nn.SiLU()]); incoming=outgoing
        self.features=nn.Sequential(*layers); self.avg_pool=nn.AdaptiveAvgPool1d(pool_length) if pooling in {"avg","avg_max"} else None; self.max_pool=nn.AdaptiveMaxPool1d(pool_length) if pooling in {"max","avg_max"} else None
        incoming*=pool_length*(2 if pooling=="avg_max" else 1); head=[nn.Flatten()]
        for width in [int(v) for v in architecture.get("dense_units",[32])]: head.extend([nn.Linear(incoming,width),nn.SiLU()]); incoming=width
        head.append(nn.Linear(incoming,1)); self.head=nn.Sequential(*head)
    def score(self,waveform):
        features=self.features(waveform[:,None,:]); pooled=[]
        if self.avg_pool is not None: pooled.append(self.avg_pool(features))
        if self.max_pool is not None: pooled.append(self.max_pool(features))
        return self.head(torch.cat(pooled,dim=1) if len(pooled)>1 else pooled[0]).squeeze(1)
    def forward(self,pair): return self.score(pair[:,0,:])-self.score(pair[:,1,:])
@dataclass
class CNNArtifact:
    model:SharedScorerCNN
    device:str
    metadata:dict[str,Any]
def candidates(config):
    p=config.get("parameters",{}); training=config.get("training",{})
    learning_rates=p.get("learning_rate",[1e-3]); weight_decays=p.get("weight_decay",[1e-5]); batch_sizes=p.get("batch_size",[training.get("batch_size",64)])
    return [{"learning_rate":float(lr),"weight_decay":float(wd),"batch_size":int(batch)} for lr,wd,batch in itertools.product(learning_rates,weight_decays,batch_sizes)]
def _device(config):
    requested=str(config.get("training",{}).get("device","auto")).lower(); return torch.device("cuda" if torch.cuda.is_available() else "cpu") if requested=="auto" else torch.device(requested)
def _loader(x,y,batch,*,shuffle,seed):
    g=torch.Generator().manual_seed(int(seed)); return DataLoader(TensorDataset(torch.from_numpy(np.ascontiguousarray(x,dtype=np.float32)),torch.from_numpy(np.asarray(y,dtype=np.float32))),batch_size=int(batch),shuffle=shuffle,generator=g)
def _predict_tensor(model,x,device,batch):
    loader=DataLoader(torch.from_numpy(np.ascontiguousarray(x,dtype=np.float32)),batch_size=int(batch),shuffle=False); values=[]; model.eval()
    with torch.no_grad():
        for pair in loader: values.append(model(pair.to(device)).detach().cpu().numpy())
    return np.concatenate(values).astype(np.float64,copy=False)
def _validation_score(residual,metric,fit_config):
    values=np.asarray(residual,dtype=np.float64); metric=str(metric).lower()
    if metric=="validation_mse": return float(np.mean(values**2))
    if metric=="validation_rmse": return float(np.sqrt(np.mean(values**2)))
    if metric in {"validation_ctr","ctr"}: return float(ctr_fwhm(values,fit_config).ctr_ps)
    raise ValueError(f"Unsupported CNN selection_metric: {metric}")
def fit(params,train_x,train_target,*,seed,config,validation_x=None,validation_target=None,final_epochs=None):
    torch.manual_seed(int(seed)); np.random.seed(int(seed));
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(int(seed))
    training=config.get("training",{}); device=_device(config); model=SharedScorerCNN(config.get("architecture",{})).to(device); optimizer=torch.optim.AdamW(model.parameters(),lr=float(params["learning_rate"]),weight_decay=float(params["weight_decay"])); loss_fn=nn.MSELoss(); batch=int(params.get("batch_size",training.get("batch_size",64))); max_epochs=int(final_epochs or training.get("epochs",350)); patience=int(training.get("patience",30)); metric=str(training.get("selection_metric","validation_ctr")).lower(); min_delta=float(training.get("min_delta",training.get("min_delta_ps",.05))); loader=_loader(train_x,train_target,batch,shuffle=True,seed=seed); best_score=float("inf"); best_epoch=max_epochs; best_state=None; stale=0
    for epoch in range(1,max_epochs+1):
        model.train()
        for pair,target in loader:
            pair=pair.to(device); target=target.to(device); optimizer.zero_grad(set_to_none=True); loss=loss_fn(model(pair),target); loss.backward(); clip=float(training.get("gradient_clip_norm",10.0));
            if clip>0: nn.utils.clip_grad_norm_(model.parameters(),clip)
            optimizer.step()
        if validation_x is None or validation_target is None or final_epochs is not None: continue
        prediction=_predict_tensor(model,validation_x,device,batch); residual=prediction-np.asarray(validation_target); score=_validation_score(residual,metric,config.get("fit"))
        if score<best_score-min_delta: best_score=score; best_epoch=epoch; best_state=copy.deepcopy(model.state_dict()); stale=0
        else:
            stale+=1
            if stale>=patience: break
    if best_state is not None: model.load_state_dict(best_state)
    return CNNArtifact(model,str(device),{"best_epoch":int(best_epoch),"best_validation_score":float(best_score),"selection_metric":metric,"batch_size":batch})
def predict(artifact,normalized_pair): return _predict_tensor(artifact.model,normalized_pair,torch.device(artifact.device),512)
def save(artifact,path:Path): path.mkdir(parents=True,exist_ok=True); torch.save({"state_dict":artifact.model.state_dict(),"metadata":artifact.metadata},path/"model.pt")
def explain(artifact,normalized_pair):
    device=torch.device(artifact.device); pair=torch.tensor(np.asarray(normalized_pair,dtype=np.float32),device=device,requires_grad=True); artifact.model.zero_grad(set_to_none=True); artifact.model(pair).sum().backward(); return pair.grad.detach().abs().mean(dim=(0,1)).cpu().numpy().astype(np.float64)
MODEL_SPEC=ModelSpec(name="cnn",candidates=candidates,fit=fit,predict=predict,save=save,explain=explain)
