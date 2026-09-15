from __future__ import annotations
import copy,itertools,os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np, torch
from torch import nn
from torch.utils.data import DataLoader,TensorDataset
from .spec import ModelSpec

def _dense_head(dense_units):
    widths=[int(v) for v in dense_units]
    layers=[nn.Flatten()]
    if widths:
        layers.extend([nn.LazyLinear(widths[0]),nn.SiLU()])
        for incoming,outgoing in zip(widths,widths[1:]):
            layers.extend([nn.Linear(incoming,outgoing),nn.SiLU()])
        layers.append(nn.Linear(widths[-1],1))
    else:
        layers.append(nn.LazyLinear(1))
    return nn.Sequential(*layers)

class SharedScorerCNN(nn.Module):
    def __init__(self,architecture:dict[str,Any]):
        super().__init__()
        channels=[int(v) for v in architecture.get("channels",[16,32,64])]
        kernels=[int(v) for v in architecture.get("kernels",[9,7,5])]
        strides=[int(v) for v in architecture.get("strides",[2,2,2])]
        dilations=[int(v) for v in architecture.get("dilations",[1,1,1])]
        if not channels or not(len(channels)==len(kernels)==len(strides)==len(dilations)):
            raise ValueError("CNN channels/kernels/strides/dilations must be non-empty and have equal length")
        layers=[]; incoming=1
        for outgoing,kernel,stride,dilation in zip(channels,kernels,strides,dilations):
            padding=dilation*(kernel-1)//2
            layers.extend([nn.Conv1d(incoming,outgoing,kernel,stride=stride,dilation=dilation,padding=padding),nn.SiLU()])
            incoming=outgoing
        self.features=nn.Sequential(*layers)
        self.head=_dense_head(architecture.get("dense_units",[32]))
    def score(self,waveform):
        return self.head(self.features(waveform[:,None,:])).squeeze(1)
    def forward(self,pair):
        return self.score(pair[:,0,:])-self.score(pair[:,1,:])

@dataclass
class CNNArtifact:
    model:SharedScorerCNN
    device:str
    metadata:dict[str,Any]
def candidates(config):
    p=config.get("parameters",{}); training=config.get("training",{})
    return [
        {"learning_rate":float(lr),"weight_decay":float(wd),"batch_size":int(batch)}
        for lr,wd,batch in itertools.product(
            p.get("learning_rate",[1e-3]),
            p.get("weight_decay",[1e-5]),
            p.get("batch_size",[training.get("batch_size",64)]),
        )
    ]
def _configure_reproducibility(seed):
    value=int(seed); np.random.seed(value); torch.manual_seed(value)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(value)
    torch.use_deterministic_algorithms(True)
    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.benchmark=False; torch.backends.cudnn.deterministic=True
    return value

def _device(config):
    requested=str(config.get("training",{}).get("device","auto")).lower(); return torch.device("cuda" if torch.cuda.is_available() else "cpu") if requested=="auto" else torch.device(requested)
def _loader(x,y,batch,*,shuffle,seed):
    g=torch.Generator().manual_seed(int(seed)); return DataLoader(TensorDataset(torch.from_numpy(np.ascontiguousarray(x,dtype=np.float32)),torch.from_numpy(np.asarray(y,dtype=np.float32))),batch_size=int(batch),shuffle=shuffle,generator=g)
def _predict_tensor(model,x,device,batch):
    loader=DataLoader(torch.from_numpy(np.ascontiguousarray(x,dtype=np.float32)),batch_size=int(batch),shuffle=False); values=[]; model.eval()
    with torch.no_grad():
        for pair in loader: values.append(model(pair.to(device)).detach().cpu().numpy())
    return np.concatenate(values).astype(np.float64,copy=False)
def _rmse(residual):
    values=np.asarray(residual,dtype=np.float64); return float(np.sqrt(np.mean(values**2)))
def _rmse_loss(prediction,target):
    return torch.sqrt(torch.mean((prediction-target)**2))

def _gradient_norm(model):
    total=0.0
    for parameter in model.parameters():
        if parameter.grad is not None:
            value=parameter.grad.detach().norm(2).item()
            total+=value*value
    return float(total**0.5)

def _internal_early_stopping_split(x,y,fraction,seed):
    x=np.asarray(x,dtype=np.float32); y=np.asarray(y,dtype=np.float64)
    if x.shape[0]!=y.shape[0]: raise ValueError("CNN train_x and train_target must contain the same number of events")
    n=int(y.shape[0]); fraction=float(fraction)
    if not 0.0<fraction<1.0: raise ValueError("training.early_stopping_fraction must lie in (0, 1)")
    if n<2: raise ValueError("CNN training requires at least two training events")
    n_early=max(1,min(n-1,int(round(n*fraction))))
    order=np.random.default_rng(int(seed)).permutation(n)
    early_idx=order[:n_early]; fit_idx=order[n_early:]
    return x[fit_idx],y[fit_idx],x[early_idx],y[early_idx]

def fit(params,train_x,train_target,*,seed,config,validation_x=None,validation_target=None):
    training_seed=_configure_reproducibility(seed)
    training=config.get("training",{}); verbose=bool(config.get("verbose",False)); logger=config.get("_logger"); device=_device(config)
    batch=int(params.get("batch_size",training.get("batch_size",64))); max_epochs=int(training.get("epochs",350)); patience=int(training.get("patience",30)); min_delta=float(training.get("min_delta",.05)); early_fraction=float(training.get("early_stopping_fraction",0.20)); output_limit=config.get('_prediction_max_abs_ps'); clip=float(training.get("gradient_clip_norm",10.0))
    split_seed=int(config.get("_early_stopping_seed",seed))
    fit_x,fit_target,early_x,early_target=_internal_early_stopping_split(train_x,train_target,early_fraction,split_seed)
    model=SharedScorerCNN(config.get("architecture",{})).to(device)
    with torch.no_grad():
        model(torch.from_numpy(np.ascontiguousarray(fit_x[:1],dtype=np.float32)).to(device))
    optimizer=torch.optim.AdamW(model.parameters(),lr=float(params["learning_rate"]),weight_decay=float(params["weight_decay"])); loss_fn=_rmse_loss; loader=_loader(fit_x,fit_target,batch,shuffle=True,seed=training_seed)
    if verbose and logger is not None:
        logger.info("cnn training | loss=RMSE | lr=%.6g | weight_decay=%.6g | batch=%d | epochs=%d | patience=%d | min_delta=%.6g | early_stop_fraction=%.3f | fit=%d | early_stop=%d | device=%s",float(params["learning_rate"]),float(params["weight_decay"]),batch,max_epochs,patience,min_delta,early_fraction,fit_target.size,early_target.size,device)
    best_score=float("inf"); best_epoch=0; best_state=None; stale=0
    for epoch in range(1,max_epochs+1):
        model.train(); epoch_gradient_norms=[]
        for pair,target in loader:
            pair=pair.to(device); target=target.to(device); optimizer.zero_grad(set_to_none=True); loss=loss_fn(model(pair),target); loss.backward()
            if verbose and logger is not None: epoch_gradient_norms.append(_gradient_norm(model))
            if clip>0: nn.utils.clip_grad_norm_(model.parameters(),clip)
            optimizer.step()
        prediction=_predict_tensor(model,early_x,device,batch)
        if output_limit is not None: prediction=np.clip(prediction,-float(output_limit),float(output_limit))
        score=_rmse(prediction-early_target)
        if verbose and logger is not None:
            fit_prediction=_predict_tensor(model,fit_x,device,batch)
            if output_limit is not None: fit_prediction=np.clip(fit_prediction,-float(output_limit),float(output_limit))
            fit_score=_rmse(fit_prediction-fit_target)
            logger.info("cnn epoch %d/%d | fit RMSE=%.4f ps | early-stop RMSE=%.4f ps | grad norm=%.6g | pred mean=%.4f ps | pred std=%.4f ps | pred min=%.4f ps | pred max=%.4f ps",epoch,max_epochs,fit_score,score,float(np.mean(epoch_gradient_norms)) if epoch_gradient_norms else float("nan"),float(np.mean(prediction)),float(np.std(prediction)),float(np.min(prediction)),float(np.max(prediction)))
        if score<best_score-min_delta: best_score=score; best_epoch=epoch; best_state=copy.deepcopy(model.state_dict()); stale=0
        else:
            stale+=1
            if stale>=patience: break
    if best_state is None: raise RuntimeError("CNN early stopping did not produce a valid checkpoint")
    model.load_state_dict(best_state)

    return CNNArtifact(model,str(device),{"best_epoch":int(best_epoch),"training_loss":"rmse","best_early_stopping_rmse_ps":float(best_score),"early_stopping_metric":"internal_train_holdout_rmse","early_stopping_fraction":early_fraction,"early_stopping_events":int(early_target.size),"optimizer_training_events":int(fit_target.size),"training_events_available":int(len(train_target)),"training_uses_full_split":False,"refit_on_full_training_split":False,"external_validation_used_for_early_stopping":False,"early_stopping_split_seed":split_seed,"learning_rate":float(params["learning_rate"]),"weight_decay":float(params["weight_decay"]),"batch_size":batch,"output_max_abs_ps":None if output_limit is None else float(output_limit),"training_seed":training_seed,"deterministic_algorithms":True})
def predict(artifact,normalized_pair): return _predict_tensor(artifact.model,normalized_pair,torch.device(artifact.device),512)
def save(artifact,path:Path): path.mkdir(parents=True,exist_ok=True); torch.save({"state_dict":artifact.model.state_dict(),"metadata":artifact.metadata},path/"model.pt")
def explain(artifact,normalized_pair):
    device=torch.device(artifact.device); pair=torch.tensor(np.asarray(normalized_pair,dtype=np.float32),device=device,requires_grad=True); artifact.model.zero_grad(set_to_none=True); artifact.model(pair).sum().backward(); return pair.grad.detach().abs().mean(dim=(0,1)).cpu().numpy().astype(np.float64)
MODEL_SPEC=ModelSpec(name="cnn",candidates=candidates,fit=fit,predict=predict,save=save,explain=explain)
