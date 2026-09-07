from __future__ import annotations
import itertools
from dataclasses import dataclass
from pathlib import Path
from typing import Any
import joblib, numpy as np
from sklearn.svm import LinearSVR
from .spec import ModelSpec
@dataclass
class LinearSVRArtifact:
    model:LinearSVR
    metadata:dict[str,Any]
def candidates(config):
    p=config.get("parameters",{}); return [{"C":float(c),"epsilon_ps":float(e)} for c,e in itertools.product(p.get("C",[1.,100.]),p.get("epsilon_ps",[10.,60.]))]
def fit(params,train_x,train_target,*,seed,config,validation_x=None,validation_target=None,final_epochs=None):
    del validation_x,validation_target,final_epochs; x=np.asarray(train_x,dtype=float)
    if x.ndim!=3 or x.shape[1]!=2: raise ValueError("Linear SVR expects [event, detector, sample]")
    model=LinearSVR(C=float(params["C"]),epsilon=float(params["epsilon_ps"]),fit_intercept=False,loss=str(config.get("loss","epsilon_insensitive")),tol=float(config.get("tolerance",1e-3)),max_iter=int(config.get("max_iterations",10000)),dual=config.get("dual","auto"),random_state=int(seed)); model.fit(x[:,0,:]-x[:,1,:],np.asarray(train_target,dtype=float)); return LinearSVRArtifact(model,{})
def predict(artifact,normalized_pair):
    x=np.asarray(normalized_pair,dtype=float); return np.asarray(artifact.model.predict(x[:,0,:]-x[:,1,:]),dtype=float)
def save(artifact,path:Path): path.mkdir(parents=True,exist_ok=True); joblib.dump(artifact.model,path/"model.joblib")
def explain(artifact,normalized_pair): del normalized_pair; return np.abs(np.asarray(artifact.model.coef_,dtype=float))
MODEL_SPEC=ModelSpec(name="linear_svr",candidates=candidates,fit=fit,predict=predict,save=save,explain=explain)
