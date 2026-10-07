from __future__ import annotations
import json,os,tempfile
from dataclasses import dataclass
from pathlib import Path
import numpy as np
from .common import atomic_json,canonical_hash
from .splits import cv_split_from_assignment,fold_assignment,make_cv_split,semantic_seed
from .stats import metric_values
from .view import model_target
CV_ARTIFACT_VERSION=1
@dataclass(frozen=True)
class SharedCVArtifacts:
    directory:Path;split:object;fold_metrics:dict[int,dict];fingerprint:str
def _atomic_npz(path,**arrays):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);fd,tmp=tempfile.mkstemp(prefix=f".{path.name}.",suffix=".tmp",dir=path.parent)
    try:
        with os.fdopen(fd,"wb") as stream:np.savez_compressed(stream,**arrays);stream.flush();os.fsync(stream.fileno())
        os.replace(tmp,path)
    finally:
        if os.path.exists(tmp):os.unlink(tmp)
class ExperimentArtifactStore:
    def __init__(self,root):self.root=Path(root).resolve();self.root.mkdir(parents=True,exist_ok=True)
    def prepare_cv(self,dataset,config):
        cv=config["cross_validation"];identity=str(dataset.manifest["event_population_identity"]);fp=canonical_hash({"version":CV_ARTIFACT_VERSION,"event_population_identity":identity,"mode":config["mode"],"window":config["window_ns"],"seed":int(config["seed"]),"folds":int(cv["folds"]),"shuffle":bool(cv["shuffle"]),"fit":config["fit"]});mode="energy" if config["mode"]=="energy_to_energy" else "timing";window=str(config.get("window_name") or "window");directory=self.root/mode/window/fp[:16];manifest_path=directory/"manifest.json";split_path=directory/"folds.npz";metrics_path=directory/"led_folds.json"
        if manifest_path.is_file() and split_path.is_file() and metrics_path.is_file():
            manifest=json.loads(manifest_path.read_text(encoding="utf-8"))
            if manifest.get("fingerprint")==fp:
                with np.load(split_path) as data:split=cv_split_from_assignment(np.asarray(data["assignment"],dtype=np.int64),seed=int(data["seed"]),shuffle=bool(int(data["shuffle"])))
                metrics=json.loads(metrics_path.read_text(encoding="utf-8"));return SharedCVArtifacts(directory,split,{int(k):v for k,v in metrics.items()},fp)
        directory.mkdir(parents=True,exist_ok=True);split=make_cv_split(dataset.n_events,population_identity=identity,batch_seed=int(config["seed"]),n_folds=int(cv["folds"]),shuffle=bool(cv["shuffle"]));minimum=int(cv["minimum_events_per_fold"]);too_small=[f.fold_id for f in split.folds if len(f.validation)<minimum]
        if too_small:raise RuntimeError(f"CV validation folds {too_small} violate minimum_events_per_fold={minimum}")
        target=np.asarray(model_target(dataset,config["mode"]),dtype=np.float64);metrics={}
        for fold in split.folds:
            val=np.asarray(fold.validation,dtype=np.int64);score=metric_values(target[val],config["fit"],seed=semantic_seed(split.seed,"led",int(fold.fold_id)));metrics[int(fold.fold_id)]={"fold_id":int(fold.fold_id),"n_validation":int(val.size),"led_ctr_ps":float(score["ctr_ps"]),"led_rmse_ps":float(score["rmse_ps"])}
        _atomic_npz(split_path,assignment=fold_assignment(split),seed=np.asarray(split.seed,dtype=np.int64),shuffle=np.asarray(int(split.shuffle),dtype=np.int8));atomic_json(metrics_path,{str(k):v for k,v in metrics.items()});atomic_json(manifest_path,{"schema_version":CV_ARTIFACT_VERSION,"fingerprint":fp,"event_population_identity":identity,"mode":config["mode"],"window_name":config.get("window_name"),"window_ns":config["window_ns"],"folds":int(cv["folds"]),"shuffle":bool(cv["shuffle"]),"seed":int(split.seed),"n_events":int(dataset.n_events),"minimum_events_per_fold":minimum});return SharedCVArtifacts(directory,split,metrics,fp)
