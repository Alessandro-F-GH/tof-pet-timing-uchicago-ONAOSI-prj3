from __future__ import annotations
import csv, os, shutil, tempfile
from pathlib import Path
import numpy as np
from .common import atomic_json

class RunStore:
    def __init__(self,root:str|Path,*,overwrite:bool=False):
        self.root=Path(root).resolve()
        if overwrite and self.root.exists(): shutil.rmtree(self.root)
        if self.root.exists() and any(self.root.iterdir()): raise FileExistsError(f"Run directory is not empty: {self.root}. Use --overwrite for a new run.")
        self.root.mkdir(parents=True,exist_ok=True)
        for name in ("models","artifacts","search","splits"): (self.root/name).mkdir(exist_ok=True)
    def write_manifest(self,value): atomic_json(self.root/"manifest.json",value)
    def write_results(self,rows):
        fields=list(dict.fromkeys(key for row in rows for key in row)); fd,tmp=tempfile.mkstemp(prefix=".results.",suffix=".csv",dir=self.root)
        try:
            with os.fdopen(fd,"w",encoding="utf-8",newline="") as stream:
                if fields:
                    writer=csv.DictWriter(stream,fieldnames=fields); writer.writeheader(); writer.writerows(rows)
            os.replace(tmp,self.root/"results.csv")
        finally:
            if os.path.exists(tmp): os.unlink(tmp)
    def save_split(self,dataset,prepared):
        target=self.root/"splits"/f"{dataset}.npz"; np.savez_compressed(target,development=np.asarray(prepared.development,dtype=np.int64),training=np.asarray(prepared.training,dtype=np.int64),validation=np.asarray(prepared.validation,dtype=np.int64),test=np.asarray(prepared.test,dtype=np.int64)); return target
    def save_search(self,dataset,name,value):
        target=self.root/"search"/dataset/f"{name}.json"; atomic_json(target,value); return target
    def model_dir(self,dataset,model):
        target=self.root/"models"/dataset/model; target.mkdir(parents=True,exist_ok=True); return target
    def save_residuals(self,dataset,method,values,*,stage="test"):
        target=self.root/"artifacts"/dataset/f"{method}_{stage}_residuals_ps.npy"; target.parent.mkdir(parents=True,exist_ok=True); np.save(target,np.asarray(values,dtype=np.float64)); return target
    def save_model_output(self,dataset,model,values,*,stage="test"):
        """Persist the actual learned correction y_theta used downstream."""
        target=self.root/"artifacts"/dataset/f"{model}_{stage}_model_output_ps.npy"; target.parent.mkdir(parents=True,exist_ok=True); np.save(target,np.asarray(values,dtype=np.float64)); return target
    def save_xai(self,dataset,model,*,time_ps,importance,example_pair_mV):
        target=self.root/"artifacts"/dataset/f"{model}_xai.npz"; target.parent.mkdir(parents=True,exist_ok=True); np.savez_compressed(target,time_ps=np.asarray(time_ps),importance=np.asarray(importance),example_pair_mV=np.asarray(example_pair_mV,dtype=np.float32)); return target
