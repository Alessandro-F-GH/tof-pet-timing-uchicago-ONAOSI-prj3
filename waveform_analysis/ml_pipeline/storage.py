from __future__ import annotations

import csv
import os
import shutil
import tempfile
from pathlib import Path

import numpy as np

from .common import atomic_json

RESULT_FIELDS=("seed","stage","model","estimator_formulation","mode","window_start_ns","window_end_ns","population_identity","event_population_identity","analysis_protocol_identity","candidate_id","selected","ctr_ps","uncorrected_ctr_ps","improvement_ps","improvement_percent","rmse_ps","uncorrected_rmse_ps","rmse_improvement_ps","rmse_improvement_percent","n","swap_rmse_ps")

class RunStore:
    def __init__(self,root,*,overwrite=False,resume=False):
        self.root=Path(root).resolve()
        if overwrite and resume:raise ValueError("overwrite and resume are mutually exclusive")
        if overwrite and self.root.exists():shutil.rmtree(self.root)
        if self.root.exists() and any(self.root.iterdir()) and not resume:raise FileExistsError(f"Run directory is not empty: {self.root}")
        self.root.mkdir(parents=True,exist_ok=True);self.resume=bool(resume)
    @property
    def results_path(self):return self.root/"results.csv"
    def write_manifest(self,value):atomic_json(self.root/"manifest.json",value)
    def write_resolved_config(self,value):atomic_json(self.root/"resolved_config.json",value)
    def write_candidates(self,value):atomic_json(self.root/"candidates.json",value)
    def read_results(self):
        if not self.results_path.is_file():return []
        with self.results_path.open("r",encoding="utf-8",newline="") as stream:return list(csv.DictReader(stream))
    def _atomic_rows(self,rows):
        fd,tmp=tempfile.mkstemp(prefix=".results.",suffix=".csv",dir=self.root)
        try:
            with os.fdopen(fd,"w",encoding="utf-8",newline="") as stream:
                writer=csv.DictWriter(stream,fieldnames=RESULT_FIELDS);writer.writeheader()
                for row in rows:writer.writerow({key:row.get(key,"") for key in RESULT_FIELDS})
            os.replace(tmp,self.results_path)
        finally:
            if os.path.exists(tmp):os.unlink(tmp)
    def upsert_result(self,row):
        rows=self.read_results();key=(str(row["seed"]),str(row["stage"]),str(row["candidate_id"]))
        rows=[r for r in rows if (str(r.get("seed")),str(r.get("stage")),str(r.get("candidate_id")))!=key];rows.append(dict(row));rows.sort(key=lambda r:(int(r["seed"]),0 if r["stage"]=="validation" else 1,str(r["candidate_id"])));self._atomic_rows(rows)
    def has_result(self,seed,stage,candidate_id):return any(str(r.get("seed"))==str(seed) and r.get("stage")==stage and r.get("candidate_id")==candidate_id for r in self.read_results())
    def save_split(self,seed,event_index,split):
        directory=self.root/"splits";directory.mkdir(exist_ok=True);np.savez_compressed(directory/f"seed_{int(seed)}.npz",train=split.train,validation=split.validation,test=split.test)
    def blind_residuals_path(self,seed,candidate_id):return self.root/"blind_residuals"/f"seed_{int(seed)}_{candidate_id}.npz"
    def save_blind_residuals(self,seed,candidate_id,corrected_ps):
        path=self.blind_residuals_path(seed,candidate_id);path.parent.mkdir(exist_ok=True);np.savez_compressed(path,corrected_ps=np.asarray(corrected_ps,np.float64));return path
    def model_dir(self,seed,candidate_id):
        path=self.root/"models"/f"seed_{int(seed)}"/candidate_id;path.mkdir(parents=True,exist_ok=True);return path
