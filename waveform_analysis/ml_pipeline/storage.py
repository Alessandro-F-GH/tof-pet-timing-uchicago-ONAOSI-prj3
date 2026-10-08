from __future__ import annotations
import csv,json,os,shutil,tempfile
from pathlib import Path
import numpy as np
from .common import atomic_json,write_csv
RUN_SCHEMA_VERSION=51
STAGE_ORDER=("cv","selection","final_fit","blind","bootstrap","xai","plots")
def _read_csv(path):
    if not Path(path).is_file(): return []
    with Path(path).open("r",encoding="utf-8",newline="") as stream:return list(csv.DictReader(stream))
def _atomic_npz(path,**arrays):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);fd,tmp=tempfile.mkstemp(prefix=f".{path.name}.",suffix=".tmp",dir=path.parent)
    try:
        with os.fdopen(fd,"wb") as stream: np.savez_compressed(stream,**arrays);stream.flush();os.fsync(stream.fileno())
        os.replace(tmp,path)
    finally:
        if os.path.exists(tmp):os.unlink(tmp)
class RunStore:
    def __init__(self,root):
        self.root=Path(root).resolve();self.root.mkdir(parents=True,exist_ok=True);self._organize_existing_layout()
    def _organize_existing_layout(self):
        moves={
            **{f"{name}.json":self.root/"metadata"/f"{name}.json" for name in ['manifest','state','config','candidates','best','final_fit','blind','bootstrap']},
            "folds.csv":self.root/"tables"/"folds.csv",
            "cv.csv":self.root/"tables"/"cv.csv",
            "pred.npz":self.root/"artifacts"/"pred.npz",
            "bootstrap.npz":self.root/"artifacts"/"bootstrap.npz",
            "xai.npz":self.root/"artifacts"/"xai.npz",
        }
        for name,destination in moves.items():
            source=self.root/name
            if source.is_file() and not destination.exists():
                destination.parent.mkdir(parents=True,exist_ok=True);shutil.move(str(source),str(destination))
    @property
    def metadata_dir(self):return self.root/"metadata"
    @property
    def manifest_path(self):return self.metadata_dir/"manifest.json"
    @property
    def state_path(self):return self.metadata_dir/"state.json"
    @property
    def tables_dir(self):return self.root/"tables"
    @property
    def artifacts_dir(self):return self.root/"artifacts"
    @property
    def folds_path(self):return self.tables_dir/"folds.csv"
    @property
    def cv_path(self):return self.tables_dir/"cv.csv"
    @property
    def candidates_path(self):return self.metadata_dir/"candidates.json"
    @property
    def best_path(self):return self.metadata_dir/"best.json"
    @property
    def final_fit_path(self):return self.metadata_dir/"final_fit.json"
    @property
    def blind_path(self):return self.metadata_dir/"blind.json"
    @property
    def bootstrap_path(self):return self.metadata_dir/"bootstrap.json"
    @property
    def bootstrap_draws_path(self):return self.artifacts_dir/"bootstrap.npz"
    @property
    def predictions_path(self):return self.artifacts_dir/"pred.npz"
    @property
    def xai_path(self):return self.artifacts_dir/"xai.npz"
    @property
    def model_dir(self):return self.root/"model"
    @property
    def plots_dir(self):return self.root/"plots"
    def write_manifest(self,v):atomic_json(self.manifest_path,v)
    def write_resolved_config(self,v):atomic_json(self.metadata_dir/"config.json",v)
    def write_candidates(self,v):atomic_json(self.candidates_path,v)
    def write_best(self,v):atomic_json(self.best_path,v)
    def write_final_fit(self,v):atomic_json(self.final_fit_path,v)
    def write_blind(self,v):atomic_json(self.blind_path,v)
    def write_bootstrap(self,v):atomic_json(self.bootstrap_path,v)
    def read_json(self,path,default=None):
        try:return json.loads(Path(path).read_text(encoding="utf-8"))
        except (OSError,json.JSONDecodeError):return default
    def read_state(self):
        v=self.read_json(self.state_path,{})
        if not isinstance(v,dict) or int(v.get("schema_version",-1))!=RUN_SCHEMA_VERSION:return {"schema_version":RUN_SCHEMA_VERSION,"stages":{}}
        v.setdefault("stages",{});return v
    def stage_status(self,stage,fingerprint):
        item=self.read_state()["stages"].get(str(stage))
        if not isinstance(item,dict):return "missing"
        if item.get("fingerprint")!=str(fingerprint):return "stale"
        return "complete" if item.get("status")=="complete" else "incomplete"
    def mark_stage(self,stage,fingerprint,*,status="complete",metadata=None):
        state=self.read_state();state["stages"][stage]={"status":str(status),"fingerprint":str(fingerprint),"metadata":dict(metadata or {})};atomic_json(self.state_path,state)
    def invalidate_from(self,stage):
        graph={"cv":("cv","selection","final_fit","blind","bootstrap","xai","plots"),"selection":("selection","final_fit","blind","bootstrap","xai","plots"),"final_fit":("final_fit","blind","bootstrap","xai","plots"),"blind":("blind","bootstrap","xai","plots"),"bootstrap":("bootstrap","plots"),"xai":("xai","plots"),"plots":("plots",)}
        files={"cv":[self.folds_path,self.cv_path,self.candidates_path,self.root/"optuna.db",self.root/"optuna_sampler.pkl"],"selection":[self.best_path],"final_fit":[self.final_fit_path,self.model_dir],"blind":[self.blind_path,self.predictions_path],"bootstrap":[self.bootstrap_path,self.bootstrap_draws_path],"xai":[self.xai_path],"plots":[self.plots_dir]}
        for current in graph[stage]:
            for path in files[current]:
                if path.is_dir():shutil.rmtree(path,ignore_errors=True)
                elif path.exists():path.unlink()
        state=self.read_state()
        for current in graph[stage]:state["stages"].pop(current,None)
        atomic_json(self.state_path,state)
    def read_fold_rows(self,candidate_id=None):
        rows=_read_csv(self.folds_path);return rows if candidate_id is None else [r for r in rows if r.get("candidate_id")==str(candidate_id)]
    def upsert_fold(self,row):
        key=(str(row["candidate_id"]),int(row["fold_id"]));rows=[r for r in _read_csv(self.folds_path) if (str(r.get("candidate_id")),int(r.get("fold_id",-1)))!=key];rows.append(dict(row));rows.sort(key=lambda r:(str(r.get("candidate_id","")),int(r.get("fold_id",0))));write_csv(self.folds_path,rows)
    def read_candidate_rows(self):return _read_csv(self.cv_path)
    def candidate_row(self,candidate_id):return next((r for r in self.read_candidate_rows() if r.get("candidate_id")==str(candidate_id)),None)
    def upsert_candidate(self,row):
        rows=[r for r in self.read_candidate_rows() if r.get("candidate_id")!=str(row["candidate_id"])];rows.append(dict(row));rows.sort(key=lambda r:str(r.get("candidate_id","")));write_csv(self.cv_path,rows)
    def save_predictions(self,*,event_id,prediction_ps,corrected_ps,led_residual_ps):
        _atomic_npz(self.predictions_path,event_id=np.asarray(event_id,dtype=np.int64),prediction_ps=np.asarray(prediction_ps,dtype=np.float64),corrected_ps=np.asarray(corrected_ps,dtype=np.float64),led_residual_ps=np.asarray(led_residual_ps,dtype=np.float64));return self.predictions_path
    def load_predictions(self):
        with np.load(self.predictions_path) as data:return {k:np.asarray(data[k]) for k in data.files}
    def save_bootstrap_draws(self,draws):_atomic_npz(self.bootstrap_draws_path,**{k:np.asarray(v,dtype=np.float64) for k,v in draws.items()})
    def save_xai(self,**arrays):_atomic_npz(self.xai_path,**arrays);return self.xai_path
    def compatible_schema(self):return int((self.read_json(self.manifest_path,{}) or {}).get("schema_version",-1))==RUN_SCHEMA_VERSION
