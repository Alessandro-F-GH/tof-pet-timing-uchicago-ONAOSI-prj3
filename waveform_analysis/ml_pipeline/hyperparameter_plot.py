from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

def _varied(candidates):
    keys=[]
    for c in candidates.values():
        for k in c:
            if k not in keys:keys.append(k)
    return [k for k in keys if len({json.dumps(c.get(k),sort_keys=True) for c in candidates.values()})>1]

def _numeric(values):
    return all(isinstance(v,(int,float,np.integer,np.floating)) and np.isfinite(float(v)) for v in values)

def plot_hyperparameter_validation(results,candidates,output_path,logger=None,*,metric="ctr_ps",metric_label="CTR"):
    varied=_varied(candidates)
    if not varied:return None
    rows=[r for r in results if r.get("stage")=="validation"]
    if not rows:return None
    scores={}
    for cid in candidates:
        v=[float(r[metric]) for r in rows if r.get("candidate_id")==cid and str(r.get(metric,"")) not in ("","nan")]
        if v:scores[cid]=float(np.mean(v))
    if not scores:return None
    numeric=[k for k in varied if _numeric([candidates[c].get(k) for c in candidates])]
    if numeric:
        counts={k:len({candidates[c].get(k) for c in candidates}) for k in numeric}
        xkey=sorted(numeric,key=lambda k:(-counts[k],varied.index(k)))[0]
    else:xkey=varied[0]
    series_keys=[k for k in varied if k!=xkey]
    groups={}
    for cid,score in scores.items():
        c=candidates[cid];label=", ".join(f"{k}={c.get(k)}" for k in series_keys) if series_keys else "candidates"
        groups.setdefault(label,[]).append((c.get(xkey),score,cid))
    fig,ax=plt.subplots()
    for label,vals in groups.items():
        vals=sorted(vals,key=lambda x:(float(x[0]) if isinstance(x[0],(int,float)) else str(x[0]),x[2]))
        ax.plot([v[0] for v in vals],[v[1] for v in vals],marker="o",label=label if series_keys else None)
    ax.set_xlabel(xkey);ax.set_ylabel(f"Mean validation {metric_label} [ps]")
    if series_keys:ax.legend()
    xs=[candidates[c].get(xkey) for c in scores]
    if _numeric(xs):
        vals=np.asarray(xs,float)
        if np.all(vals>0):ax.set_xscale("log")
        elif logger:logger.warning("Hyperparameter %s contains zero/negative values; using linear x scale",xkey)
    fig.tight_layout();output_path=Path(output_path);output_path.parent.mkdir(parents=True,exist_ok=True);fig.savefig(output_path);plt.close(fig)
    return output_path
