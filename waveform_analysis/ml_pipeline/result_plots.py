from __future__ import annotations
from pathlib import Path
import csv
import numpy as np
import matplotlib.pyplot as plt


def _save(fig,path):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);fig.tight_layout();fig.savefig(path);plt.close(fig);return path


def _blind_rows(rows):
    return [r for r in rows if r.get("stage")=="blind"]


def plot_blind_ctr_distribution(rows,path,title="Blind CTR distribution"):
    blind=_blind_rows(rows)
    values=np.asarray([float(r["ctr_ps"]) for r in blind],float)
    values=values[np.isfinite(values)]
    if not values.size:return None
    fig,ax=plt.subplots()
    bins=max(5,min(20,int(np.ceil(np.sqrt(values.size)))))
    ax.hist(values,bins=bins,histtype="stepfilled",alpha=.45)
    mean=float(np.mean(values));std=float(np.std(values,ddof=1)) if values.size>1 else 0.0
    ax.axvline(mean,linestyle="--",label=f"mean = {mean:.2f} ps")
    ax.set_xlabel("Blind CTR [ps]");ax.set_ylabel("Repeated-holdout runs");ax.set_title(title)
    ax.legend(title=f"n={values.size}, std={std:.2f} ps")
    return _save(fig,path)


def _paired_bootstrap_values(run_dir):
    directory=Path(run_dir)/"paired_bootstrap"
    arrays=[]
    if directory.is_dir():
        for path in sorted(directory.glob("*.npz")):
            with np.load(path) as data:
                values=np.asarray(data["improvement_ps"],float).reshape(-1)
            values=values[np.isfinite(values)]
            if values.size:arrays.append(values)
    return np.concatenate(arrays) if arrays else np.empty(0,float)


def plot_paired_improvement_bootstrap(run_dir,path,title="Paired LED-to-ML bootstrap improvement"):
    values=_paired_bootstrap_values(run_dir)
    if not values.size:return None
    fig,ax=plt.subplots()
    bins=max(10,min(60,int(np.ceil(np.sqrt(values.size)))))
    ax.hist(values,bins=bins,histtype="stepfilled",alpha=.45)
    mean=float(np.mean(values));std=float(np.std(values,ddof=1)) if values.size>1 else 0.0
    q025,q975=np.quantile(values,[.025,.975])
    ax.axvline(0.0,linestyle=":",label="no improvement")
    ax.axvline(mean,linestyle="--",label=f"mean = {mean:.2f} ps")
    ax.set_xlabel(r"CTR improvement, LED - ML [ps]");ax.set_ylabel("Paired bootstrap samples");ax.set_title(title)
    ax.legend(title=f"n={values.size}, std={std:.2f} ps\n95% interval [{q025:.2f}, {q975:.2f}] ps")
    return _save(fig,path)


def write_blind_summary(rows,path):
    blind=_blind_rows(rows)
    if not blind:return None
    ctr=np.asarray([float(r["ctr_ps"]) for r in blind],float)
    improvement=np.asarray([float(r.get("improvement_ps","nan")) for r in blind],float)
    percent=np.asarray([float(r.get("improvement_percent","nan")) for r in blind],float)
    def stats(values):
        values=values[np.isfinite(values)]
        return (float(np.mean(values)) if values.size else float("nan"),
                float(np.std(values,ddof=1)) if values.size>1 else 0.0,
                int(values.size))
    ctr_mean,ctr_std,n_ctr=stats(ctr);imp_mean,imp_std,n_imp=stats(improvement);pct_mean,pct_std,n_pct=stats(percent)
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open("w",encoding="utf-8",newline="") as stream:
        writer=csv.DictWriter(stream,fieldnames=["metric","mean","std","n"]);writer.writeheader()
        writer.writerow({"metric":"blind_ctr_ps","mean":ctr_mean,"std":ctr_std,"n":n_ctr})
        writer.writerow({"metric":"paired_improvement_ps","mean":imp_mean,"std":imp_std,"n":n_imp})
        writer.writerow({"metric":"paired_improvement_percent","mean":pct_mean,"std":pct_std,"n":n_pct})
    return path


def make_study_result_plots(rows,run_dir,*,model,mode,window_ns):
    run_dir=Path(run_dir);label=f"{model} | {mode} | [{window_ns['start']}, {window_ns['end']}] ns"
    return {
        "blind_ctr":plot_blind_ctr_distribution(rows,run_dir/"blind_ctr_distribution.png",f"Blind CTR distribution\n{label}"),
        "paired_improvement":plot_paired_improvement_bootstrap(run_dir,run_dir/"paired_led_improvement_bootstrap_distribution.png",f"Paired LED-to-ML bootstrap improvement\n{label}"),
        "summary":write_blind_summary(rows,run_dir/"blind_summary.csv"),
    }
