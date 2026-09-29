from __future__ import annotations
from pathlib import Path
import csv
import numpy as np
import matplotlib.pyplot as plt


def _save(fig,path):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);fig.tight_layout();fig.savefig(path);plt.close(fig);return path


def _blind_rows(rows):
    return [r for r in rows if r.get("stage")=="blind"]


def _finite_row_values(rows,key):
    values=[]
    for row in rows:
        try:value=float(row.get(key,"nan"))
        except (TypeError,ValueError):continue
        if np.isfinite(value):values.append(value)
    return np.asarray(values,float)


def _plot_blind_distribution(rows,path,key,xlabel,title):
    values=_finite_row_values(_blind_rows(rows),key)
    if not values.size:return None
    fig,ax=plt.subplots()
    bins=max(5,min(20,int(np.ceil(np.sqrt(values.size)))))
    ax.hist(values,bins=bins,histtype="stepfilled",alpha=.45)
    mean=float(np.mean(values));std=float(np.std(values,ddof=1)) if values.size>1 else 0.0
    ax.axvline(mean,linestyle="--",label=f"mean = {mean:.2f} ps")
    ax.set_xlabel(xlabel);ax.set_ylabel("Repeated-holdout splits");ax.set_title(title)
    ax.legend(title=f"n={values.size} splits, std={std:.2f} ps")
    return _save(fig,path)


def plot_blind_ctr_distribution(rows,path,title="Blind CTR distribution"):
    return _plot_blind_distribution(rows,path,"ctr_ps","Blind CTR [ps]",title)


def plot_blind_rmse_distribution(rows,path,title="Blind RMSE distribution"):
    return _plot_blind_distribution(rows,path,"rmse_ps","Blind RMSE [ps]",title)


def plot_paired_improvement_splits(rows,path,title="Paired LED-to-ML CTR improvement across splits"):
    return _plot_blind_distribution(
        rows,path,"improvement_ps",r"CTR improvement, LED - ML [ps]",title
    )


def plot_paired_rmse_improvement_splits(rows,path,title="Paired LED-to-ML RMSE improvement across splits"):
    return _plot_blind_distribution(
        rows,path,"rmse_improvement_ps",r"RMSE improvement, LED - ML [ps]",title
    )


def plot_rmse_ctr_correlation(rows,path,title="Blind RMSE vs CTR"):
    pairs=[]
    for row in _blind_rows(rows):
        try:
            ctr=float(row.get("ctr_ps","nan"));rmse=float(row.get("rmse_ps","nan"))
        except (TypeError,ValueError):continue
        if np.isfinite(ctr) and np.isfinite(rmse):pairs.append((rmse,ctr))
    if not pairs:return None
    values=np.asarray(pairs,float);rmse=values[:,0];ctr=values[:,1]
    correlation=float(np.corrcoef(rmse,ctr)[0,1]) if values.shape[0]>1 and np.std(rmse)>0 and np.std(ctr)>0 else float("nan")
    fig,ax=plt.subplots()
    ax.scatter(rmse,ctr)
    ax.set_xlabel("Blind RMSE [ps]");ax.set_ylabel("Blind CTR [ps]");ax.set_title(title)
    ax.text(.03,.97,f"Pearson r = {correlation:.3f}\nn = {values.shape[0]}",transform=ax.transAxes,ha="left",va="top")
    return _save(fig,path)


def blind_rmse_ctr_correlation(rows):
    pairs=[]
    for row in _blind_rows(rows):
        try:pairs.append((float(row.get("rmse_ps","nan")),float(row.get("ctr_ps","nan"))))
        except (TypeError,ValueError):continue
    values=np.asarray([(x,y) for x,y in pairs if np.isfinite(x) and np.isfinite(y)],float)
    if values.shape[0]<2 or np.std(values[:,0])==0 or np.std(values[:,1])==0:return float("nan"),int(values.shape[0])
    return float(np.corrcoef(values[:,0],values[:,1])[0,1]),int(values.shape[0])


def write_blind_summary(rows,path):
    blind=_blind_rows(rows)
    if not blind:return None
    metrics=(
        ("blind_ctr_ps","ctr_ps"),
        ("blind_rmse_ps","rmse_ps"),
        ("paired_ctr_improvement_ps","improvement_ps"),
        ("paired_ctr_improvement_percent","improvement_percent"),
        ("paired_rmse_improvement_ps","rmse_improvement_ps"),
        ("paired_rmse_improvement_percent","rmse_improvement_percent"),
    )
    def stats(values):
        values=np.asarray(values,float);values=values[np.isfinite(values)]
        return (float(np.mean(values)) if values.size else float("nan"),
                float(np.std(values,ddof=1)) if values.size>1 else 0.0,int(values.size))
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open("w",encoding="utf-8",newline="") as stream:
        writer=csv.DictWriter(stream,fieldnames=["metric","mean","std","n"]);writer.writeheader()
        for label,key in metrics:
            values=[]
            for row in blind:
                try:values.append(float(row.get(key,"nan")))
                except (TypeError,ValueError):pass
            mean,std,n=stats(values);writer.writerow({"metric":label,"mean":mean,"std":std,"n":n})
        correlation,n_corr=blind_rmse_ctr_correlation(rows)
        writer.writerow({"metric":"blind_rmse_ctr_pearson_r","mean":correlation,"std":"","n":n_corr})
    return path


def make_study_result_plots(rows,run_dir,*,model,mode,window_ns):
    run_dir=Path(run_dir);label=f"{model} | {mode} | [{window_ns['start']}, {window_ns['end']}] ns"
    return {
        "blind_ctr":plot_blind_ctr_distribution(rows,run_dir/"blind_ctr_distribution.png",f"Blind CTR distribution\n{label}"),
        "blind_rmse":plot_blind_rmse_distribution(rows,run_dir/"blind_rmse_distribution.png",f"Blind RMSE distribution\n{label}"),
        "paired_ctr_improvement":plot_paired_improvement_splits(rows,run_dir/"paired_led_ctr_improvement_split_distribution.png",f"Paired LED-to-ML CTR improvement across splits\n{label}"),
        "paired_rmse_improvement":plot_paired_rmse_improvement_splits(rows,run_dir/"paired_led_rmse_improvement_split_distribution.png",f"Paired LED-to-ML RMSE improvement across splits\n{label}"),
        "rmse_ctr_correlation":plot_rmse_ctr_correlation(rows,run_dir/"blind_rmse_vs_ctr.png",f"Blind RMSE vs CTR\n{label}"),
        "summary":write_blind_summary(rows,run_dir/"blind_summary.csv"),
    }
