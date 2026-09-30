from __future__ import annotations

import csv
import json
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from .control_preprocessing import fit_control_artifact
from .data import preprocess_selected
from .dataset import load_prepared_dataset
from .event_selection import apply_selection_rules
from .models import get_model
from .prepared_data import prepare_ml_dataset
from .stats import ctr_estimate, rmse_ps
from .storage import RunStore
from .train import load_fitted_model, predict_indices
from .view import model_target


def _pearson(x,y):
    x=np.asarray(x,float);y=np.asarray(y,float);mask=np.isfinite(x)&np.isfinite(y);x=x[mask];y=y[mask]
    if x.size<2 or np.std(x)==0 or np.std(y)==0:return float("nan")
    return float(np.corrcoef(x,y)[0,1])


def _prepare_dataset(config,logger,run_dir=None):
    if run_dir is not None:
        manifest_path=Path(run_dir)/"manifest.json"
        if manifest_path.is_file():
            manifest=json.loads(manifest_path.read_text(encoding="utf-8"));prepared=manifest.get("prepared_dataset")
            if prepared and Path(prepared).is_dir():return load_prepared_dataset(prepared)
    control,_=fit_control_artifact(config["reference"]["root_file"],config["reference"],config["preprocessing"],config["fit"],
        fit_by_mode={config["mode"]:config["fit"]},modes=[config["mode"]],cache_root=config["preprocessing"]["cache_dir"],rebuild=False,logger=logger)
    selection=apply_selection_rules(config["analysis"]["root_file"],config["analysis"],config["preprocessing"],control["selection_rules"],config["mode"],
        cache_dir=Path(config["preprocessing"]["cache_dir"])/"analysis_selection",rebuild=False,logger=logger)
    native=preprocess_selected(config["analysis"]["root_file"],selection,config["analysis"],config["preprocessing"],config["mode"],
        cache_dir=Path(config["preprocessing"]["cache_dir"])/"analysis_native",rebuild=False,logger=logger)
    return prepare_ml_dataset(native,control,config,cache_dir=config["preprocessing"]["cache_dir"],rebuild=False,logger=logger)


def _selected_blind_rows(rows):
    return sorted([r for r in rows if r.get("stage")=="blind" and str(r.get("selected","")).lower() in {"true","1"}],key=lambda r:int(r["seed"]))


def _split_path_for_seed(store,seed):
    manifest_path=store.root/"manifest.json"
    if manifest_path.is_file():
        manifest=json.loads(manifest_path.read_text(encoding="utf-8"));shared=(manifest.get("shared_replicas") or {}).get(str(seed))
        if shared:
            path=Path(shared)/"split.npz"
            if path.is_file():return path
    return store.root/"splits"/f"seed_{seed}.npz"


def _residuals_for_row(store,spec,dataset,config,candidates,row,target,logger):
    seed=int(row["seed"]);cid=row["candidate_id"];path=store.blind_residuals_path(seed,cid)
    if path.is_file():
        with np.load(path) as data:return np.asarray(data["corrected_ps"],float)
    split_path=_split_path_for_seed(store,seed)
    if not split_path.is_file():raise FileNotFoundError(f"Missing saved split: {split_path}")
    with np.load(split_path) as split_data:test=np.asarray(split_data["test"],np.int64)
    model_dir=store.root/"models"/f"seed_{seed}"/cid
    if not model_dir.is_dir():raise FileNotFoundError(f"Missing saved model: {model_dir}")
    params=candidates[cid];fitted=load_fitted_model(spec,model_dir,params,config)
    prediction=predict_indices(spec,fitted,dataset,config["mode"],test);corrected=np.asarray(target[test],float)-np.asarray(prediction,float)
    store.save_blind_residuals(seed,cid,corrected)
    logger.info("Blind residuals reconstructed | seed=%s | candidate=%s | events=%d | inference=1",seed,cid,len(test))
    return corrected


def run_ctr_binning_scan(config,bin_widths_ps,*,logger=None):
    log=logger or logging.getLogger("ctr-binning-scan");run_dir=Path(config["output_dir"]).resolve();store=RunStore(run_dir,resume=True)
    rows=store.read_results();blind=_selected_blind_rows(rows)
    if not blind:raise RuntimeError(f"No selected blind results found in {store.results_path}")
    candidates_path=run_dir/"candidates.json"
    if not candidates_path.is_file():raise FileNotFoundError(candidates_path)
    candidates=json.loads(candidates_path.read_text(encoding="utf-8"));spec=get_model(config["model"]["name"])
    widths=sorted({float(v) for v in bin_widths_ps})
    if not widths or any((not np.isfinite(v) or v<=0) for v in widths):raise ValueError("bin widths must be finite positive values")
    dataset=_prepare_dataset(config,log,run_dir);target=model_target(dataset,config["mode"]);residuals=[]
    for row in blind:
        corrected=_residuals_for_row(store,spec,dataset,config,candidates,row,target,log);residuals.append((int(row["seed"]),corrected,rmse_ps(corrected)))
    output_rows=[]
    for width in widths:
        fit_cfg=dict(config["fit"]);fit_cfg["histogram_bin_width_ps"]=float(width);ctrs=[];rmses=[]
        for seed,corrected,rmse in residuals:
            try:ctr=float(ctr_estimate(corrected,fit_cfg,seed=seed,bootstrap=False).ctr_ps)
            except ValueError:ctr=float("nan")
            output_rows.append({"seed":seed,"bin_width_ps":width,"rmse_ps":rmse,"ctr_ps":ctr})
            if np.isfinite(ctr):ctrs.append(ctr);rmses.append(rmse)
        r=_pearson(rmses,ctrs)
        for item in output_rows:
            if item["bin_width_ps"]==width:item["rmse_ctr_pearson_r"]=r
        log.info("CTR binning scan | width=%.3f ps | valid=%d/%d | RMSE-vs-CTR r=%.4f",width,len(ctrs),len(residuals),r)
    csv_path=run_dir/"ctr_binning_scan.csv"
    with csv_path.open("w",encoding="utf-8",newline="") as stream:
        writer=csv.DictWriter(stream,fieldnames=["seed","bin_width_ps","rmse_ps","ctr_ps","rmse_ctr_pearson_r"]);writer.writeheader();writer.writerows(output_rows)
    correlations=[]
    for width in widths:
        values=[r for r in output_rows if r["bin_width_ps"]==width];correlations.append(float(values[0]["rmse_ctr_pearson_r"]))
    fig,ax=plt.subplots();ax.plot(widths,correlations,marker="o");ax.axhline(0.0,linestyle=":");ax.set_xlabel("Histogram bin width [ps]");ax.set_ylabel("Pearson r (RMSE, CTR)")
    ax.set_title(f"CTR binning sensitivity\n{spec.name} | {config['mode']} | [{config['window_ns']['start']}, {config['window_ns']['end']}] ns")
    for x,y in zip(widths,correlations):
        if np.isfinite(y):ax.annotate(f"{y:.3f}",(x,y),textcoords="offset points",xytext=(0,6),ha="center")
    fig.tight_layout();plot_path=run_dir/"rmse_ctr_correlation_vs_bin_width.png";fig.savefig(plot_path);plt.close(fig)
    summary_path=run_dir/"ctr_binning_scan_summary.csv"
    with summary_path.open("w",encoding="utf-8",newline="") as stream:
        writer=csv.DictWriter(stream,fieldnames=["bin_width_ps","rmse_ctr_pearson_r","n_resamples"]);writer.writeheader()
        for width,r in zip(widths,correlations):writer.writerow({"bin_width_ps":width,"rmse_ctr_pearson_r":r,"n_resamples":len(residuals)})
    log.info("CTR binning scan complete | %s",run_dir);return {"plot":plot_path,"csv":csv_path,"summary":summary_path}
