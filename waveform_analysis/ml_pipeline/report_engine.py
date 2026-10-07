from __future__ import annotations
import csv,json
from pathlib import Path
import numpy as np
from .common import atomic_json,write_csv
from .plotting import grouped_bar,heatmap,load_plot_config,output_path,render_run_plots,scatter_with_labels
from .splits import semantic_seed
from .stats import align_by_event_id,paired_model_bootstrap,pearson_r

def _json(path):return json.loads(Path(path).read_text(encoding="utf-8"))
def _csv(path):
    with Path(path).open("r",encoding="utf-8",newline="") as s:return list(csv.DictReader(s))
def _matrix(path,labels,matrix):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open("w",encoding="utf-8",newline="") as s:
        w=csv.writer(s);w.writerow(["model",*labels])
        for label,row in zip(labels,np.asarray(matrix)):w.writerow([label,*row.tolist()])
def collect_runs(root):
    root=Path(root).resolve();index=root/"runs.csv"
    if not index.is_file():raise FileNotFoundError(f"Result root has no runs.csv: {root}")
    runs=[]
    for row in _csv(index):
        d=(root/row["path"]).resolve();m=d/"manifest.json"
        if not m.is_file():continue
        manifest=_json(m)
        if manifest.get("status")!="complete":continue
        runs.append({"directory":d,"manifest":manifest,"best":_json(d/"best.json"),"blind":_json(d/"blind.json"),"bootstrap":_json(d/"bootstrap.json")})
    if not runs:raise RuntimeError("No complete model results found")
    return runs
def _summaries(runs):
    validation=[];blind=[]
    for run in runs:
        m,b,f,u=run["manifest"],run["best"],run["blind"],run["bootstrap"];base={"mode":m["mode"],"window":m.get("window_name"),"model":m["model"],"formulation":m["estimator_formulation"],"candidate_id":b["candidate_id"]}
        validation.append({**base,"selection_metric":b["selection_metric"],"ctr_mean_ps":b["validation_ctr_mean_ps"],"ctr_std_ps":b["validation_ctr_std_ps"],"rmse_mean_ps":b["validation_rmse_mean_ps"],"rmse_std_ps":b["validation_rmse_std_ps"],"folds":b["folds"]})
        blind.append({**base,"n_events":f["n_events"],"ctr_ps":f["ctr_ps"],"ctr_std_ps":u["ctr_bootstrap_std_ps"],"rmse_ps":f["rmse_ps"],"rmse_std_ps":u["rmse_bootstrap_std_ps"],"led_ctr_ps":f["led_ctr_ps"],"led_rmse_ps":f["led_rmse_ps"],"ctr_improvement_ps":f["ctr_improvement_ps"],"ctr_improvement_std_ps":u["ctr_improvement_bootstrap_std_ps"],"rmse_improvement_ps":f["rmse_improvement_ps"],"rmse_improvement_std_ps":u["rmse_improvement_bootstrap_std_ps"]})
    return validation,blind
def _groups(runs):
    out={}
    for run in runs:out.setdefault((run["manifest"]["mode"],run["manifest"].get("window_name")),[]).append(run)
    return out
def _payload(run):
    with np.load(run["directory"]/"pred.npz") as d:return {k:np.asarray(d[k]) for k in d.files}
def _correlation(group):
    labels=[r["manifest"]["model"] for r in group];payloads=[_payload(r) for r in group];matrix=np.eye(len(group));counts=np.zeros((len(group),len(group)),int)
    for i in range(len(group)):
        counts[i,i]=len(payloads[i]["event_id"])
        for j in range(i+1,len(group)):
            ids,a,b=align_by_event_id(payloads[i]["event_id"],payloads[i]["prediction_ps"],payloads[j]["event_id"],payloads[j]["prediction_ps"]);matrix[i,j]=matrix[j,i]=pearson_r(a,b);counts[i,j]=counts[j,i]=len(ids)
    return labels,matrix,counts,payloads
def _paired(group,payloads,metric,root_config,seed):
    labels=[r["manifest"]["model"] for r in group];central=np.zeros((len(group),len(group)));std=np.zeros_like(central);counts=np.zeros_like(central,dtype=int);fit=root_config["protocol"]["fit"];mode=group[0]["manifest"]["mode"];fit=fit[mode] if isinstance(fit,dict) and mode in fit else fit;n=int(root_config["protocol"]["bootstrap"]["n_resamples"])
    for i in range(len(group)):
        counts[i,i]=len(payloads[i]["event_id"])
        for j in range(i+1,len(group)):
            r=paired_model_bootstrap(payloads[i]["event_id"],payloads[i]["corrected_ps"],payloads[j]["event_id"],payloads[j]["corrected_ps"],fit,metric=metric,n_resamples=n,seed=semantic_seed(seed,mode,group[0]["manifest"].get("window_name"),metric,labels[i],labels[j],"paired_model"));central[i,j]=r["difference_ps"];central[j,i]=-r["difference_ps"];std[i,j]=std[j,i]=r["bootstrap_std_ps"];counts[i,j]=counts[j,i]=r["n_matched"]
    return labels,central,std,counts
def _scatters(group,directory,cfg):
    labels=[r["manifest"]["model"] for r in group];ctr=np.asarray([r["blind"]["ctr_ps"] for r in group],float);rmse=np.asarray([r["blind"]["rmse_ps"] for r in group],float);ctr_e=np.asarray([r["bootstrap"]["ctr_bootstrap_std_ps"] for r in group],float);rmse_e=np.asarray([r["bootstrap"]["rmse_bootstrap_std_ps"] for r in group],float);r=pearson_r(rmse,ctr);scatter_with_labels(rmse,ctr,labels,output_path(directory,"rmse_vs_ctr",cfg),cfg,xlabel="Blind RMSE [ps]",ylabel="Blind CTR [ps]",title="Blind RMSE vs CTR across models",xerr=rmse_e,yerr=ctr_e,annotation=f"Pearson r = {r:.3f}" if np.isfinite(r) else "Pearson r = n/a")
    vctr=np.asarray([r["best"]["validation_ctr_mean_ps"] for r in group]);vrmse=np.asarray([r["best"]["validation_rmse_mean_ps"] for r in group])
    for metric,x,y,e in (("ctr",vctr,ctr,ctr_e),("rmse",vrmse,rmse,rmse_e)):
        rr=pearson_r(x,y);scatter_with_labels(x,y,labels,output_path(directory,f"validation_vs_blind_{metric}",cfg),cfg,xlabel=f"Development CV {metric.upper()} mean [ps]",ylabel=f"Blind {metric.upper()} [ps]",title=f"Development validation vs blind {metric.upper()}",yerr=e,annotation=f"Pearson r = {rr:.3f}" if np.isfinite(rr) else "Pearson r = n/a")
def _windows(runs,root,cfg):
    for mode in sorted({r["manifest"]["mode"] for r in runs}):
        mr=[r for r in runs if r["manifest"]["mode"]==mode];windows=sorted({r["manifest"].get("window_name") for r in mr});directory=root/("energy" if mode=="energy_to_energy" else "timing")
        for metric in ("ctr","rmse"):
            labels=[];values={"shared":[],"direct":[]};errors={"shared":[],"direct":[]}
            for window in windows:
                wr=[r for r in mr if r["manifest"].get("window_name")==window];selected={}
                for form in ("shared","direct"):
                    candidates=[r for r in wr if r["manifest"]["estimator_formulation"]==form]
                    if candidates:
                        selection_metric=str(candidates[0]["best"]["selection_metric"]);selected[form]=min(candidates,key=lambda r:float(r["best"][f"validation_{selection_metric}_mean_ps"]))
                labels.append(str(window)+"\n"+" ".join(f"{f[0].upper()}:{selected[f]['manifest']['model']}" for f in ("shared","direct") if f in selected))
                for form in ("shared","direct"):
                    run=selected.get(form);values[form].append(float(run["blind"][f"{metric}_ps"]) if run else np.nan);errors[form].append(float(run["bootstrap"][f"{metric}_bootstrap_std_ps"]) if run else np.nan)
            grouped_bar(labels,[{"label":cfg["formulations"][f]["label"],"values":values[f],"errors":errors[f]} for f in ("shared","direct")],output_path(directory,f"windows_{metric}",cfg),cfg,ylabel=f"Blind {metric.upper()} [ps]",title=f"Waveform-window comparison — winners selected by development CV {metric.upper()}")
def _read_matrix(path):
    rows=_csv(path);return [r["model"] for r in rows],np.asarray([[float(v) for k,v in r.items() if k!="model"] for r in rows],dtype=float)

def generate_report(result_root,*,logger=None,reuse_numeric=False):
    root=Path(result_root).resolve();cfg=load_plot_config(root);root_config=_json(root/"config.json");runs=collect_runs(root);report=root/"report";tables=report/"tables";tables.mkdir(parents=True,exist_ok=True)
    for run in runs:render_run_plots(run["directory"],cfg)
    validation,blind=_summaries(runs);write_csv(tables/"validation.csv",validation);write_csv(tables/"blind.csv",blind);seed=int(root_config["protocol"]["seed"])
    for (mode,window),group in _groups(runs).items():
        d=report/("energy" if mode=="energy_to_energy" else "timing")/str(window);d.mkdir(parents=True,exist_ok=True);labels,corr,corr_n,payloads=_correlation(group);_matrix(d/"output_correlation.csv",labels,corr);_matrix(d/"output_correlation_n.csv",labels,corr_n);heatmap(corr,labels,output_path(d,"output_correlation",cfg),cfg,title="Blind model-output correlation",correlation=True,value_format=".2f")
        for metric in ("ctr","rmse"):
            central_path=d/f"paired_{metric}.csv";std_path=d/f"paired_{metric}_std.csv";count_path=d/f"paired_{metric}_n.csv"
            if reuse_numeric and central_path.is_file() and std_path.is_file() and count_path.is_file():
                labels,central=_read_matrix(central_path);_,std=_read_matrix(std_path);_,counts=_read_matrix(count_path)
            else:
                labels,central,std,counts=_paired(group,payloads,metric,root_config,seed);_matrix(central_path,labels,central);_matrix(std_path,labels,std);_matrix(count_path,labels,counts)
            heatmap(central,labels,output_path(d,f"paired_{metric}",cfg),cfg,title=f"Paired blind Δ {metric.upper()} (row − column) [ps]");heatmap(std,labels,output_path(d,f"paired_{metric}_std",cfg),cfg,title=f"Paired blind Δ {metric.upper()} bootstrap std [ps]")
        _scatters(group,d,cfg)
    _windows(runs,report,cfg);atomic_json(report/"manifest.json",{"source_result_root":str(root),"runs":len(runs),"plot_regeneration_requires_training":False,"pairwise_difference_convention":"metric(row)-metric(column); negative means row model is better","window_winner_source":"development_cv_only"})
    if logger:logger.info("Report generated from persisted results only | %s",report)
    return report
