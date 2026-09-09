from __future__ import annotations

import csv, json
from pathlib import Path
from typing import Any
import numpy as np

from .common import voltage_from_name
from .dataset import load_prepared_dataset
from .view import inverse_pair, waveform_view

MODEL_ORDER=("led","cfd","linear_svr","cnn")
LABELS={"led":"LED","cfd":"CFD","linear_svr":"Linear SVR","cnn":"CNN"}


def read_results(run_dir:str|Path)->list[dict[str,Any]]:
    with (Path(run_dir)/"results.csv").open(encoding="utf-8",newline="") as stream:return list(csv.DictReader(stream))
def _float(value,default=float("nan")):
    try:return float(value)
    except (TypeError,ValueError):return default
def _voltage(row):
    value=_float(row.get("voltage_V")); return value if np.isfinite(value) else voltage_from_name(row.get("dataset", ""))
def _residual(run,dataset,method,stage="test"):
    path=run/"artifacts"/dataset/f"{method}_{stage}_residuals_ps.npy"; return np.asarray(np.load(path),dtype=float) if path.is_file() else None
def _measurement_text(value,uncertainty):
    value=float(value); uncertainty=float(uncertainty)
    if not np.isfinite(value):return "nan"
    if not np.isfinite(uncertainty) or uncertainty<=0:return f"{value:.1f}"
    rounded_unc=float(f"{uncertainty:.1g}"); exponent=int(np.floor(np.log10(abs(rounded_unc)))) if rounded_unc else 0; decimals=max(0,-exponent); return f"{value:.{decimals}f} ± {rounded_unc:.{decimals}f}"
def _distribution_methods(rows,dataset,stage):
    available={r["method"] for r in rows if r["dataset"]==dataset and r.get("stage")==stage}; ordered=["led"]+[m for m in MODEL_ORDER if m not in {"led","cfd"}]; ordered.extend(sorted(available-set(ordered)-{"cfd"})); return [m for m in ordered if m in available]
def _stripe_importance(time_ns,importance,width_ns=1.0):
    t=np.asarray(time_ns,dtype=float); imp=np.asarray(importance,dtype=float)
    if t.size==0:return []
    edges=np.arange(np.floor(np.nanmin(t)/width_ns)*width_ns,np.ceil(np.nanmax(t)/width_ns)*width_ns+width_ns,width_ns); out=[]
    for a,b in zip(edges[:-1],edges[1:]):
        mask=(t>=a)&(t<(b if b<edges[-1] else b+1e-12)); values=imp[mask]; out.append((float(a),float(b),float(np.nanmean(values)) if values.size else 0.0))
    maximum=max((v for _,_,v in out),default=0.0); return [(a,b,v/maximum if maximum>0 else 0.0) for a,b,v in out]
def _xai_plot(output,artifact,mode,model,paths):
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap,Normalize
    from matplotlib.cm import ScalarMappable
    dataset=artifact.parent.name
    with np.load(artifact) as data: time=np.asarray(data["time_ps"],dtype=float)/1000.0; importance=np.asarray(data["importance"],dtype=float); pair=np.asarray(data["example_pair_mV"],dtype=float)
    if not time.size or not importance.size:return
    stripes=_stripe_importance(time,importance,1.0); cmap=LinearSegmentedColormap.from_list("xai",["white","orange","red"]); norm=Normalize(0,1); fig,(top,bottom)=plt.subplots(2,1,figsize=(8.6,5.8),sharex=True,height_ratios=(2,1))
    for a,b,value in stripes: top.axvspan(a,b,color=cmap(norm(value)),alpha=.7,lw=0)
    top.plot(time,pair[0],label="detector 1"); top.plot(time,pair[1],label="detector 2"); top.set_ylabel("Signal [mV]"); top.legend(); top.grid(True,alpha=.2); centers=np.asarray([(a+b)/2 for a,b,_ in stripes]); values=np.asarray([v for _,_,v in stripes]); bottom.plot(centers,values,marker="o"); bottom.set_ylim(0,1.05); bottom.set_xlabel("Time relative to LED anchor [ns]"); bottom.set_ylabel("1 ns mean importance"); bottom.grid(True,alpha=.2); cbar=fig.colorbar(ScalarMappable(norm=norm,cmap=cmap),ax=top,pad=.015,fraction=.04); cbar.set_label("Normalized importance"); voltage=voltage_from_name(dataset); label=f"{voltage:g} V" if np.isfinite(voltage) else dataset; fig.suptitle(f"{LABELS.get(model,model)} · {mode.replace('_',' ')} · {label}"); fig.tight_layout(); target=output/f"xai_{dataset}_{model}.pdf"; fig.savefig(target); plt.close(fig); paths.append(target)
def _correction_rankings(run,model,dataset):
    led=_residual(run,dataset,"led","test"); corrected=_residual(run,dataset,model,"test")
    if led is None or corrected is None or led.size!=corrected.size:return [],[]
    finite=np.isfinite(led)&np.isfinite(corrected)
    if not np.any(finite):return [],[]
    center=float(np.median(led[finite])); improvement=np.abs(led-center)-np.abs(corrected-center); event_indices=np.full(led.size,-1,dtype=np.int64)
    try:
        with np.load(run/"splits"/f"{dataset}.npz") as split:test=np.asarray(split["test"],dtype=np.int64)
        manifest=json.loads((run/"manifest.json").read_text(encoding="utf-8")); prepared=load_prepared_dataset(manifest["datasets"][dataset]["prepared_dir"]); event_indices=np.asarray(prepared.event_index[test],dtype=np.int64)
    except Exception:pass
    rows=[{"dataset":dataset,"voltage_V":voltage_from_name(dataset),"position":int(i),"event_index":int(event_indices[i]) if i<event_indices.size else -1,"led_residual_ps":float(led[i]),"corrected_residual_ps":float(corrected[i]),"led_bias_ps":center,"improvement_ps":float(improvement[i])} for i in np.flatnonzero(finite)]; rows.sort(key=lambda r:r["improvement_ps"],reverse=True); return rows[:3],rows[-3:]
def _write_rankings(output,dataset,model,top,worst):
    target=output/f"correction_top_worst_{dataset}_{model}.csv"; fields=["rank_group","rank","dataset","voltage_V","event_index","led_residual_ps","corrected_residual_ps","led_bias_ps","improvement_ps"]
    with target.open("w",encoding="utf-8",newline="") as stream:
        writer=csv.DictWriter(stream,fieldnames=fields); writer.writeheader()
        for group,items in (("top",top),("worst",list(reversed(worst)))):
            for rank,row in enumerate(items,1):writer.writerow({"rank_group":group,"rank":rank,**{k:row[k] for k in fields if k not in {"rank_group","rank"}}})
    return target
def _correction_examples(output,run,mode,model,dataset,top,worst,paths):
    import matplotlib.pyplot as plt
    selected=[("Top",r) for r in top]+[("Worst",r) for r in reversed(worst)]
    if not selected:return
    manifest=json.loads((run/"manifest.json").read_text(encoding="utf-8")); fig,axes=plt.subplots(3,2,figsize=(11,9),squeeze=False)
    for ax,(group,row) in zip(axes.flat,selected):
        try:
            prepared=load_prepared_dataset(manifest["datasets"][dataset]["prepared_dir"])
            with np.load(run/"splits"/f"{dataset}.npz") as split:test=np.asarray(split["test"],dtype=np.int64)
            index=int(test[row["position"]]); view=waveform_view(prepared,mode,np.asarray([index])); pair=inverse_pair(prepared,mode,view.materialize())[0]; time=np.asarray(view.time_ps)/1000.0; ax.plot(time,pair[0],label="detector 1"); ax.plot(time,pair[1],label="detector 2"); ax.axvline(0.0,ls="--",lw=1.0,alpha=.8); ax.set_title(f"{group} #{row['event_index']} | improvement {row['improvement_ps']:.1f} ps"); ax.set_xlabel("Time [ns]"); ax.set_ylabel("Signal [mV]"); ax.grid(True,alpha=.2)
        except Exception as exc:ax.text(.5,.5,f"Unable to load example\n{exc}",ha="center",va="center",transform=ax.transAxes)
    axes[0,0].legend(); fig.suptitle(f"{dataset} · {mode.replace('_',' ')} · {LABELS.get(model,model)} top/worst corrections"); fig.tight_layout(); target=output/f"correction_examples_{dataset}_{model}.pdf"; fig.savefig(target); plt.close(fig); paths.append(target)
def _distribution_plot(output,run,rows,mode,dataset,stage,paths):
    import matplotlib.pyplot as plt
    series=[]
    for method in _distribution_methods(rows,dataset,stage):
        residual=_residual(run,dataset,method,stage)
        if residual is None:continue
        residual=residual[np.isfinite(residual)]
        if not residual.size:continue
        row=next((r for r in rows if r["dataset"]==dataset and r["method"]==method and r.get("stage")==stage),None)
        if row is not None:series.append((method,residual,row))
    if not series:return
    all_values=np.concatenate([r for _,r,_ in series]); lo,hi=np.nanpercentile(all_values,[0.5,99.5]); bins=np.linspace(lo,hi,101) if np.isfinite(lo) and np.isfinite(hi) and hi>lo else 100; fig,ax=plt.subplots(figsize=(8.6,4.8))
    for method,residual,row in series:
        label=f"{LABELS.get(method,method)} · CTR {_measurement_text(_float(row.get('ctr_ps')),_float(row.get('ctr_uncertainty_ps')))} ps · mean {np.mean(residual):+.1f} ps"; ax.hist(residual,bins=bins,histtype="step",density=True,label=label)
    ax.set_xlabel(f"{stage.capitalize()} residual [ps]"); ax.set_ylabel("Density"); ax.set_title(f"{mode.replace('_',' ')} · {dataset} · {stage}"); ax.legend(); ax.grid(True,alpha=.2); fig.tight_layout(); target=output/f"ctr_distribution_{stage}_{dataset}.pdf"; fig.savefig(target); plt.close(fig); paths.append(target)
def make_plots(run_dir:str|Path,output_dir:str|Path|None=None)->list[Path]:
    import matplotlib.pyplot as plt
    run=Path(run_dir).resolve(); plot_root=Path(output_dir).resolve() if output_dir else run/"plots"; categories={name:plot_root/name for name in ("corrections","train_distribution","test_distribution","xai")}
    for directory in categories.values():directory.mkdir(parents=True,exist_ok=True)
    manifest=json.loads((run/"manifest.json").read_text(encoding="utf-8")); mode=str(manifest.get("mode") or manifest["config"]["mode"]); all_rows=read_results(run); test_rows=[r for r in all_rows if r.get("stage")=="test"]; datasets=sorted({r["dataset"] for r in test_rows},key=voltage_from_name); paths=[]; fig,ax=plt.subplots(figsize=(8.2,4.6))
    for method in MODEL_ORDER:
        points=sorted([r for r in test_rows if r["method"]==method and np.isfinite(_voltage(r))],key=_voltage)
        if not points:continue
        voltage=np.asarray([_voltage(r) for r in points]); ctr=np.asarray([_float(r["ctr_ps"]) for r in points]); error=np.asarray([_float(r["ctr_uncertainty_ps"]) for r in points]); ax.errorbar(voltage,ctr,yerr=error,marker="o",label=LABELS.get(method,method))
    ax.set_xlabel("Bias voltage [V]"); ax.set_ylabel("CTR [ps]"); ax.set_title(mode.replace("_"," ")); ax.grid(True,alpha=.25); ax.legend(); fig.tight_layout(); target=run/"ctr_vs_voltage.pdf"; fig.savefig(target); plt.close(fig); paths.append(target)
    for dataset in datasets:
        _distribution_plot(categories["test_distribution"],run,all_rows,mode,dataset,"test",paths); _distribution_plot(categories["train_distribution"],run,all_rows,mode,dataset,"train",paths)
    models=[m for m in MODEL_ORDER if m not in {"led","cfd"} and any(r["method"]==m for r in test_rows)]
    for model in models:
        for artifact in sorted((run/"artifacts").glob(f"*/{model}_xai.npz"),key=lambda p:voltage_from_name(p.parent.name)):_xai_plot(categories["xai"],artifact,mode,model,paths)
        for dataset in datasets:
            top,worst=_correction_rankings(run,model,dataset)
            if top or worst:paths.append(_write_rankings(categories["corrections"],dataset,model,top,worst)); _correction_examples(categories["corrections"],run,mode,model,dataset,top,worst,paths)
    return paths
