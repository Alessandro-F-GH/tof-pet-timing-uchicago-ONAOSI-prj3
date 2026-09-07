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
    with (Path(run_dir)/"results.csv").open(encoding="utf-8",newline="") as stream:
        return list(csv.DictReader(stream))


def _float(value,default=float("nan")):
    try:return float(value)
    except (TypeError,ValueError):return default


def _voltage(row):
    value=_float(row.get("voltage_V"))
    return value if np.isfinite(value) else voltage_from_name(row.get("dataset", ""))


def _residual(run,dataset,mode,method):
    path=run/"artifacts"/dataset/mode/f"{method}_test_residuals_ps.npy"
    return np.asarray(np.load(path),dtype=float) if path.is_file() else None


def _measurement_text(value,uncertainty):
    value=float(value); uncertainty=float(uncertainty)
    if not np.isfinite(value):return "nan"
    if not np.isfinite(uncertainty) or uncertainty<=0:return f"{value:.1f}"
    rounded_unc=float(f"{uncertainty:.1g}")
    exponent=int(np.floor(np.log10(abs(rounded_unc)))) if rounded_unc else 0
    decimals=max(0,-exponent)
    return f"{value:.{decimals}f} ± {rounded_unc:.{decimals}f}"


def _mean_text(value,uncertainty):
    value=float(value); uncertainty=float(uncertainty)
    if not np.isfinite(value):return "nan"
    if not np.isfinite(uncertainty) or uncertainty<=0:return f"{value:+.1f}"
    rounded_unc=float(f"{uncertainty:.1g}")
    exponent=int(np.floor(np.log10(abs(rounded_unc)))) if rounded_unc else 0
    decimals=max(0,-exponent)
    return f"{value:+.{decimals}f}"


def _distribution_methods(subset,dataset):
    available={r["method"] for r in subset if r["dataset"]==dataset}
    # Distributions compare the LED reference directly with ML models. CFD is
    # intentionally omitted here; it remains available in CTR-vs-voltage plots.
    ordered=["led"]+[m for m in MODEL_ORDER if m not in {"led","cfd"}]
    ordered.extend(sorted(available-set(ordered)-{"cfd"}))
    return [m for m in ordered if m in available]


def _stripe_importance(time_ns,importance,width_ns=1.0):
    t=np.asarray(time_ns,dtype=float); imp=np.asarray(importance,dtype=float)
    if t.size==0:return []
    edges=np.arange(np.floor(np.nanmin(t)/width_ns)*width_ns,np.ceil(np.nanmax(t)/width_ns)*width_ns+width_ns,width_ns)
    out=[]
    for a,b in zip(edges[:-1],edges[1:]):
        mask=(t>=a)&(t<(b if b<edges[-1] else b+1e-12)); values=imp[mask]
        out.append((float(a),float(b),float(np.nanmean(values)) if values.size else 0.0))
    maximum=max((v for _,_,v in out),default=0.0)
    return [(a,b,v/maximum if maximum>0 else 0.0) for a,b,v in out]


def _xai_plot_dataset(mode_dir,artifact,mode,model,paths):
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap,Normalize

    dataset=artifact.parent.parent.name
    with np.load(artifact) as data:
        time=np.asarray(data["time_ps"],dtype=float)/1000.0
        importance=np.asarray(data["importance"],dtype=float)
        pair=np.asarray(data["example_pair_mV"],dtype=float)
    if not time.size or not importance.size:return
    stripes=_stripe_importance(time,importance,1.0)
    cmap=LinearSegmentedColormap.from_list("xai",["white","orange","red"]); norm=Normalize(0,1)
    fig,(top,bottom)=plt.subplots(2,1,figsize=(8.4,5.8),sharex=True,height_ratios=(2,1))
    for a,b,value in stripes:
        top.axvspan(a,b,color=cmap(norm(value)),alpha=.65,lw=0)
        bottom.axvspan(a,b,color=cmap(norm(value)),alpha=.8,lw=0)
    top.plot(time,pair[0],label="detector 1"); top.plot(time,pair[1],label="detector 2")
    top.set_ylabel("Signal [mV]"); top.legend(); top.grid(True,alpha=.2)
    centers=np.asarray([(a+b)/2 for a,b,_ in stripes]); values=np.asarray([v for _,_,v in stripes])
    bottom.plot(centers,values,marker="o")
    bottom.set_ylim(0,1.05); bottom.set_xlabel("Time relative to LED anchor [ns]"); bottom.set_ylabel("1 ns mean importance"); bottom.grid(True,alpha=.2)
    voltage=voltage_from_name(dataset)
    label=f"{voltage:g} V" if np.isfinite(voltage) else dataset
    fig.suptitle(f"{LABELS.get(model,model)} · {mode.replace('_',' ')} · {label}")
    fig.tight_layout(); target=mode_dir/f"xai_{dataset}_{model}.pdf"; fig.savefig(target); plt.close(fig); paths.append(target)


def _correction_rankings(run,mode,model,datasets):
    rows=[]
    for dataset in datasets:
        led=_residual(run,dataset,mode,"led"); corrected=_residual(run,dataset,mode,model)
        if led is None or corrected is None or led.size!=corrected.size:continue
        finite=np.isfinite(led)&np.isfinite(corrected)
        if not np.any(finite):continue
        center=float(np.median(led[finite]))
        improvement=np.abs(led-center)-np.abs(corrected-center)
        split_path=run/"splits"/f"{dataset}.npz"; event_indices=None
        try:
            with np.load(split_path) as split:test=np.asarray(split["test"],dtype=np.int64)
            manifest=json.loads((run/"manifest.json").read_text(encoding="utf-8")); prepared=load_prepared_dataset(manifest["datasets"][dataset]["prepared_dir"]); event_indices=np.asarray(prepared.event_index[test],dtype=np.int64)
        except Exception:event_indices=np.full(led.size,-1,dtype=np.int64)
        for i in np.flatnonzero(finite):
            rows.append({"dataset":dataset,"voltage_V":voltage_from_name(dataset),"position":int(i),"event_index":int(event_indices[i]) if i<event_indices.size else -1,"led_residual_ps":float(led[i]),"corrected_residual_ps":float(corrected[i]),"led_bias_ps":center,"led_distance_from_bias_ps":float(abs(led[i]-center)),"corrected_distance_from_led_bias_ps":float(abs(corrected[i]-center)),"improvement_ps":float(improvement[i])})
    rows.sort(key=lambda r:r["improvement_ps"],reverse=True)
    return rows[:3],rows[-3:]


def _write_rankings(mode_dir,mode,model,top,worst):
    target=mode_dir/f"correction_top_worst_{model}.csv"; fields=["rank_group","rank","dataset","voltage_V","event_index","led_residual_ps","corrected_residual_ps","led_bias_ps","led_distance_from_bias_ps","corrected_distance_from_led_bias_ps","improvement_ps"]
    with target.open("w",encoding="utf-8",newline="") as stream:
        writer=csv.DictWriter(stream,fieldnames=fields); writer.writeheader()
        for group,items in (("top",top),("worst",list(reversed(worst)))):
            for rank,row in enumerate(items,1): writer.writerow({"rank_group":group,"rank":rank,**{k:row[k] for k in fields if k not in {"rank_group","rank"}}})
    return target


def _correction_examples(mode_dir,run,mode,model,top,worst,paths):
    import matplotlib.pyplot as plt

    selected=[("Top",r) for r in top]+[("Worst",r) for r in reversed(worst)]
    if not selected:return
    manifest=json.loads((run/"manifest.json").read_text(encoding="utf-8"))
    fig,axes=plt.subplots(3,2,figsize=(11,9),squeeze=False)
    for ax,(group,row) in zip(axes.flat,selected):
        try:
            prepared=load_prepared_dataset(manifest["datasets"][row["dataset"]]["prepared_dir"])
            with np.load(run/"splits"/f'{row["dataset"]}.npz') as split:test=np.asarray(split["test"],dtype=np.int64)
            index=int(test[row["position"]]); view=waveform_view(prepared,mode,np.asarray([index])); pair=inverse_pair(prepared,mode,view.materialize())[0]; time=np.asarray(view.time_ps)/1000.0
            ax.plot(time,pair[0],label="detector 1"); ax.plot(time,pair[1],label="detector 2")
            ax.axvline(0.0,ls="--",lw=1.0,alpha=.8)
            ax.set_title(f"{group}: {row['dataset']} event {row['event_index']} | improvement {row['improvement_ps']:.1f} ps")
            ax.set_xlabel("Time [ns]"); ax.set_ylabel("Signal [mV]"); ax.grid(True,alpha=.2)
        except Exception as exc: ax.text(.5,.5,f"Unable to load example\n{exc}",ha="center",va="center",transform=ax.transAxes)
    axes[0,0].legend(); fig.suptitle(f"{mode.replace('_',' ')} — {LABELS.get(model,model)} top/worst corrections"); fig.tight_layout(); target=mode_dir/f"correction_examples_{model}.pdf"; fig.savefig(target); plt.close(fig); paths.append(target)


def make_plots(run_dir:str|Path,output_dir:str|Path|None=None)->list[Path]:
    import matplotlib.pyplot as plt
    run=Path(run_dir).resolve(); output=Path(output_dir).resolve() if output_dir else run/"plots"; output.mkdir(parents=True,exist_ok=True); rows=[r for r in read_results(run) if r.get("stage")=="test"]; paths=[]
    modes=sorted({r["mode"] for r in rows})
    for mode in modes:
        mode_dir=output/mode; mode_dir.mkdir(parents=True,exist_ok=True); subset=[r for r in rows if r["mode"]==mode]; datasets=sorted({r["dataset"] for r in subset},key=voltage_from_name)
        fig,ax=plt.subplots(figsize=(8.2,4.6))
        for method in MODEL_ORDER:
            points=sorted([r for r in subset if r["method"]==method and np.isfinite(_voltage(r))],key=_voltage)
            if not points:continue
            voltage=np.asarray([_voltage(r) for r in points]); ctr=np.asarray([_float(r["ctr_ps"]) for r in points]); error=np.asarray([_float(r["ctr_uncertainty_ps"]) for r in points]); ax.errorbar(voltage,ctr,yerr=error,marker="o",label=LABELS[method])
        ax.set_xlabel("Bias voltage [V]"); ax.set_ylabel("CTR [ps]"); ax.set_title(mode.replace("_"," ")); ax.grid(True,alpha=.25); ax.legend(); fig.tight_layout(); target=mode_dir/"ctr_vs_voltage.pdf"; fig.savefig(target); plt.close(fig); paths.append(target)

        # Remove legacy method-centric distribution plots so regenerated reports
        # contain one comparison plot per source file instead.
        for method in MODEL_ORDER:
            legacy=mode_dir/f"ctr_distribution_{method}.pdf"
            if legacy.is_file():legacy.unlink()
        for dataset in datasets:
            methods=_distribution_methods(subset,dataset); series=[]
            for method in methods:
                residual=_residual(run,dataset,mode,method)
                if residual is None:continue
                residual=residual[np.isfinite(residual)]
                if not residual.size:continue
                row=next((r for r in subset if r["dataset"]==dataset and r["method"]==method),None)
                if row is None:continue
                series.append((method,residual,row))
            if not series:continue
            all_values=np.concatenate([residual for _,residual,_ in series]); lo,hi=np.nanpercentile(all_values,[0.5,99.5])
            if not np.isfinite(lo) or not np.isfinite(hi) or hi<=lo:lo,hi=float(np.nanmin(all_values)),float(np.nanmax(all_values))
            if hi<=lo:lo,hi=lo-1.0,hi+1.0
            bins=np.linspace(lo,hi,101)
            fig,ax=plt.subplots(figsize=(8.6,4.8))
            for method,residual,row in series:
                mean=float(np.mean(residual)); ctr=_float(row.get("ctr_ps")); unc=_float(row.get("ctr_uncertainty_ps"))
                label=f"{LABELS.get(method,method)} · CTR {_measurement_text(ctr,unc)} ps · mean {_mean_text(mean,unc)} ps"
                ax.hist(residual,bins=bins,histtype="step",density=True,label=label)
            voltage=voltage_from_name(dataset); voltage_label=f" · {voltage:g} V" if np.isfinite(voltage) else ""
            ax.set_xlabel("Test residual [ps]"); ax.set_ylabel("Density"); ax.set_title(f"{mode.replace('_',' ')} · {dataset}{voltage_label}"); ax.legend(); ax.grid(True,alpha=.2); fig.tight_layout(); target=mode_dir/f"ctr_distribution_{dataset}.pdf"; fig.savefig(target); plt.close(fig); paths.append(target)

        for model in ("linear_svr","cnn"):
            for artifact in sorted((run/"artifacts").glob(f"*/{mode}/{model}_xai.npz"),key=lambda p:voltage_from_name(p.parent.parent.name)):
                _xai_plot_dataset(mode_dir,artifact,mode,model,paths)
            top,worst=_correction_rankings(run,mode,model,datasets)
            if top or worst:
                paths.append(_write_rankings(mode_dir,mode,model,top,worst)); _correction_examples(mode_dir,run,mode,model,top,worst,paths)
    return paths
