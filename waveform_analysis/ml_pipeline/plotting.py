from __future__ import annotations
import csv,json
from contextlib import contextmanager
from pathlib import Path
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from .stats import residual_summary
def load_plot_config(root):return json.loads((Path(root).resolve()/"plots.json").read_text(encoding="utf-8"))
@contextmanager
def plot_context(config):
    f=config["font"]
    with mpl.rc_context({"font.family":f["family"],"font.size":float(f["size"]),"axes.titlesize":float(f["title_size"]),"axes.labelsize":float(f["label_size"]),"xtick.labelsize":float(f["tick_size"]),"ytick.labelsize":float(f["tick_size"]),"legend.fontsize":float(f["legend_size"]),"figure.dpi":float(config["output"]["dpi"]),"savefig.dpi":float(config["output"]["dpi"]),"lines.linewidth":float(config["line"]["width"])}):yield
def output_path(directory,stem,config):return Path(directory)/f"{stem}.{str(config['output'].get('format','png')).lstrip('.')}"
def _finish(ax,cfg):
    g=cfg["grid"];ax.grid(bool(g["enabled"]),alpha=float(g["alpha"]),linestyle=g["linestyle"])
def _save(fig,path):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);fig.tight_layout();fig.savefig(path);plt.close(fig);return path
def _csv(path):
    with Path(path).open("r",encoding="utf-8",newline="") as s:return list(csv.DictReader(s))
def plot_run_cv(run,cfg):
    run=Path(run);path=run/"tables"/"cv.csv"
    if not path.is_file():return None
    rows=_csv(path);manifest=json.loads((run/"manifest.json").read_text());metric=manifest["cv"]["metric"];mean=f"{metric}_mean_ps";std=f"{metric}_std_ps";rows=sorted(rows,key=lambda r:float(r.get(mean,"inf")));x=np.arange(len(rows));v=np.asarray([float(r[mean]) for r in rows]);e=np.asarray([float(r[std]) for r in rows]);pr=np.asarray([str(r.get("pruned","")).lower() in {"true","1"} for r in rows])
    with plot_context(cfg):
        fig,ax=plt.subplots(figsize=tuple(cfg["cv"]["figsize"]));ax.errorbar(x[~pr],v[~pr],yerr=e[~pr],fmt=cfg["markers"]["default"],capsize=3,label="complete")
        if np.any(pr):ax.scatter(x[pr],v[pr],marker=cfg["cv"]["pruned_marker"],label="pruned")
        ax.set_xticks(x);ax.set_xticklabels([r["candidate_id"] for r in rows],rotation=float(cfg["cv"]["label_rotation"]),ha="right");ax.set_ylabel(f"Development CV {metric.upper()} [ps]");ax.legend();_finish(ax,cfg);return _save(fig,output_path(run/"plots","cv",cfg))
def plot_run_blind(run,cfg):
    run=Path(run);path=run/"artifacts"/"pred.npz"
    if not path.is_file():return None
    with np.load(path) as d:
        corrected=np.asarray(d["corrected_ps"]);led=np.asarray(d["led_residual_ps"])
    summaries=[residual_summary(a) for a in (led,corrected)]
    valid=[s for s in summaries if s["n_finite"]]
    if not valid:return None
    lo=min(s["q01_ps"] for s in valid);hi=max(s["q99_ps"] for s in valid)
    span=hi-lo
    if span<=0:
        span=max(abs(lo)*0.1,1.0);lo-=span/2;hi+=span/2
    else:
        lo-=0.05*span;hi+=0.05*span
    with plot_context(cfg):
        fig,ax=plt.subplots(figsize=tuple(cfg["histogram"]["figsize"]))
        for values,label in ((led,"LED"),(corrected,"ML corrected")):
            values=np.asarray(values,float).ravel()
            ax.hist(values[np.isfinite(values)],bins=int(cfg["histogram"]["bins"]),range=(lo,hi),alpha=float(cfg["histogram"]["alpha"]),label=label)
        ax.set_xlabel("Blind residual [ps]");ax.set_ylabel("Events");ax.legend();_finish(ax,cfg)
        return _save(fig,output_path(run/"plots","blind",cfg))
def _xai_one_ns(time,importance):
    # Aggregate per-sample group importance in fixed 1 ns time bins.
    edges=np.arange(np.floor(time.min()),np.ceil(time.max())+1,1.0)
    if edges.size<2:edges=np.array([time.min()-0.5,time.max()+0.5])
    centers=(edges[:-1]+edges[1:])/2
    values=np.full(centers.size,np.nan)
    indices=np.clip(np.searchsorted(edges,time,side="right")-1,0,centers.size-1)
    for i in range(centers.size):
        portion=importance[indices==i]
        portion=portion[np.isfinite(portion)]
        if portion.size:values[i]=float(np.mean(portion))
    maximum=float(np.nanmax(values)) if np.any(np.isfinite(values)) else 0.0
    normalized=np.nan_to_num(values/maximum,nan=0.0,posinf=0.0,neginf=0.0) if maximum>0 else np.zeros_like(values)
    return edges,centers,normalized

def plot_run_xai(run,cfg):
    run=Path(run);path=run/"artifacts"/"xai.npz"
    if not path.is_file():return None
    with np.load(path) as d:
        time=np.asarray(d["time_ps"],float).reshape(-1)/1000.0
        importance=np.asarray(d["importance_ps"],float).reshape(-1)
        example=np.asarray(d["example_waveforms_mV"],float)
    if time.size<2 or importance.size!=time.size or not np.all(np.isfinite(time)):return None
    if example.shape!=(2,time.size):raise ValueError(f"XAI waveform pair shape mismatch in {path}: {example.shape}")
    edges,centers,values=_xai_one_ns(time,importance)
    style=cfg["xai"];width,height=style["figsize"]
    with plot_context(cfg):
        fig,(top,bottom)=plt.subplots(2,1,figsize=(max(float(width),7.2),max(float(height),5.4)),sharex=True,gridspec_kw={"height_ratios":[2.0,1.0]},layout="constrained")
        norm=mpl.colors.Normalize(vmin=0,vmax=1);cmap=mpl.colormaps.get_cmap(style.get("cmap","viridis"))
        for index,(color,linestyle) in enumerate((("#0072B2","-"),("#D55E00","--"))):
            top.plot(time,example[index],color=color,linestyle=linestyle,label=f"Detector {index+1}",zorder=3)
        top.legend(loc="upper right")
        top.set_ylabel("Signal [mV]")
        for left,right,value in zip(edges[:-1],edges[1:],values):
            top.axvspan(left,right,color=cmap(norm(value)),alpha=float(style.get("band_alpha",0.25)),linewidth=0,zorder=0)
        bottom.bar(centers,values,width=np.diff(edges),color=[cmap(norm(v)) for v in values],edgecolor="white",linewidth=0.5,align="center")
        bottom.plot(centers,values,color="#222222",linewidth=1,marker="o",markersize=2.5)
        bottom.set_ylim(0,1.08);bottom.set_ylabel("Normalized importance");bottom.set_xlabel("Time relative to LED [ns]")
        top.set_xlim(time.min(),time.max())
        colorbar=fig.colorbar(mpl.cm.ScalarMappable(norm=norm,cmap=cmap),ax=[top,bottom],location="right",fraction=0.035,pad=0.035)
        colorbar.set_label("Normalized importance",labelpad=9)
        _finish(top,cfg);_finish(bottom,cfg)
        output=output_path(run/"plots","xai",cfg);output.parent.mkdir(parents=True,exist_ok=True)
        fig.savefig(output,bbox_inches="tight",pad_inches=0.16);plt.close(fig)
        return output
def render_run_plots(run,cfg):return {"cv":plot_run_cv(run,cfg),"blind":plot_run_blind(run,cfg),"xai":plot_run_xai(run,cfg)}
def heatmap(matrix,labels,path,cfg,*,title,correlation=False,value_format=".1f"):
    matrix=np.asarray(matrix,float);style=cfg["heatmap"];kwargs={"cmap":style["correlation_cmap"] if correlation else style["difference_cmap"]}
    if correlation:kwargs.update(vmin=float(style["correlation_limits"][0]),vmax=float(style["correlation_limits"][1]))
    elif np.any(np.isfinite(matrix)):
        limit=float(np.nanmax(np.abs(matrix)))
        if limit>0:kwargs.update(vmin=-limit,vmax=limit)
    with plot_context(cfg):
        fig,ax=plt.subplots(figsize=tuple(style["min_figsize"]));im=ax.imshow(matrix,**kwargs);ax.set_xticks(range(len(labels)));ax.set_yticks(range(len(labels)));ax.set_xticklabels(labels,rotation=float(style["label_rotation"]),ha="right");ax.set_yticklabels(labels);ax.set_title(title)
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                if np.isfinite(matrix[i,j]):ax.text(j,i,format(matrix[i,j],value_format),ha="center",va="center",fontsize=float(style["cell_text_size"]))
        fig.colorbar(im,ax=ax);return _save(fig,path)
def scatter_with_labels(x,y,labels,path,cfg,*,xlabel,ylabel,title,xerr=None,yerr=None,annotation=None):
    with plot_context(cfg):
        fig,ax=plt.subplots(figsize=tuple(cfg["scatter"]["figsize"]));x=np.asarray(x,float);y=np.asarray(y,float)
        if xerr is not None or yerr is not None:ax.errorbar(x,y,xerr=xerr,yerr=yerr,fmt="none",capsize=float(cfg["scatter"]["error_capsize"]))
        ax.scatter(x,y,s=float(cfg["scatter"]["marker_size"]))
        for xi,yi,label in zip(x,y,labels):ax.annotate(label,(xi,yi),xytext=tuple(cfg["scatter"]["annotation_offset"]),textcoords="offset points")
        if annotation:ax.text(.03,.97,annotation,transform=ax.transAxes,ha="left",va="top")
        ax.set_xlabel(xlabel);ax.set_ylabel(ylabel);ax.set_title(title);_finish(ax,cfg);return _save(fig,path)
def grouped_bar(labels,series,path,cfg,*,ylabel,title):
    style=cfg["bar"];width=max(float(style["figure_width_min"]),float(style["width_per_category"])*len(labels))
    with plot_context(cfg):
        fig,ax=plt.subplots(figsize=(width,float(style["figure_height"])));x=np.arange(len(labels),dtype=float);bar_width=float(style["group_width"])/max(1,len(series))
        for index,item in enumerate(series):
            off=(index-(len(series)-1)/2)*bar_width;v=np.asarray(item["values"],float);e=np.asarray(item.get("errors",np.zeros_like(v)),float);bars=ax.bar(x+off,v,width=bar_width,yerr=e,capsize=float(style["error_capsize"]),label=item["label"])
            for bar,value,error in zip(bars,v,e):
                if not np.isfinite(value):continue
                text=f"{int(round(value))} ± {int(round(error))} ps" if np.isfinite(error) and error>0 else f"{int(round(value))} ps"
                ax.annotate(text,(bar.get_x()+bar.get_width()/2,bar.get_height()),xytext=(0,float(style["annotation_offset_points"])),textcoords="offset points",ha="center",va="bottom",fontsize=float(cfg["font"]["annotation_size"]),rotation=90)
        ax.set_xticks(x);ax.set_xticklabels(labels);ax.set_ylabel(ylabel);ax.set_title(title);ax.legend();_finish(ax,cfg);return _save(fig,path)
