from __future__ import annotations
import csv,json
from contextlib import contextmanager
from pathlib import Path
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
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
    with np.load(path) as d:corrected=np.asarray(d["corrected_ps"]);led=np.asarray(d["led_residual_ps"])
    with plot_context(cfg):
        fig,ax=plt.subplots(figsize=tuple(cfg["histogram"]["figsize"]));ax.hist(led,bins=int(cfg["histogram"]["bins"]),alpha=float(cfg["histogram"]["alpha"]),label="LED");ax.hist(corrected,bins=int(cfg["histogram"]["bins"]),alpha=float(cfg["histogram"]["alpha"]),label="ML corrected");ax.set_xlabel("Blind residual [ps]");ax.set_ylabel("Events");ax.legend();_finish(ax,cfg);return _save(fig,output_path(run/"plots","blind",cfg))
def plot_run_xai(run,cfg):
    run=Path(run);path=run/"artifacts"/"xai.npz"
    if not path.is_file():return None
    with np.load(path) as d:time=np.asarray(d["time_ps"],float)/1000.0;importance=np.asarray(d["importance_ps"],float)
    with plot_context(cfg):
        fig,ax=plt.subplots(figsize=tuple(cfg["xai"]["figsize"]));ax.plot(time,importance);ax.fill_between(time,0,importance,alpha=float(cfg["xai"]["fill_alpha"]));ax.set_xlabel("Waveform time [ns]");ax.set_ylabel("Mean |prediction change| [ps]");_finish(ax,cfg);return _save(fig,output_path(run/"plots","xai",cfg))
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
