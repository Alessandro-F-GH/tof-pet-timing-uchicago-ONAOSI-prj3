from __future__ import annotations
import json
from contextvars import ContextVar
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
_CONFIG=ContextVar("waveform_plot_config",default=None)
_DEFAULT=Path(__file__).resolve().parents[1]/"config"/"plots"/"default.json"
def configure_plotting(config):_CONFIG.set(config)
def _config():return _CONFIG.get() or json.loads(_DEFAULT.read_text(encoding="utf-8"))
def _save(fig,path):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);fig.tight_layout();fig.savefig(path,bbox_inches="tight",dpi=float(_config()["output"]["dpi"]));plt.close(fig)
def _finish(ax):
    g=_config()["grid"];ax.grid(bool(g["enabled"]),alpha=float(g["alpha"]),linestyle=g["linestyle"])
def _finite(values):
    v=np.asarray(values,float).reshape(-1);return v[np.isfinite(v)]
def _range(series,important=()):
    arrays=[_finite(v) for v in series];nonempty=[v for v in arrays if v.size]
    if not nonempty:return None
    lo,hi=map(float,np.quantile(np.concatenate(nonempty),_config()["preprocessing"]["display_quantiles"]));imp=[float(v) for v in important if np.isfinite(v)]
    if imp:lo=min(lo,min(imp));hi=max(hi,max(imp))
    pad=max(1e-3,0.02*(hi-lo)) if hi>lo else max(1e-3,abs(lo)*0.01);return lo-pad,hi+pad
def _selection_range(intervals):
    values=[float(v) for interval in intervals for v in interval if v is not None and np.isfinite(v)]
    if not values:return None
    lo,hi=min(values),max(values);span=hi-lo;cfg=_config()["preprocessing"];pad=max(float(cfg["selection_minimum_pad"]),float(cfg["selection_relative_pad"])*span);return lo-pad,hi+pad
def _selected(values,selection):
    values=_finite(values);lo,hi=selection;mask=np.ones(values.size,bool)
    if lo is not None and np.isfinite(lo):mask&=values>=float(lo)
    if hi is not None and np.isfinite(hi):mask&=values<=float(hi)
    return values[mask]
def _hist(ax,values,bins,label,display,selection,color):
    values=_finite(values)
    if not values.size:return
    counts,edges=np.histogram(values,bins=int(bins),range=display);selected=_selected(values,selection);chosen,_=np.histogram(selected,bins=edges);ax.stairs(chosen,edges,fill=True,color=color,alpha=float(_config()["preprocessing"]["hist_alpha"]),linewidth=0);ax.stairs(counts,edges,color=color,linewidth=float(_config()["line"]["width"]),label=f"{label} ({selected.size}/{values.size} selected)")
def _legend(ax):ax.legend(loc="upper left",bbox_to_anchor=(1.02,1.0),borderaxespad=0.0,frameon=True,title="Filled histogram = selected range")
def _fig(wave=False):
    cfg=_config()["preprocessing"];return tuple(cfg["waveform_figsize"] if wave else cfg["figsize"])
def plot_photopeak(amplitudes,intervals,path,title):
    cfg=_config()["preprocessing"];a=np.asarray(amplitudes,float);series=[a[:,d] for d in range(2)];display=_range(series,[v for i in intervals for v in i]);fig,ax=plt.subplots(figsize=_fig())
    for d in range(2):lo,hi=map(float,intervals[d]);_hist(ax,series[d],cfg["photopeak_bins"],f"detector {d+1}",display,(lo,hi),cfg["detector_colors"][d])
    if display:ax.set_xlim(*display)
    ax.set_xlabel("Energy-channel amplitude [mV]");ax.set_ylabel("Events");ax.set_title(title);_legend(ax);_finish(ax);_save(fig,path)
def plot_baseline_noise(rms,candidate,limits,path,title):
    cfg=_config()["preprocessing"];rms=np.asarray(rms,float);candidate=np.asarray(candidate,bool);series=[rms[candidate,d] for d in range(2)];display=_range(series,limits);fig,ax=plt.subplots(figsize=_fig())
    for d in range(2):_hist(ax,series[d],cfg["noise_bins"],f"detector {d+1}",display,(None,float(limits[d])),cfg["detector_colors"][d])
    if display:ax.set_xlim(*display)
    ax.set_xlabel("Baseline RMS [mV]");ax.set_ylabel("Events");ax.set_title(title);_legend(ax);_finish(ax);_save(fig,path)
def plot_baseline_clipping(clearance,candidate,margin_mV,path,title):
    cfg=_config()["preprocessing"];clearance=np.asarray(clearance,float);candidate=np.asarray(candidate,bool);series=[clearance[candidate,d] for d in range(2)];display=_range(series,[margin_mV]);fig,ax=plt.subplots(figsize=_fig())
    for d in range(2):_hist(ax,series[d],cfg["noise_bins"],f"detector {d+1}",display,(float(np.nextafter(float(margin_mV),np.inf)),None),cfg["detector_colors"][d])
    if display:ax.set_xlim(*display)
    ax.set_xlabel("Baseline clearance to vertical-scale limit [mV]");ax.set_ylabel("Events");ax.set_title(title);_legend(ax);_finish(ax);_save(fig,path)
def plot_tot(hits,photo_mask,limits,path,title):
    cfg=_config()["preprocessing"];photo_mask=np.asarray(photo_mask,bool);series=[np.asarray([float(h.duration_ns) for row in np.flatnonzero(photo_mask) for h in hits[row][d]],float) for d in range(2)];display=_selection_range(limits);fig,ax=plt.subplots(figsize=_fig())
    for d in range(2):lo,hi=map(float,limits[d]);_hist(ax,series[d],cfg["tot_bins"],f"detector {d+1}",display,(lo,hi),cfg["detector_colors"][d])
    if display:ax.set_xlim(*display)
    ax.set_xlabel("ToT [ns]");ax.set_ylabel("Hits");ax.set_title(title);_legend(ax);_finish(ax);_save(fig,path)
def plot_materialized_event(preprocessed,family,channel_numbers,path,title):
    waves=preprocessed.energy_windows_mV if family=="energy" else preprocessed.timing_windows_mV;intervals=preprocessed.energy_sample_interval_s if family=="energy" else preprocessed.timing_sample_interval_s
    if waves is None or intervals is None or preprocessed.n_events<=0:return None
    row=preprocessed.n_events//2;event=np.asarray(waves[row],float);dt=np.asarray(intervals[row],float).reshape(-1);channels=np.asarray(channel_numbers).reshape(-1);before=float(preprocessed.manifest["materialized_window_ns"]["before"]);idx=int(np.asarray(preprocessed.event_index)[row]);colors=_config()["preprocessing"]["detector_colors"];fig,axes=plt.subplots(2,1,figsize=_fig(True))
    for d,ax in enumerate(axes):
        ns=float(dt[d])*1e9;offset=int(np.ceil(before/ns));time=(np.arange(event[d].size)-offset)*ns;ax.plot(time,event[d],color=colors[d]);ax.axvline(0.0,linestyle=":");ax.set_ylabel("Amplitude [mV]");ax.set_title(f"Detector {d+1} — channel {int(channels[d])}");_finish(ax)
    axes[-1].set_xlabel("Time relative to selected trigger [ns]");fig.suptitle(f"{title} — event {idx}");_save(fig,path);return Path(path)
def plot_led_selection(scan,selected,path,title):
    rows=list(scan);fig,ax=plt.subplots(figsize=_fig());ax.plot([float(r["threshold_mV"]) for r in rows],[float(r["ctr_ps"]) for r in rows],marker=_config()["markers"]["default"]);ax.axvline(float(selected),linestyle=_config()["reference"]["linestyle"]);ax.set_xlabel("LED threshold [mV]");ax.set_ylabel("Control CTR [ps]");ax.set_title(title);_finish(ax);_save(fig,path)
