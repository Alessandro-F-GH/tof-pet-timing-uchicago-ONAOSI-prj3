from __future__ import annotations
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

_DISPLAY_QUANTILES=(0.005,0.995)

def _save(fig,path):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);fig.tight_layout();fig.savefig(path);plt.close(fig)

def _finite(values):
    v=np.asarray(values,float).reshape(-1);return v[np.isfinite(v)]

def _robust_range(series,important=(),quantiles=_DISPLAY_QUANTILES):
    arrays=[_finite(v) for v in series];nonempty=[v for v in arrays if v.size]
    if not nonempty:return None
    joined=np.concatenate(nonempty);qlo,qhi=np.quantile(joined,quantiles)
    lo=float(qlo);hi=float(qhi)
    finite_important=[float(v) for v in important if np.isfinite(v)]
    if finite_important:
        lo=min(lo,min(finite_important));hi=max(hi,max(finite_important))
    if not np.isfinite(lo) or not np.isfinite(hi):return None
    if hi<=lo:
        pad=max(1e-9,abs(lo)*0.01,1e-3);lo-=pad;hi+=pad
    else:
        pad=0.02*(hi-lo);lo-=pad;hi+=pad
    return lo,hi

def _range_label(name,values,display_range):
    v=_finite(values)
    if display_range is None:return f"{name} (n={v.size})"
    lo,hi=display_range;below=int(np.count_nonzero(v<lo));above=int(np.count_nonzero(v>hi))
    return f"{name} (n={v.size}, outside: {below} low + {above} high)"

def _hist(ax,values,bins,label,display_range):
    v=_finite(values)
    if not v.size:return
    ax.hist(v,bins=bins,range=display_range,histtype="step",label=_range_label(label,v,display_range))

def plot_photopeak(amplitudes,intervals,path,title):
    a=np.asarray(amplitudes,float);series=[a[:,d] for d in range(2)]
    important=[float(x) for interval in intervals for x in interval];display=_robust_range(series,important)
    fig,ax=plt.subplots()
    for d in range(2):
        _hist(ax,series[d],100,f"detector {d+1}",display)
        lo,hi=map(float,intervals[d]);ax.axvline(lo,linestyle="--");ax.axvline(hi,linestyle="--")
    if display is not None:ax.set_xlim(*display)
    ax.set_xlabel("Energy-channel amplitude [mV]");ax.set_ylabel("Events");ax.set_title(title);ax.legend();_save(fig,path)

def plot_baseline_noise(rms,candidate,limits,path,title):
    r=np.asarray(rms,float);m=np.asarray(candidate,bool);series=[r[m,d] for d in range(2)]
    display=_robust_range(series,[float(v) for v in limits]);fig,ax=plt.subplots()
    for d in range(2):
        _hist(ax,series[d],80,f"detector {d+1}",display)
        ax.axvline(float(limits[d]),linestyle="--")
    if display is not None:ax.set_xlim(*display)
    ax.set_xlabel("Baseline RMS [mV]");ax.set_ylabel("Events");ax.set_title(title);ax.legend();_save(fig,path)

def plot_baseline_clipping(clipped,candidate,path,title):
    c=np.asarray(clipped,bool);m=np.asarray(candidate,bool)
    passed=np.count_nonzero(m & ~np.any(c,axis=1));failed=np.count_nonzero(m & np.any(c,axis=1))
    fig,ax=plt.subplots();ax.bar(["pass","clipped"],[passed,failed]);ax.set_ylabel("Events");ax.set_title(title);_save(fig,path)

def plot_tot(hits,photo_mask,limits,path,title):
    m=np.asarray(photo_mask,bool);series=[]
    for d in range(2):
        vals=[]
        for r in np.flatnonzero(m):vals.extend(float(h.duration_ns) for h in hits[r][d])
        series.append(np.asarray(vals,float))
    important=[float(x) for interval in limits for x in interval];display=_robust_range(series,important);fig,ax=plt.subplots()
    for d in range(2):
        _hist(ax,series[d],100,f"detector {d+1}",display)
        lo,hi=map(float,limits[d]);ax.axvline(lo,linestyle="--");ax.axvline(hi,linestyle="--")
    if display is not None:ax.set_xlim(*display)
    ax.set_xlabel("ToT [ns]");ax.set_ylabel("Hits");ax.set_title(title);ax.legend();_save(fig,path)

def plot_led_selection(scan,selected,path,title):
    rows=list(scan);x=[float(r["threshold_mV"]) for r in rows];y=[float(r["ctr_ps"]) for r in rows]
    fig,ax=plt.subplots();ax.plot(x,y,marker="o");ax.axvline(float(selected),linestyle="--")
    ax.set_xlabel("LED threshold [mV]");ax.set_ylabel("Control CTR [ps]");ax.set_title(title);_save(fig,path)
