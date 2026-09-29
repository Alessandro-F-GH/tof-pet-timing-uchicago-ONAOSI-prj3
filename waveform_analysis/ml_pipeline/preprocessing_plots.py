from __future__ import annotations
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

def _save(fig,path):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);fig.tight_layout();fig.savefig(path);plt.close(fig)

def plot_photopeak(amplitudes,intervals,path,title):
    a=np.asarray(amplitudes,float);fig,ax=plt.subplots()
    for d in range(2):
        v=a[:,d];v=v[np.isfinite(v)]
        if v.size:ax.hist(v,bins=100,histtype="step",label=f"detector {d+1}")
        lo,hi=map(float,intervals[d]);ax.axvline(lo,linestyle="--");ax.axvline(hi,linestyle="--")
    ax.set_xlabel("Energy-channel amplitude [mV]");ax.set_ylabel("Events");ax.set_title(title);ax.legend();_save(fig,path)

def plot_baseline_noise(rms,candidate,limits,path,title):
    r=np.asarray(rms,float);m=np.asarray(candidate,bool);fig,ax=plt.subplots()
    for d in range(2):
        v=r[m,d];v=v[np.isfinite(v)]
        if v.size:ax.hist(v,bins=80,histtype="step",label=f"detector {d+1}")
        ax.axvline(float(limits[d]),linestyle="--")
    ax.set_xlabel("Baseline RMS [mV]");ax.set_ylabel("Events");ax.set_title(title);ax.legend();_save(fig,path)

def plot_baseline_clipping(clipped,candidate,path,title):
    c=np.asarray(clipped,bool);m=np.asarray(candidate,bool)
    passed=np.count_nonzero(m & ~np.any(c,axis=1));failed=np.count_nonzero(m & np.any(c,axis=1))
    fig,ax=plt.subplots();ax.bar(["pass","clipped"],[passed,failed]);ax.set_ylabel("Events");ax.set_title(title);_save(fig,path)

def plot_tot(hits,photo_mask,limits,path,title):
    fig,ax=plt.subplots();m=np.asarray(photo_mask,bool)
    for d in range(2):
        vals=[]
        for r in np.flatnonzero(m):
            vals.extend(float(h.duration_ns) for h in hits[r][d])
        if vals:ax.hist(vals,bins=100,histtype="step",label=f"detector {d+1}")
        lo,hi=map(float,limits[d]);ax.axvline(lo,linestyle="--");ax.axvline(hi,linestyle="--")
    ax.set_xlabel("ToT [ns]");ax.set_ylabel("Hits");ax.set_title(title);ax.legend();_save(fig,path)

def plot_led_selection(scan,selected,path,title):
    rows=list(scan);x=[float(r["threshold_mV"]) for r in rows];y=[float(r["ctr_ps"]) for r in rows]
    fig,ax=plt.subplots();ax.plot(x,y,marker="o");ax.axvline(float(selected),linestyle="--")
    ax.set_xlabel("LED threshold [mV]");ax.set_ylabel("Control CTR [ps]");ax.set_title(title);_save(fig,path)
