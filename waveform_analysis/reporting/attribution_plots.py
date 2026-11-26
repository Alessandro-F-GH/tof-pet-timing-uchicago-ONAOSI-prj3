"""Common global attribution renderer; no estimator inference."""
from pathlib import Path
import numpy as np

PLOT_STYLE = "signed-window-importance-v1"

def plot_metadata():
    return {"style": PLOT_STYLE, "importance": "abs(mean_positive)+abs(mean_negative), averaged across overlapping windows",
            "importance_units": "ps", "waveform_dispersion": "not displayed",
            "output_formats": ["PDF", "PNG"]}

def window_importance(time_ps, starts, width, positive, negative):
    t = np.asarray(time_ps, float)
    starts = np.asarray(starts, int)
    p, n = np.asarray(positive, float), np.asarray(negative, float)
    if t.ndim != 1 or len(t)<2 or not np.all(np.isfinite(t)) or np.any(np.diff(t)<=0):
        raise ValueError("Invalid time grid")
    if p.shape != n.shape or p.ndim != 2 or p.shape[0] not in (1,2) or p.shape[1]!=len(starts):
        raise ValueError("Invalid signed attribution shapes")
    if not np.all(np.isfinite(p)) or not np.all(np.isfinite(n)) or np.any(p<0) or np.any(n>0):
        raise ValueError("Invalid signed contributions")
    if width<1 or np.any(starts<0) or np.any(starts>=len(t)):
        raise ValueError("Invalid windows")
    dt = np.min(np.diff(t))
    total = np.zeros((p.shape[0],len(t)))
    count = np.zeros(len(t))
    groups=[]
    for j,s in enumerate(starts):
        end=min(s+width,len(t))
        idx=np.arange(s,end)
        total[:,idx] += (p[:,j]-n[:,j])[:,None]
        count[idx]+=1
        runs=np.split(idx,np.flatnonzero(np.diff(t[idx])>1.5*dt)+1)
        groups.append([(t[r[0]]-dt/2,t[r[-1]]+dt/2) for r in runs])
    return total/np.maximum(count,1), groups, dt

def plot_global_attributions(destination, *, time_ps, mean_waveform_mv, window_starts,
                             window_samples, mean_positive_ps, mean_negative_ps,
                             method_name, model_name, window_name, std_waveform_mv=None,
                             importance_vmax=None, contribution_ylim=None,
                             waveform_ylim=None, time_xlim=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection
    from matplotlib.colors import LinearSegmentedColormap, Normalize
    from matplotlib.cm import ScalarMappable
    t=np.asarray(time_ps,float)
    y=np.asarray(mean_waveform_mv,float)
    p,n=np.asarray(mean_positive_ps,float),np.asarray(mean_negative_ps,float)
    importance,groups,dt=window_importance(t,window_starts,int(window_samples),p,n)
    if y.shape!=(2,len(t)) or not np.all(np.isfinite(y)):
        raise ValueError("Invalid mean waveforms")
    vmax=float(np.max(importance)) if importance_vmax is None else float(importance_vmax)
    norm=Normalize(0,vmax if vmax>0 else 1,clip=True)
    cmap=LinearSegmentedColormap.from_list("importance",["#b8b8b8","#4c78a8","#1f5f99","#f2d94e"])
    with plt.rc_context({"font.family":"serif","font.size":10,"axes.labelsize":11,
                         "xtick.labelsize":10,"ytick.labelsize":10,"legend.fontsize":10,
                         "pdf.fonttype":42,"ps.fonttype":42}):
        fig=plt.figure(figsize=(12.2,7.8 if len(p)==1 else 9.5),constrained_layout=True)
        try:
            gs=fig.add_gridspec(2+len(p),2,width_ratios=[1,.035],
                                height_ratios=[1,1]+[.65]*len(p))
            axes=[fig.add_subplot(gs[0,0])]
            axes.extend(fig.add_subplot(gs[i,0],sharex=axes[0]) for i in range(1,2+len(p)))
            cax=fig.add_subplot(gs[:2,1])
            model=str(model_name).replace("_"," ").title()
            for a,b in (("Cnn","CNN"),("Mlp","MLP"),("Rdst","RDST"),("Minirocket","MiniRocket"),("Multirocket","MultiRocket")):
                model=model.replace(a,b)
            fig.suptitle(f"Global {method_name} — {model}",fontsize=14)
            lo,hi=float(y.min()),float(y.max())
            margin=max((hi-lo)*.06,1e-6)
            for k,ax in enumerate(axes[:2]):
                points=np.column_stack((t/1000,y[k]))
                seg=np.stack((points[:-1],points[1:]),axis=1)
                valid=np.diff(t)<=1.5*dt
                values=importance[0 if len(p)==1 else k]
                coll=LineCollection(seg[valid],cmap=cmap,norm=norm,linewidths=3,
                                    capstyle="round")
                coll.set_array((.5*(values[:-1]+values[1:]))[valid])
                ax.add_collection(coll)
                ax.set_ylim(waveform_ylim or (lo-margin,hi+margin))
                ax.set_ylabel("Amplitude (mV)")
                ax.text(.98,.95,f"Detector {k+1}",transform=ax.transAxes,
                        ha="right",va="top",fontsize=11,
                        bbox=dict(facecolor="white",edgecolor="none",alpha=.8))
            for k,ax in enumerate(axes[2:]):
                for j,parts in enumerate(groups):
                    for left,right in parts:
                        center=(left+right)/2000
                        width=(right-left)/1000
                        ax.bar(center,p[k,j],width=width*.95,color="#4c78a8",
                               label="Positive" if j==0 and left==parts[0][0] else None)
                        ax.bar(center,n[k,j],width=width*.95,color="#e45756",
                               label="Negative" if j==0 and left==parts[0][0] else None)
                ax.axhline(0,color=".4",linewidth=.7)
                ax.set_ylabel(("Pair" if len(p)==1 else f"Detector {k+1}")+" contribution (ps)")
                if contribution_ylim is not None: ax.set_ylim(contribution_ylim)
                ax.legend(loc="best")
            for ax in axes:
                if str(window_name).lower()=="wide":
                    ax.axvspan(-1.5,2,color=".75",alpha=.25,zorder=0,linewidth=0)
                ax.grid(alpha=.18,linewidth=.5)
                ax.set_axisbelow(True)
                ax.set_xlim(time_xlim or (t[0]/1000,t[-1]/1000))
            for ax in axes[:-1]: ax.tick_params(labelbottom=False)
            axes[-1].set_xlabel("Waveform time (ns)")
            cb=fig.colorbar(ScalarMappable(norm=norm,cmap=cmap),cax=cax)
            cb.set_label("Absolute importance (ps)",fontsize=11)
            cb.ax.tick_params(labelsize=10)
            destination=Path(destination)
            destination.mkdir(parents=True,exist_ok=True)
            fig.savefig(destination/"global_importance.pdf",bbox_inches="tight")
            fig.savefig(destination/"global_importance.png",dpi=240,bbox_inches="tight")
        finally:
            plt.close(fig)
