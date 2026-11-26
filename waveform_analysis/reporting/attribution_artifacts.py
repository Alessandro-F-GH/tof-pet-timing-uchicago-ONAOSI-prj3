"""Regenerate global attribution figures from persisted arrays."""
import json
from pathlib import Path
import numpy as np
from waveform_analysis.reporting.attribution_plots import plot_global_attributions, plot_metadata, window_importance

def load_saved(directory):
    d=Path(directory)
    summary=json.loads((d/"summary.json").read_text(encoding="utf-8"))
    linear=(d/"global_attributions.npz").is_file()
    identity=summary.get("identity",{})
    settings=summary.get("settings",identity.get("settings",{}))
    with np.load(d/("global_attributions.npz" if linear else "attributions.npz"),allow_pickle=False) as a:
        if linear:
            starts=a["group_starts"]
            width=int(settings["group_samples"])
            p,n=a["mean_positive_contribution_ps"],a["mean_negative_contribution_ps"]
        else:
            starts=a["window_starts"]
            width=int(a["window_samples"].item()) if "window_samples" in a else int(summary.get("tshap_window_samples",settings.get("window_samples",0)))
            scores=a["raw_window_scores_ps"]
            if scores.ndim==2: scores=scores[:,None,:]
            p,n=np.maximum(scores,0).mean(axis=0),np.minimum(scores,0).mean(axis=0)
        args=dict(time_ps=a["time_ps"],mean_waveform_mv=a["mean_waveform_mv"],
                  window_starts=starts,window_samples=width,mean_positive_ps=p,
                  mean_negative_ps=n,method_name="Linear SHAP" if linear else "TSHAP",
                  model_name=summary.get("model",identity.get("model",summary.get("model_name",d.parent.name))),
                  window_name=summary.get("window",summary.get("window_name",d.parent.parent.name)))
        ids=a["event_id"] if "event_id" in a else None
    return args,summary,ids

def render_saved(directory, **scales):
    d=Path(directory)
    args,summary,_=load_saved(d)
    plot_global_attributions(d,**args,**scales)
    summary["global_plot"]=plot_metadata()
    summary["global_plot"]["scales"]=scales
    summary["plots"]=["global_importance.pdf","global_importance.png"]
    if "waveform_dispersion" in summary:
        summary["waveform_dispersion"]="not displayed in the unified global plot"
    if "global_importance_aggregation" in summary:
        summary["global_importance_aggregation"]["normalization"]="linear colour scale from signed-window contributions in ps; original sample-wise exports unchanged"
    from waveform_analysis.core.io import atomic_json
    atomic_json(d/"summary.json",summary)
