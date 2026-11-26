"""Replot persisted Linear SHAP and TSHAP artifacts without model inference."""
import argparse
from pathlib import Path
from collections import defaultdict
import numpy as np
from waveform_analysis.reporting.attribution_artifacts import load_saved, render_saved
from waveform_analysis.reporting.attribution_plots import window_importance

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results",type=Path,nargs="+",required=True)
    parser.add_argument("--shared-scales",action="store_true")
    args=parser.parse_args()
    dirs=set()
    for root in args.results:
        if not root.is_dir(): raise FileNotFoundError(root)
        for name in ("global_attributions.npz","attributions.npz"):
            dirs.update(p.parent.resolve() for p in root.rglob(name) if (p.parent/"summary.json").is_file())
    groups=defaultdict(list)
    for d in sorted(dirs):
        try:
            data,summary,ids=load_saved(d)
        except KeyError as exc:
            print(f"SKIP {d}: missing saved field {exc}")
            continue
        board=next((x.upper() for x in reversed(d.parts) if x.upper() in ("UC","FBK")),summary.get("board","unknown"))
        mode=summary.get("mode",summary.get("identity",{}).get("mode",d.parent.parent.parent.name))
        key=(board,str(mode).replace("_to_energy","").replace("_to_timing",""),data["model_name"],data["window_name"])
        groups[key].append((d,data,ids))
    if not groups: raise ValueError("No compatible global attribution artifacts found")
    for key,items in groups.items():
        scales={}
        if args.shared_scales:
            vmax=0.; ymin=0.; ymax=0.; lo=float("inf"); hi=-lo; tlo=lo; thi=hi
            first=items[0][2]
            widths=set()
            for d,data,ids in items:
                imp,_,dt=window_importance(data["time_ps"],data["window_starts"],data["window_samples"],data["mean_positive_ps"],data["mean_negative_ps"])
                vmax=max(vmax,float(imp.max()))
                ymin=min(ymin,float(data["mean_negative_ps"].min()))
                ymax=max(ymax,float(data["mean_positive_ps"].max()))
                lo=min(lo,float(data["mean_waveform_mv"].min())); hi=max(hi,float(data["mean_waveform_mv"].max()))
                tlo=min(tlo,float(data["time_ps"][0]/1000)); thi=max(thi,float(data["time_ps"][-1]/1000))
                widths.add(round(dt*data["window_samples"],6))
                if first is not None and ids is not None and not np.array_equal(first,ids):
                    print(f"WARNING different event IDs: {d}")
            if len(widths)>1: print(f"WARNING different window durations (ps): {key}: {widths}")
            margin=max((hi-lo)*.06,1e-6)
            scales=dict(importance_vmax=vmax or 1.,contribution_ylim=[ymin*1.08 if ymin<0 else -.01,ymax*1.08 if ymax>0 else .01],
                        waveform_ylim=[lo-margin,hi+margin],time_xlim=[tlo,thi])
        for d,_,_ in items:
            render_saved(d,**scales)
            print(f"REPLOTTED {d}")

if __name__=="__main__":
    main()
