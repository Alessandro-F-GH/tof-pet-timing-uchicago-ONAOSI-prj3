"""Post-hoc asymmetric/single-detector photopeak cuts versus original ML tail events.

Reads frozen blind predictions and native energy preprocessing, never ROOT or ML.
The original top-absolute-residual outlier groups remain fixed across all cuts.

A cut of 'left' narrows ONLY the lower bound (rejects low amplitudes), 'right'
narrows ONLY the upper bound (rejects high amplitudes), 'symmetric' narrows both.
The selected detector scope is D1, D2 or both. Other detectors remain unrestricted.
Each side's maximal observed excursion is its own normalizer; fraction=1 keeps all.
"""
from __future__ import annotations
import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from waveform_analysis.scripts.scan_photopeak_ctr import (
    FRACTIONS, OUTLIER_KINDS, _load_pair, _original_outlier_groups, _extreme_report,
)
from waveform_analysis.signal.metrics import fit_ctr_ps, rmse_ps

SCOPES=("D1","D2","both")
SIDES=("left","right","symmetric")


def _bounds(centers, low_span, high_span, fraction, scope, side):
    lo=centers-low_span
    hi=centers+high_span
    indices=(0,) if scope=="D1" else (1,) if scope=="D2" else (0,1)
    for index in indices:
        if side in ("left","symmetric"):
            lo[index]=centers[index]-fraction*low_span[index]
        if side in ("right","symmetric"):
            hi[index]=centers[index]+fraction*high_span[index]
    return lo,hi


def _metrics(values, *, bin_ps, bootstrap, seed, min_events):
    a=np.asarray(values,dtype=np.float64)
    result={
        "n":len(a),"ctr_ps":None,"ctr_bootstrap_error_ps":None,
        "rmse_ps":None,"sigma_ps":None,"mean_ps":None,"status":"ok",
    }
    if len(a)<min_events:
        result["status"]="too_few_events"
        return result
    result.update({
        "rmse_ps":float(rmse_ps(a)),
        "sigma_ps":float(np.std(a,ddof=1)),
        "mean_ps":float(np.mean(a)),
    })
    try:
        fit=fit_ctr_ps(
            a,
            {"histogram_bin_width_ps":bin_ps,"bootstrap_samples":bootstrap},
            seed=seed,bootstrap=bootstrap>1,
        )
        result["ctr_ps"]=float(fit.ctr_ps)
        result["ctr_bootstrap_error_ps"]=float(fit.ctr_error_ps)
    except ValueError as exc:
        result["status"]=f"CTR unavailable: {exc}"
    return result


def _write_rows(path, rows):
    with path.open("w",encoding="utf-8",newline="") as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _side_distribution(amps, centers, groups, pred):
    """Diagnostic of whether original tail events cluster below/above each median."""
    rows=[]
    n=len(amps)
    for d in range(2):
        low=amps[:,d]<centers[d]
        high=amps[:,d]>centers[d]
        equal=~(low|high)
        for name,kind_mask in (("all",np.ones(n,bool)),*groups.items()):
            total=int(kind_mask.sum())
            for side,side_mask in (("below",low),("above",high),("equal",equal)):
                count=int((kind_mask&side_mask).sum())
                denominator=int(side_mask.sum())
                rows.append({
                    "detector":f"D{d+1}","population":name,
                    "relative_to_center":side,
                    "n_population":total,
                    "n_in_side":count,
                    "fraction_of_population":count/total if total else None,
                    "n_all_in_side":denominator,
                    "outlier_rate_within_side":count/denominator if denominator else None,
                    "center_mV":float(centers[d]),
                })
    return rows


def _plot_extreme_metrics(records,out):
    """One legible figure per detector scope and outlier group."""
    for scope in SCOPES:
        for kind in OUTLIER_KINDS:
            fig,axes=plt.subplots(1,3,figsize=(15,4.4),constrained_layout=True)
            for side in SIDES:
                a=sorted((r for r in records
                          if r["scope"]==scope and r["side"]==side
                          and r["outlier_group"]==kind),
                         key=lambda r:r["acceptance_vs_original"])
                x=[100*r["acceptance_vs_original"] for r in a]
                axes[0].plot(x,[100*r["outlier_survival_fraction"] for r in a],
                             marker="o",label=side)
                axes[1].plot(x,[r["timing_ml_outlier_rmse_ps"] for r in a],
                             marker="o",label=side)
                axes[2].plot(x,[100*r["timing_ml_outlier_fraction_of_retained_sse"] for r in a],
                             marker="o",label=side)
            axes[0].plot([0,100],[0,100],":",color="gray",label="random retention")
            for ax,title,ylabel in zip(
                axes,
                ("Original outlier survival","RMSE of surviving outliers",
                 "Outlier contribution to retained timing ML SSE"),
                ("Outlier survival [%]","Outlier RMSE [ps]","Fraction of SSE [%]"),
            ):
                ax.set_title(title)
                ax.set_xlabel("Total event acceptance [%]")
                ax.set_ylabel(ylabel)
                ax.grid(alpha=.2)
                ax.legend(fontsize=8)
            fig.suptitle(f"Cut on {scope} | original ML outliers: {kind}")
            fig.savefig(out/f"asymmetric_{scope}_{kind}.png",dpi=170)
            plt.close(fig)


def _plot_ctr(rows,out):
    fig,axes=plt.subplots(1,3,figsize=(15,4.5),constrained_layout=True)
    for ax,scope in zip(axes,SCOPES):
        for side in SIDES:
            a=sorted((r for r in rows
                      if r["scope"]==scope and r["side"]==side
                      and r["channel"]=="timing" and r["estimator"]=="ML"
                      and r["ctr_ps"] is not None),
                     key=lambda r:r["acceptance"])
            ax.plot([100*r["acceptance"] for r in a],
                    [r["ctr_ps"] for r in a],marker="o",label=side)
        ax.set_title(f"Timing ML F1 CTR: cut on {scope}")
        ax.set_xlabel("Total event acceptance [%]")
        ax.set_ylabel("CTR FWHM [ps]")
        ax.grid(alpha=.2)
        ax.legend(fontsize=8)
    fig.savefig(out/"asymmetric_ctr_timing_ml.png",dpi=170)
    plt.close(fig)


def analyze(energy_run,timing_run,cache_root,output,*,
            fractions,centers,bin_ps,bootstrap,seed,min_events,tail_fraction):
    ids,amps,pred,sources=_load_pair(energy_run,timing_run,cache_root)
    n=len(ids)
    groups=_original_outlier_groups(pred,tail_fraction)
    if centers is None:
        centers=np.median(amps,axis=0)
        center_origin="medians of the existing accepted blind population (exploratory)"
    else:
        centers=np.asarray(centers,dtype=float)
        center_origin="supplied frozen reference centers"
    if centers.shape!=(2,) or not np.all(np.isfinite(centers)):
        raise ValueError("Expected two finite photopeak centers")
    low_span=centers-np.min(amps,axis=0)
    high_span=np.max(amps,axis=0)-centers
    if np.any(low_span<=0) or np.any(high_span<=0):
        raise ValueError("Each center must be strictly inside observed amplitude range")
    fractions=sorted(set(map(float,fractions)),reverse=True)
    if not fractions or any(not (0<f<=1) for f in fractions):
        raise ValueError("Each fraction must be in (0,1]")
    out=Path(output).expanduser().resolve()
    out.mkdir(parents=True,exist_ok=True)
    rows=[]
    extreme_rows=[]
    for scope in SCOPES:
        for side in SIDES:
            for fraction in fractions:
                lo,hi=_bounds(centers,low_span,high_span,fraction,scope,side)
                keep=np.all((amps>=lo[None,:]-1e-9)&(amps<=hi[None,:]+1e-9),axis=1)
                retained=int(keep.sum())
                metadata={
                    "scope":scope,"side":side,
                    "fraction_of_side_span":fraction,
                    "acceptance":retained/n,"n_retained":retained,
                    "d1_low_mV":float(lo[0]),"d1_high_mV":float(hi[0]),
                    "d2_low_mV":float(lo[1]),"d2_high_mV":float(hi[1]),
                }
                for item in _extreme_report(groups,keep,pred,fraction,n):
                    extreme_rows.append({**metadata,**item})
                for channel in ("timing","energy"):
                    for estimator in ("LED","ML"):
                        result=_metrics(pred[channel][estimator][keep],bin_ps=bin_ps,
                            bootstrap=bootstrap,seed=seed,min_events=min_events)
                        rows.append({**metadata,"channel":channel,
                                     "estimator":estimator,**result})
    side_rows=_side_distribution(amps,centers,groups,pred)
    _write_rows(out/"asymmetric_photopeak_outliers.csv",extreme_rows)
    _write_rows(out/"asymmetric_photopeak_ctr.csv",rows)
    _write_rows(out/"original_outliers_by_amplitude_side.csv",side_rows)
    _plot_extreme_metrics(extreme_rows,out)
    _plot_ctr(rows,out)
    (out/"asymmetric_photopeak_scan.json").write_text(json.dumps({
        "n_blind_events":n,
        "sources":sources,
        "center_mV":centers.tolist(),
        "center_origin":center_origin,
        "left_spans_mV":low_span.tolist(),
        "right_spans_mV":high_span.tolist(),
        "fractions":fractions,"scopes":list(SCOPES),"sides":list(SIDES),
        "tail_fraction":tail_fraction,
        "original_outliers":{k:int(v.sum()) for k,v in groups.items()},
        "ctr_bin_width_ps":bin_ps,"bootstrap_draws":bootstrap,
        "selection":"Within existing blind accepted population; scope D1/D2/both. Left narrows lower bound only, right upper bound only, symmetric both. Opposite side remains at original observed extreme. Fraction=1 retains all. Fractions normalized separately by side and detector.",
        "statistical_note":"Compare strategies at matched *acceptance*, not necessarily at matched fractional window width. Exploration on blind data is not independent selection validation.",
        "sse_note":"Outlier RMSE is computed only for surviving members of each fixed original outlier group. SSE share divides that group's squared-error sum by the squared-error sum of all retained events.",
        "peak_note":"Uses max in frozen native energy window; centers default to median of accepted blind population, not fitted photopeak mean. No original ROOT reads.",
    },indent=2),encoding="utf-8")
    print(f"Analyzed {n} common blind events. Saved {len(rows)} CTR and {len(extreme_rows)} outlier rows in {out}")
    print("Outputs: asymmetric_photopeak_outliers.csv, asymmetric_photopeak_ctr.csv, original_outliers_by_amplitude_side.csv, asymmetric_*.png")


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--energy-run",type=Path,required=True)
    p.add_argument("--timing-run",type=Path,required=True)
    p.add_argument("--cache-root",type=Path,required=True)
    p.add_argument("--output-dir",type=Path,required=True)
    p.add_argument("--centers-mv",nargs=2,type=float,metavar=("D1","D2"))
    p.add_argument("--fractions",nargs="+",type=float,default=FRACTIONS)
    p.add_argument("--histogram-bin-ps",type=float,default=5.)
    p.add_argument("--bootstrap",type=int,default=0,
                   help="CTR F1 event-bootstrap draws per selection; 0 is fast exploratory run")
    p.add_argument("--min-events",type=int,default=500)
    p.add_argument("--tail-fraction",type=float,default=.002)
    p.add_argument("--seed",type=int,default=42)
    args=p.parse_args()
    if args.bootstrap<0 or args.min_events<5 or args.histogram_bin_ps<=0:
        p.error("bootstrap >=0, min-events >=5, histogram-bin-ps >0 required")
    analyze(args.energy_run.expanduser().resolve(),
            args.timing_run.expanduser().resolve(),
            args.cache_root.expanduser().resolve(),args.output_dir,
            fractions=args.fractions,centers=args.centers_mv,
            bin_ps=args.histogram_bin_ps,bootstrap=args.bootstrap,
            seed=args.seed,min_events=args.min_events,
            tail_fraction=args.tail_fraction)


if __name__=="__main__":
    main()
