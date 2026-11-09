"""Compare energy/timing residual correlations of original outliers removed by
restoring a symmetric [-2,+2] sigma photopeak selection.

Uses frozen native-energy cache, saved blind residuals and frozen CONTROL
photopeak fit. Does not refit, read ROOT, rerun inference, or relabel outliers.
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
from scipy.stats import spearmanr

from waveform_analysis.data.results import resolve_run_config
from waveform_analysis.data.named_cache import control_alias, resolve_alias, alias_exists
from waveform_analysis.scripts.scan_photopeak_ctr import (
    _load_pair, _original_outlier_groups,
)


def _load_frozen_fit(cache_root, control_dataset):
    """Use authoritative named control artifact, not recursive JSON searches.

    The frozen photopeak fits are in manifest.json under selection_rules.
    The cache alias may be a Windows .cache_link.json pointer or a symlink.
    """
    from waveform_analysis.data.named_cache import dataset_name
    root=Path(cache_root).expanduser().resolve()
    control_parent=root/"control"/dataset_name(control_dataset)
    # Control artifact can cover energy alone or energy+timing.
    preferred=[
        control_alias(root,control_dataset,("energy_to_energy","timing_to_timing")),
        control_alias(root,control_dataset,("energy_to_energy",)),
    ]
    aliases=[p for p in preferred if alias_exists(p)]
    if not aliases:
        raise FileNotFoundError(
            f"Named frozen control artifact absent under {control_parent}. "
            "Provide --mu-mv D1 D2 and --sigma-mv D1 D2 from the frozen control fit.")
    found=[]
    for alias in aliases:
        target=resolve_alias(alias)
        manifest_path=target/"manifest.json"
        if not manifest_path.is_file():
            raise FileNotFoundError(f"Control manifest missing: {manifest_path}")
        manifest=json.loads(manifest_path.read_text(encoding="utf-8-sig"))
        selection=manifest.get("selection_rules") or {}
        fits=selection.get("photopeak_fits")
        if not isinstance(fits,list) or len(fits)!=2:
            raise ValueError(
                f"Control selection_rules.photopeak_fits missing/invalid: {manifest_path}")
        mu=np.asarray([f["mean_mV"] for f in fits],dtype=float)
        sigma=np.asarray([f["sigma_mV"] for f in fits],dtype=float)
        if not np.isfinite(mu).all() or not np.isfinite(sigma).all() or np.any(sigma<=0):
            raise ValueError(f"Invalid frozen photopeak values: {manifest_path}")
        # Check we used the control acquisition intended for this run.
        expected=Path(str(control_dataset["root_file"]).replace("\\\\","/")).name
        actual=Path(str(manifest.get("control_source","")).replace("\\\\","/")).name
        if expected!=actual:
            raise ValueError(
                f"Control source mismatch: expected {expected}, got {actual}")
        found.append((mu,sigma,str(manifest_path)))
    reference=found[0]
    for another in found[1:]:
        if not (np.allclose(reference[0],another[0],rtol=0,atol=1e-8)
                and np.allclose(reference[1],another[1],rtol=0,atol=1e-8)):
            raise ValueError(
                "Conflicting frozen control fits for energy and energy+timing; "
                "choose --mu-mv and --sigma-mv explicitly.")
    return reference



def _stats(x,y):
    n=len(x)
    result={"n":n,"pearson":None,"spearman":None,"cov_ps2":None,
            "sigma_energy_ps":None,"sigma_timing_ps":None,
            "sigma_difference_ps":None,
            "rmse_energy_ps":None,"rmse_timing_ps":None,
            "same_sign_fraction":None}
    if n<3:
        return result
    x=np.asarray(x,float);y=np.asarray(y,float)
    sx=float(np.std(x,ddof=1));sy=float(np.std(y,ddof=1))
    c=float(np.cov(x,y,ddof=1)[0,1])
    result.update({
        "pearson":float(c/(sx*sy)) if sx>0 and sy>0 else None,
        "spearman":float(spearmanr(x,y).statistic) if sx>0 and sy>0 else None,
        "cov_ps2":c,
        "sigma_energy_ps":sx,"sigma_timing_ps":sy,
        "sigma_difference_ps":float(np.std(x-y,ddof=1)),
        "rmse_energy_ps":float(np.sqrt(np.mean(x*x))),
        "rmse_timing_ps":float(np.sqrt(np.mean(y*y))),
        "same_sign_fraction":float(np.mean((x*y)>0)),
    })
    return result


def _ci_corr(x,y,draws,seed):
    n=len(x)
    if n<5 or draws==0:
        return (None,None)
    rng=np.random.default_rng(seed)
    vals=[]
    for _ in range(draws):
        idx=rng.integers(0,n,n)
        rho=_stats(x[idx],y[idx])["pearson"]
        if rho is not None and np.isfinite(rho):
            vals.append(rho)
    if len(vals)<max(10,draws//2):
        return None,None
    return tuple(map(float,np.percentile(vals,[2.5,97.5])))


def _ci_difference(a,b,draws,seed):
    # Groups are disjoint; bootstrap independently inside each fixed group.
    if len(a)<5 or len(b)<5 or draws==0:
        return (None,None)
    rng=np.random.default_rng(seed)
    diffs=[]
    for _ in range(draws):
        ai=rng.integers(len(a),size=len(a))
        bi=rng.integers(len(b),size=len(b))
        ra=_stats(a[ai,0],a[ai,1])["pearson"]
        rb=_stats(b[bi,0],b[bi,1])["pearson"]
        if ra is not None and rb is not None and np.isfinite(ra-rb):
            diffs.append(ra-rb)
    if len(diffs)<max(10,draws//2):
        return None,None
    return tuple(map(float,np.percentile(diffs,[2.5,97.5])))


def _write_csv(path,rows):
    with Path(path).open("w",encoding="utf-8",newline="") as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]))
        w.writeheader();w.writerows(rows)


def analyze(energy_run,timing_run,cache_root,output,*,mu,sigma,
            low_sigma,high_sigma,tail_fraction,bootstrap,seed):
    ec=resolve_run_config(energy_run)
    ids,amp,pred,sources=_load_pair(energy_run,timing_run,cache_root)
    if mu is None or sigma is None:
        mu,sigma,fit_source=_load_frozen_fit(cache_root,ec["control"])
    else:
        mu=np.asarray(mu,dtype=float);sigma=np.asarray(sigma,dtype=float)
        fit_source="explicit frozen control-fit parameters supplied via CLI"
    if mu.shape!=(2,) or sigma.shape!=(2,) or not np.isfinite(mu).all() or np.any(sigma<=0):
        raise ValueError("Two finite means and two positive sigmas are required")
    z=(amp-mu[None,:])/sigma[None,:]
    # Population has already passed the frozen selection; explicitly preserve it.
    keep=np.all((z>=low_sigma)&(z<=high_sigma),axis=1)
    removed=~keep
    outliers=_original_outlier_groups(pred,tail_fraction)
    n=len(ids)
    rows=[];comparisons=[]
    for kind in ("timing","energy","either","both"):
        original=outliers[kind]
        for state in ("LED","ML"):
            x=np.asarray(pred["energy"][state],float)
            y=np.asarray(pred["timing"][state],float)
            groups={
                "removed_original_outliers":original&removed,
                "retained_original_outliers":original&keep,
                "all_original_outliers":original,
                "removed_nonoutliers":(~original)&removed,
                "retained_nonoutliers":(~original)&keep,
                "all_events":np.ones(n,bool),
            }
            for name,mask in groups.items():
                a,b=x[mask],y[mask]
                row={"outlier_definition":kind,"state":state,"group":name,
                     "fraction_of_original_events":float(np.mean(mask)),
                     "n_original_outliers":int(original.sum()),
                     "n_cut_removed":int(removed.sum())}
                row.update(_stats(a,b))
                ci=_ci_corr(a,b,bootstrap,seed)
                row.update(pearson_ci_low=ci[0],pearson_ci_high=ci[1])
                rows.append(row)
            a=np.column_stack([x[original&removed],y[original&removed]])
            b=np.column_stack([x[original&keep],y[original&keep]])
            ra=_stats(a[:,0],a[:,1])["pearson"]
            rb=_stats(b[:,0],b[:,1])["pearson"]
            ci=_ci_difference(a,b,bootstrap,seed)
            comparisons.append({
                "outlier_definition":kind,"state":state,
                "n_removed_outliers":len(a),"n_retained_outliers":len(b),
                "pearson_removed":ra,"pearson_retained":rb,
                "pearson_difference_removed_minus_retained":(
                    ra-rb if ra is not None and rb is not None else None),
                "difference_ci_low":ci[0],"difference_ci_high":ci[1],
            })
    out=Path(output).expanduser().resolve()
    out.mkdir(parents=True,exist_ok=True)
    _write_csv(out/"symmetry_outlier_correlations.csv",rows)
    _write_csv(out/"symmetry_outlier_correlation_differences.csv",comparisons)
    eventrows=[]
    for d in range(2):
        for name,mask in (("removed_outlier_timing",removed&outliers["timing"]),
                          ("retained_outlier_timing",keep&outliers["timing"]),
                          ("removed_all",removed),("retained_all",keep)):
            eventrows.append({
                "detector":f"D{d+1}","group":name,"n":int(mask.sum()),
                "z_mean":float(np.mean(z[mask,d])) if mask.any() else None,
                "z_median":float(np.median(z[mask,d])) if mask.any() else None,
                "z_over_plus2_count":int(np.sum(mask&(z[:,d]>2))),
                "z_over_plus2_fraction":float(np.mean(z[mask,d]>2)) if mask.any() else None,
            })
    _write_csv(out/"symmetry_amplitude_composition.csv",eventrows)
    for kind in ("timing","both"):
        fig,axes=plt.subplots(1,2,figsize=(11,4.6),constrained_layout=True)
        original=outliers[kind]
        for ax,state in zip(axes,("LED","ML")):
            x=pred["energy"][state];y=pred["timing"][state]
            for label,mask in (("Removed",original&removed),("Retained",original&keep)):
                ax.scatter(x[mask],y[mask],s=15,alpha=.55,
                           label=f"{label} (n={int(mask.sum())})")
            ax.set_title(f"{state}: original {kind} ML outliers")
            ax.set_xlabel("Energy residual [ps]")
            ax.set_ylabel("Timing residual [ps]")
            ax.axhline(0,color="gray",lw=.5)
            ax.axvline(0,color="gray",lw=.5)
            ax.legend()
            ax.grid(alpha=.2)
        fig.savefig(out/f"symmetry_outliers_{kind}_scatter.png",dpi=180)
        plt.close(fig)
    (out/"symmetry_outlier_analysis.json").write_text(json.dumps({
        "n_original":n,"n_removed":int(removed.sum()),
        "n_retained":int(keep.sum()),
        "frozen_fit_mu_mV":mu.tolist(),"frozen_fit_sigma_mV":sigma.tolist(),
        "fit_source":fit_source,"z_low":low_sigma,"z_high":high_sigma,
        "tail_fraction":tail_fraction,
        "outlier_counts_original":{k:int(v.sum()) for k,v in outliers.items()},
        "outlier_counts_removed":{k:int((v&removed).sum()) for k,v in outliers.items()},
        "outlier_counts_retained":{k:int((v&keep).sum()) for k,v in outliers.items()},
        "prediction_sources":sources,"bootstrap":bootstrap,
        "inference_caution":"Conditioning on extreme residual outcomes changes correlations. Lower correlation is not proof of amplifier noise. Shared target calibration error can induce correlation. CI is conditional on frozen populations/cuts and fitted models.",
    },indent=2),encoding="utf-8")
    print(f"Analyzed {n} original blind events; retained {int(keep.sum())}, removed {int(removed.sum())}")
    for v in comparisons:
        if v["state"]=="ML":
            print(f'{v["outlier_definition"]}: removed n={v["n_removed_outliers"]} '
                  f'rho={v["pearson_removed"]}, retained n={v["n_retained_outliers"]} '
                  f'rho={v["pearson_retained"]}; difference CI='
                  f'[{v["difference_ci_low"]},{v["difference_ci_high"]}]')
    print(f"Saved correlation tables, comparison CIs, and scatter plots: {out}")


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--energy-run",type=Path,required=True)
    p.add_argument("--timing-run",type=Path,required=True)
    p.add_argument("--cache-root",type=Path,required=True)
    p.add_argument("--output-dir",type=Path,required=True)
    p.add_argument("--mu-mv",nargs=2,type=float,metavar=("D1","D2"),
                   help="Frozen control photopeak mean in mV (autodetected from control cache if omitted)")
    p.add_argument("--sigma-mv",nargs=2,type=float,metavar=("D1","D2"),
                   help="Frozen control photopeak sigma in mV (required with --mu-mv)")
    p.add_argument("--low-sigma",type=float,default=-2.0)
    p.add_argument("--high-sigma",type=float,default=2.0)
    p.add_argument("--tail-fraction",type=float,default=.002)
    p.add_argument("--bootstrap",type=int,default=500)
    p.add_argument("--seed",type=int,default=42)
    args=p.parse_args()
    if (args.mu_mv is None)!=(args.sigma_mv is None):
        p.error("Supply both --mu-mv and --sigma-mv, or neither")
    if not (0<args.tail_fraction<1) or args.bootstrap<0 or args.low_sigma>=args.high_sigma:
        p.error("Invalid tail fraction, bootstrap count, or sigma interval")
    analyze(args.energy_run.expanduser().resolve(),args.timing_run.expanduser().resolve(),
            args.cache_root.expanduser().resolve(),args.output_dir,
            mu=args.mu_mv,sigma=args.sigma_mv,
            low_sigma=args.low_sigma,high_sigma=args.high_sigma,
            tail_fraction=args.tail_fraction,bootstrap=args.bootstrap,seed=args.seed)


if __name__=="__main__":
    main()
