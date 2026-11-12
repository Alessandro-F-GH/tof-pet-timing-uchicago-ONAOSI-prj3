"""Acquisition-order diagnostics for frozen TOF-PET extreme residuals.

Order is based on original event_id, not time: event IDs are a proxy for
chronology, not elapsed seconds. Null permutations draw uniformly from the
exact existing paired blind-event population; no rate/time calibration.
No ROOT access, inference, retraining or new selection.
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
from scipy.stats import binomtest

from waveform_analysis.scripts.scan_photopeak_ctr import _load_pair, _original_outlier_groups
from waveform_analysis.scripts.compare_symmetric_photopeak_outliers import _load_frozen_fit
from waveform_analysis.data.results import resolve_run_config


def _write(path, rows):
    with path.open("w",encoding="utf-8",newline="") as stream:
        w=csv.DictWriter(stream,fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def _deviation(indices, population_size):
    """Max separation of empirical event positions from population ranks."""
    if not len(indices):
        return float("nan")
    u=np.sort((indices+0.5)/population_size)
    k=np.arange(1,len(indices)+1)/len(indices)
    return float(max(np.max(k-u),np.max(u-(k-1/len(indices)))))


def _block_metrics(indices, n, bins):
    edges=np.linspace(0,n,bins+1)
    counts=np.histogram(indices,bins=edges)[0]
    expected=len(indices)/bins
    statistic=float(np.sum((counts-expected)**2/expected)) if expected else float("nan")
    return counts,statistic


def _pair_count(indices, n):
    """Number of adjacent selected records, conditional on the chosen sample size."""
    if len(indices)<2:
        return 0
    return int(np.sum(np.diff(np.sort(indices))==1))


def _permutation_test(indices,n,bins,draws,rng):
    """Permutation null conditioned on n events, k outliers and each block."""
    k=len(indices)
    if k<2:
        return dict(ks_statistic=None,ks_permutation_p=None,
                    block_chisq=None,block_permutation_p=None,
                    adjacent_pairs=None,adjacent_pairs_p_high=None,
                    adjacent_pairs_null_mean=None,
                    adjacent_pairs_null_ci_low=None,adjacent_pairs_null_ci_high=None)
    d=_deviation(indices,n)
    count,chi=_block_metrics(indices,n,bins)
    adjacency=_pair_count(indices,n)
    null_d=np.empty(draws)
    null_chi=np.empty(draws)
    null_adj=np.empty(draws)
    for j in range(draws):
        sample=np.sort(rng.choice(n,k,replace=False))
        null_d[j]=_deviation(sample,n)
        null_chi[j]=_block_metrics(sample,n,bins)[1]
        null_adj[j]=_pair_count(sample,n)
    return dict(
        ks_statistic=d,
        ks_permutation_p=float((1+np.sum(null_d>=d))/(draws+1)),
        block_chisq=chi,
        block_permutation_p=float((1+np.sum(null_chi>=chi))/(draws+1)),
        adjacent_pairs=adjacency,
        adjacent_pairs_p_high=float((1+np.sum(null_adj>=adjacency))/(draws+1)),
        adjacent_pairs_null_mean=float(np.mean(null_adj)),
        adjacent_pairs_null_ci_low=float(np.quantile(null_adj,.025)),
        adjacent_pairs_null_ci_high=float(np.quantile(null_adj,.975)),
    ),count


def _sign_stats(values):
    pos=int(np.sum(values>0));neg=int(np.sum(values<0))
    valid=pos+neg
    if not valid:
        return dict(n_positive=0,n_negative=0,n_zero=int(len(values)),
                    fraction_positive=None,binomial_p_two_sided=None,
                    positive_fraction_ci_low=None,positive_fraction_ci_high=None)
    test=binomtest(pos,valid,p=.5,alternative="two-sided")
    ci=test.proportion_ci(confidence_level=.95,method="exact")
    return dict(
        n_positive=pos,n_negative=neg,n_zero=int(len(values)-valid),
        fraction_positive=pos/valid,binomial_p_two_sided=float(test.pvalue),
        positive_fraction_ci_low=float(ci.low),
        positive_fraction_ci_high=float(ci.high),
    )


def analyze(energy_run,timing_run,cache_root,output,*,
            tail_fraction,bins,permutations,seed,fit_requested):
    ids,amp,pred,sources=_load_pair(energy_run,timing_run,cache_root)
    n=len(ids)
    order=np.argsort(ids,kind="stable")
    ids=ids[order]
    amp=amp[order]
    pred={mode:{state:np.asarray(vals)[order] for state,vals in states.items()}
          for mode,states in pred.items()}
    if np.any(np.diff(ids)<=0):
        raise ValueError("event_id must be unique and monotonically rankable")
    extremes=_original_outlier_groups(pred,tail_fraction)
    populations={
        "all_events":np.ones(n,dtype=bool),
        "timing_outliers":extremes["timing"],
        "energy_outliers":extremes["energy"],
        "either_outliers":extremes["either"],
        "shared_outliers":extremes["both"],
    }
    fit_source=None
    if fit_requested:
        ec=resolve_run_config(energy_run)
        mu,sigma,fit_source=_load_frozen_fit(cache_root,ec["control"])
        z=(amp-mu[None,:])/sigma[None,:]
        removed=np.any(z>2,axis=1)
        populations.update({
            "timing_outliers_above_plus2sigma":extremes["timing"]&removed,
            "timing_outliers_within_plus2sigma":extremes["timing"]&(~removed),
            "shared_outliers_above_plus2sigma":extremes["both"]&removed,
            "shared_outliers_within_plus2sigma":extremes["both"]&(~removed),
        })
    rows=[]
    blockrows=[]
    np.random.seed(seed)  # plots are deterministic; permutations use local RNG below
    for label,mask in populations.items():
        positions=np.flatnonzero(mask)
        if label=="all_events":
            permutation=dict(ks_statistic=None,ks_permutation_p=None,
                block_chisq=None,block_permutation_p=None,
                adjacent_pairs=None,adjacent_pairs_p_high=None,
                adjacent_pairs_null_mean=None,adjacent_pairs_null_ci_low=None,
                adjacent_pairs_null_ci_high=None)
            counts=np.histogram(positions,bins=np.linspace(0,n,bins+1))[0]
        else:
            permutation,counts=_permutation_test(
                positions,n,bins,permutations,
                np.random.default_rng(seed+sum(ord(c) for c in label)))
        for j,value in enumerate(counts):
            blockrows.append({
                "population":label,"block":j+1,
                "block_start_percent":100*j/bins,
                "block_end_percent":100*(j+1)/bins,
                "n_outliers":int(value),
                "n_all":int(np.histogram(np.arange(n),bins=np.linspace(0,n,bins+1))[0][j]),
            })
        for state in ("LED","ML"):
            for channel in ("timing","energy"):
                values=pred[channel][state][mask]
                row={
                    "population":label,"channel":channel,"state":state,
                    "n_events":int(mask.sum()),"fraction_original":float(mask.mean()),
                    "first_event_id":int(ids[positions[0]]) if len(positions) else None,
                    "last_event_id":int(ids[positions[-1]]) if len(positions) else None,
                    **_sign_stats(values),**permutation,
                }
                rows.append(row)
    out=Path(output).expanduser().resolve()
    out.mkdir(parents=True,exist_ok=True)
    _write(out/"outlier_acquisition_order_tests.csv",rows)
    _write(out/"outlier_acquisition_order_blocks.csv",blockrows)
    for label in ("timing_outliers","shared_outliers",
                  "timing_outliers_above_plus2sigma",
                  "timing_outliers_within_plus2sigma"):
        if label not in populations:continue
        mask=populations[label]
        selected=np.flatnonzero(mask)
        fig,(ax1,ax2)=plt.subplots(2,1,figsize=(10,6.5),constrained_layout=True)
        if len(selected):
            ax1.step((selected+.5)/n*100,
                     np.arange(1,len(selected)+1)/len(selected),
                     where="post",label="Observed outlier CDF")
        ax1.plot([0,100],[0,1],"--",color="gray",label="Uniform-per-event null")
        ax1.set_ylabel("Cumulative fraction")
        ax1.set_xlabel("Acquisition order percentile")
        ax1.grid(alpha=.2);ax1.legend()
        data=[r for r in blockrows if r["population"]==label]
        ax2.bar([r["block"] for r in data],[r["n_outliers"] for r in data])
        ax2.axhline(len(selected)/bins,color="gray",linestyle="--")
        ax2.set_xlabel("Equal-count acquisition block")
        ax2.set_ylabel("Outlier count")
        ax2.grid(alpha=.2)
        fig.suptitle(label.replace("_"," ")+" — original event ID ordering")
        fig.savefig(out/f"{label}_acquisition_order.png",dpi=180)
        plt.close(fig)
    summary={
        "n_events":n,"min_event_id":int(ids[0]),"max_event_id":int(ids[-1]),
        "event_id_ordering":"Sort original event IDs; no assumption that ID differences are elapsed time",
        "null_hypothesis":"Fixed number of outlier labels uniformly permuted across the existing paired blind population; accounts for preselection gaps in source event IDs",
        "test_definitions":{
            "uniform_cdf":"Empirical outlier acquisition-rank distribution, permutation calibrated",
            "block_chisq":"Equal event-count blocks, permutation calibrated",
            "adjacent_pairs":"Number of neighboring selected-event indices both outliers, upper-tail permutation p",
            "binomial_sign":"Two-sided exact test against 50/50 positive/negative within each selected group; not entailed by stationary pileup",
        },
        "tail_fraction":tail_fraction,"n_bins":bins,"permutations":permutations,
        "sources":sources,"frozen_fit_source":fit_source,
        "interpretation_limitations":[
            "Event index is not a timestamp; cannot test constant rate per second",
            "ML outliers are defined using absolute ML residual; sign test need not be symmetric",
            "Temporal uniformity/independence is compatible with pile-up but cannot identify it",
            "Statistical exploration on blind data does not constitute independent confirmatory validation",
        ],
    }
    (out/"outlier_acquisition_order.json").write_text(
        json.dumps(summary,indent=2),encoding="utf-8")
    print(f"Analyzed {n} blind events in original ID order; outputs: {out}")
    for row in rows:
        if row["channel"]=="timing" and row["state"]=="ML" and row["population"]!="all_events":
            print(f'{row["population"]}: n={row["n_events"]}, '
                  f'positive={row["n_positive"]}, negative={row["n_negative"]}, '
                  f'binomial_p={row["binomial_p_two_sided"]}, '
                  f'uniform_p={row["ks_permutation_p"]}, '
                  f'blocks_p={row["block_permutation_p"]}, '
                  f'adjacency_p={row["adjacent_pairs_p_high"]}')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--energy-run",type=Path,required=True)
    p.add_argument("--timing-run",type=Path,required=True)
    p.add_argument("--cache-root",type=Path,required=True)
    p.add_argument("--output-dir",type=Path,required=True)
    p.add_argument("--tail-fraction",type=float,default=.002)
    p.add_argument("--bins",type=int,default=20)
    p.add_argument("--permutations",type=int,default=2000)
    p.add_argument("--seed",type=int,default=42)
    p.add_argument("--split-high-photopeak",action="store_true",
        help="Also analyze original timing/shared outliers above +2 fitted control sigma in either energy detector")
    args=p.parse_args()
    if not 0<args.tail_fraction<1 or args.bins<2 or args.permutations<100:
        p.error("Need 0<tail-fraction<1, bins>=2, permutations>=100")
    analyze(args.energy_run.expanduser().resolve(),
            args.timing_run.expanduser().resolve(),
            args.cache_root.expanduser().resolve(),args.output_dir,
            tail_fraction=args.tail_fraction,bins=args.bins,
            permutations=args.permutations,seed=args.seed,
            fit_requested=args.split_high_photopeak)


if __name__=="__main__":
    main()
