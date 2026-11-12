"""Explore a uniform timing-residual background and its contribution beneath CTR peak.

Uses original full blind LED/ML residual distributions, not tail-selected samples.
The +/- coincidence gate is a MODEL assumption, not assumed to truncate
ML-corrected residuals. Background is estimated from fixed symmetric sidebands
and is extrapolated to the core ONLY if sideband checks are plausible.
No ROOT access, retraining or model predictions.
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
from scipy.stats import binomtest, kstest, chi2

from waveform_analysis.scripts.plot_residual_correlation_percentiles import _read
from waveform_analysis.signal.metrics import fit_ctr_ps, direct_fwhm_from_histogram
from utils_fit.binning import fixed_width_histogram_edges


def _write(path,rows):
    with path.open("w",newline="",encoding="utf-8") as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]))
        writer.writeheader();writer.writerows(rows)


def _fit_fwhm(residual,bin_ps,background_density=None):
    edges=fixed_width_histogram_edges(residual,bin_ps)
    counts,_=np.histogram(residual,edges)
    if background_density is not None:
        # Uniform pedestal in expected *counts* per bin. F1 width after
        # subtracting it from the original histogram; never delete events.
        counts=np.maximum(counts-background_density*np.diff(edges),0.)
    try:
        fit=direct_fwhm_from_histogram(counts,edges)
        return float(fit.ctr_ps),float(fit.center_ps)
    except ValueError:
        return None,None


def _analyse(x,gate,inner,outer,core_width,bin_ps,coarse_bins,alpha):
    n=len(x)
    bounds=(np.abs(x)>=inner)&(np.abs(x)<=outer)
    tail=x[bounds]
    k=len(tail)
    sideband_length=2*(outer-inner)
    density=k/sideband_length if sideband_length>0 else float("nan")
    # With a fixed, predeclared sideband interval the uniform CDF null
    # has no estimated shape parameter. Conditioning on k is valid.
    positions=np.where(tail<0,(tail+outer)/(outer-inner)*.5,
                       .5+(tail-inner)/(outer-inner)*.5)
    # This transformation is continuous for symmetric sidebands.
    uniform_ks=float(kstest(positions,"uniform").pvalue) if k>=5 else None
    positive=int(np.sum(tail>0)); negative=int(np.sum(tail<0))
    side_sign=float(binomtest(positive,k,p=.5).pvalue) if k else None
    per_bin=np.histogram(positions,bins=np.linspace(0,1,coarse_bins+1))[0]
    expected=k/coarse_bins
    chi_stat=float(np.sum((per_bin-expected)**2/expected)) if expected else None
    chi_p=float(chi2.sf(chi_stat,coarse_bins-1)) if chi_stat is not None else None
    # Exact Poisson interval for the sideband intensity, per ps.
    ci_low=(.5*chi2.ppf(alpha/2,2*k)/sideband_length) if k else 0.
    ci_high=.5*chi2.ppf(1-alpha/2,2*(k+1))/sideband_length
    in_core=int(np.sum(np.abs(x)<=core_width))
    predicted_uniform_core=density*2*core_width
    original_fwhm,peak=_fit_fwhm(x,bin_ps)
    corrected_fwhm,_=_fit_fwhm(x,bin_ps,density)
    full_gate_count=int(np.sum(np.abs(x)<=gate))
    # Extrapolation of uniformly distributed contamination conditional on
    # the selected gate; fraction of all input residuals, not a physical
    # pileup incidence estimate.
    uniform_gate_count=density*2*gate
    return {
        "n_events":n,
        "n_within_assumed_gate":full_gate_count,
        "n_outside_assumed_gate":n-full_gate_count,
        "sideband_abs_low_ps":inner,
        "sideband_abs_high_ps":outer,
        "sideband_n":k,
        "sideband_density_events_per_ps":density,
        "density_ci_low":float(ci_low),
        "density_ci_high":float(ci_high),
        "uniform_ks_p":uniform_ks,
        "sideband_sign_p":side_sign,
        "coarse_uniform_chisq_p":chi_p,
        "coarse_bins":coarse_bins,
        "sideband_positive":positive,
        "sideband_negative":negative,
        "predicted_uniform_within_gate":uniform_gate_count,
        "predicted_uniform_fraction_all_events":uniform_gate_count/n,
        "predicted_uniform_within_core":predicted_uniform_core,
        "predicted_uniform_within_core_ci_low":float(ci_low*2*core_width),
        "predicted_uniform_within_core_ci_high":float(ci_high*2*core_width),
        "observed_within_core":in_core,
        "predicted_uniform_fraction_of_observed_core":predicted_uniform_core/in_core if in_core else None,
        "F1_CTR_full_ps":original_fwhm,
        "F1_CTR_subtract_pedestal_ps":corrected_fwhm,
        "delta_F1_CTR_ps":original_fwhm-corrected_fwhm if original_fwhm is not None and corrected_fwhm is not None else None,
        "histogram_peak_center_ps":peak,
    },per_bin


def _plot(x,gate,inner,outer,core_width,bin_ps,r,out):
    fig,axs=plt.subplots(1,2,figsize=(13,4.8),constrained_layout=True)
    edges=np.arange(-gate,gate+bin_ps,bin_ps)
    count,_=np.histogram(x,bins=edges)
    center=(edges[1:]+edges[:-1])/2
    axs[0].step(center,count,where="mid",lw=1.2,label="Observed",color="C0")
    axs[0].axhline(r["sideband_density_events_per_ps"]*bin_ps,
                   color="C1",label="Uniform sideband estimate")
    for v in (-outer,-inner,inner,outer):
        axs[0].axvline(v,color="gray",lw=.7,linestyle=":")
    axs[0].set_yscale("symlog",linthresh=1)
    axs[0].set_xlabel("Residual [ps]")
    axs[0].set_ylabel(f"Counts per {bin_ps:g} ps")
    axs[0].set_title("Full residuals, assumed coincidence gate")
    axs[0].legend(fontsize=8)
    zoom=150
    view=(center>=-zoom)&(center<=zoom)
    axs[1].step(center[view],count[view],where="mid",
                color="C0",lw=1.3,label="Observed")
    axs[1].axhline(r["sideband_density_events_per_ps"]*bin_ps,
                   color="C1",label="Estimated pedestal")
    axs[1].axvspan(-core_width,core_width,color="gray",alpha=.08,
                   label="Requested central interval")
    axs[1].set_xlabel("Residual [ps]")
    axs[1].set_ylabel(f"Counts per {bin_ps:g} ps")
    axs[1].set_title("CTR peak and extrapolated pedestal")
    axs[1].legend(fontsize=8)
    for ax in axs:ax.grid(alpha=.15)
    fig.savefig(out,dpi=180)
    plt.close(fig)


def run(energy_run,timing_run,cache_root,output,*,
        gate,inner_values,outer,core_width,bin_ps,coarse_bins,alpha):
    if not (0<core_width<min(inner_values) and max(inner_values)<outer<=gate):
        raise ValueError("Need 0<core_width<min(sideband starts)<outer<=gate")
    root=cache_root/"artifact_store"
    energy=_read(energy_run,root)
    timing=_read(timing_run,root)
    ids,ei,ti=np.intersect1d(energy[0],timing[0],return_indices=True,assume_unique=True)
    if len(ids)<100:
        raise ValueError("Not enough common blind predictions")
    out=Path(output).expanduser().resolve();out.mkdir(parents=True,exist_ok=True)
    rows=[];uniform_bins=[]
    for name,data,index in (("energy",energy,ei),("timing",timing,ti)):
        for state,position in (("LED",1),("ML",2)):
            x=np.asarray(data[position][index],float)
            for inner in inner_values:
                row,counts=_analyse(x,gate,inner,outer,core_width,
                                    bin_ps,coarse_bins,alpha)
                row={"channel":name,"state":state,**row}
                rows.append(row)
                for j,count in enumerate(counts):
                    uniform_bins.append({
                        "channel":name,"state":state,
                        "sideband_abs_low_ps":inner,
                        "sideband_abs_high_ps":outer,
                        "sideband_bin":j+1,
                        "observed":int(count),
                        "expected_uniform":float(row["sideband_n"]/coarse_bins),
                    })
                if inner==inner_values[0]:
                    _plot(x,gate,inner,outer,core_width,bin_ps,row,
                          out/f"uniform_background_{name}_{state.lower()}.png")
    _write(out/"uniform_background_summary.csv",rows)
    _write(out/"uniform_background_sideband_bins.csv",uniform_bins)
    summary={
        "n_matched_blind_events":int(len(ids)),
        "energy_predictions":energy[3],"timing_predictions":timing[3],
        "gate_half_width_ps":gate,
        "sideband_start_ps":list(inner_values),
        "sideband_end_ps":outer,
        "central_half_width_ps":core_width,
        "histogram_bin_ps":bin_ps,
        "coarse_sideband_bins":coarse_bins,
        "key_cautions":[
            "The +/- gate is a hypothesized selection on raw coincidence timestamps; corrected residuals are not guaranteed to remain within it.",
            "Conditional uniformity in sidebands does NOT establish uniformity beneath the core.",
            "Only extrapolate a pedestal if the sideband profile is approximately constant across multiple boundaries and both signs.",
            "High-p-values do not prove a uniform distribution; small sideband counts have limited power.",
            "The pedestal-subtracted F1 is an exploratory histogram statistic, NOT the original repository's standard CTR estimator.",
            "Core broadening from non-uniform artifacts remains unidentified; uniform pedestal extrapolation cannot quantify it.",
        ],
    }
    (out/"uniform_background_assumptions.json").write_text(
        json.dumps(summary,indent=2),encoding="utf-8")
    for row in rows:
        if row["channel"]=="timing" and row["state"]=="ML":
            print(f'sideband |r|={row["sideband_abs_low_ps"]:g}..{outer:g} ps '
                  f'| N={row["sideband_n"]} | KS p={row["uniform_ks_p"]} '
                  f'| predicted core={row["predicted_uniform_within_core"]:.2f} '
                  f'| CTR F1={row["F1_CTR_full_ps"]} ps '
                  f'| pedestal-subtracted={row["F1_CTR_subtract_pedestal_ps"]} ps')
    print(f"Saved results for {len(ids)} paired blind events: {out}")


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--energy-run",type=Path,required=True)
    p.add_argument("--timing-run",type=Path,required=True)
    p.add_argument("--cache-root",type=Path,required=True)
    p.add_argument("--output-dir",type=Path,required=True)
    p.add_argument("--gate-ps",type=float,default=2000.)
    p.add_argument("--sideband-start-ps",type=float,nargs="+",default=[150.,250.,400.])
    p.add_argument("--sideband-end-ps",type=float,default=1800.)
    p.add_argument("--core-half-width-ps",type=float,default=50.)
    p.add_argument("--histogram-bin-ps",type=float,default=5.)
    p.add_argument("--sideband-bins",type=int,default=12)
    args=p.parse_args()
    if args.histogram_bin_ps<=0 or args.sideband_bins<4:
        p.error("Invalid histogram bin or sideband count")
    run(args.energy_run.expanduser().resolve(),
        args.timing_run.expanduser().resolve(),
        args.cache_root.expanduser().resolve(),args.output_dir,
        gate=args.gate_ps,inner_values=sorted(set(args.sideband_start_ps)),
        outer=args.sideband_end_ps,core_width=args.core_half_width_ps,
        bin_ps=args.histogram_bin_ps,coarse_bins=args.sideband_bins,alpha=.05)


if __name__=="__main__":
    main()
