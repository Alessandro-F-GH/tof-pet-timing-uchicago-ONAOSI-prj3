"""Post-hoc photopeak-window CTR scan, using frozen native preprocessing and blind predictions.

No ROOT scans, refits or inference. Conditions on the existing blind/selection population.
Widths refer to fractions of the full *observed accepted* amplitude excursion about
the center in each detector; they do not re-estimate the original photopeak fit.
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

from waveform_analysis.data.results import resolve_run_config
from waveform_analysis.data.named_cache import native_alias, resolve_alias, alias_exists
from waveform_analysis.scripts.plot_residual_correlation_percentiles import _read
from waveform_analysis.signal.metrics import fit_ctr_ps, rmse_ps

FRACTIONS = (1.0, .9, .8, .7, .6, .5, .4, .3, .2, .1)
OUTLIER_KINDS = ("timing", "energy", "either", "both")


def _native_energy(run, cache_root):
    cfg=resolve_run_config(run)
    alias=native_alias(cache_root, cfg["blind"], cfg["blind"], "energy_to_energy")
    if not alias_exists(alias):
        raise FileNotFoundError(
            f"Native frozen energy artifact missing: {alias}. "
            "No ROOT fallback or recomputation is performed.")
    path=resolve_alias(alias)
    ids=np.load(path/"event_index.npy",mmap_mode="r")
    waves=np.load(path/"energy_windows_mV.npy",mmap_mode="r")
    if waves.ndim!=3 or waves.shape[1]!=2 or len(ids)!=len(waves):
        raise ValueError(f"Invalid frozen native energy waveforms: {path}")
    # The native window includes the energy rising edge/peak by construction.
    # We intentionally use the peak in physical oriented mV, not clipped ML input.
    amp=np.nanmax(waves,axis=2).astype(np.float64)
    if not np.all(np.isfinite(amp)):
        raise ValueError("Nonfinite energy peaks in native preprocessing")
    ids=np.asarray(ids,np.int64)
    if len(np.unique(ids))!=len(ids):
        raise ValueError("Duplicate event IDs in native preprocessing")
    return ids,amp,str(path)


def _load_pair(energy_run,timing_run,cache_root):
    ec,tc=resolve_run_config(energy_run),resolve_run_config(timing_run)
    if ec["mode"]!="energy_to_energy" or tc["mode"]!="timing_to_timing":
        raise ValueError("Expected energy and timing runs, respectively")
    e_name=Path(str(ec["blind"]["root_file"]).replace("\\","/")).name
    t_name=Path(str(tc["blind"]["root_file"]).replace("\\","/")).name
    if e_name!=t_name or ec["blind"]["channels"]!=tc["blind"]["channels"]:
        raise ValueError("Runs do not refer to the same blind acquisition/channel mapping")
    artifact_root=cache_root/"artifact_store"
    energy=_read(energy_run,artifact_root)
    timing=_read(timing_run,artifact_root)
    native=_native_energy(energy_run,cache_root)
    ids,ei,ti=np.intersect1d(energy[0],timing[0],assume_unique=True,return_indices=True)
    ids2,bi,ni=np.intersect1d(ids,native[0],assume_unique=True,return_indices=True)
    if len(ids2)<100:
        raise ValueError(f"Only {len(ids2)} paired blind events also have native amplitude")
    # Do not silently assign predictions to the wrong acquisition.
    if len(ids2)!=len(ids):
        raise ValueError(f"Native preprocessing missing {len(ids)-len(ids2)} paired prediction IDs")
    i,j=ei[bi],ti[bi]
    return ids2,native[1][ni],{
        "energy":{"LED":energy[1][i],"ML":energy[2][i]},
        "timing":{"LED":timing[1][j],"ML":timing[2][j]},
    },{"energy_predictions":energy[3],"timing_predictions":timing[3],
       "native_energy":native[2],"blind_root_name":e_name}


def _score(residuals, bin_ps, bootstrap, seed):
    fit=fit_ctr_ps(residuals,{
        "histogram_bin_width_ps":bin_ps,
        "bootstrap_samples":bootstrap,
    },seed=seed,bootstrap=bootstrap>1)
    return {"ctr_ps":float(fit.ctr_ps),
            "ctr_bootstrap_error_ps":float(fit.ctr_error_ps),
            "rmse_ps":rmse_ps(residuals),
            "sigma_ps":float(np.std(residuals,ddof=1)),
            "mean_ps":float(np.mean(residuals))}


def _original_outlier_groups(pred, tail_fraction):
    """Freeze the original ML extreme-event IDs once, before energy cuts.

    energy/timing: largest absolute residuals in that channel.
    either: union of individual-channel tails.
    both: intersection of individual-channel tails.
    """
    if not 0 < tail_fraction < 1:
        raise ValueError("tail_fraction must be between zero and one")
    n=len(pred["timing"]["ML"])
    count=max(1,int(np.ceil(n*tail_fraction)))
    tail={}
    for mode in ("energy","timing"):
        value=np.abs(pred[mode]["ML"])
        chosen=np.argsort(value,kind="stable")[-count:]
        mask=np.zeros(n,dtype=bool)
        mask[chosen]=True
        tail[mode]=mask
    tail["either"]=tail["energy"]|tail["timing"]
    tail["both"]=tail["energy"]&tail["timing"]
    return tail


def _extreme_report(groups, keep, pred, fraction, n_original):
    """All contributions are measured on a fixed baseline-defined group."""
    records=[]
    n_selected=int(keep.sum())
    for kind,original in groups.items():
        original_count=int(original.sum())
        surviving=original&keep
        survivor_count=int(surviving.sum())
        record={
            "fraction_of_observed_halfspan":float(fraction),
            "outlier_group":kind,
            "outliers_original":original_count,
            "outliers_surviving":survivor_count,
            "outlier_survival_fraction":(
                survivor_count/original_count if original_count else None),
            "outliers_removed":original_count-survivor_count,
            "n_events_retained":n_selected,
            "acceptance_vs_original":n_selected/n_original,
            "outlier_prevalence_among_retained":(
                survivor_count/n_selected if n_selected else None),
            "retention_enrichment_vs_all":(
                (survivor_count/original_count)/(n_selected/n_original)
                if original_count and n_selected else None),
        }
        for mode in ("timing","energy"):
            for state in ("LED","ML"):
                residual=np.asarray(pred[mode][state],dtype=float)
                key=f"{mode}_{state.lower()}"
                tail_mse_sum=float(np.sum(np.square(residual[surviving])))
                all_mse_sum=float(np.sum(np.square(residual[keep])))
                initial_mse_sum=float(np.sum(np.square(residual[original])))
                record[key+"_outlier_rmse_ps"]=(
                    float(np.sqrt(tail_mse_sum/survivor_count))
                    if survivor_count else None)
                record[key+"_outlier_rmse_initial_ps"]=(
                    float(np.sqrt(initial_mse_sum/original_count))
                    if original_count else None)
                record[key+"_outlier_mse_contribution_to_retained_mse_ps2"]=(
                    tail_mse_sum/n_selected if n_selected else None)
                record[key+"_outlier_fraction_of_retained_sse"]=(
                    tail_mse_sum/all_mse_sum if all_mse_sum>0 else None)
                record[key+"_outlier_fraction_of_original_tail_sse_surviving"]=(
                    tail_mse_sum/initial_mse_sum if initial_mse_sum>0 else None)
        records.append(record)
    return records


def _plot_extremes(records,output):
    fig,axes=plt.subplots(1,3,figsize=(16,4.5),constrained_layout=True)
    for kind in OUTLIER_KINDS:
        rr=sorted((r for r in records if r["outlier_group"]==kind),
                  key=lambda r:r["acceptance_vs_original"])
        x=np.array([r["acceptance_vs_original"]*100 for r in rr])
        axes[0].plot(x,[100*r["outlier_survival_fraction"] for r in rr],
                     marker="o",label=kind)
        axes[1].plot(x,[r["timing_ml_outlier_rmse_ps"] for r in rr],
                     marker="o",label=kind)
        axes[2].plot(x,[100*r["timing_ml_outlier_fraction_of_retained_sse"]
                         for r in rr],marker="o",label=kind)
    axes[0].plot([0,100],[0,100],linestyle=":",color="gray",label="Random retention")
    for ax,title,ylabel in zip(axes,
            ("Survival of original ML outliers","RMSE of surviving original ML outliers",
             "Original outliers' share of timing ML squared error"),
            ("Outlier survival [%]","Outlier RMSE [ps]","Fraction of retained SSE [%]")):
        ax.set_xlabel("Overall retained blind events [%]")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(alpha=.2)
        ax.legend(fontsize=8)
    fig.savefig(output/"photopeak_outlier_persistence.png",dpi=180)
    plt.close(fig)


def analyze(energy_run,timing_run,cache_root,output,*,
            fractions,centers,bin_ps,bootstrap,seed,min_events,tail_fraction,amplitude_bin_mv):
    ids,amps,pred,sources=_load_pair(energy_run,timing_run,cache_root)
    n=len(ids)
    groups=_original_outlier_groups(pred,tail_fraction)
    if centers is None:
        centers=np.median(amps,axis=0)
        center_origin="median of already selected blind-event peak amplitudes (exploratory proxy)"
    else:
        centers=np.asarray(centers,dtype=float)
        center_origin="user-provided frozen photopeak peak centers"
    if not np.all(np.isfinite(centers)):
        raise ValueError("Nonfinite center")
    # Define one fixed denominator per detector, so fraction=1 includes all
    # events regardless of asymmetry in the original accepted photopeak window.
    halfspan=np.max(np.abs(amps-centers[None,:]),axis=0)
    if np.any(halfspan<=0):
        raise ValueError("Degenerate photopeak excursion")
    out=Path(output).expanduser().resolve()
    out.mkdir(parents=True,exist_ok=True)
    rows=[]
    extreme_rows=[]
    all_masks={}
    for fraction in sorted(set(fractions),reverse=True):
        if not 0<fraction<=1:
            raise ValueError("All fractions must be in (0,1]")
        radius=fraction*halfspan
        mask=np.all(np.abs(amps-centers[None,:])<=radius[None,:]+1e-9,axis=1)
        all_masks[fraction]=mask
        count=int(mask.sum())
        extreme_rows.extend(_extreme_report(groups,mask,pred,fraction,n))
        for mode in ("timing","energy"):
            for state in ("LED","ML"):
                row={"fraction_of_observed_halfspan":fraction,
                     "mode":mode,"state":state,"n":count,
                     "acceptance_vs_paired_population":count/n,
                     "d1_low_mV":float(centers[0]-radius[0]),
                     "d1_high_mV":float(centers[0]+radius[0]),
                     "d2_low_mV":float(centers[1]-radius[1]),
                     "d2_high_mV":float(centers[1]+radius[1]),
                     "ctr_ps":None,"ctr_bootstrap_error_ps":None,
                     "rmse_ps":None,"sigma_ps":None,"mean_ps":None,
                     "status":"insufficient events" if count<min_events else "ok"}
                if count>=min_events:
                    try:
                        row.update(_score(pred[mode][state][mask],bin_ps,bootstrap,seed))
                    except ValueError as exc:
                        row["status"]=f"CTR fit failure: {exc}"
                rows.append(row)
    with (out/"photopeak_ctr_scan.csv").open("w",newline="",encoding="utf-8") as f:
        writer=csv.DictWriter(f,fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    with (out/"photopeak_outlier_persistence.csv").open(
            "w",newline="",encoding="utf-8") as f:
        writer=csv.DictWriter(f,fieldnames=list(extreme_rows[0]))
        writer.writeheader()
        writer.writerows(extreme_rows)
    _plot_extremes(extreme_rows,out)
    (out/"photopeak_ctr_scan.json").write_text(json.dumps({
        "n_paired_events":n,"sources":sources,
        "center_mV":centers.tolist(),"center_origin":center_origin,
        "observed_halfspan_mV":halfspan.tolist(),
        "definition":"Both native energy peaks must be within +/- fraction * each detector's observed halfspan about its center; fraction=1 retains all baseline accepted events",
        "population":"Same frozen blind-event intersection, frozen LED and ML predictions; no reselection of baseline photopeak and no recalibration",
        "limitation":"Maximum inside native saved window, not a full ROOT waveform scan; verify window contains original energy peak. A narrower cut is post-hoc exploratory and not blind model selection.",
        "ctr":"Repository utils_fit.fit_ctr_ps direct F1 interpolated half-max histogram FWHM",
        "histogram_bin_width_ps":bin_ps,"bootstrap_draws":bootstrap,
        "fractions":list(fractions),"seed":seed,
        "outlier_definition":"Original ML absolute-error top fraction per energy/timing channel; either=union, both=intersection. Original IDs never redefined after cuts.",
        "outlier_tail_fraction_per_channel":tail_fraction,
        "outlier_initial_counts":{k:int(v.sum()) for k,v in groups.items()},
        "outlier_sse":"Sum of squared residuals; fraction_of_retained_sse uses total retained population in denominator",
        "amplitude_histogram_bin_width_mV":amplitude_bin_mv,
    },indent=2),encoding="utf-8")
    for mode in ("timing","energy"):
        fig,axes=plt.subplots(1,2,figsize=(12,4.5),constrained_layout=True)
        for state in ("LED","ML"):
            rr=[r for r in rows if r["mode"]==mode and r["state"]==state and r["status"]=="ok"]
            x=[r["acceptance_vs_paired_population"]*100 for r in rr]
            axes[0].plot(x,[r["ctr_ps"] for r in rr],marker="o",label=state)
            axes[1].plot(x,[r["rmse_ps"] for r in rr],marker="o",label=state)
        for ax,title,ylabel in zip(axes,("F1 CTR FWHM","RMSE"),("CTR [ps]","RMSE [ps]")):
            ax.set_title(f"{mode.capitalize()} — {title}")
            ax.set_xlabel("Retained events among paired blind population [%]")
            ax.set_ylabel(ylabel)
            ax.grid(alpha=.2)
            ax.legend()
        fig.savefig(out/f"photopeak_{mode}_ctr_vs_acceptance.png",dpi=180)
        plt.close(fig)
    fig,ax=plt.subplots(figsize=(9,5),constrained_layout=True)
    min_amp=float(np.min(amps))
    max_amp=float(np.max(amps))
    bins=np.arange(
        np.floor(min_amp/amplitude_bin_mv)*amplitude_bin_mv,
        np.ceil(max_amp/amplitude_bin_mv)*amplitude_bin_mv+1.01*amplitude_bin_mv,
        amplitude_bin_mv)
    for det in range(2):
        ax.hist(amps[:,det],bins=bins,histtype="step",label=f"Energy peak D{det+1}")
        ax.axvline(centers[det],linestyle="--",alpha=.6)
    ax.set_xlabel("Native energy waveform maximum [mV]")
    ax.set_ylabel("Events")
    ax.set_title("Existing blind population; independent per-detector centers")
    ax.legend()
    fig.savefig(out/"photopeak_energy_amplitudes.png",dpi=180)
    plt.close(fig)
    print(f"Read {n} matched blind events; outputs: {out}")
    print(f"Amplitude centers [mV]: {centers.tolist()} ({center_origin})")


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--energy-run",type=Path,required=True)
    p.add_argument("--timing-run",type=Path,required=True)
    p.add_argument("--cache-root",type=Path,required=True)
    p.add_argument("--output-dir",type=Path,required=True)
    p.add_argument("--centers-mv",nargs=2,type=float,metavar=("D1","D2"),
                   help="Frozen reference photopeak centers; omit to use exploratory medians of accepted blind events")
    p.add_argument("--fractions",nargs="+",type=float,default=FRACTIONS)
    p.add_argument("--histogram-bin-ps",type=float,default=5.)
    p.add_argument("--bootstrap",type=int,default=100)
    p.add_argument("--min-events",type=int,default=500)
    p.add_argument("--tail-fraction",type=float,default=.002,
                   help="Original top absolute-error fraction per channel (default 0.002)")
    p.add_argument("--amplitude-bin-mv",type=float,default=1.0,
                   help="Energy amplitude histogram bin width, in mV (minimum 1)")
    p.add_argument("--seed",type=int,default=42)
    args=p.parse_args()
    if args.amplitude_bin_mv<1 or not 0<args.tail_fraction<1:
        p.error("Amplitude bins must be >=1 mV and 0 < tail-fraction < 1")
    if args.bootstrap<0 or args.min_events<5 or args.histogram_bin_ps<=0:
        p.error("Bootstrap >=0, min-events >=5, histogram-bin-ps >0 required")
    analyze(args.energy_run.expanduser().resolve(),
            args.timing_run.expanduser().resolve(),
            args.cache_root.expanduser().resolve(),
            args.output_dir, fractions=args.fractions,
            centers=args.centers_mv,bin_ps=args.histogram_bin_ps,
            bootstrap=args.bootstrap,seed=args.seed,min_events=args.min_events,
            tail_fraction=args.tail_fraction,amplitude_bin_mv=args.amplitude_bin_mv)


if __name__=="__main__":
    main()
