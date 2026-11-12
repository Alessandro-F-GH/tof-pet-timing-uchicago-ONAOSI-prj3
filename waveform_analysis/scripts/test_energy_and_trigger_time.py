"""Diagnostic for a possible energy-AND oscilloscope trigger timing shortcut.

Read original ROOT energy channels 1/2, without LED alignment, event
selection, ML cache, or predictions. Estimate hardware discriminator levels
from the lower envelope of observed pulse maxima (a hypothesis, not known
hardware settings). Compare peak amplitude with absolute threshold-crossing
time in ROOT coordinates, and whether the latest crossing is near t=0.

Important: A trigger fixes acquisition time by design. This script diagnoses
its geometry; it does NOT by itself demonstrate unwanted information in ML.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import awkward as ak
import numpy as np
import uproot

from waveform_analysis.signal.pulses import decode_oriented


def rising_crossing(signal, level, start_s, dt_s):
    """First upward threshold crossing, linearly interpolated in native time."""
    y = np.asarray(signal, dtype=np.float64)
    candidates = np.flatnonzero((y[:-1] < level) & (y[1:] >= level))
    if not len(candidates):
        return float("nan")
    i = int(candidates[0])
    fraction = (level - y[i]) / (y[i + 1] - y[i])
    return float((start_s + (i + fraction) * dt_s) * 1e9)


def spearman(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    ok = np.isfinite(a) & np.isfinite(b)
    if np.count_nonzero(ok) < 3:
        return None
    from scipy.stats import spearmanr
    result = spearmanr(a[ok], b[ok])
    return float(result.statistic) if np.isfinite(result.statistic) else None


def load_root(root, max_events):
    waveforms = []
    meta = []
    with uproot.open(root) as file:
        tree = file["events"]
        required = ["vertical_gain_v_per_count", "vertical_offset_v",
                    "horizontal_interval_s", "horizontal_offset_s",
                    "samples_ch1", "samples_ch2"]
        optional = ["event_id"] if "event_id" in tree.keys() else []
        for chunk in tree.iterate(filter_name=required + optional,
                                  step_size="32 MB", library="ak"):
            gains = ak.to_numpy(chunk["vertical_gain_v_per_count"])
            offs = ak.to_numpy(chunk["vertical_offset_v"])
            dt = ak.to_numpy(chunk["horizontal_interval_s"])
            t0 = ak.to_numpy(chunk["horizontal_offset_s"])
            for i in range(len(gains)):
                s = [
                    decode_oriented(chunk[f"samples_ch{ch}"][i],
                                    gains[i,ch-1], offs[i,ch-1], 1)
                    for ch in (1,2)
                ]
                if any(len(v) < 3 or not np.isfinite(v).all() for v in s):
                    continue
                waveforms.append(s)
                meta.append({
                    "event_id": int(chunk["event_id"][i])
                                if optional else len(meta),
                    "dt_s": np.asarray(dt[i,:2],dtype=float),
                    "t0_s": np.asarray(t0[i,:2],dtype=float),
                })
                if len(meta) >= max_events:
                    break
            if len(meta) >= max_events:
                break
    if len(meta) < 3:
        raise ValueError("ROOT contains too few valid events")
    return waveforms, meta


def analyze(waveforms, meta, thresholds=None, *, near_fraction=0.1):
    peak = np.array([[float(np.max(v)) for v in pair]
                     for pair in waveforms], dtype=float)
    peak_time = np.array([
        [(m["t0_s"][d] + np.argmax(pair[d])*m["dt_s"][d])*1e9
         for d in (0,1)]
        for pair,m in zip(waveforms,meta)
    ], dtype=float)
    minimum = np.min(peak,axis=0)
    q01 = np.percentile(peak,0.1,axis=0)
    q1 = np.percentile(peak,1.,axis=0)
    if thresholds is None:
        # Minimum of observed maxima, per channel, used exactly as a tentative
        # discriminator bound. Allow user-supplied hardware values instead.
        levels = minimum.copy()
        source = "observed minimum pulse maximum (tentative)"
    else:
        levels = np.asarray(thresholds,dtype=float)
        source = "user-specified trigger levels"
    crossing = np.array([
        [rising_crossing(pair[d], levels[d],
                         m["t0_s"][d], m["dt_s"][d])
         for d in (0,1)]
        for pair,m in zip(waveforms,meta)
    ])
    and_cross = np.max(crossing,axis=1)
    and_cross[~np.isfinite(crossing).all(axis=1)] = np.nan
    amplitude_gap = peak - levels[None,:]
    # Near-trigger events defined by bottom 10% of positive amplitude gaps,
    # separately for the two channels, not by proximity in time.
    near = np.zeros(peak.shape,dtype=bool)
    for d in (0,1):
        valid = np.isfinite(amplitude_gap[:,d]) & (amplitude_gap[:,d]>=0)
        if valid.sum():
            cutoff = np.quantile(amplitude_gap[valid,d],near_fraction)
            near[:,d] = valid & (amplitude_gap[:,d] <= cutoff)
    d_cross = crossing[:,1]-crossing[:,0]
    rows = []
    for k,m in enumerate(meta):
        row = {"event_id":m["event_id"],
               "and_latest_crossing_ns":float(and_cross[k]),
               "crossing_difference_ch2_minus_ch1_ns":float(d_cross[k])}
        for d in (0,1):
            tag=f"ch{d+1}"
            row.update({
                f"{tag}_peak_mV":float(peak[k,d]),
                f"{tag}_peak_time_ns":float(peak_time[k,d]),
                f"{tag}_gap_above_trigger_mV":float(amplitude_gap[k,d]),
                f"{tag}_threshold_crossing_ns":float(crossing[k,d]),
                f"{tag}_near_threshold":bool(near[k,d]),
                f"{tag}_horizontal_offset_ns":float(m["t0_s"][d]*1e9),
            })
        rows.append(row)
    summary = {
        "events":len(rows),"observed_min_peak_mV":minimum.tolist(),
        "observed_p0p1_peak_mV":q01.tolist(),
        "observed_p1_peak_mV":q1.tolist(),
        "thresholds_mV":levels.tolist(),"threshold_source":source,
        "crossing_definition":"first upward crossing, native ROOT horizontal_offset_s + sample*dt",
        "time_units":"ns; zero is ROOT acquisition time reference, not LED",
        "near_threshold_fraction":near_fraction,
        "valid_pair_crossings":int(np.isfinite(and_cross).sum()),
        "median_latest_and_crossing_ns":(
            float(np.nanmedian(and_cross)) if np.isfinite(and_cross).any() else None),
        "median_abs_latest_and_crossing_ns":(
            float(np.nanmedian(np.abs(and_cross))) if np.isfinite(and_cross).any() else None),
        "detectors":{},
        "interpretation_note":(
            "Observed minimum peak is not a calibrated hardware discriminator. "
            "AND triggering can make the latest crossing close to time zero "
            "by construction; this is not itself proof of TOF leakage or source-shift bias. "
            "The nearest-to-threshold group is selected solely by amplitude."),
    }
    for d in (0,1):
        valid = np.isfinite(crossing[:,d])
        selected = near[:,d] & valid
        summary["detectors"][str(d+1)] = {
            "n_crossings":int(valid.sum()),
            "n_near_threshold":int(selected.sum()),
            "spearman_peak_vs_crossing":spearman(peak[:,d],crossing[:,d]),
            "spearman_gap_vs_abs_crossing":spearman(
                amplitude_gap[:,d],np.abs(crossing[:,d])),
            "median_crossing_ns":(
                float(np.median(crossing[valid,d])) if valid.any() else None),
            "near_threshold_median_crossing_ns":(
                float(np.median(crossing[selected,d])) if selected.any() else None),
            "near_threshold_median_abs_crossing_ns":(
                float(np.median(np.abs(crossing[selected,d]))) if selected.any() else None),
            "far_threshold_median_abs_crossing_ns":(
                float(np.median(np.abs(crossing[valid&~near[:,d],d])))
                if (valid&~near[:,d]).any() else None),
        }
    return rows,summary


def plot(rows,summary,out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    for d in (1,2):
        a=np.asarray([r[f"ch{d}_peak_mV"] for r in rows])
        t=np.asarray([r[f"ch{d}_threshold_crossing_ns"] for r in rows])
        p=np.asarray([r[f"ch{d}_peak_time_ns"] for r in rows])
        near=np.asarray([r[f"ch{d}_near_threshold"] for r in rows])
        level=summary["thresholds_mV"][d-1]
        fig,ax=plt.subplots(1,3,figsize=(15,4.5),constrained_layout=True)
        ax[0].hist(a,bins=90)
        ax[0].axvline(level,ls="--",color="k",label="Inferred trigger")
        ax[0].set(xlabel="Maximum pulse amplitude (mV)",ylabel="Events",
                  title=f"Energy detector {d}: peak distribution")
        ax[0].legend()
        ax[1].scatter(a[~near],t[~near],s=3,alpha=.2,label="Other events")
        ax[1].scatter(a[near],t[near],s=8,alpha=.5,label="Lowest 10% amplitudes")
        ax[1].axhline(0,ls="--",color="k")
        ax[1].set(xlabel="Maximum pulse amplitude (mV)",
                  ylabel="First trigger-level crossing (ns)",
                  title="Amplitude versus absolute crossing time")
        ax[1].legend()
        ax[2].scatter(a,p,s=3,alpha=.2)
        ax[2].axhline(0,ls="--",color="k")
        ax[2].set(xlabel="Maximum pulse amplitude (mV)",
                  ylabel="Peak time in ROOT coordinates (ns)",
                  title="Amplitude versus peak position")
        for axis in ax:axis.grid(alpha=.2)
        fig.savefig(out/f"detector{d}_amplitude_vs_absolute_time.png",dpi=170)
        plt.close(fig)
    t1=np.asarray([r["ch1_threshold_crossing_ns"] for r in rows])
    t2=np.asarray([r["ch2_threshold_crossing_ns"] for r in rows])
    delta=t2-t1
    tand=np.asarray([r["and_latest_crossing_ns"] for r in rows])
    fig,axs=plt.subplots(1,3,figsize=(15,4.5),constrained_layout=True)
    axs[0].scatter(t1,t2,s=3,alpha=.2)
    axs[0].set(xlabel="Detector 1 crossing (ns)",ylabel="Detector 2 crossing (ns)",
               title="AND trigger geometry")
    axs[1].hist(tand[np.isfinite(tand)],bins=100)
    axs[1].axvline(0,color="k",ls="--")
    axs[1].set(xlabel="Latest of two crossing times (ns)",ylabel="Events",
               title="Is AND crossing near ROOT t=0?")
    axs[2].scatter(tand,delta,s=3,alpha=.2)
    axs[2].axvline(0,color="k",ls="--")
    axs[2].set(xlabel="Latest crossing (ns)",ylabel="t2 - t1 (ns)",
               title="Absolute trigger position vs relative timing")
    for ax in axs:ax.grid(alpha=.2)
    fig.savefig(out/"and_trigger_geometry.png",dpi=170)
    plt.close(fig)


def run(args):
    waveforms,meta=load_root(args.root_file,args.max_events)
    rows,summary=analyze(
        waveforms,meta,thresholds=args.trigger_mV,
        near_fraction=args.near_fraction)
    summary["root_file"]=str(Path(args.root_file).expanduser().resolve())
    out=Path(args.output_root).expanduser().resolve()
    out.mkdir(parents=True,exist_ok=True)
    with (out/"event_amplitudes_and_times.csv").open("w",newline="",encoding="utf-8") as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (out/"summary.json").write_text(json.dumps(summary,indent=2,allow_nan=False),
                                    encoding="utf-8")
    plot(rows,summary,out)
    print("Peak minima (ch1, ch2) mV:",summary["observed_min_peak_mV"])
    print("Inferred/entered trigger thresholds mV:",summary["thresholds_mV"])
    print("Pair crossings:",summary["valid_pair_crossings"],"of",summary["events"])
    print("Median latest AND crossing ns:",summary["median_latest_and_crossing_ns"])
    for d in (1,2):
        print(f"Detector {d}:",summary["detectors"][str(d)])
    return summary


def parser():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root-file",type=Path,required=True)
    p.add_argument("--output-root",type=Path,required=True)
    p.add_argument("--max-events",type=int,default=10000)
    p.add_argument("--trigger-mV",type=float,nargs=2,default=None,
                   help="Optional known hardware trigger levels for energy ch1/ch2")
    p.add_argument("--near-fraction",type=float,default=.1)
    return p


if __name__=="__main__":
    run(parser().parse_args())
