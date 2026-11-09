"""UC vs FBK paired energy/timing residual comparison (saved blind results only).

Independent acquisitions: align energy/timing events *within each board*, never
between boards. Descriptive, exploratory comparison; no causal decomposition.
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

from waveform_analysis.scripts.plot_residual_correlation_percentiles import _read
from waveform_analysis.scripts.residual_exploratory_diagnostics import (
    _stats, _tail_dependence, exploratory_diagnostics,
)

PERCENTILES = (50, 80, 90, 95, 99, 99.8, 100)
METRICS = (
    "pearson", "spearman", "covariance_ps2",
    "variance_energy_ps2", "variance_timing_ps2",
    "variance_difference_ps2", "shared_fraction_timing_proxy",
    "same_sign_fraction",
)


def _load_board(energy_run, timing_run, artifact_root):
    e = _read(Path(energy_run).expanduser().resolve(), artifact_root)
    t = _read(Path(timing_run).expanduser().resolve(), artifact_root)
    ids, ei, ti = np.intersect1d(e[0], t[0], assume_unique=True, return_indices=True)
    if len(ids) < 20:
        raise ValueError(f"Only {len(ids)} matched energy/timing events")
    led = np.column_stack([e[1][ei], t[1][ti]])
    ml = np.column_stack([e[2][ei], t[2][ti]])
    return {"ids": ids, "led": led, "ml": ml,
            "sources": {"energy": e[3], "timing": t[3]}}


def _selection(points, percentile):
    if percentile == 100:
        return np.ones(len(points), dtype=bool)
    score = np.max(np.abs(points), axis=1)
    # Use exact rank for consistency with percentile analysis; ties deterministic.
    order = np.argsort(score, kind="stable")
    n_keep = int(np.floor(len(points)*percentile/100))
    mask = np.zeros(len(points), dtype=bool)
    mask[order[:n_keep]] = True
    return mask


def _flatten(stats, prefix):
    return {prefix+k: stats[k] for k in METRICS}


def _bootstrap_board_contrast(uc, fbk, draws, rng):
    """Independent-board event bootstrap for ML UC minus FBK."""
    res = {key: [] for key in METRICS}
    for _ in range(draws):
        a = _stats(uc[rng.integers(len(uc), size=len(uc))])
        b = _stats(fbk[rng.integers(len(fbk), size=len(fbk))])
        for key in METRICS:
            if a[key] is not None and b[key] is not None:
                res[key].append(a[key]-b[key])
    return {k: list(map(float, np.quantile(vals, [.025,.975])))
            if len(vals) >= max(10, draws//2) else None
            for k, vals in res.items()}


def _bootstrap_within(led, ml, draws, rng):
    """Paired event-level bootstrap of LED-minus-ML within one board."""
    res={key:[] for key in METRICS}
    for _ in range(draws):
        indices=rng.integers(len(ml), size=len(ml))
        a,b=_stats(led[indices]),_stats(ml[indices])
        for key in METRICS:
            if a[key] is not None and b[key] is not None:
                res[key].append(a[key]-b[key])
    return {k:list(map(float,np.quantile(vals,[.025,.975])))
            if len(vals)>=max(10,draws//2) else None
            for k, vals in res.items()}


def _tail_summary(points):
    data=_tail_dependence(points)
    for row in data.values():
        expected=row["independent_expected_joint_fraction"]
        row["joint_excess_over_independence"]= (
            row["joint_fraction"]/expected if expected>0 else None)
    return data


def _plot(rows, path):
    fig,axs=plt.subplots(2,2,figsize=(12,8),constrained_layout=True)
    metrics=[
        ("pearson","Pearson rho"),
        ("covariance_ps2","Cov(energy,timing) [ps²]"),
        ("variance_timing_ps2","Timing variance [ps²]"),
        ("variance_difference_ps2","Var(energy - timing) [ps²]"),
    ]
    for ax,(key,title) in zip(axs.flat,metrics):
        for board in ("UC","FBK"):
            sub=[r for r in rows if r["board"]==board]
            x=np.array([r["percentile"] for r in sub])
            ax.plot(x,[r["ml_"+key] for r in sub],marker="o",label=board)
        ax.set_title(title)
        ax.set_xlabel("Central percentile, ranked by ML shared score")
        ax.grid(alpha=.2)
        ax.legend()
    fig.savefig(path,dpi=180)
    plt.close(fig)


def _plot_within(rows,path):
    fig,axs=plt.subplots(1,2,figsize=(12,4),constrained_layout=True)
    for ax,key,title in zip(axs,
            ("pearson","variance_timing_ps2"),
            ("Pearson correlation","Timing variance [ps²]")):
        for board in ("UC","FBK"):
            sub=[r for r in rows if r["board"]==board]
            x=[r["percentile"] for r in sub]
            ax.plot(x,[r["led_"+key] for r in sub],marker="o",linestyle="--",label=board+" LED")
            ax.plot(x,[r["ml_"+key] for r in sub],marker="o",label=board+" ML")
        ax.set_title(title+" (same events per board)")
        ax.set_xlabel("Central percentile ranked by ML")
        ax.grid(alpha=.2)
        ax.legend(fontsize=8)
    fig.savefig(path,dpi=180)
    plt.close(fig)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for board in ("uc","fbk"):
        p.add_argument(f"--{board}-energy-run",type=Path,required=True)
        p.add_argument(f"--{board}-timing-run",type=Path,required=True)
        p.add_argument(f"--{board}-cache-root",type=Path)
        p.add_argument(f"--{board}-artifact-root",type=Path)
    p.add_argument("--output-dir",type=Path,required=True)
    p.add_argument("--bootstrap",type=int,default=100)
    p.add_argument("--seed",type=int,default=42)
    p.add_argument("--full-per-board",action="store_true",
                   help="Also generate all single-board exploratory diagnostics")
    args=p.parse_args()
    if args.bootstrap<20:
        p.error("--bootstrap must be >=20")
    boards={}
    for name in ("UC","FBK"):
        prefix=name.lower()
        root=getattr(args,prefix+"_cache_root")
        artifact=getattr(args,prefix+"_artifact_root")
        if root and artifact:
            p.error(f"Specify either --{prefix}-cache-root or --{prefix}-artifact-root")
        ar=(artifact.expanduser().resolve() if artifact else
            root.expanduser().resolve()/"artifact_store" if root else None)
        boards[name]=_load_board(
            getattr(args,prefix+"_energy_run"),
            getattr(args,prefix+"_timing_run"),ar)
    out=args.output_dir.expanduser().resolve()
    out.mkdir(parents=True,exist_ok=True)
    rng=np.random.default_rng(args.seed)
    rows=[]
    raw={}
    selected={}
    for name,b in boards.items():
        if args.full_per_board:
            exploratory_diagnostics(b["led"],b["ml"],out/name,
                                    draws=args.bootstrap,seed=args.seed)
        raw[name]={
            "n_events":len(b["ids"]),
            "source_files":b["sources"],
            "full_population_led":_stats(b["led"]),
            "full_population_ml":_stats(b["ml"]),
            "tail_dependence_led":_tail_summary(b["led"]),
            "tail_dependence_ml":_tail_summary(b["ml"]),
            "percentile_selection":"ML shared-score ranks, reused for LED on same events",
            "per_percentile":{},
        }
        selected[name]={}
        for q in PERCENTILES:
            keep=_selection(b["ml"],q)
            led,ml=b["led"][keep],b["ml"][keep]
            selected[name][q]=ml
            a, c=_stats(led), _stats(ml)
            delta={k:(a[k]-c[k] if a[k] is not None and c[k] is not None else None)
                   for k in METRICS}
            raw[name]["per_percentile"][str(q)]={
                "n":len(ml),"LED":a,"ML":c,
                "LED_minus_ML":delta,
                "LED_minus_ML_paired_bootstrap_ci95":
                    _bootstrap_within(led,ml,args.bootstrap,rng),
            }
            rows.append({"board":name,"percentile":q,"n":len(ml),
                         **_flatten(a,"led_"),**_flatten(c,"ml_"),
                         **{"led_minus_ml_"+k:delta[k] for k in METRICS}})
    contrasts={}
    for q in PERCENTILES:
        a=_stats(selected["UC"][q])
        b=_stats(selected["FBK"][q])
        contrasts[str(q)]={
            "n_uc":len(selected["UC"][q]),"n_fbk":len(selected["FBK"][q]),
            "ML_UC_minus_FBK":{
                key:(a[key]-b[key] if a[key] is not None and b[key] is not None else None)
                for key in METRICS},
            "ML_UC_minus_FBK_independent_bootstrap_ci95":
                _bootstrap_board_contrast(selected["UC"][q],selected["FBK"][q],
                                          args.bootstrap,rng),
        }
    with (out/"cross_board_by_percentile.csv").open("w",newline="",encoding="utf-8") as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (out/"cross_board_exploratory.json").write_text(json.dumps({
        "UC":raw["UC"],"FBK":raw["FBK"],
        "UC_minus_FBK":contrasts,
        "methodology":{
            "pairing":"Within-board energy/timing matching by event ID; boards never matched or pooled",
            "selection":"Each board ML shared severity=max(abs(R_E),abs(R_T)); same selected events used for LED/ML comparison",
            "bootstrap":"Within-board paired LED/ML resampling; UC and FBK independent for cross-board CI; fixed selected memberships",
            "limits":"Different amplifier boards may also differ in crystal/SiPM realization, acquisition and calibration",
            "common_proxy":"Covariance is not physically identified as irreducible jitter",
            "ctr":"No new CTR FWHM fit: standard deviation is not CTR",
        },
    },indent=2),encoding="utf-8")
    _plot(rows,out/"cross_board_ml_components.png")
    _plot_within(rows,out/"cross_board_led_vs_ml.png")
    print(f"UC events: {len(boards['UC']['ids'])}; FBK events: {len(boards['FBK']['ids'])}")
    print(f"Cross-board results: {out}")
    print("Files: cross_board_by_percentile.csv, cross_board_exploratory.json, "
          "cross_board_ml_components.png, cross_board_led_vs_ml.png")
    if args.full_per_board:
        print("Also wrote UC/ and FBK/ exploratory diagnostics")


if __name__=="__main__":
    main()
