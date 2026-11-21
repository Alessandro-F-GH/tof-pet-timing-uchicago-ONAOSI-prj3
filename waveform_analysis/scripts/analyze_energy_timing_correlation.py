"""Definitive paired Energy/Timing LED and ML correlation diagnostics.

No waveform loading, inference, or refitting. One acquisition per board; align
energy/timing by event_id, NEVER between boards. Tukey fences are calculated
independently for each board and each correction state; no Gaussian fit.

Outlier correlation uses only outliers, core correlation only inliers.
Correction correlation uses events in the intersection of LED and ML cores.
Event-id ordering is a proxy for acquisition order ONLY if verified upstream.
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
from scipy.stats import chi2


def read_predictions(run):
    """Load canonical saved blind predictions directly (no cache dependencies)."""
    path = Path(run).expanduser().resolve() / "artifacts" / "predictions.npz"
    if not path.is_file():
        raise FileNotFoundError(f"Missing blind predictions: {path}")
    with np.load(path, allow_pickle=False) as z:
        required = {"event_id", "led_residual_ps", "corrected_ps", "prediction_ps"}
        missing = required - set(z.files)
        if missing:
            raise ValueError(f"{path}: missing {sorted(missing)}")
        ids = np.asarray(z["event_id"])
        led = np.asarray(z["led_residual_ps"], dtype=np.float64)
        ml = np.asarray(z["corrected_ps"], dtype=np.float64)
        pred = np.asarray(z["prediction_ps"], dtype=np.float64)
    if ids.ndim != 1 or ids.dtype.kind not in "iu" or len(ids) < 20:
        raise ValueError(f"{path}: invalid event_id")
    if np.unique(ids).size != len(ids):
        raise ValueError(f"{path}: duplicate event ids")
    if any(v.shape != ids.shape or not np.isfinite(v).all() for v in (led, ml, pred)):
        raise ValueError(f"{path}: invalid/nonfinite prediction vectors")
    if not np.allclose(led - pred, ml, atol=1e-6, rtol=1e-8):
        raise ValueError(f"{path}: predictions inconsistent with LED - ML")
    return ids.astype(np.int64), led, ml, pred


def paired_board(energy_run, timing_run):
    e, t = read_predictions(energy_run), read_predictions(timing_run)
    ids, ei, ti = np.intersect1d(e[0], t[0], assume_unique=True, return_indices=True)
    if ids.size < 20:
        raise ValueError("Too few paired blind event IDs")
    led = np.column_stack((e[1][ei], t[1][ti]))
    ml = np.column_stack((e[2][ei], t[2][ti]))
    pred = np.column_stack((e[3][ei], t[3][ti]))
    return {"event_id": ids, "led": led, "ml": ml, "prediction": pred,
            "matched": int(ids.size),
            "energy_events": int(e[0].size), "timing_events": int(t[0].size)}


def tukey(points, k):
    if k <= 0 or not np.isfinite(k):
        raise ValueError("IQR multiplier must be finite and positive")
    q1, q3 = np.quantile(points, [0.25, 0.75], axis=0)
    iqr = q3-q1
    if not np.all(iqr > 0):
        raise ValueError("Zero IQR in a channel")
    lo, hi = q1-k*iqr, q3+k*iqr
    flags = (points < lo) | (points > hi)
    return {"q1_ps":q1.tolist(), "q3_ps":q3.tolist(), "iqr_ps":iqr.tolist(),
            "low_ps":lo.tolist(), "high_ps":hi.tolist(), "multiplier":k}, flags


def moments(points):
    """Fast covariance-only metrics, no sorting or higher moments."""
    if len(points) < 4:
        return {"n":int(len(points)), "rho":None, "covariance_ps2":None,
                "sigma_energy_ps":None, "sigma_timing_ps":None}
    x = np.asarray(points, float)
    centered = x - x.mean(axis=0)
    co = centered.T @ centered / (len(x)-1)
    std = np.sqrt(np.diag(co))
    rho = float(co[0,1] / np.prod(std)) if np.all(std>0) else None
    return {"n":int(len(x)), "rho":rho,
            "covariance_ps2":float(co[0,1]),
            "sigma_energy_ps":float(std[0]), "sigma_timing_ps":float(std[1])}


def variance_contribution(points, mask):
    """Share of global centered sum-of-squares attributable to masked events.

    The remaining share is complementary. Includes within-group and between-
    group location effects; NOT a variance of a fitted Gaussian component.
    """
    center = points.mean(axis=0)
    ss = (points-center)**2
    denominator = ss.sum(axis=0)
    return (ss[mask].sum(axis=0)/denominator).tolist()


def bootstrap_metrics(points, draws, rng):
    """Percentile CI conditional on fixed group membership."""
    if len(points)<4:
        return {"rho_ci95":None, "covariance_ps2_ci95":None}
    result = np.empty((draws,2))
    for j in range(draws):
        sample = points[rng.integers(0,len(points),size=len(points))]
        d = moments(sample)
        result[j] = [d["rho"] if d["rho"] is not None else np.nan,
                     d["covariance_ps2"]]
    return {
        key:[float(v) for v in np.quantile(result[:,i][np.isfinite(result[:,i])],
                                            [0.025,0.975])]
        if np.count_nonzero(np.isfinite(result[:,i])) >= max(10,draws//2) else None
        for i,key in enumerate(("rho_ci95","covariance_ps2_ci95"))
    }


def event_bins(ids, flags, bins):
    """Fixed-size acquisition-index bins, after sorting by event_id."""
    n = len(ids)
    edges = np.linspace(0,n,bins+1,dtype=int)
    if np.any(np.diff(edges)==0):
        raise ValueError("Too many acquisition bins")
    out = []
    for i,(a,b) in enumerate(zip(edges[:-1],edges[1:])):
        out.append({"bin":i+1, "first_event_id":int(ids[a]),
                    "last_event_id":int(ids[b-1]), "n":int(b-a),
                    "outliers":int(flags[a:b].sum()),
                    "rate":float(np.mean(flags[a:b]))})
    observed = np.array([r["outliers"] for r in out],float)
    sizes = np.array([r["n"] for r in out],float)
    p = flags.mean()
    # Pearson chi-square on 2 x B contingency counts. Useful screening,
    # not proof of stationarity (requires independent Bernoulli events).
    expected_yes = sizes*p
    expected_no = sizes*(1-p)
    if 0<p<1 and np.all(expected_yes>=5) and np.all(expected_no>=5):
        stat = float(np.sum((observed-expected_yes)**2/expected_yes
                            +((sizes-observed)-expected_no)**2/expected_no))
        pvalue = float(chi2.sf(stat,bins-1))
    else:
        stat,pvalue=None,None
    return out, {"test":"Pearson chi-square homogeneity across acquisition-index bins",
                 "chi2":stat,"df":bins-1,"p_value":pvalue,
                 "warning":"Exploratory: assumes independent event labels; does not establish full stationarity."}


def summarize_board(data, k, bins, draws, rng):
    led,ml,pred,ids=(data[x] for x in ("led","ml","prediction","event_id"))
    states = {}
    masks = {}
    for name,points in (("led",led),("ml",ml)):
        fences, channel_flags = tukey(points,k)
        outliers = channel_flags.any(axis=1)
        masks[name] = outliers
        categories = {
            "core":~outliers, "outliers":outliers,
            "energy_only":channel_flags[:,0]&~channel_flags[:,1],
            "timing_only":channel_flags[:,1]&~channel_flags[:,0],
            "both":channel_flags[:,0]&channel_flags[:,1],
        }
        groups = {}
        for group,mask in categories.items():
            stats = moments(points[mask])
            if group in ("core","outliers"):
                stats.update(bootstrap_metrics(points[mask], draws, rng))
            groups[group] = stats
        groups["outliers"]["variance_share_energy"] = variance_contribution(points,outliers)[0]
        groups["outliers"]["variance_share_timing"] = variance_contribution(points,outliers)[1]
        groups["core"]["variance_share_energy"] = 1-groups["outliers"]["variance_share_energy"]
        groups["core"]["variance_share_timing"] = 1-groups["outliers"]["variance_share_timing"]
        series, test = event_bins(ids,outliers,bins)
        states[name] = {
            "fences":fences, "counts":{c:int(m.sum()) for c,m in categories.items()},
            "outlier_rate":float(outliers.mean()),"groups":groups,
            "acquisition_bins":series,"homogeneity_test":test,
        }
    # Exclude either kind of extreme event before comparing ML corrections;
    # this single, common cohort is not defined by the magnitude of corrections.
    common_core = ~(masks["led"]|masks["ml"])
    predictions = moments(pred[common_core])
    predictions.update(bootstrap_metrics(pred[common_core], draws, rng))
    predictions["selection"] = "intersection of LED and ML IQR cores"
    predictions["n_excluded"] = int((~common_core).sum())
    return {"matched":data["matched"],
            "energy_events":data["energy_events"],"timing_events":data["timing_events"],
            "states":states, "model_outputs_core":predictions},masks,common_core


def paired_differences(uc,fbk,uc_data,fbk_data,uc_masks,fbk_masks,
                       uc_core,fbk_core,draws,rng):
    """Independent-board bootstrap CIs, conditioning on selected groups."""
    rows = []
    for state in ("led","ml"):
        a,b=uc_data[state],fbk_data[state]
        ma,mb=uc_masks[state],fbk_masks[state]
        for group,sa,sb in (("core",~ma,~mb),("outliers",ma,mb)):
            pa,pb=a[sa],b[sb]
            va,vb=moments(pa),moments(pb)
            for metric in ("rho","covariance_ps2"):
                if va[metric] is None or vb[metric] is None:continue
                samples=[]
                for _ in range(draws):
                    xa=pa[rng.integers(len(pa),size=len(pa))]
                    xb=pb[rng.integers(len(pb),size=len(pb))]
                    da,db=moments(xa)[metric],moments(xb)[metric]
                    if da is not None and db is not None:samples.append(da-db)
                rows.append({"state":state,"group":group,"metric":metric,
                    "UC":va[metric],"FBK":vb[metric],
                    "UC_minus_FBK":va[metric]-vb[metric],
                    "ci95":np.quantile(samples,[.025,.975]).tolist()
                    if len(samples)>=max(10,draws//2) else None})
        rate_a,rate_b=ma.mean(),mb.mean()
        # Exact resampling of Bernoulli labels via binomial counts.
        d=(rng.binomial(len(ma),rate_a,size=draws)/len(ma)
           -rng.binomial(len(mb),rate_b,size=draws)/len(mb))
        rows.append({"state":state,"group":"all","metric":"outlier_rate",
                     "UC":float(rate_a),"FBK":float(rate_b),
                     "UC_minus_FBK":float(rate_a-rate_b),
                     "ci95":np.quantile(d,[.025,.975]).tolist()})
        for channel,field in (("energy","variance_share_energy"),
                              ("timing","variance_share_timing")):
            # Fixed labels, but sample whole paired events to retain
            # between-population location and frequency uncertainty.
            observed=[uc,fbk]
            delta=[]
            for _ in range(draws):
                vals=[]
                for board,mask,points in ((uc,ma,a),(fbk,mb,b)):
                    ix=rng.integers(len(points),size=len(points))
                    vals.append(variance_contribution(points[ix],mask[ix])[0 if channel=="energy" else 1])
                delta.append(vals[0]-vals[1])
            va=uc["states"][state]["groups"]["outliers"][field]
            vb=fbk["states"][state]["groups"]["outliers"][field]
            rows.append({"state":state,"group":"outliers","metric":field,
                "UC":va,"FBK":vb,"UC_minus_FBK":va-vb,
                "ci95":np.quantile(delta,[.025,.975]).tolist()})
    a,b=uc_data["prediction"][uc_core],fbk_data["prediction"][fbk_core]
    for metric in ("rho","covariance_ps2"):
        va,vb=moments(a)[metric],moments(b)[metric]
        if va is None or vb is None:continue
        differences=[]
        for _ in range(draws):
            da=moments(a[rng.integers(len(a),size=len(a))])[metric]
            db=moments(b[rng.integers(len(b),size=len(b))])[metric]
            if da is not None and db is not None:differences.append(da-db)
        rows.append({"state":"model_outputs","group":"joint_core","metric":metric,
                     "UC":va,"FBK":vb,"UC_minus_FBK":va-vb,
                     "ci95":np.quantile(differences,[.025,.975]).tolist()
                     if len(differences)>=max(10,draws//2) else None})
    return rows


def plot_scatter(board,data,masks,out):
    fig,axes=plt.subplots(1,2,figsize=(13,5),constrained_layout=True)
    for ax,state in zip(axes,("led","ml")):
        pts=data[state]; mask=masks[state]
        good=np.flatnonzero(~mask)
        if len(good)>25000:
            good=np.random.default_rng(42).choice(good,25000,replace=False)
        ax.scatter(pts[good,0],pts[good,1],s=4,color="0.55",
                   alpha=.2,rasterized=True,label="Core (sample)")
        ax.scatter(pts[mask,0],pts[mask,1],s=8,color="tab:orange",
                   alpha=.6,rasterized=True,label="IQR outliers (all)")
        ax.set(xlabel="Energy residual [ps]",ylabel="Timing residual [ps]",
               title=f"{board}: {state.upper()}")
        ax.legend(fontsize=8)
        ax.grid(alpha=.2)
    for ext in ("png","pdf"):fig.savefig(out/f"scatter_led_ml.{ext}",dpi=170)
    plt.close(fig)


def plot_acquisition(board,summary,out):
    fig,axes=plt.subplots(2,1,figsize=(10,6),sharex=True,constrained_layout=True)
    for ax,state in zip(axes,("led","ml")):
        bins=summary["states"][state]["acquisition_bins"]
        ax.plot([b["bin"] for b in bins],[b["outliers"] for b in bins],
                marker="o",ms=3)
        expected=summary["states"][state]["outlier_rate"]*np.mean([b["n"] for b in bins])
        ax.axhline(expected,color="0.45",linestyle="--")
        ax.set_ylabel(f"{state.upper()} outliers / bin")
        ax.grid(alpha=.2)
    axes[-1].set_xlabel("Acquisition-order proxy: ascending event_id bin")
    for ext in ("png","pdf"):fig.savefig(out/f"outlier_acquisition_order.{ext}",dpi=170)
    plt.close(fig)


def _json(obj):
    if isinstance(obj,np.generic):return obj.item()
    if isinstance(obj,np.ndarray):return obj.tolist()
    raise TypeError(type(obj).__name__)


def write_csv(path,rows):
    if not rows:return
    with path.open("w",newline="",encoding="utf-8") as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]))
        writer.writeheader();writer.writerows(rows)


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    for board in ("uc","fbk"):
        p.add_argument(f"--{board}-energy-run",type=Path,required=True)
        p.add_argument(f"--{board}-timing-run",type=Path,required=True)
    p.add_argument("--output-dir",type=Path,required=True)
    p.add_argument("--iqr-multiplier",type=float,default=3.0,
                   help="Extreme Tukey fences, 3 x IQR (default).")
    p.add_argument("--acquisition-bins",type=int,default=20)
    p.add_argument("--bootstrap",type=int,default=300)
    p.add_argument("--seed",type=int,default=42)
    args=p.parse_args(argv)
    if args.bootstrap<20 or args.acquisition_bins<2:
        p.error("--bootstrap >= 20 and --acquisition-bins >= 2 required")
    rng=np.random.default_rng(args.seed)
    results,raw,flags,cores={}, {}, {}, {}
    for name in ("UC","FBK"):
        key=name.lower()
        raw[name]=paired_board(getattr(args,key+"_energy_run"),
                               getattr(args,key+"_timing_run"))
        results[name],flags[name],cores[name]=summarize_board(
            raw[name],args.iqr_multiplier,args.acquisition_bins,args.bootstrap,rng)
        print(f"{name} | paired={results[name]['matched']:,} | "
              f"LED outliers={results[name]['states']['led']['counts']['outliers']:,} | "
              f"ML outliers={results[name]['states']['ml']['counts']['outliers']:,} | "
              f"joint core={results[name]['model_outputs_core']['n']:,}",flush=True)
    diffs=paired_differences(results["UC"],results["FBK"],
                             raw["UC"],raw["FBK"],flags["UC"],flags["FBK"],
                             cores["UC"],cores["FBK"],args.bootstrap,rng)
    out=args.output_dir.expanduser().resolve()
    out.mkdir(parents=True,exist_ok=True)
    for name in ("UC","FBK"):
        folder=out/name
        folder.mkdir(parents=True,exist_ok=True)
        plot_scatter(name,raw[name],flags[name],folder)
        plot_acquisition(name,results[name],folder)
        for state in ("led","ml"):
            write_csv(folder/f"{state}_acquisition_bins.csv",
                      results[name]["states"][state]["acquisition_bins"])
        (folder/"summary.json").write_text(
            json.dumps(results[name],indent=2,default=_json),encoding="utf-8")
    write_csv(out/"board_differences.csv",[
        {**{k:v for k,v in r.items() if k!="ci95"},
         "ci95_low":r["ci95"][0] if r["ci95"] else None,
         "ci95_high":r["ci95"][1] if r["ci95"] else None} for r in diffs])
    (out/"summary.json").write_text(json.dumps({
        "method":"separate LED/ML 3xIQR classification (configurable)",
        "iqr_multiplier":args.iqr_multiplier,
        "bootstrap_draws":args.bootstrap,
        "bootstrap_interpretation":"percentile CI conditional on original IQR group definition; fences not re-estimated",
        "variance_share_definition":"sum of squared deviations from full mean among tail events / total sum of squared deviations",
        "acquisition_order_warning":"Ascending event_id assumes identifiers track acquisition order; verify from source dataset",
        "core_prediction_selection":"intersection of LED and ML non-outliers",
        "board_differences":diffs,
        "boards":results,
    },indent=2,default=_json),encoding="utf-8")
    print(f"Results: {out}",flush=True)


if __name__=="__main__":
    main()
