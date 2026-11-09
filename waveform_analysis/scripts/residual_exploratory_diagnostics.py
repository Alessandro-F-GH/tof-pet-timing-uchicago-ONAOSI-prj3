"""Exploratory paired timing-residual diagnostics; no physical-floor claims.

Works on aligned saved LED and ML corrected residuals. Percentile-conditioned
statistics are descriptive; bootstrap conditions on fixed selected membership.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PERCENTILES = (50, 80, 90, 95, 99, 99.8, 100)
TAILS = (50, 100, 200, 500, 1000)
Q = (0.001, 0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99, 0.999)


def _safe(value):
    value = float(value)
    return value if np.isfinite(value) else None


def _corr(a, b):
    if len(a) < 3 or np.std(a) == 0 or np.std(b) == 0:
        return None
    return _safe(np.corrcoef(a, b)[0, 1])


def _rank(values):
    order = np.argsort(values, kind="stable")
    ranks = np.empty(len(order), dtype=float)
    ranks[order] = np.arange(len(order), dtype=float)
    return ranks


def _spearman(a, b):
    return _corr(_rank(a), _rank(b))


def _moment(x):
    center = np.mean(x)
    sd = np.std(x, ddof=1) if len(x) > 1 else np.nan
    if not np.isfinite(sd) or sd == 0:
        return None, None
    z = (x - center) / sd
    return _safe(np.mean(z ** 3)), _safe(np.mean(z ** 4) - 3)


def _vector(x):
    if not len(x):
        return {}
    q = np.quantile(x, Q)
    skew, kurtosis = _moment(x)
    med = np.median(x)
    return {
        "mean_ps": _safe(np.mean(x)),
        "median_ps": _safe(med),
        "bias_abs_ps": _safe(abs(np.mean(x))),
        "std_ps": _safe(np.std(x, ddof=1)) if len(x)>1 else None,
        "variance_ps2": _safe(np.var(x, ddof=1)) if len(x)>1 else None,
        "rmse_ps": _safe(np.sqrt(np.mean(x*x))),
        "mae_ps": _safe(np.mean(np.abs(x))),
        "mad_normal_ps": _safe(1.4826 * np.median(np.abs(x - med))),
        "iqr_ps": _safe(q[5] - q[3]),
        "skewness": skew, "excess_kurtosis": kurtosis,
        **{f"q{100*p:g}_ps": _safe(v) for p,v in zip(Q,q)},
        **{f"abs_gt_{int(t)}ps_fraction": _safe(np.mean(np.abs(x)>t)) for t in TAILS},
    }


def _stats(x):
    a, b = np.asarray(x[:,0],float), np.asarray(x[:,1],float)
    difference = a-b
    c = np.cov(a,b) if len(x)>1 else np.full((2,2),np.nan)
    va, vt, cov = c[0,0], c[1,1], c[0,1]
    var_diff = float(np.var(difference,ddof=1)) if len(x)>1 else np.nan
    common_proxy = cov
    return {
        "n": int(len(x)),
        "energy": _vector(a), "timing": _vector(b), "energy_minus_timing": _vector(difference),
        "pearson": _corr(a,b),
        "spearman": _spearman(a,b),
        "covariance_ps2": _safe(cov),
        "variance_energy_ps2": _safe(va),
        "variance_timing_ps2": _safe(vt),
        "variance_difference_ps2": _safe(var_diff),
        "covariance_identity_difference_ps2": _safe(va+vt-2*cov),
        "variance_ratio_energy_over_timing": _safe(va/vt) if vt>0 else None,
        "sigma_ratio_energy_over_timing": _safe(np.sqrt(va/vt)) if vt>0 else None,
        "slope_timing_on_energy": _safe(cov/va) if va>0 else None,
        "slope_energy_on_timing": _safe(cov/vt) if vt>0 else None,
        "linear_r_squared": _safe((cov*cov)/(va*vt)) if va>0 and vt>0 else None,
        "same_sign_fraction": _safe(np.mean(np.sign(a)==np.sign(b))),
        "mean_sum_ps": _safe(np.mean(a+b)),
        "mean_difference_ps": _safe(np.mean(difference)),
        "shared_variance_proxy_ps2": _safe(common_proxy),
        "specific_energy_proxy_ps2": _safe(va-common_proxy),
        "specific_timing_proxy_ps2": _safe(vt-common_proxy),
        "shared_fraction_energy_proxy": _safe(common_proxy/va) if va>0 else None,
        "shared_fraction_timing_proxy": _safe(common_proxy/vt) if vt>0 else None,
        "decomposition_valid_nonnegative": bool(
            np.isfinite(cov) and cov>=0 and va>=cov and vt>=cov),
        "decomposition_assumption": "Only if residuals equal independent shared + independent channel-specific components",
    }


def _bootstrap(points, draws, rng):
    if len(points)<4:
        return {}
    keys = ("pearson", "covariance_ps2", "variance_difference_ps2",
            "shared_fraction_energy_proxy", "shared_fraction_timing_proxy")
    collected = {key: [] for key in keys}
    for _ in range(draws):
        sample = points[rng.integers(len(points),size=len(points))]
        stats = _stats(sample)
        for key in keys:
            value = stats[key]
            if value is not None:
                collected[key].append(value)
    return {key+"_ci95": [float(q) for q in np.quantile(vals,[.025,.975])]
            if len(vals)>=max(10,draws//2) else None
            for key,vals in collected.items()}


def _tail_dependence(points):
    a,b=np.abs(points[:,0]),np.abs(points[:,1])
    result={}
    for p in (90,95,99,99.5,99.8):
        ta=a>=np.percentile(a,p)
        tb=b>=np.percentile(b,p)
        both=ta & tb
        result[str(p)]={
            "n_joint":int(both.sum()),
            "joint_fraction":float(np.mean(both)),
            "P_timing_tail_given_energy_tail":float(np.mean(tb[ta])),
            "P_energy_tail_given_timing_tail":float(np.mean(ta[tb])),
            "independent_expected_joint_fraction":float(np.mean(ta)*np.mean(tb)),
            "sign_agreement_joint":float(np.mean(np.sign(points[both,0])==np.sign(points[both,1])))
            if both.any() else None,
        }
    return result


def _plot_components(rows, destination):
    fig,axes=plt.subplots(2,1,figsize=(10,8),sharex=True,constrained_layout=True)
    for state in ("led","ml"):
        rr=[r for r in rows if r["state"]==state]
        p=np.array([r["percentile"] for r in rr])
        c=np.array([r["covariance_ps2"] for r in rr])
        d=np.array([r["variance_difference_ps2"] for r in rr])
        axes[0].plot(p,c,marker="o",label=state)
        axes[1].plot(p,d,marker="o",label=state)
    axes[0].set_ylabel("Cov(energy, timing) [ps²]")
    axes[1].set_ylabel("Var(energy - timing) [ps²]")
    axes[1].set_xlabel("Central population retained [%]")
    for ax in axes:
        ax.grid(alpha=.2)
        ax.legend()
    fig.savefig(destination,dpi=180)
    plt.close(fig)


def _plot_scatter(led, ml, destination):
    fig,axes=plt.subplots(1,2,figsize=(12,5),constrained_layout=True)
    for ax,x,title in zip(axes,(led,ml),("LED","ML corrected")):
        keep=np.max(np.abs(x),axis=1)<=np.percentile(np.max(np.abs(x),axis=1),99)
        ax.hexbin(x[keep,0],x[keep,1],gridsize=70,mincnt=1,bins="log")
        ax.axhline(0,color=".6",lw=.8)
        ax.axvline(0,color=".6",lw=.8)
        ax.set_xlabel("Energy residual [ps]")
        ax.set_ylabel("Timing residual [ps]")
        ax.set_title(title+" (central 99%, own score)")
    fig.savefig(destination,dpi=180)
    plt.close(fig)


def exploratory_diagnostics(led, ml, output_dir, *, draws=100,seed=42):
    """Compute descriptive diagnostics on identical event IDs in both models."""
    out=Path(output_dir)
    out.mkdir(parents=True,exist_ok=True)
    rng=np.random.default_rng(seed)
    populations={}
    rows=[]
    for state, points in (("led",led),("ml",ml)):
        score=np.max(np.abs(points),axis=1)
        population={}
        for p in PERCENTILES:
            mask=score<=np.percentile(score,p) if p<100 else np.ones(len(score),bool)
            x=points[mask]
            stats=_stats(x)
            stats["bootstrap_fixed_membership"]=_bootstrap(x,draws,rng)
            population[str(p)]=stats
            rows.append({"state":state,"percentile":p,
                         **{key:stats[key] for key in (
                             "n","pearson","spearman","covariance_ps2",
                             "variance_energy_ps2","variance_timing_ps2",
                             "variance_difference_ps2","shared_fraction_energy_proxy",
                             "shared_fraction_timing_proxy",
                             "specific_energy_proxy_ps2","specific_timing_proxy_ps2",
                             "same_sign_fraction")}})
        populations[state]={
            "central_percentiles":population,
            "absolute_tail_dependence":_tail_dependence(points),
        }
    # Work on one and the same event population, not separately re-ranked LED/ML.
    comparison={}
    for p in (50,90,99,99.8,100):
        led_score=np.max(np.abs(led),axis=1)
        ml_score=np.max(np.abs(ml),axis=1)
        mask=np.ones(len(led),bool) if p==100 else (
            (led_score<=np.percentile(led_score,p))
            & (ml_score<=np.percentile(ml_score,p)))
        l,m=led[mask],ml[mask]
        correction=l-m
        comparison[str(p)]={
            "n":int(mask.sum()),
            "led":_stats(l),"ml":_stats(m),
            "correction_energy_timing":_stats(correction),
            "covariance_change_led_minus_ml_ps2":_safe(np.cov(l,rowvar=False)[0,1]
                                                        -np.cov(m,rowvar=False)[0,1]),
            "energy_residual_vs_correction_rho":_corr(m[:,0],correction[:,0]),
            "timing_residual_vs_correction_rho":_corr(m[:,1],correction[:,1]),
            "correction_energy_vs_timing_rho":_corr(correction[:,0],correction[:,1]),
        }
    with (out/"exploratory_components.csv").open("w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    (out/"exploratory_diagnostics.json").write_text(json.dumps({
        "n_events":len(led),
        "populations":populations,
        "same_event_LED_ML_comparisons":comparison,
        "definitions":{
            "R_E_minus_R_T":"Difference of residual methods on the same event, cancels exactly shared additive error",
            "shared_proxy":"Cov(R_E,R_T) only represents common variance under independent additive noise assumptions",
            "specific_proxies":"Var(R_i)-Cov(R_E,R_T), not generally positive or physically identified",
            "percentile_conditioning":"Selection on residual magnitude can induce/change correlations",
            "bootstrap":"Event-level resampling, percentile membership held fixed",
            "CTRFWHM":"Not estimated from standard deviation; CTR needs separate histogram FWHM fit",
        }},indent=2),encoding="utf-8")
    _plot_components(rows,out/"covariance_and_difference_variance.png")
    _plot_scatter(led,ml,out/"paired_residual_hexbin_central99.png")
    return rows
