#!/usr/bin/env python3
"""Compare energy and timing residuals for events having the same event_id.

Read two saved prediction NPZ files; no model/data provenance checks, model
loading, retraining, or prediction. Users choose matching input populations.
The corrected_ps values are residuals AFTER the model correction.
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
from scipy.stats import pearsonr, spearmanr, linregress, t as student_t
from waveform_analysis.reporting.stats import metric_values


def read_residuals(path: Path):
    if not path.is_file():
        raise FileNotFoundError(path)
    with np.load(path, allow_pickle=False) as data:
        if "event_id" not in data or "corrected_ps" not in data:
            raise ValueError(f"{path}: requires event_id and corrected_ps")
        event_id = np.asarray(data["event_id"])
        residual = np.asarray(data["corrected_ps"], dtype=np.float64)
    if event_id.ndim != 1 or residual.ndim != 1 or len(event_id) != len(residual):
        raise ValueError(f"{path}: event_id and corrected_ps must be same-length 1D arrays")
    if event_id.dtype.kind not in "iu":
        raise ValueError(f"{path}: event_id must contain integers")
    if len(event_id) != len(np.unique(event_id)):
        raise ValueError(f"{path}: duplicate event_id would produce ambiguous matches")
    if len(event_id) == 0:
        raise ValueError(f"{path}: empty prediction file")
    return event_id.astype(np.int64, copy=False), residual


def align_residuals(energy_ids, energy_residuals, timing_ids, timing_residuals):
    """Join on exact event ID; sorted ascending to guarantee reproducibility."""
    ids, indices_energy, indices_timing = np.intersect1d(
        energy_ids, timing_ids, assume_unique=True, return_indices=True)
    if len(ids) < 3:
        raise ValueError(f"Only {len(ids)} matched event IDs; need at least three")
    energy = np.asarray(energy_residuals[indices_energy], dtype=np.float64)
    timing = np.asarray(timing_residuals[indices_timing], dtype=np.float64)
    valid = np.isfinite(energy) & np.isfinite(timing)
    if int(np.sum(valid)) < 3:
        raise ValueError("Fewer than three finite matched residual pairs")
    return ids[valid], energy[valid], timing[valid], int((~valid).sum())


def correlations(energy, timing):
    if np.ptp(energy) == 0 or np.ptp(timing) == 0:
        raise ValueError("Correlation undefined: constant residual values")
    pearson = pearsonr(energy, timing)
    spearman = spearmanr(energy, timing)
    return {
        "n_matched_finite_events": int(len(energy)),
        "pearson_r": float(pearson.statistic),
        "pearson_p_value": float(pearson.pvalue),
        "spearman_rho": float(spearman.statistic),
        "spearman_p_value": float(spearman.pvalue),
        "energy_mean_ps": float(np.mean(energy)),
        "timing_mean_ps": float(np.mean(timing)),
        "energy_std_ps": float(np.std(energy, ddof=1)),
        "timing_std_ps": float(np.std(timing, ddof=1)),
    }


def common_component_statistics(energy, timing):
    """Common-plus-channel-specific variance decomposition, if its assumptions hold.

    Assumes r_E=C+e_E and r_T=C+e_T, where all three centered
    components are pairwise uncorrelated and C has unit loading in both.
    H0: common timing fraction <= 0.5, H1: > 0.5.
    Standard OLS inference assumes independent, homoscedastic events.
    """
    if len(energy) < 3 or np.ptp(timing) == 0:
        raise ValueError("Slope test requires three events and nonconstant timing residuals")
    fit = linregress(timing, energy)
    beta = float(fit.slope)
    se = float(fit.stderr)
    df = len(energy) - 2
    threshold = 0.5
    t_stat = (beta - threshold) / se if se > 0 else (
        float("inf") if beta > threshold else
        float("-inf") if beta < threshold else 0.0)
    p_value = float(student_t.sf(t_stat, df)) if se > 0 else (
        0.0 if beta > threshold else 1.0)
    margin = float(student_t.ppf(0.975, df) * se)
    variance_e = float(np.var(energy, ddof=1))
    variance_t = float(np.var(timing, ddof=1))
    common_variance = float(np.cov(energy, timing, ddof=1)[0, 1])
    specific_e = variance_e - common_variance
    specific_t = variance_t - common_variance
    compatible = bool(common_variance >= 0 and specific_e >= 0 and specific_t >= 0)
    return {
        "beta_energy_on_timing": beta,
        "beta_intercept_ps": float(fit.intercept),
        "beta_standard_error": se,
        "beta_ci_95": [beta - margin, beta + margin],
        "common_variance_ps2": common_variance,
        "energy_specific_variance_ps2": specific_e,
        "timing_specific_variance_ps2": specific_t,
        "common_fraction_timing": beta,
        "common_fraction_energy": common_variance / variance_e,
        "common_component_variances_nonnegative": compatible,
        "dominance_null_max_fraction": threshold,
        "dominance_test_t_statistic": float(t_stat),
        "dominance_test_df": int(df),
        "dominance_test_p_value_one_sided": p_value,
        "dominance_reject_h0_at_0_05": bool(p_value < 0.05),
        "note_common_component": "Fractions interpreted as independent additive "
            "noise contributions ONLY if both channels have the same unit-loading "
            "common component and the three components are pairwise uncorrelated. "
            "Negative inferred specific variances would invalidate these assumptions. "
            "One-sided slope t-test H0: beta<=0.5; OLS p-value/CI assume "
            "independent observations and homoscedastic regression errors. "
            "Shared time-reference errors may also generate common covariance.",
    }


def gaussian_component_fit_by_central_range(energy, timing,
                                            retained_percentages=(100.0, 99.9, 99.0, 95.0)):
    """Moment-based Gaussian common-component fit on paired central subsets.

    Each channel is clipped to its own symmetric empirical quantiles and
    only the intersection of event masks is used. Clipped-sample moments are
    NOT a truncation-corrected Gaussian MLE: parameters describe the selected
    core and cannot be extrapolated to the untruncated population.
    """
    n_all = len(energy)
    fits = []
    for percentage in retained_percentages:
        if not 0 < percentage <= 100:
            raise ValueError("Central percentage must be in (0, 100]")
        if percentage == 100:
            keep = np.ones(n_all, dtype=bool)
        else:
            tail = (100.0 - percentage) / 200.0
            low_e, high_e = np.quantile(energy, [tail, 1.0 - tail])
            low_t, high_t = np.quantile(timing, [tail, 1.0 - tail])
            keep = ((energy >= low_e) & (energy <= high_e)
                    & (timing >= low_t) & (timing <= high_t))
        e, t = energy[keep], timing[keep]
        if len(e) < 3 or np.var(t) == 0 or np.var(e) == 0:
            raise ValueError(f"Insufficient varying matched events at {percentage}%")
        # Gaussian maximum-likelihood covariance uses population normalization.
        var_e = float(np.var(e, ddof=0))
        var_t = float(np.var(t, ddof=0))
        covariance = float(np.mean((e - e.mean()) * (t - t.mean())))
        var_components = {
            "common": covariance,
            "energy_specific": var_e - covariance,
            "timing_specific": var_t - covariance,
        }
        admissible = all(v >= 0 for v in var_components.values())
        fits.append({
            "central_percent_per_channel": float(percentage),
            "n_retained": int(len(e)),
            "retained_fraction": float(len(e) / n_all),
            "mean_energy_ps": float(np.mean(e)),
            "mean_timing_ps": float(np.mean(t)),
            "observed_energy_variance_ps2": var_e,
            "observed_timing_variance_ps2": var_t,
            "component_variances_ps2": var_components,
            "sigma_common_ps": float(np.sqrt(var_components["common"]))
                if var_components["common"] >= 0 else None,
            "sigma_energy_specific_ps": float(np.sqrt(var_components["energy_specific"]))
                if var_components["energy_specific"] >= 0 else None,
            "sigma_timing_specific_ps": float(np.sqrt(var_components["timing_specific"]))
                if var_components["timing_specific"] >= 0 else None,
            "fraction_common_timing": covariance / var_t,
            "fraction_common_energy": covariance / var_e,
            "physically_admissible": admissible,
        })
    return fits


def save_gaussian_component_plot(fits, output):
    percentages = [x["central_percent_per_channel"] for x in fits]
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    for key, label in (
        ("sigma_common_ps", "Common component"),
        ("sigma_energy_specific_ps", "Energy-specific"),
        ("sigma_timing_specific_ps", "Timing-specific"),
    ):
        values = [np.nan if f[key] is None else f[key] for f in fits]
        ax.plot(percentages, values, marker="o", label=label)
    ax.set_xlabel("Central percentile retained in each channel (%)")
    ax.set_ylabel("Estimated component sigma (ps)")
    ax.set_title("Gaussian core variance decomposition (paired events)")
    ax.legend()
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(output / "gaussian_component_vs_clipping.png", dpi=180)
    fig.savefig(output / "gaussian_component_vs_clipping.pdf")
    plt.close(fig)


def difference_statistics(energy, timing):
    """Observed difference and variance-subtraction test on paired events."""
    difference = energy - timing
    energy_std = float(np.std(energy, ddof=1))
    timing_std = float(np.std(timing, ddof=1))
    variance_excess = energy_std**2 - timing_std**2
    return {
        "difference_mean_ps": float(np.mean(difference)),
        "difference_std_ps": float(np.std(difference, ddof=1)),
        "difference_rmse_ps": float(np.sqrt(np.mean(difference**2))),
        "energy_variance_minus_timing_variance_ps2": variance_excess,
        "predicted_extra_std_ps_if_timing_is_common_only":
            float(np.sqrt(variance_excess)) if variance_excess >= 0 else None,
        "note_difference": "The quadrature prediction uses full-sample standard "
            "deviations, not fitted CTR. It assumes timing=C and energy=C+N, "
            "with C independent of N.",
    }


def ctr_quadrature_comparison(energy, timing, *, bin_width_ps, seed):
    """Fitted CTR for energy, timing, their difference; common+extra prediction."""
    config = {"histogram_bin_width_ps": float(bin_width_ps)}
    ctr_energy = float(metric_values(energy, config, seed=seed)["ctr_ps"])
    ctr_timing = float(metric_values(timing, config, seed=seed)["ctr_ps"])
    ctr_difference = float(metric_values(energy-timing, config, seed=seed)["ctr_ps"])
    excess = ctr_energy**2 - ctr_timing**2
    predicted = float(np.sqrt(excess)) if excess >= 0 else None
    return {
        "ctr_energy_ps": ctr_energy,
        "ctr_timing_ps": ctr_timing,
        "ctr_difference_ps": ctr_difference,
        "predicted_difference_ctr_ps_if_timing_is_common_only": predicted,
        "difference_ctr_minus_prediction_ps": (
            float(ctr_difference-predicted) if predicted is not None else None),
        "ctr_bin_width_ps": float(bin_width_ps),
        "note_ctr": "CTR comparison requires approximately Gaussian residual "
            "distributions with consistent fitted-width definitions. No "
            "independent-source decomposition can be established from this alone.",
    }


def save_difference_plot(energy, timing, output):
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    ax.hist(energy - timing, bins=100, density=True, alpha=0.75)
    ax.set_xlabel("Energy residual - timing residual (ps)")
    ax.set_ylabel("Density")
    ax.set_title("Difference of matched corrected residuals")
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(output / "residual_difference.png", dpi=180)
    fig.savefig(output / "residual_difference.pdf")
    plt.close(fig)


def save_plot(energy, timing, output):
    fig, ax = plt.subplots(figsize=(6.2, 5.4))
    ax.scatter(energy, timing, s=7, alpha=0.25, rasterized=True)
    ax.axhline(0, color="0.5", lw=0.8)
    ax.axvline(0, color="0.5", lw=0.8)
    ax.set_xlabel("Energy corrected residual (ps)")
    ax.set_ylabel("Timing corrected residual (ps)")
    ax.set_title("Matched-event energy vs timing residuals")
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(output / "residual_correlation.png", dpi=180)
    fig.savefig(output / "residual_correlation.pdf")
    plt.close(fig)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--energy", type=Path, required=True,
                        help="Energy predictions .npz with event_id and corrected_ps")
    parser.add_argument("--timing", type=Path, required=True,
                        help="Timing predictions .npz with event_id and corrected_ps")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--fit-bin-width-ps", type=float, default=10.0,
                        help="Histogram bin width for CTR fits (default: 10 ps)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)
    if args.fit_bin_width_ps <= 0:
        parser.error("--fit-bin-width-ps must be positive")
    energy_ids, energy_residuals = read_residuals(args.energy)
    timing_ids, timing_residuals = read_residuals(args.timing)
    ids, energy, timing, n_nonfinite = align_residuals(
        energy_ids, energy_residuals, timing_ids, timing_residuals)
    results = correlations(energy, timing)
    results.update(common_component_statistics(energy, timing))
    results["gaussian_component_by_clipping"] = gaussian_component_fit_by_central_range(energy, timing)
    results["gaussian_clipping_note"] = ("Clipped fits use Gaussian sample moments on the "
        "jointly selected central events, not truncation-corrected likelihood. "
        "Interpret component sigmas only for the retained subset, under the "
        "unit-loading, pairwise-independent additive model.")
    results.update(difference_statistics(energy, timing))
    results.update(ctr_quadrature_comparison(
        energy, timing, bin_width_ps=args.fit_bin_width_ps, seed=args.seed))
    results.update(
        n_energy_events=int(len(energy_ids)),
        n_timing_events=int(len(timing_ids)),
        n_common_ids=int(len(np.intersect1d(energy_ids, timing_ids, assume_unique=True))),
        n_nonfinite_pairs=int(n_nonfinite),
        energy_file=str(args.energy),
        timing_file=str(args.timing),
        residual_column="corrected_ps",
        note="Event IDs paired without source provenance verification. "
             "Development predictions can be in-sample, not out-of-fold.",
    )
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / "matched_residuals.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["event_id", "energy_corrected_ps", "timing_corrected_ps", "energy_minus_timing_ps"])
        writer.writerows(zip(ids.tolist(), energy.tolist(), timing.tolist(), (energy-timing).tolist()))
    (args.output / "correlation_summary.json").write_text(
        json.dumps(results, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    save_plot(energy, timing, args.output)
    save_difference_plot(energy, timing, args.output)
    save_gaussian_component_plot(results["gaussian_component_by_clipping"], args.output)
    print(f"Matched events: {len(ids)} | Pearson r={results['pearson_r']:.5f} "
          f"(p={results['pearson_p_value']:.3g}) | "
          f"Spearman rho={results['spearman_rho']:.5f} "
          f"(p={results['spearman_p_value']:.3g})", flush=True)
    print(f"Common-component timing fraction (beta)={results['common_fraction_timing']:.5f} "
          f"| 95% CI=({results['beta_ci_95'][0]:.5f}, {results['beta_ci_95'][1]:.5f}) "
          f"| H0: fraction<=0.5, one-sided p={results['dominance_test_p_value_one_sided']:.3g} "
          f"| nonnegative variance decomposition={results['common_component_variances_nonnegative']}",
          flush=True)
    print("Gaussian component fit by central percentile (per channel):", flush=True)
    for fit in results["gaussian_component_by_clipping"]:
        def fmt(value):
            return f"{value:.3f}" if value is not None else "N/A"
        print(f"  {fit['central_percent_per_channel']:5g}% "
              f"| n={fit['n_retained']} ({100*fit['retained_fraction']:.2f}%) "
              f"| sigma_common={fmt(fit['sigma_common_ps'])} ps "
              f"| sigma_energy={fmt(fit['sigma_energy_specific_ps'])} ps "
              f"| sigma_timing={fmt(fit['sigma_timing_specific_ps'])} ps "
              f"| f_timing={fit['fraction_common_timing']:.4f} "
              f"| admissible={fit['physically_admissible']}", flush=True)
    print(f"Difference std={results['difference_std_ps']:.3f} ps | "
          f"quadrature prediction={results['predicted_extra_std_ps_if_timing_is_common_only']} ps",
          flush=True)
    print(f"Fitted CTR: energy={results['ctr_energy_ps']:.2f} ps, "
          f"timing={results['ctr_timing_ps']:.2f} ps, "
          f"difference={results['ctr_difference_ps']:.2f} ps | "
          f"quadrature prediction={results['predicted_difference_ctr_ps_if_timing_is_common_only']}",
          flush=True)
    print(f"Results: {args.output}", flush=True)


if __name__ == "__main__":
    main()
