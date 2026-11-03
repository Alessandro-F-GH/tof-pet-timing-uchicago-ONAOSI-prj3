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
import sys
from pathlib import Path

import matplotlib
if "--show-waveforms" not in sys.argv:
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import pearsonr, spearmanr, linregress, t as student_t
from sklearn.mixture import GaussianMixture
from matplotlib.patches import Ellipse
from waveform_analysis.data.dataset import load_prepared_dataset
from waveform_analysis.data.named_cache import resolve_alias
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


def fit_bivariate_gmm(energy, timing, *, seed=42, n_init=5):
    """Two full-covariance Gaussian components, on matched residual pairs.

    Labels are assigned using the determinant of each covariance matrix:
    smaller area = core, larger area = tail. This is a statistical naming
    convention, not evidence of distinct physical hit populations.
    """
    x = np.column_stack((energy, timing))
    if len(x) < 10:
        raise ValueError("GMM requires at least 10 paired events")
    scale = np.std(x, axis=0)
    if np.any(scale <= 0):
        raise ValueError("Both residual coordinates must vary for GMM")
    center = np.mean(x, axis=0)
    normalized = (x - center) / scale
    model = GaussianMixture(
        n_components=2, covariance_type="full", n_init=n_init,
        random_state=seed, reg_covar=1e-6, max_iter=300,
    ).fit(normalized)
    single = GaussianMixture(
        n_components=1, covariance_type="full", n_init=1,
        random_state=seed, reg_covar=1e-6, max_iter=300,
    ).fit(normalized)
    means = model.means_ * scale + center
    covariances = model.covariances_ * scale[None, :, None] * scale[None, None, :]
    order = np.argsort(np.linalg.det(covariances))
    responsibilities = model.predict_proba(normalized)[:, order]
    parts = []
    for label, index in zip(("core", "tail"), order):
        cov = covariances[index]
        sd_energy, sd_timing = np.sqrt(np.diag(cov))
        covariance = float(cov[0, 1])
        shared = {
            "common_variance_ps2": covariance,
            "energy_specific_variance_ps2": float(cov[0, 0] - covariance),
            "timing_specific_variance_ps2": float(cov[1, 1] - covariance),
        }
        valid = all(v >= 0 for v in shared.values())
        parts.append({
            "label": label,
            "weight": float(model.weights_[index]),
            "mean_energy_ps": float(means[index, 0]),
            "mean_timing_ps": float(means[index, 1]),
            "sigma_energy_ps": float(sd_energy),
            "sigma_timing_ps": float(sd_timing),
            "correlation": float(covariance / (sd_energy * sd_timing)),
            "covariance_ps2": covariance,
            "covariance_matrix_ps2": cov.tolist(),
            "common_noise_decomposition_admissible": valid,
            "sigma_common_ps": float(np.sqrt(shared["common_variance_ps2"]))
                if shared["common_variance_ps2"] >= 0 else None,
            "sigma_energy_specific_ps": float(np.sqrt(shared["energy_specific_variance_ps2"]))
                if shared["energy_specific_variance_ps2"] >= 0 else None,
            "sigma_timing_specific_ps": float(np.sqrt(shared["timing_specific_variance_ps2"]))
                if shared["timing_specific_variance_ps2"] >= 0 else None,
        })
    summary = {
        "n_events": int(len(x)),
        "components": parts,
        "delta_correlation_tail_minus_core":
            parts[1]["correlation"] - parts[0]["correlation"],
        "bic_one_component": float(single.bic(normalized)),
        "bic_two_components": float(model.bic(normalized)),
        "delta_bic_one_minus_two": float(single.bic(normalized) - model.bic(normalized)),
        "converged": bool(model.converged_),
        "n_iterations": int(model.n_iter_),
        "n_initializations": int(n_init),
        "core_tail_label_rule": "Increasing determinant of full 2D covariance matrix",
        "note": "Correlations are within-component parameters, not correlations "
                "after hard event assignment. Components are descriptive, "
                "not proven physical populations. Variance decomposition requires "
                "additive unit-loading common noise and independent channel-specific noise.",
    }
    return summary, responsibilities


def save_bivariate_gmm_plot(energy, timing, summary, responsibilities, output, seed=42):
    """Scatter colored by tail responsibility with 2-sigma component ellipses."""
    from matplotlib.colors import Normalize
    x = np.column_stack((energy, timing))
    rng = np.random.default_rng(seed)
    sample = rng.choice(len(x), min(len(x), 10000), replace=False)
    fig, ax = plt.subplots(figsize=(7.2, 5.8))
    scatter = ax.scatter(
        energy[sample], timing[sample], c=responsibilities[sample, 1],
        cmap="viridis", norm=Normalize(0, 1), s=5, alpha=0.45,
        rasterized=True,
    )
    for part, color in zip(summary["components"], ("tab:blue", "tab:orange")):
        covariance = np.asarray(part["covariance_matrix_ps2"])
        eigenvalues, eigenvectors = np.linalg.eigh(covariance)
        direction = eigenvectors[:, -1]
        angle = np.degrees(np.arctan2(direction[1], direction[0]))
        ellipse = Ellipse(
            (part["mean_energy_ps"], part["mean_timing_ps"]),
            width=4*np.sqrt(eigenvalues[-1]),
            height=4*np.sqrt(eigenvalues[0]), angle=angle,
            facecolor="none", edgecolor=color, linewidth=2,
            label=f'{part["label"]} (rho={part["correlation"]:.3f})',
        )
        ax.add_patch(ellipse)
    fig.colorbar(scatter, ax=ax, label="P(tail | event)")
    ax.set_xlabel("Energy residual (ps)")
    ax.set_ylabel("Timing residual (ps)")
    ax.set_title("Two-component bivariate Gaussian mixture")
    ax.legend()
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(output / "bivariate_gmm.png", dpi=180)
    fig.savefig(output / "bivariate_gmm.pdf")
    plt.close(fig)


def waveform_groups_from_gmm(energy, timing, gmm_summary, responsibilities):
    """Classify tail events by their common-mode residual direction.

    A probability >= 0.5 assigns an event to the broad GMM component;
    the signed average of residuals centered on the tail-component means
    separates the left and right tails. Remaining events form the core.
    """
    tail = responsibilities[:, 1] >= 0.5
    params = gmm_summary["components"][1]
    displacement = 0.5 * (
        (energy - params["mean_energy_ps"]) +
        (timing - params["mean_timing_ps"])
    )
    groups = {
        "left tail": tail & (displacement < 0),
        "core": ~tail,
        "right tail": tail & (displacement >= 0),
    }
    return groups


def prepared_waveform_means(prepared_path, channel, ids, groups):
    """Read the already-prepared normalized waveforms, aligned by event_index.

    Inverse normalization uses the dataset's frozen transform, returning
    signal amplitudes in mV without retraining or additional preprocessing.
    """
    prepared = load_prepared_dataset(resolve_alias(prepared_path))
    windows = getattr(prepared, f"{channel}_windows")
    time_ps = getattr(prepared, f"{channel}_time_ps")
    transform = getattr(prepared, f"{channel}_transform")
    if windows is None or time_ps is None or transform is None:
        raise ValueError(f"{prepared_path}: missing prepared {channel} waveform/time/transform")
    if windows.ndim != 3 or windows.shape[1] != 2:
        raise ValueError(f"{prepared_path}: expected [event, detector, sample] waveforms")
    cache_ids = np.asarray(prepared.event_index)
    if len(cache_ids) != len(np.unique(cache_ids)):
        raise ValueError(f"{prepared_path}: duplicated event_index")
    common, indices, cache_positions = np.intersect1d(
        ids, cache_ids, assume_unique=True, return_indices=True)
    if len(common) != len(ids):
        raise ValueError(
            f"{prepared_path}: {len(ids)-len(common)} prediction event IDs "
            f"missing in prepared {channel} data. Check dataset role/cache.")
    result = {}
    for name, selected in groups.items():
        rows = cache_positions[selected[indices]]
        if len(rows) == 0:
            result[name] = None
            continue
        # Load only selected rows and invert frozen amplitude normalization.
        accum = np.zeros((2, windows.shape[-1]), dtype=np.float64)
        accum_sq = np.zeros_like(accum)
        for chunk in np.array_split(rows, max(1, (len(rows) + 1023) // 1024)):
            wave = transform.inverse(np.asarray(windows[chunk], dtype=np.float32))
            accum += wave.sum(axis=0, dtype=np.float64)
            accum_sq += np.square(wave, dtype=np.float64).sum(axis=0)
        mean = accum / len(rows)
        std = np.sqrt(np.maximum(accum_sq / len(rows) - mean**2, 0))
        result[name] = {"count": int(len(rows)), "mean": mean, "std": std}
    return np.asarray(time_ps, dtype=np.float64), result


def show_grouped_waveforms(ids, energy, timing, summary, responsibilities,
                           energy_prepared, timing_prepared):
    """Interactive 2x2 comparison: energy/timing x detector 1/2."""
    groups = waveform_groups_from_gmm(energy, timing, summary, responsibilities)
    e_times, e_groups = prepared_waveform_means(energy_prepared, "energy", ids, groups)
    t_times, t_groups = prepared_waveform_means(timing_prepared, "timing", ids, groups)
    fig, axes = plt.subplots(2, 2, figsize=(13, 8), squeeze=False)
    palette = {"left tail": "tab:blue", "core": "tab:green", "right tail": "tab:orange"}
    for row, (channel, times, data) in enumerate((
        ("Energy", e_times, e_groups), ("Timing", t_times, t_groups)
    )):
        for detector in range(2):
            ax = axes[row, detector]
            for label in ("left tail", "core", "right tail"):
                entry = data[label]
                if entry is None:
                    continue
                mean, std = entry["mean"][detector], entry["std"][detector]
                ax.plot(times / 1000., mean, color=palette[label],
                        label=f"{label} (n={entry['count']})", linewidth=1.5)
                ax.fill_between(times / 1000., mean-std, mean+std,
                                color=palette[label], alpha=0.10)
            ax.set_title(f"{channel}, detector {detector+1}")
            ax.set_xlabel("Time (ns)")
            ax.set_ylabel("Signal (mV)")
            ax.grid(alpha=0.2)
            ax.legend(fontsize=8)
    fig.suptitle("Waveform shapes by joint residual GMM group (mean +/- 1 std)")
    fig.tight_layout()
    plt.show()


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
    parser.add_argument("--show-waveforms", action="store_true",
                        help="Open interactive waveform comparison for GMM left/core/right groups")
    parser.add_argument("--energy-prepared", type=Path,
                        help="Prepared energy dataset directory or named alias, same role as predictions")
    parser.add_argument("--timing-prepared", type=Path,
                        help="Prepared timing dataset directory or named alias, same role as predictions")
    parser.add_argument("--gmm-n-init", type=int, default=5,
                        help="GMM random initializations (default: 5)")
    args = parser.parse_args(argv)
    if args.show_waveforms and (args.energy_prepared is None or args.timing_prepared is None):
        parser.error("--show-waveforms requires --energy-prepared and --timing-prepared")
    if args.gmm_n_init < 1:
        parser.error("--gmm-n-init must be >= 1")
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
    gmm_summary, gmm_responsibilities = fit_bivariate_gmm(
        energy, timing, seed=args.seed, n_init=args.gmm_n_init)
    results["bivariate_gmm"] = gmm_summary
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
    save_bivariate_gmm_plot(energy, timing, gmm_summary, gmm_responsibilities,
                            args.output, seed=args.seed)
    with (args.output / "gmm_event_responsibilities.csv").open(
            "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["event_id", "p_core", "p_tail"])
        writer.writerows(zip(ids.tolist(), gmm_responsibilities[:, 0].tolist(),
                             gmm_responsibilities[:, 1].tolist()))
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
    print("Bivariate 2-Gaussian mixture:", flush=True)
    for part in gmm_summary["components"]:
        print(f"  {part['label']}: weight={part['weight']:.4f} "
              f"| sigma_E={part['sigma_energy_ps']:.3f} ps "
              f"| sigma_T={part['sigma_timing_ps']:.3f} ps "
              f"| rho={part['correlation']:.5f}", flush=True)
    print(f"  delta_rho(tail-core)={gmm_summary['delta_correlation_tail_minus_core']:.5f} "
          f"| delta_BIC(1-2)={gmm_summary['delta_bic_one_minus_two']:.1f} "
          f"| converged={gmm_summary['converged']}", flush=True)
    print(f"Difference std={results['difference_std_ps']:.3f} ps | "
          f"quadrature prediction={results['predicted_extra_std_ps_if_timing_is_common_only']} ps",
          flush=True)
    print(f"Fitted CTR: energy={results['ctr_energy_ps']:.2f} ps, "
          f"timing={results['ctr_timing_ps']:.2f} ps, "
          f"difference={results['ctr_difference_ps']:.2f} ps | "
          f"quadrature prediction={results['predicted_difference_ctr_ps_if_timing_is_common_only']}",
          flush=True)
    print(f"Results: {args.output}", flush=True)
    if args.show_waveforms:
        show_grouped_waveforms(
            ids, energy, timing, gmm_summary, gmm_responsibilities,
            args.energy_prepared, args.timing_prepared,
        )


if __name__ == "__main__":
    main()
