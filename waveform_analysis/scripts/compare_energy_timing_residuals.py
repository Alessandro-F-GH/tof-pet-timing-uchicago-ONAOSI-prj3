#!/usr/bin/env python3
"""Standalone robust Gaussian core and empirical-tail analysis of paired PET residuals.

Input NPZ files: event_id, led_residual_ps, corrected_ps.
Uses only numpy, scipy, matplotlib. Energy and timing events are matched by ID.
No GMM, Gaussian model for tails, cache, fitted ML models or waveform data.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import chi2, norm


def load_predictions(path):
    path = Path(path)
    with np.load(path, allow_pickle=False) as data:
        required = ("event_id", "led_residual_ps", "corrected_ps")
        missing = [k for k in required if k not in data]
        if missing:
            raise ValueError(f"{path}: missing {missing}")
        ids = np.asarray(data["event_id"])
        led = np.asarray(data["led_residual_ps"], dtype=float)
        ml = np.asarray(data["corrected_ps"], dtype=float)
    if ids.ndim != 1 or led.ndim != 1 or ml.ndim != 1:
        raise ValueError(f"{path}: arrays must be 1D")
    if len(ids) < 10 or len(ids) != len(led) or len(ids) != len(ml):
        raise ValueError(f"{path}: inconsistent lengths")
    if ids.dtype.kind not in "iu" or len(np.unique(ids)) != len(ids):
        raise ValueError(f"{path}: event_id must contain unique integers")
    if not np.isfinite(led).all() or not np.isfinite(ml).all():
        raise ValueError(f"{path}: nonfinite residuals")
    return ids.astype(np.int64), led, ml


def align_prediction_pairs(energy, timing):
    ids, ei, ti = np.intersect1d(energy[0], timing[0], assume_unique=True,
                                  return_indices=True)
    if len(ids) < 20:
        raise ValueError(f"Only {len(ids)} matched event IDs")
    return ids, np.column_stack((energy[1][ei], timing[1][ti])), np.column_stack(
        (energy[2][ei], timing[2][ti]))


def mahalanobis_squared(points, mean, covariance):
    centered = np.asarray(points) - mean
    return np.einsum("ij,jk,ik->i", centered, np.linalg.inv(covariance), centered)


def robust_start(points):
    """Coordinate medians and covariance of a compact robust initial subset."""
    center = np.median(points, axis=0)
    mad = np.median(np.abs(points - center), axis=0) * 1.4826
    mad = np.maximum(mad, np.maximum(np.std(points, axis=0)*1e-5, 1e-6))
    radial = np.sum(((points-center)/mad)**2, axis=1)
    subset = points[radial <= np.quantile(radial, .60)]
    cov = np.cov(subset, rowvar=False, ddof=0)
    # This is only an initializer; subsequent truncated-MLE moment
    # correction is computed analytically at the selected chi-square radius.
    cov = cov / max(chi2.cdf(chi2.ppf(.60, 2), 4)/.60, .1)
    return center, cov


def fit_gaussian_core(points, fit_fraction=.95, max_iter=100, tol=1e-6):
    """Iteratively fit a bivariate Gaussian core with radial-truncation correction.

    An inlier is selected using squared Mahalanobis distance <= chi2.ppf(p,2).
    For a truly bivariate Gaussian, conditional radial truncation multiplies
    covariance by c=F_chi2(df=4,q)/F_chi2(df=2,q). The selected sample covariance
    is divided by c, avoiding naive sigma underestimation from clipping.

    This is an iterative robust moment estimator, NOT an exact likelihood MLE.
    Its correction is only exact for a Gaussian population with fixed
    elliptical selection; contamination and estimated center introduce error.
    """
    x = np.asarray(points, float)
    if x.ndim != 2 or x.shape[1] != 2 or len(x) < 20:
        raise ValueError("Expected >=20 paired events")
    if not 0 < fit_fraction < 1:
        raise ValueError("fit_fraction must lie strictly between zero and one")
    q = chi2.ppf(fit_fraction, df=2)
    factor = chi2.cdf(q, df=4) / fit_fraction
    mean, cov = robust_start(x)
    converged = False
    iterations = 0
    for iterations in range(1, max_iter+1):
        try:
            d2 = mahalanobis_squared(x, mean, cov)
        except np.linalg.LinAlgError as exc:
            raise ValueError("Singular core covariance") from exc
        keep = d2 <= q
        if keep.sum() < max(20, int(len(x)*.2)):
            raise ValueError("Too few core events; robust fit failed")
        new_mean = x[keep].mean(axis=0)
        new_cov = np.cov(x[keep], rowvar=False, ddof=0) / factor
        if np.linalg.eigvalsh(new_cov).min() <= 0:
            raise ValueError("Core covariance is not positive definite")
        displacement = np.linalg.norm(new_mean-mean) / max(
            np.linalg.norm(np.sqrt(np.diag(cov))), 1e-9)
        scale_change = np.linalg.norm(new_cov-cov) / max(np.linalg.norm(cov), 1e-9)
        mean, cov = new_mean, new_cov
        if max(displacement, scale_change) < tol:
            converged = True
            break
    final_d2 = mahalanobis_squared(x, mean, cov)
    keep = final_d2 <= q
    sigmas = np.sqrt(np.diag(cov))
    return {
        "mean_ps": mean,
        "covariance_matrix_ps2": cov,
        "sigma_energy_ps": float(sigmas[0]),
        "sigma_timing_ps": float(sigmas[1]),
        "correlation": float(cov[0, 1]/np.prod(sigmas)),
        "cov_energy_timing_ps2": float(cov[0, 1]),
        "fit_fraction_nominal": float(fit_fraction),
        "fit_radius_squared": float(q),
        "fit_events": int(keep.sum()),
        "fit_events_fraction": float(keep.mean()),
        "truncation_covariance_factor": float(factor),
        "numerically_converged": converged,
        "iterations": iterations,
    }


def empirical_statistics(points):
    """Untruncated sample statistics on selected points, no parametric tail fit."""
    x = np.asarray(points, float)
    if len(x) < 3:
        return None
    cov = np.cov(x, rowvar=False)
    sd = np.sqrt(np.diag(cov))
    if np.any(sd <= 0):
        rho = None
    else:
        rho = float(cov[0, 1]/np.prod(sd))
    return {
        "n": int(len(x)),
        "mean_energy_ps": float(x[:, 0].mean()),
        "mean_timing_ps": float(x[:, 1].mean()),
        "sigma_energy_ps": float(sd[0]),
        "sigma_timing_ps": float(sd[1]),
        "correlation": rho,
        "cov_energy_timing_ps2": float(cov[0, 1]),
    }


def gaussian_reference_tail(core, threshold, seed=42, n_draw=300000):
    """Expected tail correlation after applying EXACTLY the same radial cut.

    This is a null reference for selection effects, not a model of observed tails.
    The Monte Carlo is purely Gaussian core and uses the fitted covariance.
    """
    rng = np.random.default_rng(seed)
    # Draw in chunks for controlled memory.
    outside = []
    remaining = n_draw
    mean = core["mean_ps"]
    cov = core["covariance_matrix_ps2"]
    while remaining:
        size = min(50000, remaining)
        z = rng.multivariate_normal(mean, cov, size=size)
        mask = mahalanobis_squared(z, mean, cov) > threshold
        if np.any(mask):
            outside.append(z[mask])
        remaining -= size
    subset = np.concatenate(outside, axis=0)
    return empirical_statistics(subset)


def core_cdf(x, mean, sigma):
    return norm.cdf(x, loc=mean, scale=sigma)


def core_expected_bin_counts(edges, core, axis, n):
    mu = core["mean_ps"][axis]
    sd = (core["sigma_energy_ps"] if axis == 0 else core["sigma_timing_ps"])
    return n * np.diff(core_cdf(edges, mu, sd))


def analyze_distribution(points, *, fit_fraction=.95, tail_fraction=.999,
                         seed=42, n_reference=300000):
    core = fit_gaussian_core(points, fit_fraction=fit_fraction)
    threshold = float(chi2.ppf(tail_fraction, df=2))
    distances = mahalanobis_squared(points, core["mean_ps"],
                                     core["covariance_matrix_ps2"])
    tail = distances > threshold
    if tail.sum() < 3:
        raise ValueError("Too few events outside core envelope for tail statistics")
    empirical_tail = empirical_statistics(points[tail])
    # Note: core statistics refer to fitted Gaussian, not empirical moments
    # of a truncated set. This distinction matters for interpretation.
    reference = gaussian_reference_tail(core, threshold, seed=seed,
                                        n_draw=n_reference)
    return {
        "core_gaussian_fit": core,
        "tail_definition": {
            "mahalanobis_squared_above": threshold,
            "nominal_gaussian_probability_outside": float(1-tail_fraction),
            "tail_count": int(tail.sum()),
            "tail_fraction": float(tail.mean()),
            "expected_count_from_core_gaussian": float(len(points)*(1-tail_fraction)),
            "excess_factor_over_core_gaussian": float(
                tail.mean()/(1-tail_fraction)),
        },
        "empirical_outside_core": empirical_tail,
        "gaussian_selection_only_reference": reference,
        "gaussian_reference_simulated_events": int(n_reference),
    }, tail, distances


def core_projection_interval(core, axis):
    """Coordinate projection of the elliptical region used in core fitting."""
    mean = float(core["mean_ps"][axis])
    sigma = core["sigma_energy_ps"] if axis == 0 else core["sigma_timing_ps"]
    half_width = np.sqrt(core["fit_radius_squared"]) * sigma
    return mean - half_width, mean + half_width


def plot_core_histograms(led, ml, summaries, output):
    """LED/ML overlays with linear count scale, restricted to the fit interval.

    Each channel uses the union of LED and ML projected *iterative fit*
    ellipses, giving both datasets exactly the same display range.
    Counts are computed on the full population with common fixed-width bins.
    """
    fig, axes = plt.subplots(1, 2, figsize=(13.2, 5), squeeze=False)
    for j, (channel, bin_width) in enumerate((("Energy", 10.0), ("Timing", 5.0))):
        ax = axes[0, j]
        intervals = [core_projection_interval(
            summaries[state]["core_gaussian_fit"], j) for state in ("led", "ml")]
        lower = min(interval[0] for interval in intervals)
        upper = max(interval[1] for interval in intervals)
        edges = np.arange(
            np.floor(lower / bin_width) * bin_width,
            np.ceil(upper / bin_width) * bin_width + 1.5 * bin_width,
            bin_width,
        )
        centers = (edges[:-1] + edges[1:]) / 2
        peak = 1.
        for state, pairs, color, label in (
            ("led", led, "tab:orange", "LED"),
            ("ml", ml, "tab:blue", "ML"),
        ):
            core = summaries[state]["core_gaussian_fit"]
            counts = np.histogram(pairs[:, j], bins=edges)[0]
            expected = core_expected_bin_counts(edges, core, j, len(pairs))
            peak = max(peak, float(counts.max()), float(expected.max()))
            ax.stairs(counts, edges, color=color, linewidth=1.3,
                      label=f"{label}: observed")
            ax.plot(centers, expected, color=color, linewidth=1.8,
                    linestyle="--", label=f"{label}: Gaussian core fit")
        ax.set_xlim(lower, upper)
        ax.set_ylim(0, peak * 1.12)
        ax.set_xlabel("Coincidence residual (ps)")
        ax.set_ylabel(f"Events per {bin_width:g} ps")
        ax.set_title(channel)
        ax.legend(fontsize=9)
        ax.grid(alpha=.2)
    fig.suptitle("LED vs ML — central iterative Gaussian-fit intervals")
    fig.tight_layout()
    fig.savefig(output / "led_vs_ml_core_distributions.png", dpi=190)
    fig.savefig(output / "led_vs_ml_core_distributions.pdf")
    plt.close(fig)


def plot_scatter(led, ml, summaries, tail_masks, output, seed=42):
    """Unclassified gray scatter with identical LED/ML x and y limits."""
    rng = np.random.default_rng(seed)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.4))
    combined = np.vstack((led, ml))
    x_min, y_min = np.min(combined, axis=0)
    x_max, y_max = np.max(combined, axis=0)
    x_padding = max(0.03 * (x_max - x_min), 1.)
    y_padding = max(0.03 * (y_max - y_min), 1.)
    for ax, (label, pairs) in zip(axes, (
        ("LED", led), ("After ML", ml),
    )):
        indices = np.arange(len(pairs))
        if len(indices) > 20000:
            indices = rng.choice(indices, 20000, replace=False)
        ax.scatter(pairs[indices, 0], pairs[indices, 1],
                   s=6, alpha=0.24, color="0.35", rasterized=True)
        ax.set_xlim(x_min - x_padding, x_max + x_padding)
        ax.set_ylim(y_min - y_padding, y_max + y_padding)
        ax.set_xlabel("Energy residual (ps)")
        ax.set_ylabel("Timing residual (ps)")
        ax.set_title(label)
        ax.grid(alpha=.2)
    fig.suptitle("Paired energy and timing residuals — identical axis limits")
    fig.tight_layout()
    fig.savefig(output / "led_vs_ml_joint_scatter.png", dpi=190)
    fig.savefig(output / "led_vs_ml_joint_scatter.pdf")
    plt.close(fig)


def plot_radial_exceedance(distances, summaries, output):
    fig, ax = plt.subplots(figsize=(8, 5))
    q = np.linspace(0, max(30., *(float(np.quantile(d, .9999)) for d in
                                   distances.values())), 350)
    for label, d in distances.items():
        observed = (len(d) - np.searchsorted(np.sort(d), q, side="right"))/len(d)
        ax.plot(q, np.maximum(observed, .5/len(d)), lw=1.4,
                label=f"{label.upper()} empirical")
    ax.plot(q, chi2.sf(q, df=2), ls="--", color="black",
            label="Bivariate Gaussian core reference")
    ax.set_yscale("log")
    ax.set_ylim(bottom=1e-5, top=1.1)
    ax.set_xlabel("Squared Mahalanobis distance from fitted core")
    ax.set_ylabel("Fraction outside radius")
    ax.legend()
    ax.grid(alpha=.2)
    fig.tight_layout()
    fig.savefig(output / "core_radial_exceedance.png", dpi=190)
    fig.savefig(output / "core_radial_exceedance.pdf")
    plt.close(fig)


def jsonable(value):
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer, np.bool_)):
        return value.item()
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    return value


def analyze(energy_file, timing_file, output, *, fit_fraction=.95,
            tail_fraction=.999, seed=42, n_reference=300000):
    _, led, ml = align_prediction_pairs(load_predictions(energy_file),
                                        load_predictions(timing_file))
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    results = {
        "n_matched_events": int(len(led)),
        "energy_prediction_file": str(energy_file),
        "timing_prediction_file": str(timing_file),
    }
    masks, distances = {}, {}
    for state, pairs in (("led", led), ("ml", ml)):
        results[state], masks[state], distances[state] = analyze_distribution(
            pairs, fit_fraction=fit_fraction, tail_fraction=tail_fraction,
            seed=seed, n_reference=n_reference)
    results["interpretation"] = (
        "Core is an iteratively fitted bivariate Gaussian, adjusted for radial "
        "truncation. Tail means events outside a fixed chi-square Mahalanobis "
        "envelope; their correlation and covariance are empirical, not fitted "
        "Gaussian tail parameters. Core fit convergence is numerical only. "
        "Gaussian selection reference applies the identical tail selection to "
        "synthetic Gaussian core observations. A common-variance interpretation "
        "of covariance requires additional equal-loading and independence "
        "assumptions. Neither selection nor correlation identifies physics.")
    (output / "core_tail_summary.json").write_text(
        json.dumps(jsonable(results), indent=2) + "\n", encoding="utf-8")
    plot_core_histograms(led, ml, results, output)
    plot_scatter(led, ml, results, masks, output, seed=seed)
    plot_radial_exceedance(distances, results, output)
    print("=" * 78, flush=True)
    print("PAIRED PET RESIDUALS | GAUSSIAN CORE + EMPIRICAL OUTSIDE-CORE EVENTS", flush=True)
    print("=" * 78, flush=True)
    print(f"Energy input: {energy_file}", flush=True)
    print(f"Timing input: {timing_file}", flush=True)
    print(f"Events matched by event_id: {len(led):,}", flush=True)
    print(f"Core fitting: iterative bivariate Gaussian, fit_fraction={fit_fraction:.4%} "
          f"(Mahalanobis squared radius <= {chi2.ppf(fit_fraction, 2):.3f})", flush=True)
    print("  Gaussian radial-truncation correction is applied to core covariance.", flush=True)
    print(f"Outside-core definition: tail_fraction={tail_fraction:.4%} "
          f"(Mahalanobis squared distance > {chi2.ppf(tail_fraction, 2):.3f})", flush=True)
    print("  The fraction specifies a Gaussian reference envelope, NOT the observed "
          "fraction of tail events.", flush=True)
    print("  No distribution is fitted or assumed for events outside that envelope.", flush=True)
    print(f"Selection-bias reference: {n_reference:,} Gaussian-core Monte Carlo draws "
          f"(seed={seed}).", flush=True)

    for state, heading in (("led", "BEFORE ML — LED"),
                           ("ml", "AFTER ML — CORRECTED")):
        fit = results[state]
        core = fit["core_gaussian_fit"]
        ext = fit["tail_definition"]
        tail = fit["empirical_outside_core"]
        reference = fit["gaussian_selection_only_reference"]
        print("\\n" + "-" * 78, flush=True)
        print(heading, flush=True)
        print("-" * 78, flush=True)
        print("GAUSSIAN CORE (corrected for elliptical fit-region truncation)", flush=True)
        print(f'  Numerical convergence: {core["numerically_converged"]} '
              f'after {core["iterations"]} iterations '
              "(this is NOT a Gaussianity test)", flush=True)
        print(f'  Selected during iterative fit: {core["fit_events"]:,} '
              f'({100 * core["fit_events_fraction"]:.2f}% of matched events)', flush=True)
        print(f'  Mean residuals: energy={core["mean_ps"][0]:+.2f} ps, '
              f'timing={core["mean_ps"][1]:+.2f} ps', flush=True)
        print(f'  Widths: sigma_energy={core["sigma_energy_ps"]:.2f} ps, '
              f'sigma_timing={core["sigma_timing_ps"]:.2f} ps', flush=True)
        print(f'  Energy/timing correlation: rho={core["correlation"]:.5f}', flush=True)
        print(f'  Energy/timing covariance: '
              f'{core["cov_energy_timing_ps2"]:.2f} ps^2', flush=True)

        print("EMPIRICAL EVENTS OUTSIDE THE CORE ENVELOPE", flush=True)
        print(f'  Observed: {ext["tail_count"]:,} events '
              f'({100 * ext["tail_fraction"]:.4f}% of matched events)', flush=True)
        print(f'  Gaussian-core reference expectation: '
              f'{ext["expected_count_from_core_gaussian"]:.1f} events', flush=True)
        print(f'  Observed / reference expectation: '
              f'{ext["excess_factor_over_core_gaussian"]:.2f}x', flush=True)
        if tail is None:
            print("  Fewer than 3 outside-core events: empirical covariance unavailable.",
                  flush=True)
        else:
            print(f'  Empirical widths: sigma_energy={tail["sigma_energy_ps"]:.2f} ps, '
                  f'sigma_timing={tail["sigma_timing_ps"]:.2f} ps', flush=True)
            print(f'  Empirical energy/timing correlation: '
                  f'rho={tail["correlation"]:.5f}', flush=True)
            print(f'  Empirical energy/timing covariance: '
                  f'{tail["cov_energy_timing_ps2"]:.2f} ps^2', flush=True)
        if reference is not None:
            print("SELECTION-ONLY GAUSSIAN REFERENCE (NOT A FIT TO OBSERVED TAILS)",
                  flush=True)
            print(f'  Reference rho after identical selection: '
                  f'{reference["correlation"]:.5f} '
                  f'({reference["n"]:,} simulated outside-envelope draws)', flush=True)
        print("  Note: empirical outside-core correlation is affected by the "
              "selection rule.", flush=True)

    print("\\n" + "=" * 78, flush=True)
    print("OUTPUT FILES", flush=True)
    for name, description in (
        ("core_tail_summary.json", "Numerical core and empirical outside-core summary"),
        ("led_vs_ml_core_distributions.png/.pdf", "Overlaid LED/ML counts and core Gaussian curves"),
        ("led_vs_ml_joint_scatter.png/.pdf", "Unclassified gray energy/timing scatter"),
        ("core_radial_exceedance.png/.pdf", "Radial survival against Gaussian-core reference"),
    ):
        print(f"  {name}: {description}", flush=True)
    print(f"Directory: {output.resolve()}", flush=True)
    print("Covariance is NOT automatically a physically identified common-noise "
          "variance.", flush=True)
    return results


def resolve_board_files(root, board):
    """Locate exactly one energy and one timing NPZ in root/UC or root/FBK.

    Supports energy_predictions.npz/timing_predictions.npz and energy.npz/
    timing.npz; never guesses from multiple ambiguous candidates.
    """
    folder = Path(root) / board
    if not folder.is_dir():
        raise FileNotFoundError(f"Missing required board directory: {folder}")
    found = {}
    for channel in ("energy", "timing"):
        candidates = [
            folder / f"{channel}_predictions.npz",
            folder / f"{channel}.npz",
        ]
        matches = [path for path in candidates if path.is_file()]
        if len(matches) != 1:
            raise ValueError(
                f"{folder}: expected exactly one prediction NPZ for {channel} "
                f"(energy_predictions.npz / timing_predictions.npz preferred), "
                f"found {[p.name for p in matches]}.")
        found[channel] = matches[0]
    return found


def bootstrap_correlations(points, samples=300, seed=42):
    """Fixed-membership nonparametric bootstrap for selected-event correlation.

    This does NOT re-estimate the robust core or tail membership. It represents
    conditional sample uncertainty only, not uncertainty in the cut boundary.
    """
    x = np.asarray(points, dtype=float)
    if len(x) < 4 or np.std(x[:, 0]) == 0 or np.std(x[:, 1]) == 0:
        return None
    rng = np.random.default_rng(seed)
    output = []
    for _ in range(samples):
        subset = x[rng.integers(0, len(x), len(x))]
        if np.std(subset[:, 0]) > 0 and np.std(subset[:, 1]) > 0:
            output.append(float(np.corrcoef(subset[:, 0], subset[:, 1])[0, 1]))
    if len(output) < samples * .9:
        return None
    return np.array(output)


def compare_boards(analyses, bootstrap_samples=300, seed=42):
    """UC-minus-FBK comparisons with conditional bootstrap uncertainty.

    Bootstrap core events are defined by each fitted core's *fit radius*,
    whereas empirical tail events use the more extreme tail radius.
    """
    outcome = {"bootstrap_resamples": bootstrap_samples,
               "uncertainty_scope": "Fixed core/tail membership; core fits and "
                                    "selection boundaries not bootstrapped.",
               "states": {}}
    for state in ("led", "ml"):
        comparison = {}
        for region in ("core", "outside_core"):
            draws = {}
            point = {}
            for board in ("UC", "FBK"):
                pair = analyses[board]["pairs"][state]
                report = analyses[board]["results"][state]
                core = report["core_gaussian_fit"]
                distance = mahalanobis_squared(
                    pair, core["mean_ps"], core["covariance_matrix_ps2"])
                mask = (distance <= core["fit_radius_squared"]) if region == "core" else (
                    distance > report["tail_definition"]["mahalanobis_squared_above"])
                selected = pair[mask]
                stats = empirical_statistics(selected)
                point[board] = stats["correlation"] if stats is not None else None
                draws[board] = bootstrap_correlations(
                    selected, samples=bootstrap_samples,
                    seed=seed + (0 if board == "UC" else 1777)
                    + (0 if region == "core" else 3009)
                    + (0 if state == "led" else 5011))
            if draws["UC"] is not None and draws["FBK"] is not None:
                difference = draws["UC"] - draws["FBK"]
                lower, upper = np.quantile(difference, [.025, .975])
                ci = [float(lower), float(upper)]
            else:
                ci = None
            difference_point = (point["UC"] - point["FBK"]) if (
                point["UC"] is not None and point["FBK"] is not None) else None
            comparison[region] = {
                "rho_UC": point["UC"],
                "rho_FBK": point["FBK"],
                "delta_rho_UC_minus_FBK": difference_point,
                "delta_rho_bootstrap_95_ci": ci,
                "zero_within_95_ci": (ci[0] <= 0 <= ci[1]) if ci else None,
            }
        # The two core fitted correlations are also reported directly.
        comparison["fitted_core"] = {
            "rho_UC": analyses["UC"]["results"][state]["core_gaussian_fit"]["correlation"],
            "rho_FBK": analyses["FBK"]["results"][state]["core_gaussian_fit"]["correlation"],
        }
        comparison["fitted_core"]["delta_rho_UC_minus_FBK"] = (
            comparison["fitted_core"]["rho_UC"] -
            comparison["fitted_core"]["rho_FBK"])
        comparison["core_widths_and_tail_frequency"] = {}
        for field, section, key in (
            ("core_sigma_energy_ps", "core_gaussian_fit", "sigma_energy_ps"),
            ("core_sigma_timing_ps", "core_gaussian_fit", "sigma_timing_ps"),
            ("outside_core_fraction", "tail_definition", "tail_fraction"),
            ("outside_core_excess_factor", "tail_definition", "excess_factor_over_core_gaussian"),
        ):
            uc = float(analyses["UC"]["results"][state][section][key])
            fbk = float(analyses["FBK"]["results"][state][section][key])
            comparison["core_widths_and_tail_frequency"][field] = {
                "UC": uc, "FBK": fbk, "UC_minus_FBK": uc - fbk,
            }
        outcome["states"][state] = comparison
    outcome["interpretation"] = (
        "If the conditional bootstrap difference interval includes zero, the "
        "observed correlation difference is not resolved by this check; it "
        "does not establish equivalence. Intervals excluding zero indicate "
        "a detectable difference under fixed selection. Measurements from "
        "different boards are treated as independent, and systematic effects "
        "or uncertainty in the fitting/selection method are not included.")
    return outcome


def plot_board_comparison(analyses, output):
    """UC/FBK overlaid central histograms per state and channel; shared axes."""
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    colors = {"UC": "tab:blue", "FBK": "tab:orange"}
    for row, state in enumerate(("led", "ml")):
        for col, (channel, width) in enumerate((("Energy", 10.), ("Timing", 5.))):
            ax = axes[row, col]
            intervals = [core_projection_interval(
                analyses[board]["results"][state]["core_gaussian_fit"], col)
                for board in ("UC", "FBK")]
            lower = min(v[0] for v in intervals)
            upper = max(v[1] for v in intervals)
            edges = np.arange(np.floor(lower / width)*width,
                              np.ceil(upper / width)*width + 1.5*width, width)
            centers = .5*(edges[:-1] + edges[1:])
            peak = 1.
            for board in ("UC", "FBK"):
                pair = analyses[board]["pairs"][state]
                core = analyses[board]["results"][state]["core_gaussian_fit"]
                count = np.histogram(pair[:, col], bins=edges)[0]
                prediction = core_expected_bin_counts(edges, core, col, len(pair))
                peak = max(peak, float(count.max()), float(prediction.max()))
                ax.stairs(count, edges, color=colors[board], lw=1.25,
                          label=f"{board} observed (N={len(pair):,})")
                ax.plot(centers, prediction, color=colors[board], lw=1.5,
                        ls="--", label=f"{board} core fit")
            ax.set_xlim(lower, upper)
            ax.set_ylim(0, peak*1.12)
            ax.set_title(f"{state.upper()} — {channel}")
            ax.set_xlabel("Residual (ps)")
            ax.set_ylabel(f"Events / {width:g} ps (different sample sizes)")
            ax.grid(alpha=.2)
            ax.legend(fontsize=8)
    fig.suptitle("UC vs FBK — empirical counts and Gaussian cores")
    fig.tight_layout()
    fig.savefig(output / "uc_vs_fbk_core_distributions.png", dpi=190)
    fig.savefig(output / "uc_vs_fbk_core_distributions.pdf")
    plt.close(fig)


def plot_board_scatter(analyses, output, seed=42):
    """Board-scatter comparison using identical limits across ALL four plots."""
    pairs = [analyses[board]["pairs"][state]
             for state in ("led", "ml") for board in ("UC", "FBK")]
    combined = np.concatenate(pairs, axis=0)
    min_xy = combined.min(axis=0)
    max_xy = combined.max(axis=0)
    padding = np.maximum((max_xy-min_xy)*.03, 1.)
    rng = np.random.default_rng(seed)
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 10))
    for row, state in enumerate(("led", "ml")):
        for col, board in enumerate(("UC", "FBK")):
            ax = axes[row, col]
            points = analyses[board]["pairs"][state]
            idx = np.arange(len(points))
            if len(idx) > 20000:
                idx = rng.choice(idx, size=20000, replace=False)
            ax.scatter(points[idx, 0], points[idx, 1], color=".35",
                       alpha=.24, s=6, rasterized=True)
            ax.set_xlim(min_xy[0]-padding[0], max_xy[0]+padding[0])
            ax.set_ylim(min_xy[1]-padding[1], max_xy[1]+padding[1])
            ax.set_title(f"{board} — {state.upper()}")
            ax.set_xlabel("Energy residual (ps)")
            ax.set_ylabel("Timing residual (ps)")
            ax.grid(alpha=.2)
    fig.tight_layout()
    fig.savefig(output / "uc_vs_fbk_joint_scatter.png", dpi=190)
    fig.savefig(output / "uc_vs_fbk_joint_scatter.pdf")
    plt.close(fig)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions-root", type=Path, required=True,
                        help="Directory with UC/ and FBK/, each containing "
                             "energy_predictions.npz and timing_predictions.npz")
    parser.add_argument("--output", type=Path, default=Path("core_tail_results"))
    parser.add_argument("--fit-fraction", type=float, default=.95,
                        help="Nominal central Gaussian coverage for iterative fit")
    parser.add_argument("--tail-fraction", type=float, default=.999,
                        help="Nominal Gaussian envelope used to flag extreme events")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--reference-events", type=int, default=300000)
    parser.add_argument("--bootstrap-resamples", type=int, default=300,
                        help="Fixed-selection bootstrap for UC-FBK correlation differences")
    args = parser.parse_args(argv)
    if not 0 < args.fit_fraction < args.tail_fraction < 1:
        parser.error("Require 0 < --fit-fraction < --tail-fraction < 1")
    if args.reference_events < 1000:
        parser.error("--reference-events must be >= 1000")
    if args.bootstrap_resamples < 50:
        parser.error("--bootstrap-resamples must be >= 50")

    args.output.mkdir(parents=True, exist_ok=True)
    analyses = {}
    for board in ("UC", "FBK"):
        paths = resolve_board_files(args.predictions_root, board)
        print("\\n" + "#" * 78)
        print(f"ANALYSIS FOR BOARD {board}")
        print("#" * 78, flush=True)
        result = analyze(paths["energy"], paths["timing"], args.output / board,
                         fit_fraction=args.fit_fraction,
                         tail_fraction=args.tail_fraction,
                         seed=args.seed, n_reference=args.reference_events)
        _, led, ml = align_prediction_pairs(
            load_predictions(paths["energy"]),
            load_predictions(paths["timing"]))
        analyses[board] = {"results": result, "pairs": {"led": led, "ml": ml}}

    comparison = compare_boards(analyses, bootstrap_samples=args.bootstrap_resamples,
                                seed=args.seed)
    (args.output / "uc_fbk_comparison.json").write_text(
        json.dumps(jsonable(comparison), indent=2) + "\\n", encoding="utf-8")
    plot_board_comparison(analyses, args.output)
    plot_board_scatter(analyses, args.output, seed=args.seed)

    print("\\n" + "=" * 78)
    print("UC vs FBK | CORRELATION COMPATIBILITY (DESCRIPTIVE BOOTSTRAP)")
    print("=" * 78)
    print("Each board was analyzed separately with identical fit/tail fractions.")
    print(f"Bootstrap samples: {args.bootstrap_resamples} per fixed selected population.")
    print("Confidence intervals exclude core-fit uncertainty and threshold uncertainty.")
    for state in ("led", "ml"):
        print(f"\\n{state.upper()}")
        for field, entry in comparison["states"][state][
                "core_widths_and_tail_frequency"].items():
            suffix = "%" if field == "outside_core_fraction" else ""
            multiplier = 100 if suffix else 1
            print(f'  {field}: UC={entry["UC"]*multiplier:.4f}{suffix}'
                  f' | FBK={entry["FBK"]*multiplier:.4f}{suffix}'
                  f' | delta={entry["UC_minus_FBK"]*multiplier:+.4f}{suffix}')
        fitted = comparison["states"][state]["fitted_core"]
        print(f'  fitted Gaussian-core rho: UC={fitted["rho_UC"]:.5f}'
              f' | FBK={fitted["rho_FBK"]:.5f}'
              f' | delta={fitted["delta_rho_UC_minus_FBK"]:+.5f}')
        for region in ("core", "outside_core"):
            entry = comparison["states"][state][region]
            print(f'  {region}: rho_UC={entry["rho_UC"]:.5f} '
                  f'| rho_FBK={entry["rho_FBK"]:.5f} '
                  f'| delta(UC-FBK)={entry["delta_rho_UC_minus_FBK"]:+.5f}')
            ci = entry["delta_rho_bootstrap_95_ci"]
            if ci is not None:
                print(f"    Conditional 95% CI of difference: [{ci[0]:+.5f}, "
                      f"{ci[1]:+.5f}] | includes zero: {entry['zero_within_95_ci']}")
            else:
                print("    Bootstrap interval unavailable (too few varying events).")
    print("\\nShared results:")
    for name in ("uc_fbk_comparison.json", "uc_vs_fbk_core_distributions.png/.pdf",
                 "uc_vs_fbk_joint_scatter.png/.pdf"):
        print(f"  {args.output / name}")
    print("Per-board figures and JSON: output/UC and output/FBK")
    print("A CI including zero does not prove UC and FBK are equivalent.")


if __name__ == "__main__":
    main()
