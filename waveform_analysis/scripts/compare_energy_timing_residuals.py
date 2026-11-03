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


def plot_core_histograms(led, ml, summaries, output, bin_width=10.):
    """Overlay LED/ML observed counts and fitted Gaussian cores per channel.

    The y-axis is set using the actual observed peak and fitted peak, not
    extrapolated low-density tails. The x-axis remains full-range to expose
    rare anomalous events, and counts are plotted with logarithmic scale.
    """
    fig, axes = plt.subplots(1, 2, figsize=(13.2, 5), squeeze=False)
    for j, channel in enumerate(("Energy", "Timing")):
        ax = axes[0, j]
        max_abs = max(float(np.max(np.abs(led[:, j]))),
                      float(np.max(np.abs(ml[:, j]))))
        extent = max(200., np.ceil(max_abs / bin_width) * bin_width)
        edges = np.arange(-extent, extent + 1.5 * bin_width, bin_width)
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
            # Zero-count bins are not drawn on the logarithmic axis.
            ax.stairs(np.where(counts > 0, counts, np.nan), edges,
                      color=color, linewidth=1.25,
                      label=f"{label}: observed")
            ax.plot(centers, np.where(expected > 0, expected, np.nan),
                    color=color, linewidth=1.6, linestyle="--",
                    label=f"{label}: fitted Gaussian core")
        ax.set_yscale("log")
        ax.set_ylim(0.8, peak * 1.25)
        ax.set_xlim(edges[0], edges[-1])
        ax.set_xlabel("Coincidence residual (ps)")
        ax.set_ylabel(f"Observed / expected events per {bin_width:g} ps")
        ax.set_title(channel)
        ax.legend(fontsize=9)
        ax.grid(alpha=.2)
    fig.suptitle("LED vs ML: observed distributions and Gaussian-core fits")
    fig.tight_layout()
    fig.savefig(output / "led_vs_ml_core_distributions.png", dpi=190)
    fig.savefig(output / "led_vs_ml_core_distributions.pdf")
    plt.close(fig)


def plot_scatter(led, ml, summaries, tail_masks, output, seed=42):
    """Show the full joint distributions in neutral gray, without a hard boundary.

    The Gaussian-core envelope is used only for numerical tail diagnostics;
    no points are colored or separated by that classification.
    """
    rng = np.random.default_rng(seed)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.4))
    for ax, (label, pairs) in zip(axes, (
        ("LED", led), ("After ML", ml),
    )):
        indices = np.arange(len(pairs))
        if len(indices) > 20000:
            indices = rng.choice(indices, 20000, replace=False)
        ax.scatter(pairs[indices, 0], pairs[indices, 1],
                   s=6, alpha=0.24, color="0.35", rasterized=True)
        ax.set_xlabel("Energy residual (ps)")
        ax.set_ylabel("Timing residual (ps)")
        ax.set_title(label)
        ax.grid(alpha=.2)
    fig.suptitle("Paired energy and timing residuals")
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


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--energy", type=Path, required=True)
    parser.add_argument("--timing", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("core_tail_results"))
    parser.add_argument("--fit-fraction", type=float, default=.95,
                        help="Core fit elliptical fraction; default .95")
    parser.add_argument("--tail-fraction", type=float, default=.999,
                        help="Core Gaussian radial probability defining tail; default .999")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--reference-events", type=int, default=300000,
                        help="Gaussian selection-reference draws")
    args = parser.parse_args(argv)
    if not 0 < args.fit_fraction < args.tail_fraction < 1:
        parser.error("Require 0 < --fit-fraction < --tail-fraction < 1")
    if args.reference_events < 1000:
        parser.error("--reference-events must be >=1000")
    analyze(args.energy, args.timing, args.output,
            fit_fraction=args.fit_fraction, tail_fraction=args.tail_fraction,
            seed=args.seed, n_reference=args.reference_events)


if __name__ == "__main__":
    main()
