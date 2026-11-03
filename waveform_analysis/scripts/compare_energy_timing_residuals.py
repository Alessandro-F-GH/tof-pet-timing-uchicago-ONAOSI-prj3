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
from scipy.stats import chi2, norm, pearsonr


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
    """Only a central Gaussian is fitted; no Gaussian curve fitted to tails."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex="col")
    for j, channel in enumerate(("Energy", "Timing")):
        extent = max(200., np.ceil(max(np.max(np.abs(led[:, j])),
                                       np.max(np.abs(ml[:, j]))) / bin_width)*bin_width)
        edges = np.arange(-extent, extent+1.5*bin_width, bin_width)
        centers = (edges[:-1] + edges[1:])/2
        for i, (label, pairs, summary) in enumerate((
            ("LED", led, summaries["led"]),
            ("After ML", ml, summaries["ml"]),
        )):
            ax = axes[i, j]
            counts = np.histogram(pairs[:, j], bins=edges)[0]
            expected = core_expected_bin_counts(edges, summary["core_gaussian_fit"],
                                                j, len(pairs))
            ax.stairs(counts, edges, color="black", label="Observed counts")
            ax.plot(centers, expected, color="tab:blue", lw=1.6,
                    label="Gaussian core extrapolation")
            ax.set_yscale("log")
            ax.set_ylim(bottom=.8)
            ax.set_title(f"{label} — {channel}")
            ax.set_xlabel("Residual (ps)")
            ax.set_ylabel(f"Events per {bin_width:g} ps (log)")
            ax.legend(fontsize=8)
            ax.grid(alpha=.2)
    fig.suptitle("Gaussian core fits; empirical tails not parametrically modeled")
    fig.tight_layout()
    fig.savefig(output / "gaussian_core_distributions.png", dpi=190)
    fig.savefig(output / "gaussian_core_distributions.pdf")
    plt.close(fig)


def plot_scatter(led, ml, summaries, tail_masks, output, seed=42):
    rng = np.random.default_rng(seed)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.4))
    for ax, (label, pairs, tail) in zip(axes, (
        ("LED", led, tail_masks["led"]),
        ("After ML", ml, tail_masks["ml"]),
    )):
        core_idx = np.flatnonzero(~tail)
        tail_idx = np.flatnonzero(tail)
        if len(core_idx) > 15000:
            core_idx = rng.choice(core_idx, 15000, replace=False)
        ax.scatter(pairs[core_idx, 0], pairs[core_idx, 1],
                   s=4, alpha=.08, color="tab:blue", label="Inside core envelope",
                   rasterized=True)
        ax.scatter(pairs[tail_idx, 0], pairs[tail_idx, 1],
                   s=12, alpha=.7, color="tab:orange", label="Outside core envelope",
                   rasterized=True)
        ax.set_xlabel("Energy residual (ps)")
        ax.set_ylabel("Timing residual (ps)")
        ax.set_title(label)
        ax.grid(alpha=.2)
        ax.legend()
    fig.tight_layout()
    fig.savefig(output / "core_and_empirical_tails.png", dpi=190)
    fig.savefig(output / "core_and_empirical_tails.pdf")
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
    fig.savefig(output / "radial_tail_exceedance.png", dpi=190)
    fig.savefig(output / "radial_tail_exceedance.pdf")
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
    print(f"Matched events: {len(led)}")
    for state in ("led", "ml"):
        result = results[state]
        core = result["core_gaussian_fit"]
        tail = result["empirical_outside_core"]
        null = result["gaussian_selection_only_reference"]
        definition = result["tail_definition"]
        print(f"\n{state.upper()} | core fit numerically converged="
              f'{core["numerically_converged"]} ({core["iterations"]} iterations)')
        print(f'  CORE Gaussian: sigmaE={core["sigma_energy_ps"]:.2f} ps'
              f' sigmaT={core["sigma_timing_ps"]:.2f} ps'
              f' rho={core["correlation"]:.4f}'
              f' Cov(E,T)={core["cov_energy_timing_ps2"]:.2f} ps^2')
        print(f'  OUTSIDE core: {definition["tail_count"]} events '
              f'({100*definition["tail_fraction"]:.4f}%); '
              f'Gaussian reference count={definition["expected_count_from_core_gaussian"]:.1f}; '
              f'excess factor={definition["excess_factor_over_core_gaussian"]:.1f}')
        print(f'  EMPIRICAL outside: rho={tail["correlation"]:.4f} '
              f'Cov(E,T)={tail["cov_energy_timing_ps2"]:.2f} ps^2'
              f' | selection-only Gaussian rho={null["correlation"]:.4f}')
    print(f"\nResults: {output.resolve()}")
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
