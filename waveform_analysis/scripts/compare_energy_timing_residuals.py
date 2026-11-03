#!/usr/bin/env python3
"""Standalone analysis of common timing errors in paired energy/timing channels.

Requires only numpy, scipy, matplotlib and scikit-learn. No project imports,
cache, models, raw waveform files or retraining are required.

Each prediction NPZ must contain:
    event_id         unique integer event IDs
    led_residual_ps  LED-corrected coincidence residual before ML, in ps
    corrected_ps     coincidence residual after ML, in ps

The two NPZ files must refer to the same physical events and correction
convention. Events are matched by event_id, never by their array positions.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import numpy as np
from scipy.stats import norm
from sklearn.mixture import GaussianMixture


def load_predictions(path):
    """Read both LED and ML residuals, applying strict source validation."""
    path = Path(path)
    with np.load(path, allow_pickle=False) as data:
        required = ("event_id", "led_residual_ps", "corrected_ps")
        missing = [name for name in required if name not in data]
        if missing:
            raise ValueError(f"{path}: missing fields {missing}; use an original predictions NPZ")
        ids = np.asarray(data["event_id"])
        led = np.asarray(data["led_residual_ps"], dtype=np.float64)
        ml = np.asarray(data["corrected_ps"], dtype=np.float64)
    if any(array.ndim != 1 for array in (ids, led, ml)):
        raise ValueError(f"{path}: expected one-dimensional arrays")
    if len(ids) < 10 or len(ids) != len(led) or len(ids) != len(ml):
        raise ValueError(f"{path}: inconsistent lengths or fewer than ten events")
    if ids.dtype.kind not in "iu" or len(np.unique(ids)) != len(ids):
        raise ValueError(f"{path}: event_id must be unique integers")
    if not (np.isfinite(led).all() and np.isfinite(ml).all()):
        raise ValueError(f"{path}: LED and ML residuals must be finite")
    return ids.astype(np.int64), led, ml


def align_prediction_pairs(energy_data, timing_data):
    """Inner-join event IDs once, applying the same selection to LED and ML."""
    eids, eled, eml = energy_data
    tids, tled, tml = timing_data
    ids, ei, ti = np.intersect1d(eids, tids, assume_unique=True, return_indices=True)
    if len(ids) < 10:
        raise ValueError(f"Only {len(ids)} matching events (need >= 10)")
    return ids, np.column_stack((eled[ei], tled[ti])), np.column_stack((eml[ei], tml[ti]))


def fit_mixture(pairs, seed=42, n_init=5):
    """Fit one/two-component 2D Gaussians; return physically scaled parameters.

    Components ordered by increasing determinant of covariance (ellipse area).
    An unconstrained Gaussian mixture does not establish distinct physical
    populations. Its one-component BIC provides a reference for model selection.
    """
    x = np.asarray(pairs, dtype=np.float64)
    if x.ndim != 2 or x.shape[1] != 2 or len(x) < 10:
        raise ValueError("Expected at least 10 paired 2D samples")
    center = x.mean(axis=0)
    scale = x.std(axis=0)
    if np.any(scale <= 0):
        raise ValueError("Both channels must have nonzero dispersion")
    normalized = (x - center) / scale
    kwargs = dict(covariance_type="full", random_state=seed, reg_covar=1e-6,
                  max_iter=400)
    mixture = GaussianMixture(n_components=2, n_init=n_init, **kwargs).fit(normalized)
    baseline = GaussianMixture(n_components=1, n_init=1, **kwargs).fit(normalized)
    means = mixture.means_ * scale + center
    covariance = mixture.covariances_ * scale[None, :, None] * scale[None, None, :]
    order = np.argsort(np.linalg.det(covariance))
    components = []
    for name, k in zip(("core", "tail"), order):
        cov = covariance[k]
        sigma = np.sqrt(np.diag(cov))
        shared_var = float(cov[0, 1])
        specific_vars = np.diag(cov) - shared_var
        admissible = bool(shared_var >= 0 and np.all(specific_vars >= 0))
        components.append({
            "name": name,
            "weight": float(mixture.weights_[k]),
            "mean_ps": means[k].tolist(),
            "sigma_energy_ps": float(sigma[0]),
            "sigma_timing_ps": float(sigma[1]),
            "correlation": float(shared_var / np.prod(sigma)),
            "covariance_matrix_ps2": cov.tolist(),
            "shared_variance_ps2": shared_var,
            "sigma_common_ps": float(np.sqrt(shared_var)) if shared_var >= 0 else None,
            "energy_specific_variance_ps2": float(specific_vars[0]),
            "timing_specific_variance_ps2": float(specific_vars[1]),
            "additive_common_noise_model_admissible": admissible,
        })
    return {
        "n_events": int(len(x)),
        "components": components,
        "bic_one_gaussian": float(baseline.bic(normalized)),
        "bic_two_gaussians": float(mixture.bic(normalized)),
        "delta_bic_one_minus_two": float(baseline.bic(normalized) - mixture.bic(normalized)),
        "converged": bool(mixture.converged_),
        "n_iterations": int(mixture.n_iter_),
        "delta_rho_tail_minus_core": float(components[1]["correlation"] -
                                            components[0]["correlation"]),
    }


def marginal_pdf(values, components, channel_index):
    """Marginal of the *bivariate* fit; no independent univariate refit."""
    individual = [
        part["weight"] * norm.pdf(
            values, loc=part["mean_ps"][channel_index],
            scale=(part["sigma_energy_ps"] if channel_index == 0
                   else part["sigma_timing_ps"]))
        for part in components
    ]
    return individual, sum(individual)


def save_distribution_plot(led, ml, results, output):
    """LED/ML by energy/timing: empirical histogram plus fitted 2-Gaussian PDFs.

    Log density exposes both the core and broad tails without clipping data.
    All histogram bins in a column are shared between LED and ML.
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex="col")
    for j, channel in enumerate(("Energy", "Timing")):
        all_values = np.concatenate((led[:, j], ml[:, j]))
        extent = max(float(np.max(np.abs(all_values))), 150.0)
        # Use sufficiently narrow bins for the core, while retaining full tails.
        bin_count = int(np.clip(2 * extent / 10, 150, 1200))
        edges = np.linspace(-extent, extent, bin_count + 1)
        xx = np.linspace(-extent, extent, 2500)
        for i, (label, pairs, params) in enumerate((
            ("LED", led, results["led"]),
            ("After ML", ml, results["ml"]),
        )):
            ax = axes[i, j]
            ax.hist(pairs[:, j], bins=edges, density=True,
                    histtype="step", linewidth=1.1, color="black",
                    label="Measured distribution")
            curves, total = marginal_pdf(xx, params["components"], j)
            ax.plot(xx, total, color="tab:red", linewidth=1.7, label="2-Gaussian mixture")
            for curve, part, color in zip(
                curves, params["components"], ("tab:blue", "tab:orange")
            ):
                ax.plot(xx, curve, "--", color=color, linewidth=1.2,
                        label=f'{part["name"]} ({part["weight"]*100:.2f}%)')
            ax.set_yscale("log")
            ax.set_ylim(bottom=1e-8)
            ax.set_title(f"{label} — {channel}")
            ax.set_xlabel("Coincidence residual (ps)")
            ax.set_ylabel("Probability density (log)")
            ax.legend(fontsize=8)
            ax.grid(alpha=.2)
    fig.suptitle("Matched-event timing residuals and bivariate Gaussian-mixture marginals")
    fig.tight_layout()
    fig.savefig(output / "gaussian_mixture_distributions.png", dpi=200)
    fig.savefig(output / "gaussian_mixture_distributions.pdf")
    plt.close(fig)


def save_joint_plot(led, ml, results, output, seed=42):
    """2D residual plots with fitted covariance ellipses (2 sigma)."""
    rng = np.random.default_rng(seed)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5))
    for ax, (label, pairs, params) in zip(
        axes, (("LED", led, results["led"]), ("After ML", ml, results["ml"]))
    ):
        sample = rng.choice(len(pairs), min(15000, len(pairs)), replace=False)
        ax.scatter(pairs[sample, 0], pairs[sample, 1],
                   s=3, alpha=.14, color="0.3", rasterized=True)
        for part, color in zip(params["components"], ("tab:blue", "tab:orange")):
            cov = np.asarray(part["covariance_matrix_ps2"])
            eigenvalues, eigenvectors = np.linalg.eigh(cov)
            angle = np.degrees(np.arctan2(eigenvectors[1, -1],
                                          eigenvectors[0, -1]))
            ax.add_patch(Ellipse(
                xy=part["mean_ps"], width=4*np.sqrt(eigenvalues[-1]),
                height=4*np.sqrt(eigenvalues[0]), angle=angle,
                edgecolor=color, facecolor="none", linewidth=2,
                label=f'{part["name"]} (rho={part["correlation"]:.3f})'))
        ax.set_xlabel("Energy residual (ps)")
        ax.set_ylabel("Timing residual (ps)")
        ax.set_title(label)
        ax.grid(alpha=.2)
        ax.legend()
    fig.suptitle("Joint energy/timing residuals: core and tail correlation")
    fig.tight_layout()
    fig.savefig(output / "gaussian_mixture_joint.png", dpi=200)
    fig.savefig(output / "gaussian_mixture_joint.pdf")
    plt.close(fig)


def analyze(energy_file, timing_file, output, *, seed=42, n_init=5):
    energy = load_predictions(energy_file)
    timing = load_predictions(timing_file)
    ids, led, ml = align_prediction_pairs(energy, timing)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    results = {
        "n_matched_events": int(len(ids)),
        "energy_prediction_file": str(energy_file),
        "timing_prediction_file": str(timing_file),
        "led": fit_mixture(led, seed=seed, n_init=n_init),
        "ml": fit_mixture(ml, seed=seed, n_init=n_init),
        "interpretation": (
            "Each component's common variance equals its energy/timing covariance "
            "only if both channels share one additive error with unit coefficients "
            "and all channel-specific errors are mutually uncorrelated. "
            "Components are statistical, not verified physical hit populations. "
            "Two-component BIC compares descriptive Gaussian models but does not "
            "prove that exactly two physical mechanisms exist. "
            "LED and ML fits use the same matched events, independently refitted.")
    }
    (output / "mixture_summary.json").write_text(
        json.dumps(results, indent=2) + "\n", encoding="utf-8")
    save_distribution_plot(led, ml, results, output)
    save_joint_plot(led, ml, results, output, seed=seed)
    print(f"Matched events: {len(ids)}")
    for state in ("led", "ml"):
        result = results[state]
        print(f"\n{state.upper()} | Delta BIC (one - two) = {result['delta_bic_one_minus_two']:.1f}"
              f" | converged={result['converged']}")
        for part in result["components"]:
            common = (f'{part["sigma_common_ps"]:.2f}' if part["sigma_common_ps"]
                      is not None else "N/A")
            print(f'  {part["name"]:4s} | weight={part["weight"]*100:.3f}%'
                  f' | rho={part["correlation"]:.4f}'
                  f' | sigma_E={part["sigma_energy_ps"]:.2f} ps'
                  f' | sigma_T={part["sigma_timing_ps"]:.2f} ps'
                  f' | Var_common={part["shared_variance_ps2"]:.2f} ps^2'
                  f' | sigma_common={common} ps'
                  f' | admissible={part["additive_common_noise_model_admissible"]}')
    print(f"\nPlots and JSON: {output.resolve()}")
    return results


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--energy", required=True, type=Path,
                        help="Energy predictions NPZ (event_id, led_residual_ps, corrected_ps)")
    parser.add_argument("--timing", required=True, type=Path,
                        help="Timing predictions NPZ (same fields)")
    parser.add_argument("--output", default=Path("mixture_results"), type=Path)
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--gmm-n-init", default=5, type=int)
    args = parser.parse_args(argv)
    if args.gmm_n_init < 1:
        parser.error("--gmm-n-init must be >= 1")
    analyze(args.energy, args.timing, args.output,
            seed=args.seed, n_init=args.gmm_n_init)


if __name__ == "__main__":
    main()
