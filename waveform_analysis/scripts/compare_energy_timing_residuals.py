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
import numpy as np
from scipy.stats import norm, kstest
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


def fit_mixture(pairs, seed=42, n_init=5, *, return_tail_posterior=False):
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
            "cov_energy_timing_ps2": shared_var,
            "sigma_common_if_unit_loading_ps": float(np.sqrt(shared_var)) if admissible else None,
            "energy_specific_variance_ps2": float(specific_vars[0]),
            "timing_specific_variance_ps2": float(specific_vars[1]),
            "unit_loading_noise_decomposition_valid": admissible,
        })
    summary = {
        "n_events": int(len(x)),
        "components": components,
        "bic_one_gaussian": float(baseline.bic(normalized)),
        "bic_two_gaussians": float(mixture.bic(normalized)),
        "delta_bic_one_minus_two": float(baseline.bic(normalized) - mixture.bic(normalized)),
        "em_numerically_converged": bool(mixture.converged_),
        "em_n_iterations": int(mixture.n_iter_),
        "delta_rho_tail_minus_core": float(components[1]["correlation"] -
                                            components[0]["correlation"]),
    }
    if return_tail_posterior:
        return summary, mixture.predict_proba(normalized)[:, order[1]]
    return summary


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


def projected_parameters(components, vector):
    """Weights, means and standard deviations for a linear 2D projection."""
    v = np.asarray(vector, float)
    return [(p["weight"], float(v @ np.asarray(p["mean_ps"])),
             float(np.sqrt(v @ np.asarray(p["covariance_matrix_ps2"]) @ v)))
            for p in components]


def projected_cdf(x, components, vector):
    return sum(weight * norm.cdf(x, mu, sigma)
               for weight, mu, sigma in projected_parameters(components, vector))


def expected_bin_counts(edges, components, vector, n):
    """Integrate fitted probability across bins exactly using the Gaussian CDF."""
    return n * np.maximum(np.diff(projected_cdf(edges, components, vector)), 0.0)


def signed_poisson_deviance(observed, expected):
    """Signed likelihood-ratio residual for Poisson counts."""
    observed = np.asarray(observed, float)
    expected = np.asarray(expected, float)
    if np.any(expected <= 0):
        raise ValueError("Expected bin counts must be positive")
    with np.errstate(divide="ignore", invalid="ignore"):
        information = np.where(observed > 0,
                               observed * np.log(observed / expected), 0.0)
    return np.sign(observed - expected) * np.sqrt(
        np.maximum(2 * (information - observed + expected), 0.0))


def goodness_of_fit(values, components, vector, bin_width=10.0):
    """Descriptive KS, merged-bin deviance and observed/predicted tail counts.

    The KS statistic is not accompanied by a standard p-value: the fitted
    distribution was estimated using these same events.
    """
    values = np.asarray(values, float)
    n = len(values)
    extent = max(150., np.ceil(np.max(np.abs(values)) / bin_width) * bin_width)
    edges = np.arange(-extent, extent + 1.5 * bin_width, bin_width)
    observed = np.histogram(values, bins=edges)[0]
    expected = expected_bin_counts(edges, components, vector, n)
    # Merge adjoining bins to keep expected counts >= 5 where possible.
    groups = []
    start = 0
    for end in range(1, len(expected) + 1):
        if expected[start:end].sum() >= 5:
            groups.append((start, end))
            start = end
    if start < len(expected):
        if groups:
            groups[-1] = (groups[-1][0], len(expected))
        else:
            groups.append((0, len(expected)))
    merged_obs = np.array([observed[a:b].sum() for a, b in groups])
    merged_exp = np.array([expected[a:b].sum() for a, b in groups])
    residuals = signed_poisson_deviance(merged_obs, merged_exp)
    ks = float(kstest(values, lambda z: projected_cdf(z, components, vector)).statistic)
    exceedance = []
    for threshold in (100., 200., 300., 500., 1000.):
        probability = float(projected_cdf(-threshold, components, vector)
                            + 1.0 - projected_cdf(threshold, components, vector))
        exceedance.append({
            "threshold_ps": threshold,
            "observed_count": int(np.count_nonzero(np.abs(values) > threshold)),
            "expected_count": n * probability,
        })
    return {
        "ks_distance_descriptive": ks,
        "poisson_deviance_merged_bins": float(np.sum(residuals ** 2)),
        "n_merged_bins": len(groups),
        "exceedance": exceedance,
        "_edges": edges,
        "_observed": observed,
        "_expected": expected,
        "_merged_edges": np.array([edges[a] for a, _ in groups] +
                                   [edges[groups[-1][1]]]),
        "_deviance_residuals": residuals,
    }


def save_fit_quality_plots(led, ml, results, output):
    projections = ((("Energy", (1., 0.)),
                    ("Timing", (0., 1.)),
                    ("Energy - timing", (1., -1.))))
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    for row, (state, samples) in enumerate((("led", led), ("ml", ml))):
        for col, (name, projection) in enumerate(projections):
            d = goodness_of_fit(samples @ np.asarray(projection),
                                results[state]["components"], projection)
            bounds, values = d["_merged_edges"], d["_deviance_residuals"]
            axes[row, col].bar((bounds[:-1] + bounds[1:])/2, values,
                               width=np.diff(bounds), color="tab:blue", alpha=.7)
            axes[row, col].axhline(0, color="black", linewidth=.8)
            axes[row, col].axhline(2, color="gray", ls="--", linewidth=.8)
            axes[row, col].axhline(-2, color="gray", ls="--", linewidth=.8)
            axes[row, col].set_title(
                f'{state.upper()} {name}: KS={d["ks_distance_descriptive"]:.3f}')
            axes[row, col].set_xlabel("Residual (ps)")
            axes[row, col].set_ylabel("Signed Poisson deviance")
            axes[row, col].grid(alpha=.2)
    fig.tight_layout()
    fig.savefig(output / "gmm_fit_residuals.png", dpi=180)
    fig.savefig(output / "gmm_fit_residuals.pdf")
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8))
    for ax, (name, projection) in zip(axes, projections):
        for state, samples, color in (("led", led, "tab:orange"),
                                       ("ml", ml, "tab:blue")):
            x = np.abs(samples @ np.asarray(projection))
            thresholds = np.linspace(0, max(1000., float(np.max(x))), 300)
            observed = (len(x) - np.searchsorted(
                np.sort(x), thresholds, side="right")) / len(x)
            fit = results[state]["components"]
            predicted = (projected_cdf(-thresholds, fit, projection) +
                         1 - projected_cdf(thresholds, fit, projection))
            ax.plot(thresholds, np.maximum(observed, 1 / len(x)),
                    color=color, label=f"{state.upper()} observed")
            ax.plot(thresholds, np.maximum(predicted, 1e-10),
                    color=color, linestyle="--", label=f"{state.upper()} fit")
        ax.set_yscale("log")
        ax.set_ylim(1e-5, 1.1)
        ax.set_title(name)
        ax.set_xlabel("Absolute residual threshold (ps)")
        ax.set_ylabel("P(|r| > threshold)")
        ax.legend(fontsize=8)
        ax.grid(alpha=.2)
    fig.tight_layout()
    fig.savefig(output / "gmm_exceedance.png", dpi=180)
    fig.savefig(output / "gmm_exceedance.pdf")
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    for ax, (state, samples) in zip(axes, (("led", led), ("ml", ml))):
        difference = samples[:, 0] - samples[:, 1]
        diagnostic = goodness_of_fit(difference, results[state]["components"],
                                     (1., -1.))
        ax.stairs(diagnostic["_observed"], diagnostic["_edges"],
                  color="black", label="Observed")
        ax.stairs(diagnostic["_expected"], diagnostic["_edges"],
                  color="tab:red", label="Predicted by joint GMM")
        ax.set_yscale("log")
        ax.set_ylim(bottom=.8)
        ax.set_title(state.upper())
        ax.set_xlabel("Energy - timing residual (ps)")
        ax.set_ylabel("Events per 10 ps")
        ax.legend()
        ax.grid(alpha=.2)
    fig.tight_layout()
    fig.savefig(output / "gmm_energy_minus_timing.png", dpi=180)
    fig.savefig(output / "gmm_energy_minus_timing.pdf")
    plt.close(fig)


def save_distribution_plot(led, ml, results, output):
    """Measured bin counts and fitted expected counts for the same bin widths.

    The plotted marginal curves come from the fitted *bivariate* mixture.
    Zero-count histogram bins are omitted on the logarithmic count axis.
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex="col")
    for channel_index, channel in enumerate(("Energy", "Timing")):
        values = np.concatenate((led[:, channel_index], ml[:, channel_index]))
        bin_width_ps = 10.0
        extent = max(
            bin_width_ps * np.ceil(np.max(np.abs(values)) / bin_width_ps),
            150.0,
        )
        edges = np.arange(-extent, extent + 1.5 * bin_width_ps, bin_width_ps)
        centers = (edges[:-1] + edges[1:]) / 2
        grid = np.linspace(edges[0], edges[-1], 3500)

        for row, (label, pairs, fit) in enumerate((
            ("LED", led, results["led"]),
            ("After ML", ml, results["ml"]),
        )):
            ax = axes[row, channel_index]
            counts, _ = np.histogram(pairs[:, channel_index], bins=edges)
            valid = counts > 0
            ax.stairs(counts, edges, color="black", linewidth=1.0,
                      label="Measured counts")

            pdf_components, fitted_pdf = marginal_pdf(
                grid, fit["components"], channel_index)
            expected_factor = len(pairs) * bin_width_ps
            total_counts = expected_factor * fitted_pdf
            ax.plot(grid, total_counts, color="tab:red", linewidth=1.9,
                    label="2-Gaussian mixture")

            for component_pdf, part, color in zip(
                pdf_components, fit["components"],
                ("tab:blue", "tab:orange"),
            ):
                ax.plot(grid, expected_factor * component_pdf,
                        "--", color=color, linewidth=1.5,
                        label=f'{part["name"]} ({100*part["weight"]:.2f}%)')
            ax.set_yscale("log")
            ax.set_ylim(0.8, max(float(counts.max()),
                                 float(total_counts.max()), 1.0) * 1.5)
            ax.set_title(f"{label} — {channel}")
            ax.set_xlabel("Coincidence residual (ps)")
            ax.set_ylabel(f"Events / {bin_width_ps:g} ps (log scale)")
            ax.legend(fontsize=8)
            ax.grid(alpha=0.2)

    fig.suptitle("Matched-event residuals and 2-Gaussian-mixture fits")
    fig.tight_layout()
    fig.savefig(output / "gaussian_mixture_distributions.png", dpi=200)
    fig.savefig(output / "gaussian_mixture_distributions.pdf")
    plt.close(fig)


def save_joint_plot(led, ml, results, tail_posteriors, output, seed=42):
    """Plot core in translucent blue and all high-posterior tail points in orange.

    Posterior threshold defines a visual classification only; correlations and
    common variances in JSON are fitted component parameters, not hard-group
    sample statistics.
    """
    rng = np.random.default_rng(seed)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5))
    for ax, (label, pairs, fit, posterior) in zip(axes, (
        ("LED", led, results["led"], tail_posteriors["led"]),
        ("After ML", ml, results["ml"], tail_posteriors["ml"]),
    )):
        is_tail = np.asarray(posterior) >= 0.5
        core_indices = np.flatnonzero(~is_tail)
        tail_indices = np.flatnonzero(is_tail)
        if len(core_indices) > 15000:
            core_indices = rng.choice(core_indices, 15000, replace=False)
        ax.scatter(pairs[core_indices, 0], pairs[core_indices, 1],
                   s=4, alpha=0.08, color="tab:blue",
                   label="Core", rasterized=True)
        ax.scatter(pairs[tail_indices, 0], pairs[tail_indices, 1],
                   s=12, alpha=0.65, color="tab:orange",
                   label="Tail", rasterized=True)
        ax.set_xlabel("Energy residual (ps)")
        ax.set_ylabel("Timing residual (ps)")
        ax.set_title(label)
        ax.grid(alpha=0.2)
        ax.legend()

    fig.suptitle("Energy vs timing residuals — GMM component assignment")
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
    led_fit, led_tail_probability = fit_mixture(
        led, seed=seed, n_init=n_init, return_tail_posterior=True)
    ml_fit, ml_tail_probability = fit_mixture(
        ml, seed=seed, n_init=n_init, return_tail_posterior=True)
    results = {
        "n_matched_events": int(len(ids)),
        "energy_prediction_file": str(energy_file),
        "timing_prediction_file": str(timing_file),
        "led": led_fit,
        "ml": ml_fit,
        "interpretation": (
            "Each component's common variance equals its energy/timing covariance "
            "only if both channels share one additive error with unit coefficients "
            "and all channel-specific errors are mutually uncorrelated. "
            "Components are statistical, not verified physical hit populations. "
            "Two-component BIC compares descriptive Gaussian models but does not "
            "prove that exactly two physical mechanisms exist. "
            "LED and ML fits use the same matched events, independently refitted.")
    }
    fit_checks = {}
    for state, pairs in (("led", led), ("ml", ml)):
        fit_checks[state] = {}
        for label, vec in (("energy", (1., 0.)),
                           ("timing", (0., 1.)),
                           ("energy_minus_timing", (1., -1.))):
            d = goodness_of_fit(pairs @ np.asarray(vec),
                                results[state]["components"], vec)
            fit_checks[state][label] = {
                key: value for key, value in d.items() if not key.startswith("_")
            }
    results["fit_quality"] = fit_checks
    (output / "mixture_summary.json").write_text(
        json.dumps(results, indent=2) + "\n", encoding="utf-8")
    save_distribution_plot(led, ml, results, output)
    save_fit_quality_plots(led, ml, results, output)
    save_joint_plot(
        led, ml, results,
        {"led": led_tail_probability, "ml": ml_tail_probability},
        output, seed=seed)
    print(f"Matched events: {len(ids)}")
    for state in ("led", "ml"):
        result = results[state]
        print(f"\n{state.upper()} | Delta BIC (one - two) = {result['delta_bic_one_minus_two']:.1f}"
              f" | EM numerically converged={result['em_numerically_converged']}")
        for part in result["components"]:
            common = (f'{part["sigma_common_if_unit_loading_ps"]:.2f}' if part["sigma_common_if_unit_loading_ps"]
                      is not None else "N/A")
            print(f'  {part["name"]:4s} | weight={part["weight"]*100:.3f}%'
                  f' | rho={part["correlation"]:.4f}'
                  f' | sigma_E={part["sigma_energy_ps"]:.2f} ps'
                  f' | sigma_T={part["sigma_timing_ps"]:.2f} ps'
                  f' | Cov(E,T)={part["cov_energy_timing_ps2"]:.2f} ps^2'
                  f' | sigma_common_if_unit_loading={common} ps'
                  f' | unit-loading decomposition valid={part["unit_loading_noise_decomposition_valid"]}')
    for state in ("led", "ml"):
        print(f"  {state.upper()} descriptive fit checks:")
        for label, check in results["fit_quality"][state].items():
            print(f'    {label}: KS distance={check["ks_distance_descriptive"]:.4f}'
                  f' | merged-bin Poisson deviance={check["poisson_deviance_merged_bins"]:.1f}'
                  f' ({check["n_merged_bins"]} bins)')
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
