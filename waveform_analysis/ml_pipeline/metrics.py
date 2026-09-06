from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import sys
from typing import Any

import numpy as np
from scipy.optimize import curve_fit

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from utils_fit import FitResult, fit_delta_times_integer_fs


FWHM_PER_SIGMA = 2.0 * np.sqrt(2.0 * np.log(2.0))


@dataclass
class _FixedGaussianFitResult:
    """Duck-compatible Gaussian fit result used only when utils_fit fails.

    The study/reporting layer consumes these public fields from ``FitResult``.
    Keeping the fallback result self-contained avoids depending on the exact
    constructor signature of whichever utils_fit version is installed.
    """

    method: str
    parameter: float
    success: bool
    n_total: int
    n_selected: int
    n_valid: int
    n_fit: int
    crossing_efficiency: float
    mean_ps: float
    mean_error_ps: float
    sigma_ps: float
    sigma_error_ps: float
    ctr_ps: float
    ctr_error_ps: float
    chi2: float
    ndof: int
    fit_low_ps: float
    fit_high_ps: float
    iterations: int
    message: str = ""
    edges_ps: np.ndarray = field(default_factory=lambda: np.empty(0), repr=False)
    counts: np.ndarray = field(default_factory=lambda: np.empty(0), repr=False)
    expected: np.ndarray = field(default_factory=lambda: np.empty(0), repr=False)
    bin_width_ps: float = float("nan")
    bin_phase_ps: float = float("nan")
    phase_ctr_std_ps: float = float("nan")

    @property
    def chi2_ndof(self) -> float:
        return self.chi2 / self.ndof if self.ndof > 0 else float("nan")


def _gaussian_counts(
    x_ps: np.ndarray,
    amplitude: float,
    mean_ps: float,
    sigma_ps: float,
) -> np.ndarray:
    sigma = max(float(sigma_ps), np.finfo(float).eps)
    return float(amplitude) * np.exp(-0.5 * ((x_ps - float(mean_ps)) / sigma) ** 2)


def _robust_location_scale(values: np.ndarray) -> tuple[float, float]:
    center = float(np.median(values))
    mad = float(np.median(np.abs(values - center)))
    sigma = 1.4826 * mad
    if not np.isfinite(sigma) or sigma <= 0.0:
        sigma = float(np.std(values, ddof=1)) if values.size > 1 else float("nan")
    return center, sigma


def _fallback_bin_width(
    values: np.ndarray,
    sigma_ps: float,
    fit_config: dict[str, Any],
) -> float:
    """Choose one fixed bin width; no phase scan is performed."""
    adaptive = fit_config.get("adaptive_binning", {}) or {}
    minimum = float(adaptive.get("min_bin_ps", 1.0))
    maximum = float(adaptive.get("max_bin_ps", 25.0))
    if not np.isfinite(minimum) or minimum <= 0.0:
        minimum = 1.0
    if not np.isfinite(maximum) or maximum < minimum:
        maximum = max(25.0, minimum)

    configured = fit_config.get("histogram_bin_ps", np.nan)
    try:
        configured = float(configured)
    except (TypeError, ValueError):
        configured = float("nan")

    q25, q75 = np.quantile(values, [0.25, 0.75])
    iqr = float(q75 - q25)
    fd = (
        2.0 * iqr / np.cbrt(float(values.size))
        if values.size > 1 and np.isfinite(iqr) and iqr > 0.0
        else float("nan")
    )
    target = float(sigma_ps) / 8.0 if np.isfinite(sigma_ps) and sigma_ps > 0.0 else minimum

    # The fallback is intentionally coarser than a pathological adaptive phase.
    # Prefer an explicitly configured width, but never use a width much finer
    # than the data support according to FD / the robust core scale.
    candidates = [minimum, target]
    if np.isfinite(configured) and configured > 0.0:
        candidates.append(configured)
    if np.isfinite(fd) and fd > 0.0:
        candidates.append(fd)

    # If the distribution is visibly quantized, never choose sub-lattice bins.
    # Sub-lattice fixed bins create alternating empty bins and can drive a
    # Gaussian least-squares fit toward an artificially tiny sigma.
    unique = np.unique(values)
    if unique.size >= 2 and unique.size <= max(512, int(0.20 * values.size)):
        differences = np.diff(unique)
        differences = differences[np.isfinite(differences) & (differences > 1.0e-9)]
        if differences.size:
            lattice = float(np.median(differences))
            if np.isfinite(lattice) and lattice > 0.0:
                candidates.append(lattice)

    return float(np.clip(max(candidates), minimum, maximum))


def _fixed_bin_gaussian_fit(
    values_ps: np.ndarray,
    *,
    method: str,
    fit_config: dict[str, Any],
    primary_failure: str,
) -> _FixedGaussianFitResult:
    """Independent single-phase iterative Gaussian histogram fit.

    This fallback deliberately does not call ``fit_delta_times_integer_fs`` and
    does not consult ``adaptive_binning.enabled``.  It therefore provides a real
    escape path when the adaptive phase scan fails for every phase.
    """
    values = np.asarray(values_ps, dtype=np.float64).reshape(-1)
    if values.size == 0 or np.any(~np.isfinite(values)):
        raise RuntimeError("fixed-bin Gaussian fit requires finite values")

    min_events = int(fit_config.get("min_events", 20))
    if values.size < min_events:
        raise RuntimeError(
            f"only {values.size} events are available; need at least {min_events}"
        )

    global_center, global_sigma = _robust_location_scale(values)
    if not np.isfinite(global_sigma) or global_sigma <= 0.0:
        raise RuntimeError("cannot determine a non-zero robust width")

    bin_width = _fallback_bin_width(values, global_sigma, fit_config)

    # Locate the dominant central peak with one fixed-phase coarse histogram.
    q_low = float(np.quantile(values, 0.002))
    q_high = float(np.quantile(values, 0.998))
    search_low = max(q_low, global_center - 10.0 * global_sigma)
    search_high = min(q_high, global_center + 10.0 * global_sigma)
    if not np.isfinite(search_low) or not np.isfinite(search_high) or search_high <= search_low:
        search_low = global_center - 10.0 * global_sigma
        search_high = global_center + 10.0 * global_sigma

    minimum_span = 12.0 * bin_width
    if search_high - search_low < minimum_span:
        midpoint = 0.5 * (search_low + search_high)
        search_low = midpoint - 0.5 * minimum_span
        search_high = midpoint + 0.5 * minimum_span

    n_search_bins = max(12, int(np.ceil((search_high - search_low) / bin_width)))
    search_edges = search_low + np.arange(n_search_bins + 1, dtype=np.float64) * bin_width
    if search_edges[-1] < search_high:
        search_edges = np.append(search_edges, search_edges[-1] + bin_width)
    search_counts, search_edges = np.histogram(values, bins=search_edges)
    if search_counts.size == 0 or int(np.max(search_counts)) <= 0:
        raise RuntimeError("fixed-bin peak search histogram is empty")
    search_centers = 0.5 * (search_edges[:-1] + search_edges[1:])
    peak_center = float(search_centers[int(np.argmax(search_counts))])

    # Estimate the seed width from a local population around the modal bin.
    local_half = max(2.5 * global_sigma, 6.0 * bin_width)
    local = values[np.abs(values - peak_center) <= local_half]
    if local.size < max(10, min_events // 2):
        local = values
    _, local_sigma = _robust_location_scale(local)
    if not np.isfinite(local_sigma) or local_sigma <= 0.0:
        local_sigma = global_sigma
    local_sigma = max(local_sigma, 0.75 * bin_width)

    # The plotted histogram covers the core and moderate tails, while the fit
    # itself is iteratively restricted to a few fitted sigmas around the peak.
    hist_half = max(8.0 * local_sigma, 16.0 * bin_width)
    hist_low = peak_center - hist_half
    hist_high = peak_center + hist_half
    n_bins = max(20, int(np.ceil((hist_high - hist_low) / bin_width)))
    edges = hist_low + np.arange(n_bins + 1, dtype=np.float64) * bin_width
    counts, edges = np.histogram(values, bins=edges)
    centers = 0.5 * (edges[:-1] + edges[1:])

    # If quantization/sparsity leaves too few occupied bins, coarsen repeatedly.
    max_width = max(float((fit_config.get("adaptive_binning", {}) or {}).get("max_bin_ps", 25.0)), bin_width)
    while np.count_nonzero(counts) < 8 and bin_width < max_width:
        bin_width = min(max_width, 2.0 * bin_width)
        n_bins = max(12, int(np.ceil((hist_high - hist_low) / bin_width)))
        edges = hist_low + np.arange(n_bins + 1, dtype=np.float64) * bin_width
        counts, edges = np.histogram(values, bins=edges)
        centers = 0.5 * (edges[:-1] + edges[1:])

    if np.count_nonzero(counts) < 4:
        raise RuntimeError("fewer than four populated bins remain after fallback binning")

    iteration_sigma = float(fit_config.get("iteration_sigma", 2.5))
    if not np.isfinite(iteration_sigma) or iteration_sigma <= 1.0:
        iteration_sigma = 2.5
    max_iterations = max(2, int(fit_config.get("max_iterations", 6)))
    tolerance = float(fit_config.get("convergence_tolerance_ps", 0.05))
    if not np.isfinite(tolerance) or tolerance <= 0.0:
        tolerance = 0.05
    minimum_fit_bins = max(4, int(fit_config.get("minimum_fit_bins", 5)))
    min_sigma = max(0.5 * bin_width, np.finfo(float).eps)

    mean = peak_center
    sigma = max(local_sigma, min_sigma)
    amplitude = max(float(np.max(counts)), 1.0)
    half_width = max(iteration_sigma * sigma, 3.0 * bin_width)
    covariance: np.ndarray | None = None
    iterations = 0

    for iteration in range(max_iterations):
        fit_low = mean - half_width
        fit_high = mean + half_width
        mask = (centers >= fit_low) & (centers <= fit_high)
        x = centers[mask]
        y = counts[mask].astype(np.float64)
        # Deterministically enlarge the interval if a narrow/quantized core
        # leaves too few populated bins.  This changes the fit range only; the
        # histogram phase remains fixed.
        for _ in range(5):
            if x.size >= minimum_fit_bins and np.count_nonzero(y) >= 4:
                break
            half_width *= 1.5
            mask = (centers >= mean - half_width) & (centers <= mean + half_width)
            x = centers[mask]
            y = counts[mask].astype(np.float64)
        if x.size < minimum_fit_bins or np.count_nonzero(y) < 4:
            raise RuntimeError("too few populated bins in fixed-bin fit interval")

        uncertainty = np.sqrt(np.maximum(y, 1.0))
        lower_mean = float(x[0] - 0.5 * bin_width)
        upper_mean = float(x[-1] + 0.5 * bin_width)
        max_sigma = max(min_sigma * 2.0, upper_mean - lower_mean)
        p0 = [max(amplitude, 1.0), float(np.clip(mean, lower_mean, upper_mean)), float(np.clip(sigma, min_sigma, max_sigma))]
        parameters, covariance = curve_fit(
            _gaussian_counts,
            x,
            y,
            p0=p0,
            sigma=uncertainty,
            absolute_sigma=True,
            bounds=(
                [0.0, lower_mean, min_sigma],
                [np.inf, upper_mean, max_sigma],
            ),
            maxfev=50_000,
        )
        new_amplitude, new_mean, new_sigma = [float(v) for v in parameters]
        if not np.all(np.isfinite(parameters)) or new_sigma <= 0.0:
            raise RuntimeError("fixed-bin Gaussian optimizer returned invalid parameters")

        iterations = iteration + 1
        converged = abs(new_mean - mean) <= tolerance and abs(new_sigma - sigma) <= tolerance
        amplitude, mean, sigma = new_amplitude, new_mean, new_sigma
        half_width = max(iteration_sigma * sigma, 3.0 * bin_width)
        if converged:
            break

    # Final fit on exactly the reported interval.
    fit_low = mean - half_width
    fit_high = mean + half_width
    final_mask = (centers >= fit_low) & (centers <= fit_high)
    x = centers[final_mask]
    y = counts[final_mask].astype(np.float64)
    for _ in range(5):
        if x.size >= minimum_fit_bins and np.count_nonzero(y) >= 4:
            break
        half_width *= 1.5
        fit_low = mean - half_width
        fit_high = mean + half_width
        final_mask = (centers >= fit_low) & (centers <= fit_high)
        x = centers[final_mask]
        y = counts[final_mask].astype(np.float64)
    if x.size < minimum_fit_bins or np.count_nonzero(y) < 4:
        raise RuntimeError("final fixed-bin fit interval contains too few populated bins")

    uncertainty = np.sqrt(np.maximum(y, 1.0))
    lower_mean = float(x[0] - 0.5 * bin_width)
    upper_mean = float(x[-1] + 0.5 * bin_width)
    max_sigma = max(min_sigma * 2.0, upper_mean - lower_mean)
    parameters, covariance = curve_fit(
        _gaussian_counts,
        x,
        y,
        p0=[amplitude, mean, sigma],
        sigma=uncertainty,
        absolute_sigma=True,
        bounds=(
            [0.0, lower_mean, min_sigma],
            [np.inf, upper_mean, max_sigma],
        ),
        maxfev=50_000,
    )
    amplitude, mean, sigma = [float(v) for v in parameters]
    if not np.all(np.isfinite(parameters)) or sigma <= 0.0:
        raise RuntimeError("final fixed-bin Gaussian fit returned invalid parameters")

    expected = _gaussian_counts(centers, amplitude, mean, sigma)
    expected_fit = _gaussian_counts(x, amplitude, mean, sigma)
    chi2 = float(np.sum((y - expected_fit) ** 2 / np.maximum(expected_fit, 1.0)))
    ndof = max(0, int(x.size) - 3)
    n_fit = int(np.count_nonzero((values >= fit_low) & (values <= fit_high)))

    errors = np.full(3, np.nan, dtype=np.float64)
    if covariance is not None and np.shape(covariance) == (3, 3):
        diagonal = np.diag(covariance)
        errors = np.sqrt(np.where(diagonal >= 0.0, diagonal, np.nan))
    mean_error = float(errors[1])
    sigma_error = float(errors[2])

    return _FixedGaussianFitResult(
        method=method,
        parameter=0.0,
        success=True,
        n_total=int(values.size),
        n_selected=int(values.size),
        n_valid=int(values.size),
        n_fit=n_fit,
        crossing_efficiency=1.0,
        mean_ps=mean,
        mean_error_ps=mean_error,
        sigma_ps=sigma,
        sigma_error_ps=sigma_error,
        ctr_ps=float(FWHM_PER_SIGMA * sigma),
        ctr_error_ps=float(FWHM_PER_SIGMA * sigma_error) if np.isfinite(sigma_error) else float("nan"),
        chi2=chi2,
        ndof=ndof,
        fit_low_ps=float(fit_low),
        fit_high_ps=float(fit_high),
        iterations=iterations,
        message=(
            "fixed-bin Gaussian fallback; primary fitter failed: "
            + str(primary_failure)
        ),
        edges_ps=np.asarray(edges, dtype=np.float64),
        counts=np.asarray(counts, dtype=np.int64),
        expected=np.asarray(expected, dtype=np.float64),
        bin_width_ps=float(bin_width),
        bin_phase_ps=0.0,
        phase_ctr_std_ps=0.0,
    )


def fit_times_ps(values_ps: np.ndarray, method: str, fit_config: dict[str, Any]):
    """Gaussian-fit timing distribution with a real fixed-bin fallback.

    Primary path: the repository's canonical ``utils_fit`` implementation.
    Fallback path: an independent single-phase fixed-bin iterative Gaussian fit.
    Both return Gaussian CTR = 2*sqrt(2*ln(2))*sigma_fit.  There is no
    sample-standard-deviation CTR fallback.
    """
    values = np.asarray(values_ps, dtype=np.float64).reshape(-1)
    if values.size == 0:
        raise RuntimeError(f"{method}: empty timing distribution")
    if np.any(~np.isfinite(values)):
        raise RuntimeError(f"{method}: Gaussian fit input contains non-finite values")

    values_fs = np.rint(values * 1000.0).astype(np.int64)
    primary_failure = "unknown primary fit failure"
    try:
        fit = fit_delta_times_integer_fs(
            values_fs,
            method=method,
            parameter=0.0,
            n_total=int(values_fs.size),
            n_selected=int(values_fs.size),
            config=fit_config,
        )
        if bool(getattr(fit, "success", False)) and np.isfinite(float(getattr(fit, "ctr_ps", np.nan))):
            return fit
        primary_failure = str(getattr(fit, "message", "primary Gaussian fit unsuccessful"))
    except Exception as exc:
        primary_failure = str(exc)

    try:
        return _fixed_bin_gaussian_fit(
            values,
            method=method,
            fit_config=fit_config,
            primary_failure=primary_failure,
        )
    except Exception as fallback_exc:
        raise RuntimeError(
            f"{method}: Gaussian fit failed in both paths: "
            f"primary={primary_failure} | fixed-bin fallback={fallback_exc}"
        ) from fallback_exc


def distribution_metrics(
    values_ps: np.ndarray,
    *,
    true_value_ps: float,
    fit: Any,
) -> dict[str, Any]:
    values = np.asarray(values_ps, dtype=np.float64).reshape(-1)
    ctr_error = getattr(fit, "ctr_error_ps", getattr(fit, "ctr_err_ps", float("nan")))
    mean_error = getattr(fit, "mean_error_ps", float("nan"))
    return {
        "event_count": int(values.size),
        "true_value_ps": float(true_value_ps),
        "ctr_ps": float(fit.ctr_ps) if fit.success else float("nan"),
        "ctr_error_ps": float(ctr_error) if fit.success else float("nan"),
        "gaussian_mean_ps": float(fit.mean_ps) if fit.success else float("nan"),
        "gaussian_mean_error_ps": float(mean_error) if fit.success else float("nan"),
        "gaussian_bias_ps": float(fit.mean_ps - true_value_ps) if fit.success else float("nan"),
        "arithmetic_mean_ps": float(np.mean(values)),
        "arithmetic_bias_ps": float(np.mean(values) - true_value_ps),
        "standard_deviation_ps": float(np.std(values, ddof=0)),
        "chi2_ndof": float(fit.chi2_ndof) if fit.success else float("nan"),
        "fit_success": bool(fit.success),
        "fit_message": str(getattr(fit, "message", "")),
    }


def residual_metrics(values_ps: np.ndarray) -> dict[str, Any]:
    """All-event arithmetic diagnostics only.

    ``ctr_ps`` is retained for backward compatibility with non-study callers,
    but the Gaussian-CTR study uses ``fit_times_ps`` for selection/reporting.
    """
    values = np.asarray(values_ps, dtype=np.float64).reshape(-1)
    values = values[np.isfinite(values)]
    n = int(values.size)
    if n == 0:
        return {
            "n": 0,
            "mean_ps": float("nan"),
            "std_ps": float("nan"),
            "ctr_ps": float("nan"),
            "rmse_ps": float("nan"),
            "bias_ps": float("nan"),
        }
    mean = float(np.mean(values))
    rmse = float(np.sqrt(np.mean(values * values)))
    if n < 2:
        std = float("nan")
        ctr = float("nan")
    else:
        std = float(np.std(values, ddof=1))
        ctr = float(FWHM_PER_SIGMA * std)
    return {
        "n": n,
        "mean_ps": mean,
        "std_ps": std,
        "ctr_ps": ctr,
        "rmse_ps": rmse,
        "bias_ps": mean,
    }


def ctr_bootstrap_uncertainty(
    values_ps: np.ndarray,
    n_bootstrap: int = 1000,
    seed: int = 12345,
) -> float:
    """Legacy sample-width bootstrap retained for older non-study callers.

    The Gaussian-CTR study does not call this function; it performs a complete
    Gaussian refit for every bootstrap draw in study.py/reporting.py.
    """
    values = np.asarray(values_ps, dtype=np.float64).reshape(-1)
    values = values[np.isfinite(values)]
    if values.size < 3 or int(n_bootstrap) <= 1:
        return float("nan")
    rng = np.random.default_rng(int(seed))
    draws = np.empty(int(n_bootstrap), dtype=np.float64)
    for index in range(draws.size):
        sample = rng.choice(values, size=values.size, replace=True)
        draws[index] = FWHM_PER_SIGMA * np.std(sample, ddof=1)
    return float(np.std(draws, ddof=1))
