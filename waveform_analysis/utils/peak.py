from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.optimize import curve_fit


@dataclass
class PeakFitResult:
    success: bool
    mean: float
    sigma: float
    mean_error: float
    sigma_error: float
    selection_low: float
    selection_high: float
    chi2: float
    ndof: int
    iterations: int
    message: str = ""
    edges: np.ndarray = field(default_factory=lambda: np.empty(0), repr=False)
    counts: np.ndarray = field(default_factory=lambda: np.empty(0), repr=False)
    expected: np.ndarray = field(default_factory=lambda: np.empty(0), repr=False)
    fit_low: float = np.nan
    fit_high: float = np.nan

    @property
    def chi2_ndof(self) -> float:
        return self.chi2 / self.ndof if self.ndof > 0 else np.nan

    def as_dict(self) -> dict[str, Any]:
        return {
            "success": self.success,
            "mean": self.mean,
            "sigma": self.sigma,
            "mean_error": self.mean_error,
            "sigma_error": self.sigma_error,
            "selection_low": self.selection_low,
            "selection_high": self.selection_high,
            "chi2": self.chi2,
            "ndof": self.ndof,
            "chi2_ndof": self.chi2_ndof,
            "iterations": self.iterations,
            "fit_low": self.fit_low,
            "fit_high": self.fit_high,
            "message": self.message,
        }


def gaussian(x: np.ndarray, amplitude: float, mean: float, sigma: float) -> np.ndarray:
    return amplitude * np.exp(-0.5 * ((x - mean) / sigma) ** 2)


def fit_histogram_peak(
    values: np.ndarray,
    *,
    config: dict[str, Any],
    unit_suffix: str,
    fit_name: str,
    value_name: str,
) -> PeakFitResult:
    """Fit the dominant histogram peak and derive a sigma-based selection window.

    The numerical procedure is shared by photopeak and ToT selection. Unit-specific
    configuration keys are ``histogram_bin_<unit_suffix>``,
    ``initial_half_width_<unit_suffix>`` and
    ``convergence_tolerance_<unit_suffix>``; all other keys are common.
    """
    xvalues = np.asarray(values, dtype=np.float64)
    xvalues = xvalues[np.isfinite(xvalues)]
    if xvalues.size < 20:
        return PeakFitResult(False, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan,
                             np.nan, 0, 0, f"Too few finite {value_name}")

    bin_width = float(config[f"histogram_bin_{unit_suffix}"])
    low_edge = np.floor(np.min(xvalues) / bin_width) * bin_width
    high_edge = np.ceil(np.max(xvalues) / bin_width) * bin_width
    if high_edge <= low_edge:
        high_edge = low_edge + bin_width
    edges = np.arange(low_edge, high_edge + 1.01 * bin_width, bin_width, dtype=np.float64)
    counts, edges = np.histogram(xvalues, bins=edges)
    centers = 0.5 * (edges[:-1] + edges[1:])

    quantile_cut = float(np.quantile(xvalues, float(config["search_quantile_min"])))
    search = centers >= quantile_cut
    if not np.any(search):
        search = np.ones_like(centers, dtype=bool)
    smooth = gaussian_filter1d(counts.astype(np.float64),
                               sigma=float(config["smoothing_sigma_bins"]),
                               mode="nearest")
    candidate_indices = np.flatnonzero(search)
    peak_index = int(candidate_indices[np.argmax(smooth[candidate_indices])])
    mean = float(centers[peak_index])
    half_width = float(config[f"initial_half_width_{unit_suffix}"])
    sigma = max(bin_width, half_width / 3.0)
    fit_low = mean - half_width
    fit_high = mean + half_width
    max_iterations = int(config["max_iterations"])
    iteration_sigma = float(config["iteration_sigma"])
    tolerance = float(config[f"convergence_tolerance_{unit_suffix}"])

    covariance = None
    fitted_counts = np.empty(0)
    iterations = 0
    message = ""
    success = False

    for iteration in range(max_iterations):
        mask = (centers >= fit_low) & (centers <= fit_high)
        fit_x = centers[mask]
        fit_y = counts[mask].astype(np.float64)
        if fit_x.size < 7 or np.count_nonzero(fit_y) < 4:
            message = f"Too few populated bins in {fit_name.lower()} fit interval"
            break
        amplitude0 = max(float(np.max(fit_y)), 1.0)
        p0 = [amplitude0, mean, max(sigma, 0.5 * bin_width)]
        uncertainty = np.sqrt(np.maximum(fit_y, 1.0))
        try:
            parameters, covariance = curve_fit(
                gaussian,
                fit_x,
                fit_y,
                p0=p0,
                sigma=uncertainty,
                absolute_sigma=True,
                bounds=([0.0, fit_low, 0.1 * bin_width],
                        [np.inf, fit_high, max(high_edge - low_edge, bin_width)]),
                maxfev=20_000,
            )
        except Exception as exc:
            message = f"{fit_name} fit failed: {exc}"
            break
        amplitude, new_mean, new_sigma = [float(item) for item in parameters]
        if not np.all(np.isfinite(parameters)) or new_sigma <= 0:
            message = f"{fit_name} fit returned invalid parameters"
            break
        iterations = iteration + 1
        fitted_counts = gaussian(fit_x, amplitude, new_mean, new_sigma)
        converged = abs(new_mean - mean) <= tolerance and abs(new_sigma - sigma) <= tolerance
        mean, sigma = new_mean, new_sigma
        fit_low = mean - iteration_sigma * sigma
        fit_high = mean + iteration_sigma * sigma
        success = True
        if converged:
            break

    if not success:
        return PeakFitResult(False, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan,
                             np.nan, 0, iterations,
                             message or f"{fit_name} fit did not converge",
                             edges=edges, counts=counts)

    final_mask = (centers >= fit_low) & (centers <= fit_high)
    final_x = centers[final_mask]
    final_y = counts[final_mask].astype(np.float64)
    amplitude = float(np.max(fitted_counts)) if fitted_counts.size else float(np.max(final_y))
    uncertainty = np.sqrt(np.maximum(final_y, 1.0))
    try:
        final_parameters, covariance = curve_fit(
            gaussian,
            final_x,
            final_y,
            p0=[amplitude, mean, sigma],
            sigma=uncertainty,
            absolute_sigma=True,
            bounds=([0.0, fit_low, 0.1 * bin_width],
                    [np.inf, fit_high, max(high_edge - low_edge, bin_width)]),
            maxfev=20_000,
        )
        amplitude, mean, sigma = [float(item) for item in final_parameters]
    except Exception:
        pass

    expected = gaussian(centers, amplitude, mean, sigma)
    expected_fit = gaussian(final_x, amplitude, mean, sigma)
    variance = np.maximum(expected_fit, 1.0)
    chi2 = float(np.sum((final_y - expected_fit) ** 2 / variance))
    ndof = max(0, int(final_x.size) - 3)

    errors = np.full(3, np.nan)
    if covariance is not None and covariance.shape == (3, 3):
        diagonal = np.diag(covariance)
        errors = np.sqrt(np.where(diagonal >= 0, diagonal, np.nan))

    selection_low = mean + float(config["selection_sigma_low"]) * sigma
    selection_high = mean + float(config["selection_sigma_high"]) * sigma
    return PeakFitResult(True, mean, sigma, float(errors[1]), float(errors[2]),
                         selection_low, selection_high, chi2, ndof, iterations,
                         edges=edges, counts=counts, expected=expected,
                         fit_low=fit_low, fit_high=fit_high)
