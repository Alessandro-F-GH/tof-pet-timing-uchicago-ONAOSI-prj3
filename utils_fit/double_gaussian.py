from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
from scipy.optimize import brentq, least_squares

from .binning import fixed_width_histogram_edges, validate_histogram_bin_width_ps

SQRT_2PI = math.sqrt(2.0 * math.pi)


@dataclass(frozen=True)
class DoubleGaussianFit:
    """Shared-mean two-Gaussian mixture fit and FWHM of the total fitted shape."""

    ctr_ps: float
    center_ps: float
    sigma_narrow_ps: float
    sigma_wide_ps: float
    narrow_fraction: float
    histogram_bins: int
    histogram_bin_width_ps: float
    fit_low_ps: float
    fit_high_ps: float


def double_gaussian_density(
    x_ps: np.ndarray | float,
    center_ps: float,
    sigma_narrow_ps: float,
    sigma_wide_ps: float,
    narrow_fraction: float,
) -> np.ndarray:
    """Normalized shared-mean double-Gaussian density."""
    x = np.asarray(x_ps, dtype=np.float64)
    center = float(center_ps)
    sigma_narrow = float(sigma_narrow_ps)
    sigma_wide = float(sigma_wide_ps)
    fraction = float(narrow_fraction)
    if sigma_narrow <= 0.0 or sigma_wide <= 0.0:
        raise ValueError("Gaussian widths must be positive")
    if not 0.0 <= fraction <= 1.0:
        raise ValueError("narrow_fraction must lie in [0, 1]")
    narrow = np.exp(-0.5 * ((x - center) / sigma_narrow) ** 2) / sigma_narrow
    wide = np.exp(-0.5 * ((x - center) / sigma_wide) ** 2) / sigma_wide
    return (fraction * narrow + (1.0 - fraction) * wide) / SQRT_2PI


def double_gaussian_fwhm(
    sigma_narrow_ps: float,
    sigma_wide_ps: float,
    narrow_fraction: float,
) -> float:
    """FWHM of a shared-mean two-Gaussian mixture."""
    sigma_narrow = float(sigma_narrow_ps)
    sigma_wide = float(sigma_wide_ps)
    fraction = float(narrow_fraction)
    if sigma_narrow <= 0.0 or sigma_wide <= 0.0:
        raise ValueError("Gaussian widths must be positive")
    if sigma_narrow > sigma_wide:
        sigma_narrow, sigma_wide = sigma_wide, sigma_narrow
        fraction = 1.0 - fraction
    if not 0.0 <= fraction <= 1.0:
        raise ValueError("narrow_fraction must lie in [0, 1]")

    peak = fraction / sigma_narrow + (1.0 - fraction) / sigma_wide

    def half_max(offset: float) -> float:
        return (
            fraction / sigma_narrow * math.exp(-0.5 * (offset / sigma_narrow) ** 2)
            + (1.0 - fraction) / sigma_wide * math.exp(-0.5 * (offset / sigma_wide) ** 2)
            - 0.5 * peak
        )

    right = max(10.0 * sigma_wide, sigma_wide + sigma_narrow)
    half_width = brentq(half_max, 0.0, right)
    return float(2.0 * half_width)


def _initial_scale(values: np.ndarray) -> float:
    q16, q84 = np.quantile(values, [0.16, 0.84])
    robust = 0.5 * float(q84 - q16)
    std = float(np.std(values))
    for candidate in (robust, std, float(np.ptp(values)) / 6.0):
        if np.isfinite(candidate) and candidate > 0.0:
            return candidate
    raise ValueError("Double-Gaussian fit requires non-degenerate residuals")


def fit_double_gaussian_fwhm(
    values_ps: np.ndarray,
    *,
    histogram_bin_width_ps: float = 10.0,
    initial: DoubleGaussianFit | None = None,
) -> DoubleGaussianFit:
    """Fit a shared-mean double Gaussian on a fixed-width, zero-anchored histogram."""
    values = np.asarray(values_ps, dtype=np.float64).reshape(-1)
    values = values[np.isfinite(values)]
    bin_width = validate_histogram_bin_width_ps(histogram_bin_width_ps)
    if values.size < 10:
        raise ValueError("double_gaussian requires at least 10 finite residuals")
    if float(np.ptp(values)) <= 0.0:
        raise ValueError("Double-Gaussian fit requires non-degenerate residuals")

    edges = fixed_width_histogram_edges(values, bin_width)
    counts, _ = np.histogram(values, bins=edges)
    centers = 0.5 * (edges[:-1] + edges[1:])
    low = float(edges[0])
    high = float(edges[-1])
    scale = _initial_scale(values)
    min_sigma = max(bin_width * 0.20, scale * 1.0e-3, 1.0e-6)
    max_sigma = max(float(high - low) * 2.0, scale * 12.0, min_sigma * 20.0)
    center_low = low - 0.25 * (high - low)
    center_high = high + 0.25 * (high - low)
    sqrt_weight = np.sqrt(np.maximum(counts.astype(np.float64), 1.0))
    n_events = float(values.size)

    def expected(parameters: np.ndarray) -> np.ndarray:
        center, sigma_narrow, delta_sigma, fraction = parameters
        sigma_wide = sigma_narrow + delta_sigma
        density = double_gaussian_density(
            centers,
            center,
            sigma_narrow,
            sigma_wide,
            fraction,
        )
        return n_events * bin_width * density

    def residual(parameters: np.ndarray) -> np.ndarray:
        return (expected(parameters) - counts) / sqrt_weight

    starts: list[np.ndarray] = []
    if initial is not None:
        starts.append(
            np.asarray(
                [
                    initial.center_ps,
                    initial.sigma_narrow_ps,
                    max(0.0, initial.sigma_wide_ps - initial.sigma_narrow_ps),
                    initial.narrow_fraction,
                ],
                dtype=np.float64,
            )
        )
    else:
        center0 = float(np.median(values))
        starts.extend(
            np.asarray([center0, scale * narrow, scale * (wide - narrow), fraction])
            for narrow, wide, fraction in (
                (0.55, 1.60, 0.65),
                (0.35, 2.00, 0.45),
                (0.80, 2.50, 0.75),
            )
        )

    lower = np.asarray([center_low, min_sigma, 0.0, 1.0e-3], dtype=np.float64)
    upper = np.asarray([center_high, max_sigma, max_sigma, 1.0 - 1.0e-3], dtype=np.float64)
    best = None
    for start in starts:
        start = np.clip(start, lower + 1.0e-12, upper - 1.0e-12)
        try:
            fitted = least_squares(
                residual,
                start,
                bounds=(lower, upper),
                method="trf",
                max_nfev=2500,
            )
        except (ValueError, FloatingPointError):
            continue
        if not fitted.success or not np.all(np.isfinite(fitted.x)):
            continue
        score = float(np.sum(residual(fitted.x) ** 2))
        if best is None or score < best[0]:
            best = (score, fitted.x)
    if best is None:
        raise ValueError("Double-Gaussian fit did not converge")

    center, sigma_narrow, delta_sigma, fraction = map(float, best[1])
    sigma_wide = sigma_narrow + delta_sigma
    ctr = double_gaussian_fwhm(sigma_narrow, sigma_wide, fraction)
    if not np.isfinite(ctr) or ctr <= 0.0:
        raise ValueError("Double-Gaussian fit produced an invalid FWHM")
    return DoubleGaussianFit(
        ctr_ps=float(ctr),
        center_ps=center,
        sigma_narrow_ps=sigma_narrow,
        sigma_wide_ps=sigma_wide,
        narrow_fraction=fraction,
        histogram_bins=int(counts.size),
        histogram_bin_width_ps=bin_width,
        fit_low_ps=low,
        fit_high_ps=high,
    )
