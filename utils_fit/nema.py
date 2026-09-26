from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class NEMAFit:
    """NEMA FWHM estimate from a uniformly binned one-dimensional profile."""

    ctr_ps: float
    center_ps: float
    peak_height: float
    half_max_left_ps: float
    half_max_right_ps: float
    histogram_bins: int
    fit_low_ps: float
    fit_high_ps: float


def _middle_maximum_index(counts: np.ndarray) -> int:
    maxima = np.flatnonzero(counts == np.max(counts))
    if maxima.size == 0:
        raise ValueError("NEMA FWHM requires a non-empty histogram")
    return int(maxima[maxima.size // 2])


def _linear_crossing(
    x0: float,
    y0: float,
    x1: float,
    y1: float,
    level: float,
) -> float:
    denominator = float(y1 - y0)
    if denominator == 0.0:
        raise ValueError("NEMA half-maximum crossing is not uniquely defined")
    fraction = (float(level) - float(y0)) / denominator
    return float(x0 + fraction * (x1 - x0))


def nema_fwhm_from_histogram(
    counts: np.ndarray,
    edges_ps: np.ndarray,
) -> NEMAFit:
    """Apply the standard NEMA parabolic-peak and half-height interpolation method."""
    y = np.asarray(counts, dtype=np.float64).reshape(-1)
    edges = np.asarray(edges_ps, dtype=np.float64).reshape(-1)
    if y.size < 3 or edges.size != y.size + 1:
        raise ValueError("NEMA FWHM requires at least three histogram bins")
    if not np.all(np.isfinite(y)) or not np.all(np.isfinite(edges)):
        raise ValueError("NEMA FWHM requires finite histogram values")
    if np.any(y < 0.0) or np.any(np.diff(edges) <= 0.0):
        raise ValueError("NEMA FWHM requires non-negative counts and increasing edges")
    if not np.any(y > 0.0):
        raise ValueError("NEMA FWHM requires at least one populated bin")

    widths = np.diff(edges)
    if not np.allclose(widths, widths[0], rtol=1.0e-10, atol=1.0e-12):
        raise ValueError("NEMA FWHM requires uniform histogram binning")
    centers = 0.5 * (edges[:-1] + edges[1:])
    peak_index = _middle_maximum_index(y)
    if peak_index == 0 or peak_index == y.size - 1:
        raise ValueError("NEMA peak bin must have neighbours on both sides")

    # NEMA: estimate the true peak height from the parabola through the peak
    # bin and its two nearest neighbours. Center x before fitting for numerical
    # stability while preserving the exact quadratic interpolation.
    local_x = centers[peak_index - 1 : peak_index + 2] - centers[peak_index]
    local_y = y[peak_index - 1 : peak_index + 2]
    a, b, c = np.polyfit(local_x, local_y, 2)
    scale = max(1.0, float(np.max(local_y)))
    if abs(float(a)) <= np.finfo(np.float64).eps * scale:
        center = float(centers[peak_index])
        peak_height = float(y[peak_index])
    else:
        vertex = -float(b) / (2.0 * float(a))
        center = float(centers[peak_index] + vertex)
        peak_height = float(a * vertex * vertex + b * vertex + c)
    if not np.isfinite(peak_height) or peak_height <= 0.0:
        raise ValueError("NEMA parabolic peak estimate is invalid")

    half_height = 0.5 * peak_height
    left_candidates = np.flatnonzero(y[:peak_index] < half_height)
    right_candidates = np.flatnonzero(y[peak_index + 1 :] < half_height)
    if left_candidates.size == 0 or right_candidates.size == 0:
        raise ValueError("NEMA half-maximum crossings are outside the histogram")

    left_outer = int(left_candidates[-1])
    left_inner = left_outer + 1
    right_outer = int(peak_index + 1 + right_candidates[0])
    right_inner = right_outer - 1
    left = _linear_crossing(
        centers[left_outer],
        y[left_outer],
        centers[left_inner],
        y[left_inner],
        half_height,
    )
    right = _linear_crossing(
        centers[right_inner],
        y[right_inner],
        centers[right_outer],
        y[right_outer],
        half_height,
    )
    width = float(right - left)
    if not np.isfinite(width) or width <= 0.0:
        raise ValueError("NEMA FWHM is invalid")

    return NEMAFit(
        ctr_ps=width,
        center_ps=center,
        peak_height=peak_height,
        half_max_left_ps=float(left),
        half_max_right_ps=float(right),
        histogram_bins=int(y.size),
        fit_low_ps=float(edges[0]),
        fit_high_ps=float(edges[-1]),
    )


def fit_nema_fwhm(
    values_ps: np.ndarray,
    *,
    histogram_bins: int = 22,
) -> NEMAFit:
    """Histogram timing residuals and evaluate FWHM with the NEMA method."""
    values = np.asarray(values_ps, dtype=np.float64).reshape(-1)
    values = values[np.isfinite(values)]
    bins = int(histogram_bins)
    if bins < 3:
        raise ValueError("nema requires at least 3 histogram bins")
    if values.size < 5:
        raise ValueError("nema requires at least 5 finite residuals")
    low = float(np.min(values))
    high = float(np.max(values))
    if not np.isfinite(low) or not np.isfinite(high) or high <= low:
        raise ValueError("NEMA FWHM requires non-degenerate residuals")
    counts, edges = np.histogram(values, bins=bins, range=(low, high))
    return nema_fwhm_from_histogram(counts, edges)
