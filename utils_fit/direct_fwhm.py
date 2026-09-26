from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .binning import fixed_width_histogram_edges, validate_histogram_bin_width_ps


@dataclass(frozen=True)
class DirectFWHMFit:
    """Direct F1 full width at half maximum from a uniformly binned profile."""

    ctr_ps: float
    center_ps: float
    peak_height: float
    half_max_left_ps: float
    half_max_right_ps: float
    histogram_bins: int
    histogram_bin_width_ps: float
    fit_low_ps: float
    fit_high_ps: float


def _middle_maximum_index(counts: np.ndarray) -> int:
    maxima = np.flatnonzero(counts == np.max(counts))
    if maxima.size == 0:
        raise ValueError("Direct FWHM requires a non-empty histogram")
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
        raise ValueError("Direct half-maximum crossing is not uniquely defined")
    fraction = (float(level) - float(y0)) / denominator
    return float(x0 + fraction * (x1 - x0))


def direct_fwhm_from_histogram(
    counts: np.ndarray,
    edges_ps: np.ndarray,
) -> DirectFWHMFit:
    """Apply method F1: direct half-maximum crossings with linear interpolation.

    The peak is the middlemost maximum histogram bin. The half-maximum level is
    one half of that bin count. On each side, the nearest bin below half maximum
    is paired with its adjacent inner bin and the crossing is linearly
    interpolated. The FWHM is the distance between the two crossings.
    """
    y = np.asarray(counts, dtype=np.float64).reshape(-1)
    edges = np.asarray(edges_ps, dtype=np.float64).reshape(-1)
    if y.size < 3 or edges.size != y.size + 1:
        raise ValueError("Direct FWHM requires at least three histogram bins")
    if not np.all(np.isfinite(y)) or not np.all(np.isfinite(edges)):
        raise ValueError("Direct FWHM requires finite histogram values")
    if np.any(y < 0.0) or np.any(np.diff(edges) <= 0.0):
        raise ValueError("Direct FWHM requires non-negative counts and increasing edges")
    if not np.any(y > 0.0):
        raise ValueError("Direct FWHM requires at least one populated bin")

    widths = np.diff(edges)
    if not np.allclose(widths, widths[0], rtol=1.0e-10, atol=1.0e-12):
        raise ValueError("Direct FWHM requires uniform histogram binning")
    bin_width = float(widths[0])
    centers = 0.5 * (edges[:-1] + edges[1:])

    peak_index = _middle_maximum_index(y)
    if peak_index == 0 or peak_index == y.size - 1:
        raise ValueError("Direct FWHM peak bin must have bins on both sides")

    peak_height = float(y[peak_index])
    half_height = 0.5 * peak_height
    left_candidates = np.flatnonzero(y[:peak_index] < half_height)
    right_candidates = np.flatnonzero(y[peak_index + 1 :] < half_height)
    if left_candidates.size == 0 or right_candidates.size == 0:
        raise ValueError("Direct half-maximum crossings are outside the histogram")

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
        raise ValueError("Direct FWHM is invalid")

    return DirectFWHMFit(
        ctr_ps=width,
        center_ps=float(centers[peak_index]),
        peak_height=peak_height,
        half_max_left_ps=float(left),
        half_max_right_ps=float(right),
        histogram_bins=int(y.size),
        histogram_bin_width_ps=bin_width,
        fit_low_ps=float(edges[0]),
        fit_high_ps=float(edges[-1]),
    )


def fit_direct_fwhm(
    values_ps: np.ndarray,
    *,
    histogram_bin_width_ps: float = 10.0,
) -> DirectFWHMFit:
    """Evaluate the direct F1 FWHM on a fixed-width residual histogram."""
    values = np.asarray(values_ps, dtype=np.float64).reshape(-1)
    values = values[np.isfinite(values)]
    bin_width = validate_histogram_bin_width_ps(histogram_bin_width_ps)
    if values.size < 5:
        raise ValueError("Direct FWHM requires at least 5 finite residuals")
    if float(np.ptp(values)) <= 0.0:
        raise ValueError("Direct FWHM requires non-degenerate residuals")
    edges = fixed_width_histogram_edges(values, bin_width)
    counts, _ = np.histogram(values, bins=edges)
    return direct_fwhm_from_histogram(counts, edges)
