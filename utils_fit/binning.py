from __future__ import annotations

import math

import numpy as np

DEFAULT_HISTOGRAM_BIN_WIDTH_PS = 10.0


def validate_histogram_bin_width_ps(value: float) -> float:
    """Return a validated positive physical histogram bin width in ps."""
    if isinstance(value, bool):
        raise ValueError("histogram_bin_width_ps must be a positive finite number")
    width = float(value)
    if not np.isfinite(width) or width <= 0.0:
        raise ValueError("histogram_bin_width_ps must be a positive finite number")
    return width


def fixed_width_histogram_edges(
    values_ps: np.ndarray,
    bin_width_ps: float,
    *,
    center_anchor_ps: float = 0.0,
    low_ps: float | None = None,
    high_ps: float | None = None,
) -> np.ndarray:
    """Uniform histogram edges with a fixed physical width and stable phase.

    ``center_anchor_ps`` is the center of one bin. With the default anchor, 0 ps
    is always a bin center. Extending the data range therefore changes only the
    number of bins, never their width or phase.
    """
    width = validate_histogram_bin_width_ps(bin_width_ps)
    anchor = float(center_anchor_ps)
    if not np.isfinite(anchor):
        raise ValueError("center_anchor_ps must be finite")

    values = np.asarray(values_ps, dtype=np.float64).reshape(-1)
    values = values[np.isfinite(values)]
    if low_ps is None:
        if not values.size:
            raise ValueError("Histogram binning requires finite values or low_ps/high_ps")
        low = float(np.min(values))
    else:
        low = float(low_ps)
    if high_ps is None:
        if not values.size:
            raise ValueError("Histogram binning requires finite values or low_ps/high_ps")
        high = float(np.max(values))
    else:
        high = float(high_ps)

    if not np.isfinite(low) or not np.isfinite(high) or high <= low:
        raise ValueError("Histogram range must be finite and non-degenerate")

    base_edge = anchor - 0.5 * width
    left_index = math.floor((low - base_edge) / width)
    right_index = math.ceil((high - base_edge) / width)
    if right_index <= left_index:
        right_index = left_index + 1

    start = base_edge + left_index * width
    n_bins = int(right_index - left_index)
    edges = start + np.arange(n_bins + 1, dtype=np.float64) * width

    # Guard against floating-point roundoff at the upper boundary.
    if edges[-1] < high:
        edges = np.append(edges, edges[-1] + width)
    return edges
