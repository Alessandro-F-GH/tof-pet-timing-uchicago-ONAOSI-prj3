"""Pure NumPy native-grid timing; independent of I/O and deep learning."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .baseline import baseline_level_mV


def crossing_ps(
    signal: ArrayLike,
    start_time_s: float,
    interval_s: float,
    rising_start: int,
    rising_stop: int,
    level_mV: float,
) -> float:
    """Interpolate the last qualifying rising crossing, returning ps or NaN.

    ``signal`` is [sample]. Strict below-to-at-least crossings take precedence
    over equal-to-above crossings, even if an equality crossing occurs later.
    Bounds, float64 conversion and arithmetic order match the original kernel.
    """
    y = np.asarray(signal, dtype=np.float64)
    a, b = int(rising_start), int(rising_stop)
    if a < 0 or b >= y.size or b <= a or not np.isfinite(level_mV):
        return float("nan")
    y0, y1 = y[a:b], y[a + 1 : b + 1]
    crossings = np.flatnonzero(
        np.isfinite(y0) & np.isfinite(y1) & (y0 < level_mV) & (y1 >= level_mV)
    )
    if crossings.size == 0:
        crossings = np.flatnonzero(
            np.isfinite(y0) & np.isfinite(y1) & (y0 == level_mV) & (y1 > level_mV)
        )
    if crossings.size == 0:
        return float("nan")
    lower = a + int(crossings[-1])
    denom = float(y[lower + 1] - y[lower])
    if denom == 0.0 or not np.isfinite(denom):
        return float("nan")
    fraction = (float(level_mV) - float(y[lower])) / denom
    if not 0.0 <= fraction <= 1.0:
        return float("nan")
    return (
        float(start_time_s) + (float(lower) + fraction) * float(interval_s)
    ) * 1.0e12


def led_times_ps(
    waves: ArrayLike,
    starts: ArrayLike,
    intervals: ArrayLike,
    rising_start: ArrayLike,
    rising_stop: ArrayLike,
    indices: ArrayLike,
    thresholds_mV: ArrayLike,
    *,
    baseline_window_ns: Sequence[float] | None = None,
    materialized_before_ns: float | None = None,
) -> NDArray[np.float64]:
    """Return LED times [selected event, detector=2, threshold] in ps.

    Waveforms have shape [event, 2, sample]; starts, intervals and rising bounds
    have shape [event, 2]. Thresholds are baseline-relative only when a baseline
    window is supplied. Invalid crossings remain NaN. The sample masks are
    vectorized; event/threshold loops preserve scalar interpolation arithmetic.
    """
    waves = np.asarray(waves)
    starts = np.asarray(starts)
    intervals = np.asarray(intervals)
    rising_start = np.asarray(rising_start)
    rising_stop = np.asarray(rising_stop)
    idx = np.asarray(indices, dtype=np.int64)
    thresholds = np.asarray(thresholds_mV, dtype=np.float64).reshape(-1)
    output = np.full((idx.size, 2, thresholds.size), np.nan, dtype=np.float64)
    before_ns = (
        float(materialized_before_ns) if baseline_window_ns is not None else None
    )
    for row, event in enumerate(idx):
        for detector in range(2):
            baseline = 0.0
            if baseline_window_ns is not None:
                baseline = baseline_level_mV(
                    waves[event, detector],
                    intervals[event, detector],
                    float(before_ns),
                    baseline_window_ns,
                )
                if not np.isfinite(baseline):
                    continue
            for column, threshold in enumerate(thresholds):
                output[row, detector, column] = crossing_ps(
                    waves[event, detector],
                    starts[event, detector],
                    intervals[event, detector],
                    rising_start[event, detector],
                    rising_stop[event, detector],
                    baseline + float(threshold),
                )
    return output


def cfd_times_ps(
    waves: ArrayLike,
    starts: ArrayLike,
    intervals: ArrayLike,
    rising_start: ArrayLike,
    rising_stop: ArrayLike,
    indices: ArrayLike,
    fractions: ArrayLike,
) -> NDArray[np.float64]:
    """Return CFD times [selected event, detector=2, fraction] in ps.

    Fractions multiply the original peak in the inclusive rising interval.
    No new baseline subtraction or peak definition is introduced.
    """
    waves = np.asarray(waves)
    starts = np.asarray(starts)
    intervals = np.asarray(intervals)
    rising_start = np.asarray(rising_start)
    rising_stop = np.asarray(rising_stop)
    idx = np.asarray(indices, dtype=np.int64)
    fractions = np.asarray(fractions, dtype=np.float64).reshape(-1)
    if np.any((fractions <= 0.0) | (fractions > 1.0)):
        raise ValueError("CFD fractions must lie in (0, 1]")
    output = np.full((idx.size, 2, fractions.size), np.nan, dtype=np.float64)
    for row, event in enumerate(idx):
        for detector in range(2):
            signal = np.asarray(waves[event, detector], dtype=np.float64)
            a, b = int(rising_start[event, detector]), int(rising_stop[event, detector])
            peak = float(np.nanmax(signal[a : b + 1])) if b >= a else np.nan
            if not np.isfinite(peak) or peak <= 0.0:
                continue
            for column, fraction in enumerate(fractions):
                output[row, detector, column] = crossing_ps(
                    signal,
                    starts[event, detector],
                    intervals[event, detector],
                    a,
                    b,
                    float(fraction) * peak,
                )
    return output


def pair_delta(times_ps: ArrayLike) -> NDArray[np.float64]:
    """Subtract detector 2 from detector 1 for times [event, detector=2]."""
    values = np.asarray(times_ps, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 2:
        raise ValueError("Expected [event, detector] timing array")
    return values[:, 0] - values[:, 1]
