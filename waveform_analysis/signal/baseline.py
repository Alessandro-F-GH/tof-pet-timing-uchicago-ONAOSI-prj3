"""Pure baseline reductions, preserving the native-grid window convention.

Finite samples are selected before each reduction. Reductions remain one
waveform at a time to retain the original NumPy summation order.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from numpy.typing import ArrayLike


def baseline_level_mV(
    signal: ArrayLike,
    interval_s: float,
    materialized_before_ns: float,
    baseline_window_ns: Sequence[float],
) -> float:
    """Return the finite-sample mean of a trigger-relative baseline window.

    ``signal`` is [sample] in mV. Window endpoints are ns relative to the
    trigger; floor/ceil rounding and the inclusive upper sample are retained.
    An empty or invalid window yields NaN, as in the original implementation.
    """
    y = np.asarray(signal, dtype=np.float64)
    dt_ns = float(interval_s) * 1.0e9
    window = np.asarray(baseline_window_ns, dtype=np.float64).reshape(-1)
    if (
        y.size == 0
        or not np.isfinite(dt_ns)
        or dt_ns <= 0.0
        or window.size != 2
        or not np.all(np.isfinite(window))
        or float(window[1]) <= float(window[0])
    ):
        return float("nan")

    trigger_index = int(np.ceil(float(materialized_before_ns) / dt_ns))
    start = max(0, trigger_index + int(np.floor(float(window[0]) / dt_ns)))
    stop = min(
        y.size,
        trigger_index + int(np.ceil(float(window[1]) / dt_ns)) + 1,
    )
    if stop <= start:
        return float("nan")
    values = y[start:stop]
    values = values[np.isfinite(values)]
    return float(np.mean(values)) if values.size else float("nan")


def baseline_quality_metrics(
    signal_mV: ArrayLike,
    trigger_index: int,
    sample_interval_s: float,
    window_ns: Sequence[float],
    vertical_limits_mV: Sequence[float],
    clipping_margin_mV: float,
) -> tuple[float, bool, float]:
    """Return baseline RMS, clipping flag and rail clearance for [sample].

    Nonfinite samples are removed before the original mean/RMS reduction.
    Clearance equal to the configured margin counts as clipped. Fewer than
    two finite samples returns (NaN, False, NaN), matching selection behavior.
    """
    y = np.asarray(signal_mV, float)
    dt = float(sample_interval_s) * 1e9
    a0, b0 = map(float, window_ns)
    a = max(0, int(trigger_index) + int(np.floor(a0 / dt)))
    b = min(y.size, int(trigger_index) + int(np.ceil(b0 / dt)) + 1)
    v = y[a:b]
    v = v[np.isfinite(v)]
    if v.size < 2:
        return float("nan"), False, float("nan")
    center = float(np.mean(v))
    rms = float(np.sqrt(np.mean((v - center) ** 2)))
    low, high = map(float, np.sort(np.asarray(vertical_limits_mV, float)))
    margin = float(clipping_margin_mV)
    clearance = float(min(np.min(v) - low, high - np.max(v)))
    return rms, bool(clearance <= margin), clearance
