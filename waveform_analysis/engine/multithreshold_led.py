"""Target-independent multi-threshold leading-edge extrapolation on prepared inputs.

Use multiple *absolute*, baseline-relative mV crossing levels and fit
t_cross(V) = t0 + slope * V for each event/detector. No timing targets are
used by the crossing or fit. The fit intercept t0 is an extrapolated timing
pickoff, not a physical photon arrival time.

The prepared grid has an approximate LED sample anchor; therefore the
original fixed-threshold LED crossing is also interpolated on the same grid.
For the repository convention detector_1 - detector_2 and
corrected_residual = LED_target - prediction, return
prediction = (LED_1 - t0_1) - (LED_2 - t0_2).
"""
from __future__ import annotations

import numpy as np


def validate_threshold_sets(threshold_sets):
    result = []
    for group in threshold_sets:
        x = np.asarray(group, dtype=np.float64)
        if (x.ndim != 1 or x.size < 3 or not np.isfinite(x).all()
                or np.any(x <= 0) or np.any(x > 50)
                or np.any(np.diff(x) <= 0)):
            raise ValueError(
                "Each threshold set requires >=3 strictly increasing, "
                "positive absolute thresholds in mV, never above 50 mV")
        result.append(tuple(float(v) for v in x))
    if not result or len(set(result)) != len(result):
        raise ValueError("Threshold sets must be nonempty and distinct")
    return tuple(result)


def interpolate_upward_crossing(waveforms, levels, time_ps):
    """Return last strictly upward, linearly interpolated crossing [N,2].

    Operates only on a *short rising-side window* already selected by caller.
    The equality fallback follows waveform_analysis.signal.timing.crossing_ps.
    This vectorization avoids searching the entire long Wide waveform.
    """
    x = np.asarray(waveforms, dtype=np.float64)
    t = np.asarray(time_ps, dtype=np.float64)
    levels = np.asarray(levels, dtype=np.float64)
    if (x.ndim != 3 or x.shape[1] != 2
            or t.shape != (x.shape[2],) or x.shape[2] < 2
            or levels.shape != x.shape[:2]):
        raise ValueError("Crossing waveform, time, or threshold dimensions invalid")
    if not np.all(np.isfinite(t)) or np.any(np.diff(t) <= 0):
        raise ValueError("Time grid must be strictly increasing")
    out = np.full(x.shape[:2], np.nan, dtype=np.float64)
    positions = np.arange(t.size - 1, dtype=np.int32)[None, :]
    # Do not permit a late oscillation or falling-edge recrossing. Each
    # interpolated threshold must lie before the first local maximum.
    peaks = np.argmax(np.where(np.isfinite(x), x, -np.inf), axis=2)
    for d in range(2):
        lower, upper = x[:, d, :-1], x[:, d, 1:]
        threshold = levels[:, d, None]
        rising = positions < peaks[:, d, None]
        strict = (rising & np.isfinite(lower) & np.isfinite(upper)
                  & (lower < threshold) & (upper >= threshold))
        present = strict.any(axis=1)
        if not present.all():
            equal = (rising & np.isfinite(lower) & np.isfinite(upper)
                     & (lower == threshold) & (upper > threshold))
            strict[~present] = equal[~present]
        idx = np.where(strict, positions, -1).max(axis=1)
        rows = np.flatnonzero(idx >= 0)
        if not rows.size:
            continue
        k = idx[rows]
        low, high = lower[rows, k], upper[rows, k]
        weight = (levels[rows, d] - low) / (high - low)
        out[rows, d] = t[k] + weight * (t[k + 1] - t[k])
    return out


def linear_pickoff(crossing_times_ps, thresholds_mV):
    """Per-detector linear OLS t(V); return intercept, slope, fit-RMSE.

    crossing_times_ps: [event, 2, K]. No event labels used.
    """
    crossings = np.asarray(crossing_times_ps, dtype=np.float64)
    levels = np.asarray(thresholds_mV, dtype=np.float64).reshape(-1)
    if (crossings.ndim != 3 or crossings.shape[1:] != (2, levels.size)
            or levels.size < 3 or np.any(np.diff(levels) <= 0)):
        raise ValueError("Requires >=3 strictly increasing thresholds")
    centered = levels - levels.mean()
    denom = float(np.sum(centered ** 2))
    valid = np.isfinite(crossings).all(axis=2)
    # Crossing times must ascend strictly along increasing voltage levels.
    valid &= (np.diff(crossings, axis=2) > 0).all(axis=2)
    safe = np.where(np.isfinite(crossings), crossings, 0.)
    slope = np.sum(safe * centered[None, None, :], axis=2) / denom
    intercept = safe.mean(axis=2) - slope * float(levels.mean())
    predicted = intercept[..., None] + slope[..., None] * levels
    fit_rmse = np.sqrt(np.mean((safe - predicted) ** 2, axis=2))
    valid &= (slope > 0) & (intercept < crossings[:, :, 0])
    intercept[~valid] = np.nan
    slope[~valid] = np.nan
    fit_rmse[~valid] = np.nan
    return intercept, slope, fit_rmse


def multithreshold_corrections(
    waves, time_ps, *, threshold_sets, baseline_indices,
    led_threshold_mV, physical_scale_mV, crossing_window_ns=(-1.0, 2.0),
    chunk_size=128,
):
    """Corrections [event, threshold_set] and fit diagnostics from prepared waves.

    The waveform remains in saved normalized units; thresholds in physical mV
    are converted by the affine InputTransform range per detector. The fixed
    LED threshold is interpolated exactly like each multi-threshold crossing.
    Only samples in crossing_window_ns are searched. Unavailable crossing
    remains NaN, never replaced by a sentinel time.
    """
    groups = validate_threshold_sets(threshold_sets)
    grid = np.asarray(time_ps, np.float64)
    bidx = np.asarray(baseline_indices, np.int64)
    scales = np.asarray(physical_scale_mV, np.float64).reshape(-1)
    if (waves.ndim != 3 or waves.shape[1] != 2
            or waves.shape[2] != grid.size or grid.size < 4
            or bidx.ndim != 1 or bidx.size < 2
            or np.any(bidx < 0) or np.any(bidx >= grid.size)):
        raise ValueError("Prepared waveform or baseline dimensions invalid")
    if (not np.all(np.isfinite(grid)) or np.any(np.diff(grid) <= 0)
            or scales.shape != (2,) or np.any(~np.isfinite(scales))
            or np.any(scales <= 0) or not np.isfinite(led_threshold_mV)
            or float(led_threshold_mV) <= 0 or chunk_size < 1):
        raise ValueError("Invalid time grid or physical scale")
    start_ns, stop_ns = map(float, crossing_window_ns)
    if start_ns >= stop_ns or start_ns * 1000 < grid[0] or stop_ns * 1000 > grid[-1]:
        raise ValueError("Crossing window must be fully covered by the prepared grid")
    idx = np.flatnonzero((grid >= start_ns * 1000 - 1.e-6)
                         & (grid <= stop_ns * 1000 + 1.e-6))
    if idx.size < 3:
        raise ValueError("Crossing window contains fewer than three samples")
    levels = tuple(sorted({float(v) for group in groups for v in group}))
    all_crossings = np.full((waves.shape[0], 2, len(levels)), np.nan, np.float64)
    led = np.full((waves.shape[0], 2), np.nan, np.float64)
    for start in range(0, waves.shape[0], chunk_size):
        end = min(start + chunk_size, waves.shape[0])
        batch = np.asarray(waves[start:end], np.float64)
        baseline = np.mean(batch[:, :, bidx], axis=2)
        rising = batch[:, :, idx]
        t = grid[idx]
        led_level = baseline + float(led_threshold_mV) / scales[None, :]
        led_cross = interpolate_upward_crossing(rising, led_level, t)
        # The prepared anchor is the nearest sample to the native LED pickoff.
        # Reject a distinct, unrelated crossing far from the anchor at zero.
        tolerance = 2.5 * float(np.max(np.diff(grid)))
        led_cross[np.abs(led_cross) > tolerance] = np.nan
        led[start:end] = led_cross
        for k, mv in enumerate(levels):
            absolute_level = baseline + float(mv) / scales[None, :]
            all_crossings[start:end, :, k] = interpolate_upward_crossing(
                rising, absolute_level, t)
    correction = np.full((waves.shape[0], len(groups)), np.nan, np.float64)
    intercepts = np.full((waves.shape[0], 2, len(groups)), np.nan, np.float64)
    slopes = np.full_like(intercepts, np.nan)
    fit_rmse = np.full_like(intercepts, np.nan)
    for j, group in enumerate(groups):
        pos = [levels.index(mv) for mv in group]
        t0, slope, spread = linear_pickoff(all_crossings[:, :, pos], group)
        intercepts[:, :, j], slopes[:, :, j], fit_rmse[:, :, j] = (
            t0, slope, spread)
        dt = led - t0
        correction[:, j] = dt[:, 0] - dt[:, 1]
    return correction, {
        "led_crossing_ps": led,
        "intercept_ps": intercepts,
        "slope_ps_per_mV": slopes,
        "fit_rmse_ps": fit_rmse,
        "crossing_levels_mV": np.asarray(levels, np.float64),
    }
