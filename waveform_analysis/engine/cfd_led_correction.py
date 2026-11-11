"""Fractional-threshold CFD correction on frozen, LED-anchored prepared waveforms.

This is a waveform-derived correction relative to the *same locally interpolated*
LED crossing on the prepared grid. The prepared zero is the nearest LED anchor
sample, not necessarily the sub-sample LED crossing. No absolute timestamps,
ROOT loading, fitted model or ML inference is required.

For the repository convention delta=t_detector_1-t_detector_2 and
residual=LED_target-prediction, the correction is
(LED1-CFD1) - (LED2-CFD2); applying it produces the CFD delta.
"""
from __future__ import annotations

import numpy as np


def crossing_ps_on_grid(waves, levels, time_ps, last_sample):
    """Last below-to-at-least crossing up to inclusive peak sample.

    Parameters
    ----------
    waves: [n,2,t] prepared normalized waveform values
    levels: [n,2] threshold in the same normalized units
    time_ps: [t] common, strictly increasing prepared LED-anchor time grid
    last_sample: [n,2] peak indices restricting search to the rising side
    """
    x = np.asarray(waves, dtype=np.float64)
    t = np.asarray(time_ps, dtype=np.float64)
    levels = np.asarray(levels, dtype=np.float64)
    last_sample = np.asarray(last_sample, dtype=np.int64)
    if x.ndim != 3 or x.shape[1] != 2 or t.shape != (x.shape[2],):
        raise ValueError("Expected paired waveforms and corresponding time grid")
    if levels.shape != x.shape[:2] or last_sample.shape != x.shape[:2]:
        raise ValueError("Crossing threshold/peak shape mismatch")
    if not np.all(np.isfinite(t)) or np.any(np.diff(t) <= 0):
        raise ValueError("Prepared time grid must be strictly increasing")
    n, _, ns = x.shape
    out = np.full((n, 2), np.nan, dtype=np.float64)
    possible = np.arange(ns - 1, dtype=np.int64)[None, :]
    for detector in range(2):
        lower = x[:, detector, :-1]
        upper = x[:, detector, 1:]
        level = levels[:, detector, None]
        in_rise = possible < last_sample[:, detector, None]
        crossed = (in_rise & np.isfinite(lower) & np.isfinite(upper) &
                   (lower < level) & (upper >= level))
        # Match the repository scalar crossing kernel's equality fallback.
        has_strict = crossed.any(axis=1)
        if not np.all(has_strict):
            equality = (in_rise & np.isfinite(lower) & np.isfinite(upper) &
                        (lower == level) & (upper > level))
            crossed[~has_strict] = equality[~has_strict]
        idx = np.where(crossed, possible, -1).max(axis=1)
        valid = idx >= 0
        if not np.any(valid):
            continue
        rows = np.flatnonzero(valid)
        k = idx[valid]
        low = lower[rows, k]
        high = upper[rows, k]
        span = high - low
        ok = np.isfinite(span) & (span > 0)
        rows, k, low, span = rows[ok], k[ok], low[ok], span[ok]
        fractional = (levels[rows, detector] - low) / span
        out[rows, detector] = (
            t[k] + fractional * (t[k + 1] - t[k]))
    return out


def prepared_cfd_corrections(waves, time_ps, *, fractions, baseline_indices,
                             led_threshold_mV, physical_scale_mV,
                             chunk_size=128):
    """Compute [event, fraction] LED-to-CFD corrections in ps.

    Both crossings are re-interpolated on the *same prepared waveform*.
    Fractions use the per-event baseline-subtracted peak of the prepared pulse.
    Unrecoverable crossings remain NaN; never fill them with fabricated times.

    physical_scale_mV is the per-detector (max-min) of the saved InputTransform.
    The baseline and waveform are in prepared units; the LED threshold in mV
    is converted to those units. This avoids allocating inverse-transformed data.
    """
    x = waves
    grid = np.asarray(time_ps, dtype=np.float64)
    fractions = np.asarray(fractions, dtype=np.float64).reshape(-1)
    baseline_indices = np.asarray(baseline_indices, dtype=np.int64)
    physical_scale = np.asarray(physical_scale_mV, dtype=np.float64).reshape(-1)
    if (x.ndim != 3 or x.shape[1] != 2 or x.shape[2] != grid.size
            or baseline_indices.ndim != 1 or baseline_indices.size < 2
            or np.any(baseline_indices < 0) or
            np.any(baseline_indices >= grid.size)):
        raise ValueError("Invalid prepared waveform/baseline grids")
    if (not fractions.size or not np.all(np.isfinite(fractions))
            or np.any((fractions <= 0) | (fractions >= 1))
            or np.unique(fractions).size != fractions.size):
        raise ValueError("CFD fractions must be distinct and in (0,1)")
    if (physical_scale.shape != (2,) or not np.all(np.isfinite(physical_scale))
            or np.any(physical_scale <= 0) or not np.isfinite(led_threshold_mV)
            or led_threshold_mV <= 0 or chunk_size < 1):
        raise ValueError("Invalid saved LED threshold or physical transform")
    if not np.all(np.isfinite(grid)) or np.any(np.diff(grid) <= 0):
        raise ValueError("Invalid prepared time axis")
    n = x.shape[0]
    output = np.full((n, fractions.size), np.nan, dtype=np.float64)
    led_offset_ps = np.full((n, 2), np.nan, dtype=np.float64)
    peak_ps = np.full((n, 2), np.nan, dtype=np.float64)
    for start in range(0, n, chunk_size):
        end = min(n, start + chunk_size)
        batch = np.asarray(x[start:end], dtype=np.float64)
        baseline = batch[:, :, baseline_indices].mean(axis=-1)
        # The waveform is positive after preprocessing polarity normalization.
        # Pick the first occurrence of its maximum, restricting all crossings
        # to the rising side of this maximum.
        peak_index = np.argmax(batch, axis=2)
        peak = np.take_along_axis(
            batch, peak_index[..., None], axis=2)[..., 0]
        amplitude = peak - baseline
        good_peak = (np.isfinite(amplitude) & (amplitude > 0) &
                     (peak_index > 0) & (peak_index < grid.size - 1))
        led_level = baseline + float(led_threshold_mV) / physical_scale[None, :]
        led_crossing = crossing_ps_on_grid(batch, led_level, grid, peak_index)
        led_crossing[~good_peak] = np.nan
        # Baseline-relative LED crossing should be near the anchor sample t=0.
        # Reject an unrelated crossing elsewhere in the waveform.
        local_dt_ps = float(np.max(np.diff(grid)))
        led_crossing[np.abs(led_crossing) > 2.5 * local_dt_ps] = np.nan
        led_offset_ps[start:end] = led_crossing
        peak_ps[start:end] = grid[peak_index]
        for j, frac in enumerate(fractions):
            cfd_level = baseline + frac * amplitude
            crossing = crossing_ps_on_grid(batch, cfd_level, grid, peak_index)
            crossing[~good_peak] = np.nan
            # The residual is target - correction; the repository uses
            # detector1 minus detector2, NOT the reverse.
            delta = led_crossing - crossing
            output[start:end, j] = delta[:, 0] - delta[:, 1]
    return output, led_offset_ps, peak_ps
