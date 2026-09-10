from __future__ import annotations

import numpy as np

from .data import PreprocessedData


def family_arrays(data: PreprocessedData, family: str):
    if family == "energy":
        values = (
            data.energy_windows_mV,
            data.energy_window_start_time_s,
            data.energy_sample_interval_s,
            data.energy_rising_start,
            data.energy_rising_stop,
        )
    elif family == "timing":
        values = (
            data.timing_windows_mV,
            data.timing_window_start_time_s,
            data.timing_sample_interval_s,
            data.timing_rising_start,
            data.timing_rising_stop,
        )
    else:
        raise ValueError(f"Unknown waveform family: {family}")
    if any(value is None for value in values):
        raise ValueError(f"{family} preprocessed waveforms are unavailable")
    return values


def _crossing_ps(signal, start_time_s, interval_s, rising_start, rising_stop, level_mV) -> float:
    y = np.asarray(signal, dtype=np.float64)
    a, b = int(rising_start), int(rising_stop)
    if a < 0 or b >= y.size or b <= a or not np.isfinite(level_mV):
        return float("nan")
    y0, y1 = y[a:b], y[a + 1:b + 1]
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


def led_grid(
    data: PreprocessedData,
    family: str,
    indices: np.ndarray,
    thresholds_mV: np.ndarray,
) -> np.ndarray:
    waves, starts, intervals, rising_start, rising_stop = family_arrays(data, family)
    idx = np.asarray(indices, dtype=np.int64)
    thresholds = np.asarray(thresholds_mV, dtype=np.float64).reshape(-1)
    output = np.full((idx.size, 2, thresholds.size), np.nan, dtype=np.float64)
    for row, event in enumerate(idx):
        for detector in range(2):
            for column, threshold in enumerate(thresholds):
                output[row, detector, column] = _crossing_ps(
                    waves[event, detector],
                    starts[event, detector],
                    intervals[event, detector],
                    rising_start[event, detector],
                    rising_stop[event, detector],
                    float(threshold),
                )
    return output


def cfd_grid(
    data: PreprocessedData,
    family: str,
    indices: np.ndarray,
    fractions: np.ndarray,
) -> np.ndarray:
    waves, starts, intervals, rising_start, rising_stop = family_arrays(data, family)
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
                output[row, detector, column] = _crossing_ps(
                    signal,
                    starts[event, detector],
                    intervals[event, detector],
                    a,
                    b,
                    float(fraction) * peak,
                )
    return output



def interpolate_relative(
    signal: np.ndarray,
    start_time_s: float,
    interval_s: float,
    reference_time_ps: float,
    relative_time_ps: np.ndarray,
) -> np.ndarray:
    """Linearly sample a waveform on a continuous grid relative to a reference time."""
    y = np.asarray(signal, dtype=np.float64).reshape(-1)
    relative = np.asarray(relative_time_ps, dtype=np.float64).reshape(-1)
    interval = float(interval_s)
    reference_s = float(reference_time_ps) * 1.0e-12
    start = float(start_time_s)
    if y.size < 2 or interval <= 0.0 or not np.isfinite(reference_s):
        raise ValueError("Cannot interpolate waveform with invalid sampling metadata/reference time")
    source_time = start + np.arange(y.size, dtype=np.float64) * interval
    target_time = reference_s + relative * 1.0e-12
    tolerance = max(abs(interval) * 1.0e-9, 1.0e-18)
    if (
        np.any(~np.isfinite(target_time))
        or float(np.min(target_time)) < float(source_time[0]) - tolerance
        or float(np.max(target_time)) > float(source_time[-1]) + tolerance
    ):
        raise ValueError("Requested relative interpolation grid exceeds materialized waveform")
    return np.interp(target_time, source_time, y).astype(np.float64, copy=False)


def pair_delta(times_ps: np.ndarray) -> np.ndarray:
    values = np.asarray(times_ps, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 2:
        raise ValueError("Expected [event, detector] timing array")
    return values[:, 0] - values[:, 1]
