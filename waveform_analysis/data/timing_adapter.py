from __future__ import annotations

import numpy as np
from collections.abc import Sequence

from waveform_analysis.data.preprocessing import PreprocessedData
from waveform_analysis.signal.baseline import baseline_level_mV as _baseline_level_mV
from waveform_analysis.signal.timing import (
    crossing_ps as _crossing_ps,
    led_times_ps,
    cfd_times_ps,
    pair_delta as pair_delta,
)


def family_arrays(
    data: PreprocessedData, family: str
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
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


def _materialized_before_ns(data: PreprocessedData) -> float:
    window = data.manifest.get("materialized_window_ns") or {}
    value = float(window.get("before", np.nan))
    if not np.isfinite(value) or value < 0.0:
        raise ValueError(
            "Preprocessed manifest does not provide a valid materialized_window_ns.before"
        )
    return value


def led_grid(
    data: PreprocessedData,
    family: str,
    indices: np.ndarray,
    thresholds_mV: np.ndarray,
    *,
    baseline_window_ns: Sequence[float] | None = None,
) -> np.ndarray:
    """Adapt prepared acquisition arrays to the pure LED kernel."""
    arrays = family_arrays(data, family)
    before = _materialized_before_ns(data) if baseline_window_ns is not None else None
    return led_times_ps(
        *arrays,
        indices,
        thresholds_mV,
        baseline_window_ns=baseline_window_ns,
        materialized_before_ns=before,
    )


def cfd_grid(
    data: PreprocessedData, family: str, indices: np.ndarray, fractions: np.ndarray
) -> np.ndarray:
    """Adapt prepared acquisition arrays to the pure CFD kernel."""
    return cfd_times_ps(*family_arrays(data, family), indices, fractions)


def anchor_grid(
    data: PreprocessedData,
    family: str,
    threshold_mV: float,
    *,
    baseline_window_ns: Sequence[float] | None = None,
) -> np.ndarray:
    """Return native-grid anchor indices nearest to the interpolated LED crossing."""
    waves, starts, intervals, rising_start, rising_stop = family_arrays(data, family)
    indices = np.full((data.n_events, 2), -1, dtype=np.int32)
    before_ns = (
        _materialized_before_ns(data) if baseline_window_ns is not None else None
    )
    for event in range(data.n_events):
        for detector in range(2):
            start = float(starts[event, detector])
            interval = float(intervals[event, detector])
            a, b = int(rising_start[event, detector]), int(rising_stop[event, detector])
            baseline = 0.0
            if baseline_window_ns is not None:
                baseline = _baseline_level_mV(
                    waves[event, detector],
                    interval,
                    float(before_ns),
                    baseline_window_ns,
                )
                if not np.isfinite(baseline):
                    continue
            led_ps = _crossing_ps(
                waves[event, detector],
                start,
                interval,
                a,
                b,
                baseline + float(threshold_mV),
            )
            if not np.isfinite(led_ps) or interval <= 0.0 or b <= a:
                continue
            fractional = (led_ps * 1.0e-12 - start) / interval
            lower = int(np.floor(fractional))
            upper = lower + 1
            candidates = [sample for sample in (lower, upper) if a <= sample <= b]
            if not candidates:
                continue
            sample = min(
                candidates,
                key=lambda index: abs(
                    (start + float(index) * interval) * 1.0e12 - led_ps
                ),
            )
            indices[event, detector] = sample
    return indices


# Retain serialized identities while legacy imports resolve to this module.
from waveform_analysis.core.compat import (
    preserve_legacy_identity as _preserve_legacy_identity,
)

_preserve_legacy_identity(__name__, "waveform_analysis.ml_pipeline.timing")
