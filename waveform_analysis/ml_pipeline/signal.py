from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Any

import numpy as np

from utils.signal import (
    FEMTOSECONDS_PER_NANOSECOND,
    INVALID_TIME_FS,
    BasicFeatures,
)

from .denoising import apply_optional_lowpass_denoising


def _vertical_limits_mV(config: dict[str, Any] | None) -> tuple[float, float] | None:
    """Resolve optional oscilloscope vertical limits in mV.

    Accepted keys are intentionally permissive for configuration migration:
    ``vertical_scale_limit_mV`` (preferred), ``vertical_scale_limit`` and the
    literal user-facing spelling ``vertical scale limit``.
    """
    if not config:
        return None
    value = None
    for key in (
        "vertical_scale_limit_mV",
        "vertical_scale_limit",
        "vertical scale limit",
    ):
        if key in config:
            value = config[key]
            break
    if value is None:
        return None

    # A single pair remains supported for backward compatibility.  The
    # preferred form is family-specific because energy and timing channels can
    # use different oscilloscope vertical ranges.
    if isinstance(value, dict):
        family = str(config.get("_vertical_scale_family", "energy")).strip().lower()
        if family not in value:
            raise ValueError(
                "vertical_scale_limit_mV must define a limit for "
                f"the {family!r} channel family"
            )
        value = value[family]

    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ValueError(
            "vertical scale limit must contain exactly two values [v1, v2] in mV"
        )
    low, high = float(value[0]), float(value[1])
    if not (np.isfinite(low) and np.isfinite(high)) or not low < high:
        raise ValueError(
            "vertical scale limit must contain finite values with v1 < v2"
        )
    return low, high


def _decode_voltage_mV(
    raw_samples: np.ndarray,
    vertical_gain_v_per_count: float,
    vertical_offset_v: float,
    *,
    extraction_config: dict[str, Any] | None = None,
) -> np.ndarray:
    """Decode ADC samples and clamp them to the configured physical scale."""
    raw = np.asarray(raw_samples, dtype=np.float64)
    values = (
        raw * float(vertical_gain_v_per_count) - float(vertical_offset_v)
    ) * 1000.0
    limits = _vertical_limits_mV(extraction_config)
    if limits is not None:
        values = np.clip(values, limits[0], limits[1])
    return values


def discover_rising_crossings_relative_ns(
    raw_samples: np.ndarray,
    *,
    vertical_gain_v_per_count: float,
    vertical_offset_v: float,
    horizontal_interval_s: float,
    polarity: int,
    trigger_threshold_mV: float,
    vertical_scale_limit_mV: tuple[float, float] | list[float] | None = None,
) -> np.ndarray:
    """Return every coarse rising edge relative to the first acquired sample."""
    cfg = (
        {"vertical_scale_limit_mV": list(vertical_scale_limit_mV)}
        if vertical_scale_limit_mV is not None
        else None
    )
    voltage_mV = _decode_voltage_mV(
        raw_samples,
        vertical_gain_v_per_count,
        vertical_offset_v,
        extraction_config=cfg,
    )
    signal = float(polarity) * np.asarray(voltage_mV, dtype=np.float64)
    threshold = float(trigger_threshold_mV)
    if signal.size < 2 or not np.isfinite(threshold) or threshold <= 0.0:
        return np.empty(0, dtype=np.float64)

    y0, y1 = signal[:-1], signal[1:]
    finite = np.isfinite(y0) & np.isfinite(y1)
    rising = finite & (
        ((y0 < threshold) & (y1 >= threshold))
        | ((y0 == threshold) & (y1 > threshold))
    )
    lower = np.flatnonzero(rising)
    if lower.size == 0:
        return np.empty(0, dtype=np.float64)

    denom = y1[lower] - y0[lower]
    good = np.isfinite(denom) & (denom != 0.0)
    lower = lower[good]
    denom = denom[good]
    if lower.size == 0:
        return np.empty(0, dtype=np.float64)

    fraction = (threshold - y0[lower]) / denom
    good = np.isfinite(fraction) & (fraction >= 0.0) & (fraction <= 1.0)
    lower = lower[good]
    fraction = fraction[good]
    if lower.size == 0:
        return np.empty(0, dtype=np.float64)

    return (
        lower.astype(np.float64) + fraction
    ) * float(horizontal_interval_s) * 1.0e9


@dataclass(frozen=True)
class TimingReference:
    trigger_index: int
    led_time_fs: np.int64
    cfd_time_fs: np.int64
    valid: bool


@dataclass(frozen=True)
class ChannelExtraction:
    amplitude_mV: float
    noise_rms_mV: float
    trigger_index: int
    led_time_fs: np.int64
    cfd_time_fs: np.int64
    window_mV: np.ndarray
    window_anchor_time_fs: np.int64
    reference_aligned_window_mV: np.ndarray | None
    reference_aligned_window_anchor_time_fs: np.int64
    valid: bool


def relative_window_grid_ps(
    waveform_config: dict[str, Any],
    native_interval_s: float,
) -> np.ndarray:
    """Return native-sample offsets for the materialized trigger window."""
    interval_s = float(native_interval_s)
    if not np.isfinite(interval_s) or interval_s <= 0.0:
        raise ValueError("native_interval_s must be finite and positive")
    interval_ps = interval_s * 1.0e12
    before_ps = float(waveform_config["ml_window_ns"]["before"]) * 1000.0
    after_ps = float(waveform_config["ml_window_ns"]["after"]) * 1000.0
    before_samples = int(np.floor(before_ps / interval_ps + 1e-9))
    after_samples = int(np.floor(after_ps / interval_ps + 1e-9))
    offsets = np.arange(-before_samples, after_samples + 1, dtype=np.int64)
    if offsets.size < 4:
        raise ValueError("ML window contains fewer than four native samples")
    return offsets.astype(np.float64) * interval_ps


def timing_channel_waveform_config(
    waveform_config: dict[str, Any],
) -> dict[str, Any]:
    """Resolve timing-channel preprocessing without any fixed LED threshold."""
    override = waveform_config.get("timing_channel_led", {})
    if not isinstance(override, dict):
        raise ValueError("waveform.timing_channel_led must be an object")

    keys = (
        "baseline_window_ns",
        "subtract_baseline",
        "search_trigger_threshold_mV",
        "analysis_crop_ns",
        "ml_window_ns",
        "denoising",
        "trigger_matching",
        "_trigger_search_window_relative_ns",
        "vertical_scale_limit_mV",
        "vertical_scale_limit",
        "vertical scale limit",
    )
    resolved: dict[str, Any] = {}
    for key in keys:
        if key in override:
            resolved[key] = deepcopy(override[key])
        elif key in waveform_config:
            resolved[key] = deepcopy(waveform_config[key])

    resolved["_vertical_scale_family"] = "timing"
    return resolved

def _all_rising_crossings_ns(
    time_ns: np.ndarray,
    signal_mV: np.ndarray,
    threshold_mV: float,
) -> np.ndarray:
    x = np.asarray(time_ns, dtype=np.float64)
    y = np.asarray(signal_mV, dtype=np.float64)
    threshold = float(threshold_mV)
    if x.ndim != 1 or y.ndim != 1 or x.size != y.size or x.size < 2:
        return np.empty(0, dtype=np.float64)
    if not np.isfinite(threshold) or threshold <= 0.0:
        return np.empty(0, dtype=np.float64)

    x0, x1 = x[:-1], x[1:]
    y0, y1 = y[:-1], y[1:]
    finite = np.isfinite(x0) & np.isfinite(x1) & np.isfinite(y0) & np.isfinite(y1)
    rising = finite & (
        ((y0 < threshold) & (y1 >= threshold))
        | ((y0 == threshold) & (y1 > threshold))
    )
    lower = np.flatnonzero(rising)
    if lower.size == 0:
        return np.empty(0, dtype=np.float64)

    denom = y1[lower] - y0[lower]
    good = (x1[lower] > x0[lower]) & np.isfinite(denom) & (denom != 0.0)
    lower = lower[good]
    denom = denom[good]
    if lower.size == 0:
        return np.empty(0, dtype=np.float64)

    fraction = (threshold - y0[lower]) / denom
    good = np.isfinite(fraction) & (fraction >= 0.0) & (fraction <= 1.0)
    lower = lower[good]
    fraction = fraction[good]
    if lower.size == 0:
        return np.empty(0, dtype=np.float64)
    return x0[lower] + fraction * (x1[lower] - x0[lower])


def _first_rising_crossing_ns(
    time_ns: np.ndarray,
    signal_mV: np.ndarray,
    threshold_mV: float,
) -> float:
    crossings = _all_rising_crossings_ns(time_ns, signal_mV, threshold_mV)
    return float(crossings[0]) if crossings.size else np.nan


def _matched_rising_crossing_ns(
    time_ns: np.ndarray,
    signal_mV: np.ndarray,
    threshold_mV: float,
    *,
    reference_ns: float,
    prefer_before: bool,
) -> float:
    crossings = _all_rising_crossings_ns(time_ns, signal_mV, threshold_mV)
    if crossings.size == 0 or not np.isfinite(reference_ns):
        return np.nan
    x = np.asarray(time_ns, dtype=np.float64)
    dt_ns = float(np.median(np.diff(x))) if x.size >= 2 else 0.0
    tolerance = max(0.0, abs(dt_ns) * 1.5)
    if prefer_before:
        preferred = crossings[crossings <= float(reference_ns) + tolerance]
        if preferred.size:
            return float(preferred[-1])
    else:
        preferred = crossings[crossings >= float(reference_ns) - tolerance]
        if preferred.size:
            return float(preferred[0])
    return float(crossings[int(np.argmin(np.abs(crossings - float(reference_ns))))])


def _last_rising_crossing_before_peak_ns(
    time_ns: np.ndarray,
    signal_mV: np.ndarray,
    threshold_mV: float,
) -> float:
    x = np.asarray(time_ns, dtype=np.float64)
    y = np.asarray(signal_mV, dtype=np.float64)
    threshold = float(threshold_mV)
    if x.ndim != 1 or y.ndim != 1 or x.size != y.size or x.size < 2:
        return np.nan
    if not np.isfinite(threshold) or threshold <= 0.0:
        return np.nan
    peak_index = int(np.argmax(y))
    if peak_index <= 0 or not np.isfinite(y[peak_index]):
        return np.nan
    finite = (
        np.isfinite(x[:peak_index])
        & np.isfinite(x[1 : peak_index + 1])
        & np.isfinite(y[:peak_index])
        & np.isfinite(y[1 : peak_index + 1])
    )
    crossing = finite & (y[:peak_index] < threshold) & (
        y[1 : peak_index + 1] >= threshold
    )
    indices = np.flatnonzero(crossing)
    if indices.size == 0:
        crossing = finite & (y[:peak_index] == threshold) & (
            y[1 : peak_index + 1] > threshold
        )
        indices = np.flatnonzero(crossing)
    if indices.size == 0:
        return np.nan
    lower = int(indices[-1])
    x0, x1 = float(x[lower]), float(x[lower + 1])
    y0, y1 = float(y[lower]), float(y[lower + 1])
    if x1 <= x0 or y1 == y0:
        return np.nan
    fraction = (threshold - y0) / (y1 - y0)
    if not 0.0 <= fraction <= 1.0:
        return np.nan
    return x0 + fraction * (x1 - x0)


def _trigger_search_interval_ns(
    extraction_config: dict[str, Any],
) -> tuple[float, float] | None:
    value = extraction_config.get("_trigger_search_window_relative_ns")
    if value is None:
        return None
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ValueError(
            "_trigger_search_window_relative_ns must contain [start_ns, stop_ns]"
        )
    start_ns, stop_ns = float(value[0]), float(value[1])
    if not (np.isfinite(start_ns) and np.isfinite(stop_ns)) or stop_ns <= start_ns:
        raise ValueError(
            "_trigger_search_window_relative_ns must be finite with stop > start"
        )
    return start_ns, stop_ns


def _find_search_trigger(
    signal_mV: np.ndarray,
    *,
    threshold_mV: float,
    horizontal_interval_s: float,
    horizontal_offset_s: float,
    search_interval_ns: tuple[float, float] | None,
) -> tuple[int, np.int64]:
    """Find the coarse search trigger before any final baseline estimation.

    The returned index is the native upper sample of the selected threshold
    crossing. The interpolated crossing timestamp is kept for absolute trigger
    diagnostics, while the native sample is the materialization anchor.
    """
    y = np.asarray(signal_mV, dtype=np.float64)
    threshold = float(threshold_mV)
    dt_s = float(horizontal_interval_s)

    if y.ndim != 1 or y.size < 2:
        return -1, np.int64(INVALID_TIME_FS)
    if not np.isfinite(threshold) or threshold <= 0.0:
        return -1, np.int64(INVALID_TIME_FS)
    if not np.isfinite(dt_s) or dt_s <= 0.0 or not np.isfinite(horizontal_offset_s):
        return -1, np.int64(INVALID_TIME_FS)

    y0, y1 = y[:-1], y[1:]
    finite = np.isfinite(y0) & np.isfinite(y1)
    rising = finite & (
        ((y0 < threshold) & (y1 >= threshold))
        | ((y0 == threshold) & (y1 > threshold))
    )
    lower = np.flatnonzero(rising)
    if lower.size == 0:
        return -1, np.int64(INVALID_TIME_FS)

    denom = y1[lower] - y0[lower]
    good = np.isfinite(denom) & (denom != 0.0)
    lower = lower[good]
    denom = denom[good]
    if lower.size == 0:
        return -1, np.int64(INVALID_TIME_FS)

    fractions = (threshold - y0[lower]) / denom
    good = np.isfinite(fractions) & (fractions >= 0.0) & (fractions <= 1.0)
    lower = lower[good]
    fractions = fractions[good]
    if lower.size == 0:
        return -1, np.int64(INVALID_TIME_FS)

    crossing_sample = lower.astype(np.float64) + fractions

    if search_interval_ns is not None:
        start_ns, stop_ns = search_interval_ns
        crossing_relative_ns = crossing_sample * dt_s * 1.0e9
        inside = (
            (crossing_relative_ns >= start_ns)
            & (crossing_relative_ns <= stop_ns)
        )
        lower = lower[inside]
        fractions = fractions[inside]
        crossing_sample = crossing_sample[inside]
        crossing_relative_ns = crossing_relative_ns[inside]
        if lower.size == 0:
            return -1, np.int64(INVALID_TIME_FS)

        centre_ns = 0.5 * (start_ns + stop_ns)
        chosen = int(np.argmin(np.abs(crossing_relative_ns - centre_ns)))
    else:
        chosen = 0

    trigger_index = int(lower[chosen]) + 1
    trigger_time_s = (
        float(horizontal_offset_s)
        + float(crossing_sample[chosen]) * dt_s
    )
    trigger_time_fs = np.int64(np.rint(trigger_time_s * 1.0e15))
    return trigger_index, trigger_time_fs


def _basic_features(
    raw_samples: np.ndarray,
    *,
    vertical_gain_v_per_count: float,
    vertical_offset_v: float,
    horizontal_interval_s: float,
    horizontal_offset_s: float,
    polarity: int,
    extraction_config: dict[str, Any],
) -> BasicFeatures:
    """Decode/clip/orient the full trace and locate only the search trigger.

    Baseline mean and baseline RMSE are deliberately NOT estimated here. They are
    computed only after the trigger-centered materialized window exists.
    """
    if int(polarity) not in (-1, 1):
        raise ValueError("polarity must be +1 or -1")

    voltage_mV = _decode_voltage_mV(
        raw_samples,
        vertical_gain_v_per_count,
        vertical_offset_v,
        extraction_config=extraction_config,
    )
    oriented = float(polarity) * np.asarray(voltage_mV, dtype=np.float64)

    trigger_index, trigger_time_fs = _find_search_trigger(
        oriented,
        threshold_mV=float(extraction_config["search_trigger_threshold_mV"]),
        horizontal_interval_s=float(horizontal_interval_s),
        horizontal_offset_s=float(horizontal_offset_s),
        search_interval_ns=_trigger_search_interval_ns(extraction_config),
    )

    if np.any(np.isfinite(oriented)):
        peak_index = int(np.nanargmax(oriented))
        amplitude = float(oriented[peak_index])
    else:
        peak_index = -1
        amplitude = float("nan")

    return BasicFeatures(
        baseline_mV=float("nan"),
        noise_rms_mV=float("nan"),
        amplitude_mV=amplitude,
        peak_index=peak_index,
        trigger_index=int(trigger_index),
        trigger_time_fs=np.int64(trigger_time_fs),
        corrected_signal_mV=oriented,
    )


def _baseline_interval_ns(
    extraction_config: dict[str, Any],
) -> tuple[float, float]:
    """Return the baseline interval in ns relative to the native trigger anchor."""
    value = extraction_config.get("baseline_window_ns")
    if value is None:
        raise ValueError(
            "baseline_window_ns is required, e.g. "
            "{'start': -5.0, 'end': -1.0}"
        )

    if isinstance(value, dict):
        if "start" not in value or "end" not in value:
            raise ValueError(
                "baseline_window_ns must contain 'start' and 'end'"
            )
        start_ns, end_ns = float(value["start"]), float(value["end"])
    elif isinstance(value, (list, tuple)) and len(value) == 2:
        start_ns, end_ns = float(value[0]), float(value[1])
    else:
        raise ValueError(
            "baseline_window_ns must be {'start': ..., 'end': ...} or [start, end]"
        )

    if not (np.isfinite(start_ns) and np.isfinite(end_ns)) or end_ns <= start_ns:
        raise ValueError(
            "baseline_window_ns must be finite with end > start"
        )
    if end_ns > 0.0:
        raise ValueError(
            "baseline_window_ns must lie before or end at the trigger (end <= 0 ns)"
        )
    return start_ns, end_ns


def _baseline_from_materialized_window(
    window_mV: np.ndarray,
    relative_grid_ps: np.ndarray,
    extraction_config: dict[str, Any],
) -> tuple[float, float, np.ndarray]:
    """Compute baseline mean/RMSE exclusively on the materialized waveform."""
    start_ns, end_ns = _baseline_interval_ns(extraction_config)
    time_ns = np.asarray(relative_grid_ps, dtype=np.float64) / 1000.0
    window = np.asarray(window_mV, dtype=np.float64)

    if window.ndim != 1 or window.size != time_ns.size:
        raise ValueError(
            "materialized waveform and relative-time grid must have equal length"
        )

    dt_ns = (
        abs(float(np.median(np.diff(time_ns))))
        if time_ns.size >= 2
        else 0.0
    )
    tolerance = max(1e-12, 0.51 * dt_ns)
    mask = (
        (time_ns >= start_ns - tolerance)
        & (time_ns <= end_ns + tolerance)
        & np.isfinite(window)
    )
    values = window[mask]
    if values.size < 2:
        raise ValueError(
            "baseline_window_ns contains fewer than two finite materialized samples"
        )

    baseline_mean = float(np.mean(values))
    baseline_rmse = float(
        np.sqrt(np.mean((values - baseline_mean) ** 2))
    )
    centered = window - baseline_mean
    return baseline_mean, baseline_rmse, centered


def _materialized_amplitude(
    centered_window_mV: np.ndarray,
    relative_grid_ps: np.ndarray,
    extraction_config: dict[str, Any],
) -> float:
    """Measure photopeak amplitude after baseline estimation on the materialized window."""
    signal = np.asarray(centered_window_mV, dtype=np.float64)
    time_ns = np.asarray(relative_grid_ps, dtype=np.float64) / 1000.0
    crop = extraction_config.get("analysis_crop_ns")

    if isinstance(crop, dict):
        before_ns = float(crop["before"])
        after_ns = float(crop["after"])
        mask = (
            (time_ns >= -before_ns - 1e-12)
            & (time_ns <= after_ns + 1e-12)
            & np.isfinite(signal)
        )
    else:
        mask = np.isfinite(signal)

    indices = np.flatnonzero(mask)
    if indices.size == 0:
        return float("nan")
    return float(np.nanmax(signal[indices]))


def extract_timing_reference(
    raw_samples: np.ndarray,
    *,
    vertical_gain_v_per_count: float,
    vertical_offset_v: float,
    horizontal_interval_s: float,
    horizontal_offset_s: float,
    polarity: int,
    waveform_config: dict[str, Any],
) -> TimingReference:
    """Compatibility helper exposing only the search-trigger reference."""
    extraction_config = timing_channel_waveform_config(waveform_config)
    basic = _basic_features(
        np.asarray(raw_samples, dtype=np.int16),
        vertical_gain_v_per_count=vertical_gain_v_per_count,
        vertical_offset_v=vertical_offset_v,
        horizontal_interval_s=horizontal_interval_s,
        horizontal_offset_s=horizontal_offset_s,
        polarity=polarity,
        extraction_config=extraction_config,
    )
    valid = (
        basic.trigger_index >= 0
        and int(basic.trigger_time_fs) != int(INVALID_TIME_FS)
    )
    return TimingReference(
        trigger_index=int(basic.trigger_index),
        led_time_fs=np.int64(INVALID_TIME_FS),
        cfd_time_fs=np.int64(INVALID_TIME_FS),
        valid=bool(valid),
    )

def _native_window(
    signal_mV: np.ndarray,
    *,
    horizontal_interval_s: float,
    horizontal_offset_s: float,
    alignment_ns: float,
    relative_grid_ps: np.ndarray,
    return_anchor: bool = False,
) -> np.ndarray | None | tuple[np.ndarray | None, np.int64]:
    interval_s = float(horizontal_interval_s)
    interval_ps = interval_s * 1.0e12
    relative = np.asarray(relative_grid_ps, dtype=np.float64)
    sample_offsets = np.rint(relative / interval_ps).astype(np.int64)
    if not np.allclose(
        relative,
        sample_offsets.astype(np.float64) * interval_ps,
        rtol=0.0,
        atol=max(1e-6, abs(interval_ps) * 1e-9),
    ):
        raise ValueError("Requested ML window is not on the waveform's native sample grid")
    first_time_ns = float(horizontal_offset_s) * 1.0e9
    interval_ns = interval_s * 1.0e9
    anchor = int(np.rint((float(alignment_ns) - first_time_ns) / interval_ns))
    indices = anchor + sample_offsets
    anchor_time_s = float(horizontal_offset_s) + anchor * interval_s
    anchor_time_fs = np.int64(np.rint(anchor_time_s * 1.0e15))
    if indices.size == 0 or int(indices[0]) < 0 or int(indices[-1]) >= signal_mV.size:
        return (None, anchor_time_fs) if return_anchor else None
    window = np.asarray(signal_mV[indices], dtype=np.float32).copy()
    return (window, anchor_time_fs) if return_anchor else window


def _native_trigger_alignment_ns(
    trigger_index: int,
    *,
    horizontal_interval_s: float,
    horizontal_offset_s: float,
) -> float:
    """Absolute time of the native acquisition sample selected by search trigger."""
    return (
        float(horizontal_offset_s)
        + int(trigger_index) * float(horizontal_interval_s)
    ) * 1.0e9


def extract_channel(
    raw_samples: np.ndarray,
    *,
    vertical_gain_v_per_count: float,
    vertical_offset_v: float,
    horizontal_interval_s: float,
    horizontal_offset_s: float,
    polarity: int,
    waveform_config: dict[str, Any],
    relative_grid_ps: np.ndarray,
    timing_reference: TimingReference | None = None,
    compute_cfd: bool = False,
) -> ChannelExtraction:
    """Trigger -> materialize -> baseline/RMSE -> optional baseline subtraction.

    No fixed LED threshold and no fixed CFD estimator are evaluated during
    preprocessing. Those are selected later from the candidate sets using the
    prepared trigger-referenced waveforms.
    """
    del compute_cfd  # retained only for compatibility with current data.py

    invalid_window = np.full(
        np.asarray(relative_grid_ps).shape,
        np.nan,
        dtype=np.float32,
    )

    basic = _basic_features(
        np.asarray(raw_samples, dtype=np.int16),
        vertical_gain_v_per_count=vertical_gain_v_per_count,
        vertical_offset_v=vertical_offset_v,
        horizontal_interval_s=horizontal_interval_s,
        horizontal_offset_s=horizontal_offset_s,
        polarity=polarity,
        extraction_config=waveform_config,
    )

    invalid = ChannelExtraction(
        amplitude_mV=float("nan"),
        noise_rms_mV=float("nan"),
        trigger_index=int(basic.trigger_index),
        led_time_fs=np.int64(INVALID_TIME_FS),
        cfd_time_fs=np.int64(INVALID_TIME_FS),
        window_mV=invalid_window,
        window_anchor_time_fs=np.int64(INVALID_TIME_FS),
        reference_aligned_window_mV=None,
        reference_aligned_window_anchor_time_fs=np.int64(INVALID_TIME_FS),
        valid=False,
    )

    if basic.trigger_index < 0:
        return invalid

    alignment_ns = _native_trigger_alignment_ns(
        basic.trigger_index,
        horizontal_interval_s=horizontal_interval_s,
        horizontal_offset_s=horizontal_offset_s,
    )
    materialized, anchor_fs = _native_window(
        np.asarray(basic.corrected_signal_mV, dtype=np.float64),
        horizontal_interval_s=horizontal_interval_s,
        horizontal_offset_s=horizontal_offset_s,
        alignment_ns=alignment_ns,
        relative_grid_ps=relative_grid_ps,
        return_anchor=True,
    )
    if materialized is None or np.any(~np.isfinite(materialized)):
        return invalid

    baseline_mean, baseline_rmse, centered = _baseline_from_materialized_window(
        materialized,
        relative_grid_ps,
        waveform_config,
    )

    amplitude = _materialized_amplitude(
        centered,
        relative_grid_ps,
        waveform_config,
    )
    if not (
        np.isfinite(baseline_mean)
        and np.isfinite(baseline_rmse)
        and np.isfinite(amplitude)
    ):
        return invalid

    final_window = (
        np.asarray(centered, dtype=np.float32)
        if bool(waveform_config.get("subtract_baseline", True))
        else np.asarray(materialized, dtype=np.float32)
    )
    if np.any(~np.isfinite(final_window)):
        return invalid

    # Compatibility only: current data.py still has historical fields for a
    # "timing-aligned" energy copy. Give it the SAME trigger-centered window so
    # it does not introduce any LED dependence. prepared_data.py does not copy
    # this duplicate into the permanent dataset.
    reference_window: np.ndarray | None = None
    reference_anchor_fs = np.int64(INVALID_TIME_FS)
    if timing_reference is not None:
        reference_window = final_window.copy()
        reference_anchor_fs = np.int64(anchor_fs)

    return ChannelExtraction(
        amplitude_mV=float(amplitude),
        noise_rms_mV=float(baseline_rmse),
        trigger_index=int(basic.trigger_index),
        led_time_fs=np.int64(INVALID_TIME_FS),
        cfd_time_fs=np.int64(INVALID_TIME_FS),
        window_mV=final_window,
        window_anchor_time_fs=np.int64(anchor_fs),
        reference_aligned_window_mV=reference_window,
        reference_aligned_window_anchor_time_fs=reference_anchor_fs,
        valid=True,
    )


def extract_timing_channel(
    raw_samples: np.ndarray,
    *,
    vertical_gain_v_per_count: float,
    vertical_offset_v: float,
    horizontal_interval_s: float,
    horizontal_offset_s: float,
    polarity: int,
    waveform_config: dict[str, Any],
    relative_grid_ps: np.ndarray,
) -> ChannelExtraction:
    """Use the same trigger-first preprocessing for timing channels."""
    resolved = timing_channel_waveform_config(waveform_config)
    return extract_channel(
        raw_samples,
        vertical_gain_v_per_count=vertical_gain_v_per_count,
        vertical_offset_v=vertical_offset_v,
        horizontal_interval_s=horizontal_interval_s,
        horizontal_offset_s=horizontal_offset_s,
        polarity=polarity,
        waveform_config=resolved,
        relative_grid_ps=relative_grid_ps,
        timing_reference=None,
        compute_cfd=False,
    )
