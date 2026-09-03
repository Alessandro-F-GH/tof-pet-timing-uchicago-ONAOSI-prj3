from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Iterable

import numpy as np

from utils.signal import INVALID_TIME_FS, prepare_timing_features

from ..dataset import PreparedDataset
from ..metrics import residual_metrics


@dataclass(frozen=True)
class FamilySelection:
    family: str
    cfd_enabled: bool
    led_threshold_mV: float
    cfd_fraction: float
    led_validation_sctr_ps: float
    cfd_validation_sctr_ps: float
    led_fold_sctr_ps: tuple[float, ...]
    cfd_fold_sctr_ps: tuple[float, ...]
    led_times_fs: np.ndarray
    cfd_times_fs: np.ndarray
    led_search_low_mV: float
    led_search_high_mV: float
    led_coarse_points: int
    led_refine_points: int
    cfd_coarse_points: int
    cfd_refine_points: int

    def as_dict(self) -> dict[str, Any]:
        return {
            "family": self.family,
            "cfd_enabled": bool(self.cfd_enabled),
            "led_threshold_mV": float(self.led_threshold_mV),
            "cfd_fraction": (
                float(self.cfd_fraction) if self.cfd_enabled else None
            ),
            "led_validation_sctr_ps": float(self.led_validation_sctr_ps),
            "cfd_validation_sctr_ps": (
                float(self.cfd_validation_sctr_ps) if self.cfd_enabled else None
            ),
            "led_fold_sctr_ps": [float(v) for v in self.led_fold_sctr_ps],
            "cfd_fold_sctr_ps": (
                [float(v) for v in self.cfd_fold_sctr_ps]
                if self.cfd_enabled else []
            ),
            "led_search_low_mV": float(self.led_search_low_mV),
            "led_search_high_mV": float(self.led_search_high_mV),
            "led_coarse_points": int(self.led_coarse_points),
            "led_refine_points": int(self.led_refine_points),
            "cfd_coarse_points": int(self.cfd_coarse_points),
            "cfd_refine_points": int(self.cfd_refine_points),
            "selection_metric": "mean_fold_sctr_sample_std",
            "blind_used_for_selection": False,
        }

class ShiftedWaveformArray:
    """Lazy native-sample re-alignment without duplicating prepared waveforms."""

    def __init__(self, base: np.ndarray, shifts_samples: np.ndarray) -> None:
        self.base = base
        self.shifts_samples = np.asarray(shifts_samples, dtype=np.int64)
        if len(base.shape) != 3 or int(base.shape[1]) != 2:
            raise ValueError("ShiftedWaveformArray expects [event, detector, sample]")
        if self.shifts_samples.shape != tuple(base.shape[:2]):
            raise ValueError("Per-event shifts must have shape [event, detector]")
        self.shape = tuple(base.shape)
        self.dtype = np.dtype(getattr(base, "dtype", np.float32))

    def __getitem__(self, key: Any) -> np.ndarray:
        keys = key if isinstance(key, tuple) else (key,)
        keys = (*keys, *([slice(None)] * (3 - len(keys))))
        if len(keys) != 3:
            raise IndexError("Waveform indexing supports at most three axes")
        event_key, detector_key, sample_key = keys

        all_events = np.arange(self.shape[0], dtype=np.int64)
        all_detectors = np.arange(self.shape[1], dtype=np.int64)
        all_samples = np.arange(self.shape[2], dtype=np.int64)
        event_scalar = isinstance(event_key, (int, np.integer))
        detector_scalar = isinstance(detector_key, (int, np.integer))
        sample_scalar = isinstance(sample_key, (int, np.integer))
        events = np.atleast_1d(all_events[event_key]).astype(np.int64, copy=False)
        detectors = np.atleast_1d(all_detectors[detector_key]).astype(np.int64, copy=False)
        samples = np.atleast_1d(all_samples[sample_key]).astype(np.int64, copy=False)

        shifts = self.shifts_samples[np.ix_(events, detectors)]
        source_samples = samples[None, None, :] + shifts[:, :, None]
        valid = (source_samples >= 0) & (source_samples < self.shape[2])
        clipped = np.clip(source_samples, 0, self.shape[2] - 1)
        event_grid = events[:, None, None]
        detector_grid = detectors[None, :, None]
        values = np.asarray(
            self.base[event_grid, detector_grid, clipped],
            dtype=self.dtype,
        )
        if not np.all(valid):
            values = values.copy()
            values[~valid] = np.nan
        if sample_scalar:
            values = np.squeeze(values, axis=2)
        if detector_scalar:
            values = np.squeeze(values, axis=1)
        if event_scalar:
            values = np.squeeze(values, axis=0)
        return values


def _alignment_shifts(
    selected_led_fs: np.ndarray,
    source_anchor_fs: np.ndarray,
    relative_time_ps: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    selected = np.asarray(selected_led_fs, dtype=np.int64)
    anchors = np.asarray(source_anchor_fs, dtype=np.int64)
    if selected.shape != anchors.shape or selected.ndim != 2 or selected.shape[1] != 2:
        raise ValueError("Selected LED and source anchors must have shape [event,2]")
    t = np.asarray(relative_time_ps, dtype=np.float64)
    if t.size < 2:
        raise ValueError("Cannot re-align a waveform with fewer than two time samples")
    dt_fs = int(np.rint(float(np.median(np.diff(t))) * 1000.0))
    if dt_fs <= 0:
        raise ValueError("Prepared waveform sample interval must be positive")
    shifts = np.rint(
        (selected.astype(np.float64) - anchors.astype(np.float64)) / float(dt_fs)
    ).astype(np.int64)
    new_anchors = anchors + shifts * np.int64(dt_fs)
    return shifts, new_anchors


def _check_shift_support(
    shifts: np.ndarray,
    relative_time_ps: np.ndarray,
    windows: list[dict[str, Any]],
    *,
    label: str,
) -> None:
    t_ns = np.asarray(relative_time_ps, dtype=np.float64) / 1000.0
    low = min(float(window["start_ns"]) for window in windows)
    high = max(float(window["end_ns"]) for window in windows)
    selected = np.flatnonzero((t_ns >= low - 1e-9) & (t_ns <= high + 1e-9))
    if selected.size == 0:
        raise ValueError(f"No prepared samples cover experiment windows for {label}")
    first = int(selected[0])
    last = int(selected[-1])
    minimum = int(np.min(shifts))
    maximum = int(np.max(shifts))
    if first + minimum < 0 or last + maximum >= t_ns.size:
        raise RuntimeError(
            f"Prepared {label} waveform has insufficient alignment padding for the "
            f"selected LED (sample shifts {minimum}..{maximum}). Increase "
            "standard_methods.alignment_padding_ns and rebuild preprocessing."
        )


def family_for_mode(mode: str) -> str:
    key = str(mode)
    return "timing" if key.endswith("_to_timing") or key == "timing_to_timing" else "energy"


def cfd_enabled_for_family(config: dict[str, Any], family: str) -> bool:
    standard = config.get("standard_methods", {}) or {}
    mapping = standard.get("cfd_enabled_by_family", {}) or {}
    if not isinstance(mapping, dict):
        raise ValueError("standard_methods.cfd_enabled_by_family must be an object")
    return bool(mapping.get(str(family), True))


def cfd_enabled_for_mode(config: dict[str, Any], mode: str) -> bool:
    return cfd_enabled_for_family(config, family_for_mode(mode))

def _family_timing_config(
    config: dict[str, Any], family: str
) -> tuple[float, float, float]:
    preprocessing = config["preprocessing"]
    common = dict(preprocessing.get("common", {}) or {})
    resolved = dict(common)
    resolved.update(dict(preprocessing.get(family, {}) or {}))

    crop = resolved.get(
        "analysis_crop_ns", {"before": 2.0, "after": 40.0}
    )
    before_ns = float(crop["before"])
    after_ns = float(crop["after"])
    reference_threshold_mV = float(
        resolved["search_trigger_threshold_mV"]
    )
    if before_ns <= 0.0 or after_ns <= 0.0:
        raise ValueError(
            f"preprocessing.{family}.analysis_crop_ns must be positive"
        )
    if not np.isfinite(reference_threshold_mV):
        raise ValueError(
            f"preprocessing.{family}.search_trigger_threshold_mV "
            "must be finite"
        )
    return before_ns, after_ns, reference_threshold_mV


def _analysis_bounds(
    relative_time_ps: np.ndarray,
    *,
    reference_time_ps: float,
    before_ns: float,
    after_ns: float,
) -> tuple[int, int]:
    t = np.asarray(relative_time_ps, dtype=np.float64)
    start = max(
        0,
        int(
            np.searchsorted(
                t,
                float(reference_time_ps) - float(before_ns) * 1000.0,
                side="left",
            )
        )
        - 1,
    )
    stop = min(
        t.size,
        int(
            np.searchsorted(
                t,
                float(reference_time_ps) + float(after_ns) * 1000.0,
                side="right",
            )
        )
        + 1,
    )
    if stop - start < 3:
        raise RuntimeError(
            "analysis_crop_ns contains fewer than three native samples"
        )
    return start, stop


def _pulse_peak_after_reference(
    signal_mV: np.ndarray,
    relative_time_ps: np.ndarray,
    *,
    reference_time_ps: float,
    before_ns: float,
    after_ns: float,
) -> tuple[int, int, int, float]:
    y = np.asarray(signal_mV, dtype=np.float64)
    t = np.asarray(relative_time_ps, dtype=np.float64)
    start, stop = _analysis_bounds(
        t,
        reference_time_ps=reference_time_ps,
        before_ns=before_ns,
        after_ns=after_ns,
    )
    ref_sample = int(
        np.clip(
            np.searchsorted(
                t, float(reference_time_ps), side="right"
            )
            - 1,
            start,
            stop - 2,
        )
    )
    peak_search_start = min(stop - 1, ref_sample + 1)
    local = y[peak_search_start:stop]
    if local.size == 0 or not np.any(np.isfinite(local)):
        raise RuntimeError(
            "Cannot identify pulse peak after preprocessing reference edge"
        )
    peak = peak_search_start + int(np.nanargmax(local))
    return start, stop, ref_sample, float(y[peak])



def _interpolate_level_crossings_into(
    output_fs: np.ndarray,
    output_indices: np.ndarray,
    levels_mV: np.ndarray,
    relative_time_ps: np.ndarray,
    signal_mV: np.ndarray,
    lower_indices: np.ndarray,
) -> None:
    targets = np.asarray(output_indices, dtype=np.int64).reshape(-1)
    levels = np.asarray(levels_mV, dtype=np.float64).reshape(-1)
    lower = np.asarray(lower_indices, dtype=np.int64).reshape(-1)

    if targets.size == 0:
        return

    t = np.asarray(relative_time_ps, dtype=np.float64)
    y = np.asarray(signal_mV, dtype=np.float64)
    upper = lower + 1

    y0 = y[lower]
    y1 = y[upper]
    t0 = t[lower]
    t1 = t[upper]
    denominator = y1 - y0

    valid = (
        np.isfinite(levels)
        & np.isfinite(y0)
        & np.isfinite(y1)
        & np.isfinite(t0)
        & np.isfinite(t1)
        & (denominator != 0.0)
        & (t1 > t0)
        & (y0 < levels)
        & (levels <= y1)
    )

    if not np.any(valid):
        return

    fraction = np.full(levels.shape, np.nan, dtype=np.float64)
    fraction[valid] = (
        (levels[valid] - y0[valid])
        / denominator[valid]
    )

    valid &= (
        np.isfinite(fraction)
        & (fraction >= 0.0)
        & (fraction <= 1.0)
    )

    if not np.any(valid):
        return

    crossing_ps = (
        t0[valid]
        + fraction[valid]
        * (t1[valid] - t0[valid])
    )

    output_fs[targets[valid]] = np.rint(
        crossing_ps * 1000.0
    ).astype(np.int64)


def _same_edge_level_times_fs(
    signal_mV: np.ndarray,
    relative_time_ps: np.ndarray,
    levels_mV: np.ndarray,
    *,
    reference_threshold_mV: float,
    reference_time_ps: float,
    before_ns: float,
    after_ns: float,
) -> np.ndarray:
    # Vectorized over all LED/CFD candidate levels.
    y = np.asarray(signal_mV, dtype=np.float64)
    t = np.asarray(relative_time_ps, dtype=np.float64)
    levels = np.asarray(levels_mV, dtype=np.float64).reshape(-1)

    result = np.full(
        levels.shape,
        INVALID_TIME_FS,
        dtype=np.int64,
    )

    if (
        y.ndim != 1
        or t.ndim != 1
        or y.size != t.size
        or y.size < 3
        or levels.size == 0
    ):
        return result

    (
        start,
        stop,
        ref_sample,
        _peak_value,
    ) = _pulse_peak_after_reference(
        y,
        t,
        reference_time_ps=reference_time_ps,
        before_ns=before_ns,
        after_ns=after_ns,
    )

    ref_upper = int(
        np.searchsorted(
            t,
            float(reference_time_ps),
            side="right",
        )
    )
    ref_upper = int(
        np.clip(
            ref_upper,
            max(1, start + 1),
            stop - 1,
        )
    )

    peak_search_start = min(
        stop - 1,
        max(ref_sample + 1, ref_upper),
    )
    peak_region = y[peak_search_start:stop]

    if (
        peak_region.size == 0
        or not np.all(np.isfinite(peak_region))
    ):
        return result

    peak = (
        peak_search_start
        + int(np.argmax(peak_region))
    )

    finite_levels = (
        np.isfinite(levels)
        & (levels > 0.0)
    )

    equal_mask = (
        finite_levels
        & np.isclose(
            levels,
            float(reference_threshold_mV),
            rtol=0.0,
            atol=1e-12,
        )
    )
    result[equal_mask] = np.int64(
        np.rint(float(reference_time_ps) * 1000.0)
    )

    low_mask = (
        finite_levels
        & (levels < float(reference_threshold_mV))
    )

    if np.any(low_mask):
        segment = y[start:ref_upper + 1]

        if (
            segment.size >= 2
            and np.all(np.isfinite(segment))
        ):
            # Monotonic suffix minimum:
            # first position with suffix_min >= L is the beginning of the
            # continuous above-L excursion connected to Tsearch.
            suffix_min = np.minimum.accumulate(
                segment[::-1]
            )[::-1]

            low_levels = levels[low_mask]
            positions = np.searchsorted(
                suffix_min,
                low_levels,
                side="left",
            )

            low_indices = np.flatnonzero(low_mask)
            valid_position = (
                (positions > 0)
                & (positions < segment.size)
            )

            if np.any(valid_position):
                target_indices = low_indices[
                    valid_position
                ]
                upper = (
                    start
                    + positions[valid_position]
                )
                lower = upper - 1

                _interpolate_level_crossings_into(
                    result,
                    target_indices,
                    levels[target_indices],
                    t,
                    y,
                    lower,
                )

    high_mask = (
        finite_levels
        & (levels > float(reference_threshold_mV))
    )

    if np.any(high_mask) and peak >= ref_upper:
        post = y[ref_upper:peak + 1]

        if (
            post.size >= 1
            and np.all(np.isfinite(post))
        ):
            # Monotonic prefix maximum:
            # first position with prefix_max >= L is the first crossing after
            # the search-threshold reference.
            prefix_max = np.maximum.accumulate(post)

            high_levels = levels[high_mask]
            positions = np.searchsorted(
                prefix_max,
                high_levels,
                side="left",
            )

            high_indices = np.flatnonzero(high_mask)
            valid_position = positions < post.size

            if np.any(valid_position):
                target_indices = high_indices[
                    valid_position
                ]
                upper = (
                    ref_upper
                    + positions[valid_position]
                )
                lower = upper - 1

                valid_lower = lower >= start
                if np.any(valid_lower):
                    _interpolate_level_crossings_into(
                        result,
                        target_indices[valid_lower],
                        levels[
                            target_indices[
                                valid_lower
                            ]
                        ],
                        t,
                        y,
                        lower[valid_lower],
                    )

    return result

def _family_arrays(
    dataset: PreparedDataset, family: str
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if family == "energy":
        waves = dataset.windows_mV
        times = dataset.relative_time_ps
        anchors = dataset.energy_window_anchor_time_fs
    elif family == "timing":
        if dataset.timing_windows_mV is None or dataset.timing_relative_time_ps is None:
            raise ValueError("Adaptive timing standard methods require timing waveforms")
        waves = dataset.timing_windows_mV
        times = dataset.timing_relative_time_ps
        anchors = dataset.timing_window_anchor_time_fs
    else:
        raise ValueError(f"Unknown standard-method family {family!r}")

    if anchors is None:
        raise ValueError(
            f"{family} waveform anchors are unavailable; rebuild preprocessing"
        )

    return (
        np.asarray(waves),
        np.asarray(times, dtype=np.float64),
        np.asarray(anchors, dtype=np.int64),
    )


def _first_rising_level_time_fs(
    relative_time_ps: np.ndarray,
    signal_mV: np.ndarray,
    threshold_mV: float,
) -> np.int64:
    t = np.asarray(relative_time_ps, dtype=np.float64)
    y = np.asarray(signal_mV, dtype=np.float64)
    threshold = float(threshold_mV)

    if t.ndim != 1 or y.ndim != 1 or t.size != y.size or t.size < 2:
        return np.int64(INVALID_TIME_FS)
    if not np.isfinite(threshold) or threshold <= 0.0:
        return np.int64(INVALID_TIME_FS)

    finite = (
        np.isfinite(t[:-1])
        & np.isfinite(t[1:])
        & np.isfinite(y[:-1])
        & np.isfinite(y[1:])
    )
    mask = finite & (
        ((y[:-1] < threshold) & (y[1:] >= threshold))
        | ((y[:-1] == threshold) & (y[1:] > threshold))
    )
    candidates = np.flatnonzero(mask)
    if candidates.size == 0:
        return np.int64(INVALID_TIME_FS)

    lower = int(candidates[0])
    y0, y1 = float(y[lower]), float(y[lower + 1])
    t0, t1 = float(t[lower]), float(t[lower + 1])

    if y1 == y0 or t1 <= t0:
        return np.int64(INVALID_TIME_FS)

    fraction = (threshold - y0) / (y1 - y0)
    if not np.isfinite(fraction) or not 0.0 <= fraction <= 1.0:
        return np.int64(INVALID_TIME_FS)

    return np.int64(np.rint((t0 + fraction * (t1 - t0)) * 1000.0))



def _search_reference_times_fs(
    waves: np.ndarray,
    relative_time_ps: np.ndarray,
    anchors_fs: np.ndarray,
    *,
    reference_threshold_mV: float,
    chunk_size: int,
) -> np.ndarray:
    # Vectorized first search-threshold crossing per event/detector.
    t = np.asarray(relative_time_ps, dtype=np.float64)
    anchors = np.asarray(anchors_fs, dtype=np.int64)
    threshold = float(reference_threshold_mV)

    n_events = int(waves.shape[0])
    result = np.full((n_events, 2), INVALID_TIME_FS, dtype=np.int64)

    if t.ndim != 1 or t.size < 2:
        return result
    if not np.isfinite(threshold) or threshold <= 0.0:
        return result

    step = max(1, int(chunk_size))

    for first in range(0, n_events, step):
        stop = min(n_events, first + step)
        block = np.asarray(waves[first:stop])

        if (
            block.ndim != 3
            or block.shape[1] != 2
            or block.shape[2] != t.size
        ):
            raise RuntimeError(
                "Prepared standard-method waveform shape does not match "
                "relative_time_ps"
            )

        y0 = block[:, :, :-1]
        y1 = block[:, :, 1:]

        finite = np.isfinite(y0) & np.isfinite(y1)
        crossing = finite & (
            ((y0 < threshold) & (y1 >= threshold))
            | ((y0 == threshold) & (y1 > threshold))
        )

        has_crossing = np.any(crossing, axis=2)
        if not np.any(has_crossing):
            continue

        lower = np.argmax(crossing, axis=2).astype(
            np.int64,
            copy=False,
        )

        row = np.arange(stop - first, dtype=np.int64)[:, None]
        detector = np.arange(2, dtype=np.int64)[None, :]

        y0_sel = block[row, detector, lower].astype(
            np.float64,
            copy=False,
        )
        y1_sel = block[row, detector, lower + 1].astype(
            np.float64,
            copy=False,
        )
        t0 = t[lower]
        t1 = t[lower + 1]

        denominator = y1_sel - y0_sel
        fraction = np.full(
            denominator.shape,
            np.nan,
            dtype=np.float64,
        )

        good = (
            has_crossing
            & np.isfinite(denominator)
            & (denominator != 0.0)
            & np.isfinite(t0)
            & np.isfinite(t1)
            & (t1 > t0)
        )

        fraction[good] = (
            (threshold - y0_sel[good])
            / denominator[good]
        )

        good &= (
            np.isfinite(fraction)
            & (fraction >= 0.0)
            & (fraction <= 1.0)
        )

        local_fs = np.full(
            denominator.shape,
            INVALID_TIME_FS,
            dtype=np.int64,
        )
        local_fs[good] = np.rint(
            (
                t0[good]
                + fraction[good]
                * (t1[good] - t0[good])
            )
            * 1000.0
        ).astype(np.int64)

        absolute = anchors[first:stop] + local_fs
        out = result[first:stop]
        out[good] = absolute[good]
        result[first:stop] = out

    return result

def _robust_center_sigma(values: np.ndarray) -> tuple[float, float]:
    data = np.asarray(values, dtype=np.float64)
    data = data[np.isfinite(data)]
    if data.size == 0:
        return float("nan"), float("nan")

    center = float(np.median(data))
    mad = float(np.median(np.abs(data - center)))
    sigma = 1.4826 * mad

    if not np.isfinite(sigma) or sigma <= 0.0:
        sigma = float(np.std(data, ddof=1)) if data.size > 1 else 0.0

    return center, sigma


def filter_search_time_outliers(
    config: dict[str, Any],
    dataset: PreparedDataset,
    development: np.ndarray,
    blind: np.ndarray,
    *,
    family: str,
    logger: Any,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    standard = config.get("standard_methods", {}) or {}
    rejection = standard.get("search_time_outlier_rejection", {}) or {}
    enabled = bool(rejection.get("enabled", True))
    z_limit = float(rejection.get("zscore_limit", 4.0))

    waves, times, anchors = _family_arrays(dataset, family)
    _before_ns, _after_ns, reference_threshold_mV = _family_timing_config(
        config, family
    )
    chunk_size = int(standard.get("waveform_scan_chunk_size", 1024))

    references = _search_reference_times_fs(
        waves,
        times,
        anchors,
        reference_threshold_mV=reference_threshold_mV,
        chunk_size=chunk_size,
    )

    valid = np.all(references != INVALID_TIME_FS, axis=1)
    residual_ps = (
        (
            references[:, 0].astype(np.float64)
            - references[:, 1].astype(np.float64)
        )
        / 1000.0
        - float(dataset.true_tof_ps)
    )
    valid &= np.isfinite(residual_ps)

    development = np.asarray(development, dtype=np.int64)
    blind = np.asarray(blind, dtype=np.int64)

    fit_indices = development[valid[development]]
    if fit_indices.size < 3:
        raise RuntimeError(
            f"Too few finite {family} search-threshold timing pairs in development"
        )

    center, sigma = _robust_center_sigma(residual_ps[fit_indices])
    if not np.isfinite(center):
        raise RuntimeError(f"Cannot estimate {family} search-time center")

    if enabled:
        if not np.isfinite(z_limit) or z_limit <= 0.0:
            raise ValueError(
                "standard_methods.search_time_outlier_rejection.zscore_limit "
                "must be positive"
            )
        half_width = z_limit * sigma if sigma > 0.0 else 0.0
        accepted = valid & (np.abs(residual_ps - center) <= half_width)
    else:
        half_width = float("inf")
        accepted = valid

    dev_filtered = development[accepted[development]]
    blind_filtered = blind[accepted[blind]]

    if dev_filtered.size < 3:
        raise RuntimeError(
            f"Only {dev_filtered.size} {family} development events remain "
            "after search-time rejection"
        )
    if blind_filtered.size < 3:
        raise RuntimeError(
            f"Only {blind_filtered.size} {family} blind events remain "
            "after search-time rejection"
        )

    summary = {
        "family": family,
        "enabled": enabled,
        "reference_threshold_mV": float(reference_threshold_mV),
        "fit_population": "development_only",
        "blind_used_for_fit": False,
        "median_ps": center,
        "robust_sigma_ps": sigma,
        "zscore_limit": z_limit if enabled else None,
        "effective_half_width_ps": half_width if enabled else None,
        "development_before": int(development.size),
        "development_after": int(dev_filtered.size),
        "blind_before": int(blind.size),
        "blind_after": int(blind_filtered.size),
    }

    logger.info(
        "Search-time rejection | %s | Tref=%.3f mV | dev %d->%d | "
        "blind %d->%d | center=%.1f ps | robust sigma=%.1f ps",
        family,
        reference_threshold_mV,
        development.size,
        dev_filtered.size,
        blind.size,
        blind_filtered.size,
        center,
        sigma,
    )

    return dev_filtered, blind_filtered, summary


def _led_support(
    waves: np.ndarray,
    indices: np.ndarray,
    *,
    reference_threshold_mV: float,
    chunk_size: int,
    configured_range: tuple[float, float] | None = None,
) -> tuple[float, float]:
    idx = np.asarray(indices, dtype=np.int64).reshape(-1)
    if idx.size == 0:
        raise RuntimeError(
            "Cannot determine LED support from an empty development cohort"
        )

    common_peak = np.inf
    step = max(1, int(chunk_size))

    for first in range(0, idx.size, step):
        block = np.asarray(
            waves[idx[first:first + step]],
            dtype=np.float64,
        )
        peaks = np.nanmax(block, axis=2)

        if np.any(~np.isfinite(peaks)):
            raise RuntimeError(
                "Non-finite pulse maximum in development cohort"
            )

        common_peak = min(common_peak, float(np.min(peaks)))

    low = (
        1.0
        if configured_range is None
        else max(0.0, float(configured_range[0]))
    )

    epsilon = max(
        1e-6,
        1e-8 * max(1.0, abs(float(reference_threshold_mV))),
    )
    search_cap = float(reference_threshold_mV) - epsilon

    high = (
        min(common_peak, search_cap)
        if configured_range is None
        else min(
            common_peak,
            search_cap,
            float(configured_range[1]),
        )
    )

    if not np.isfinite(low) or not np.isfinite(high) or high <= low:
        raise RuntimeError(
            f"No valid LED scan range below search threshold: "
            f"{low:.6g}..{high:.6g} mV "
            f"(Tsearch={reference_threshold_mV:.6g} mV)"
        )

    return max(1e-6, low), high


def _extract_grids(
    waves: np.ndarray,
    relative_time_ps: np.ndarray,
    anchors_fs: np.ndarray,
    reference_times_fs: np.ndarray,
    *,
    reference_threshold_mV: float,
    led_thresholds_mV: np.ndarray,
    cfd_fractions: np.ndarray,
    before_ns: float,
    after_ns: float,
    chunk_size: int,
    indices: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    thresholds = np.asarray(
        led_thresholds_mV,
        dtype=np.float64,
    ).reshape(-1)
    fractions = np.asarray(
        cfd_fractions,
        dtype=np.float64,
    ).reshape(-1)

    n_events = int(waves.shape[0])
    led = np.full(
        (n_events, 2, thresholds.size),
        INVALID_TIME_FS,
        dtype=np.int64,
    )
    cfd = np.full(
        (n_events, 2, fractions.size),
        INVALID_TIME_FS,
        dtype=np.int64,
    )

    selected = (
        np.arange(n_events, dtype=np.int64)
        if indices is None
        else np.asarray(indices, dtype=np.int64).reshape(-1)
    )

    t = np.asarray(relative_time_ps, dtype=np.float64)
    step = max(1, int(chunk_size))

    for first in range(0, selected.size, step):
        block_idx = selected[first:first + step]
        block = np.asarray(waves[block_idx], dtype=np.float64)

        for local_row, event_value in enumerate(block_idx):
            event = int(event_value)

            for detector in range(2):
                anchor_fs = np.int64(anchors_fs[event, detector])
                reference_fs = np.int64(
                    reference_times_fs[event, detector]
                )

                if (
                    anchor_fs == INVALID_TIME_FS
                    or reference_fs == INVALID_TIME_FS
                ):
                    continue

                reference_time_ps = (
                    float(reference_fs - anchor_fs) / 1000.0
                )
                signal = block[local_row, detector]

                if thresholds.size:
                    local_led = _same_edge_level_times_fs(
                        signal,
                        t,
                        thresholds,
                        reference_threshold_mV=reference_threshold_mV,
                        reference_time_ps=reference_time_ps,
                        before_ns=before_ns,
                        after_ns=after_ns,
                    )
                    good = local_led != INVALID_TIME_FS
                    led[event, detector, good] = (
                        anchor_fs + local_led[good]
                    )

                if fractions.size:
                    try:
                        (
                            _start,
                            _stop,
                            _reference_sample,
                            peak_value,
                        ) = _pulse_peak_after_reference(
                            signal,
                            t,
                            reference_time_ps=reference_time_ps,
                            before_ns=before_ns,
                            after_ns=after_ns,
                        )
                    except RuntimeError:
                        continue

                    levels = float(peak_value) * fractions
                    local_cfd = _same_edge_level_times_fs(
                        signal,
                        t,
                        levels,
                        reference_threshold_mV=reference_threshold_mV,
                        reference_time_ps=reference_time_ps,
                        before_ns=before_ns,
                        after_ns=after_ns,
                    )
                    good = local_cfd != INVALID_TIME_FS
                    cfd[event, detector, good] = (
                        anchor_fs + local_cfd[good]
                    )

    return led, cfd

def _candidate_score(
    grid_fs: np.ndarray,
    candidate_index: int,
    development: np.ndarray,
    splits: list[tuple[np.ndarray, np.ndarray]],
    true_tof_ps: float,
) -> tuple[float, tuple[float, ...]]:
    development = np.asarray(development, dtype=np.int64)
    pair = np.asarray(grid_fs[:, :, candidate_index], dtype=np.int64)
    if np.any(pair[development] == INVALID_TIME_FS):
        return float("inf"), tuple()

    fold_values: list[float] = []
    for _train, score in splits:
        idx = np.asarray(score, dtype=np.int64)
        if idx.size < 2 or np.any(pair[idx] == INVALID_TIME_FS):
            return float("inf"), tuple()
        delta = (
            pair[idx, 0].astype(np.float64)
            - pair[idx, 1].astype(np.float64)
        ) / 1000.0
        metrics = residual_metrics(delta - float(true_tof_ps))
        fold_values.append(float(metrics["ctr_ps"]))

    if not fold_values or not np.all(np.isfinite(fold_values)):
        return float("inf"), tuple(fold_values)
    return float(np.mean(fold_values)), tuple(fold_values)


def _best_candidate(
    grid_fs: np.ndarray,
    parameters: np.ndarray,
    development: np.ndarray,
    splits: list[tuple[np.ndarray, np.ndarray]],
    true_tof_ps: float,
) -> tuple[int, float, tuple[float, ...]]:
    finite: list[tuple[float, int, tuple[float, ...]]] = []
    for index in range(int(parameters.size)):
        score, folds = _candidate_score(
            grid_fs, index, development, splits, true_tof_ps
        )
        if np.isfinite(score):
            finite.append((score, index, folds))
    if not finite:
        raise RuntimeError(
            "No standard-method candidate has complete crossing coverage on the "
            "development selection population"
        )
    score, index, folds = min(finite, key=lambda item: (item[0], item[1]))
    return int(index), float(score), tuple(folds)


def _refined_axis(values: np.ndarray, best_index: int, points: int) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    if arr.size <= 1 or int(points) <= 1:
        return arr[[best_index]]
    lo = arr[max(0, best_index - 1)]
    hi = arr[min(arr.size - 1, best_index + 1)]
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        return arr[[best_index]]
    return np.linspace(float(lo), float(hi), int(points), dtype=np.float64)



def optimize_family(
    config: dict[str, Any],
    dataset: PreparedDataset,
    development: np.ndarray,
    splits: list[tuple[np.ndarray, np.ndarray]],
    *,
    family: str,
    logger: Any,
    application_indices: np.ndarray | None = None,
) -> FamilySelection:
    standard = config.get("standard_methods", {}) or {}
    cfd_enabled = cfd_enabled_for_family(config, family)

    waves, times, anchors = _family_arrays(dataset, family)
    development = np.asarray(development, dtype=np.int64)

    before_ns, after_ns, reference_threshold_mV = _family_timing_config(
        config,
        family,
    )
    chunk_size = int(
        standard.get("waveform_scan_chunk_size", 1024)
    )

    reference_times = _search_reference_times_fs(
        waves,
        times,
        anchors,
        reference_threshold_mV=reference_threshold_mV,
        chunk_size=chunk_size,
    )

    if np.any(reference_times[development] == INVALID_TIME_FS):
        raise RuntimeError(
            f"{family} development cohort contains missing search-threshold "
            "references after search-time filtering"
        )

    raw_led_thresholds = standard.get(
        "led_thresholds_mV",
        [float(value) for value in range(5, 100, 10)],
    )
    led_axis = np.asarray(raw_led_thresholds, dtype=np.float64).reshape(-1)
    if (
        led_axis.size == 0
        or np.any(~np.isfinite(led_axis))
        or np.any(led_axis <= 0.0)
    ):
        raise ValueError(
            "standard_methods.led_thresholds_mV must contain positive finite values"
        )
    led_axis = np.unique(led_axis)
    if led_axis.size == 0:
        raise ValueError(
            "standard_methods.led_thresholds_mV must not be empty"
        )
    low = float(led_axis[0])
    high = float(led_axis[-1])

    if cfd_enabled:
        cfd_points = max(
            3,
            int(standard.get("cfd_grid_points", 81)),
        )
        cfd_axis = np.linspace(
            float(standard.get("cfd_min_fraction", 0.02)),
            float(standard.get("cfd_max_fraction", 0.80)),
            cfd_points,
            dtype=np.float64,
        )
        if not (
            0.0
            < float(cfd_axis[0])
            < float(cfd_axis[-1])
            <= 1.0
        ):
            raise ValueError(
                "standard_methods CFD range must satisfy 0 < min < max <= 1"
            )
    else:
        cfd_axis = np.empty(0, dtype=np.float64)

    logger.info(
        "Adaptive standards scan | %s | Tref=%.6g mV | "
        "LED candidates %.3f..%.3f mV (%d) | CFD %s",
        family,
        reference_threshold_mV,
        low,
        high,
        led_axis.size,
        (
            f"{cfd_axis[0]:.4f}..{cfd_axis[-1]:.4f} ({cfd_axis.size})"
            if cfd_enabled
            else "disabled"
        ),
    )

    led_grid, cfd_grid = _extract_grids(
        waves,
        times,
        anchors,
        reference_times,
        reference_threshold_mV=reference_threshold_mV,
        led_thresholds_mV=led_axis,
        cfd_fractions=cfd_axis,
        before_ns=before_ns,
        after_ns=after_ns,
        chunk_size=chunk_size,
        indices=development,
    )

    led_index, _led_score, _led_folds = _best_candidate(
        led_grid,
        led_axis,
        development,
        splits,
        dataset.true_tof_ps,
    )

    if cfd_enabled:
        cfd_index, _cfd_score, _cfd_folds = _best_candidate(
            cfd_grid,
            cfd_axis,
            development,
            splits,
            dataset.true_tof_ps,
        )
    else:
        cfd_index = -1

    # LED uses the explicit sparse scan only: no second refinement pass.
    led_fine = led_axis[[led_index]]

    cfd_fine = (
        _refined_axis(
            cfd_axis,
            cfd_index,
            max(3, int(standard.get("cfd_refine_points", 41))),
        )
        if cfd_enabled
        else np.empty(0, dtype=np.float64)
    )

    led_grid_fine, cfd_grid_fine = _extract_grids(
        waves,
        times,
        anchors,
        reference_times,
        reference_threshold_mV=reference_threshold_mV,
        led_thresholds_mV=led_fine,
        cfd_fractions=cfd_fine,
        before_ns=before_ns,
        after_ns=after_ns,
        chunk_size=chunk_size,
        indices=development,
    )

    led_index, led_score, led_folds = _best_candidate(
        led_grid_fine,
        led_fine,
        development,
        splits,
        dataset.true_tof_ps,
    )

    if cfd_enabled:
        cfd_index, cfd_score, cfd_folds = _best_candidate(
            cfd_grid_fine,
            cfd_fine,
            development,
            splits,
            dataset.true_tof_ps,
        )
        selected_cfd_fraction = float(cfd_fine[cfd_index])
    else:
        cfd_score = float("nan")
        cfd_folds = tuple()
        selected_cfd_fraction = float("nan")

    active = (
        np.arange(int(waves.shape[0]), dtype=np.int64)
        if application_indices is None
        else np.asarray(application_indices, dtype=np.int64).reshape(-1)
    )

    selected_led_grid, _ = _extract_grids(
        waves,
        times,
        anchors,
        reference_times,
        reference_threshold_mV=reference_threshold_mV,
        led_thresholds_mV=np.asarray(
            [led_fine[led_index]],
            dtype=np.float64,
        ),
        cfd_fractions=np.empty(0, dtype=np.float64),
        before_ns=before_ns,
        after_ns=after_ns,
        chunk_size=chunk_size,
        indices=active,
    )

    selected_led = np.asarray(
        selected_led_grid[:, :, 0],
        dtype=np.int64,
    )

    missing_led = np.any(
        selected_led[active] == INVALID_TIME_FS,
        axis=1,
    )

    if np.any(missing_led):
        raise RuntimeError(
            f"Selected {family} LED threshold "
            f"{float(led_fine[led_index]):.6g} mV is missing a same-edge "
            f"crossing for {int(np.count_nonzero(missing_led))} events "
            "that passed search-time rejection. No LED-derived event "
            "rejection or fallback is allowed."
        )

    inactive = np.ones(selected_led.shape[0], dtype=bool)
    inactive[active] = False

    if np.any(inactive):
        bootstrap_led = (
            dataset.energy_led_time_fs
            if family == "energy"
            else dataset.timing_led_time_fs
        )
        if bootstrap_led is None:
            raise RuntimeError(
                f"{family} bootstrap LED timestamps unavailable"
            )
        selected_led[inactive] = np.asarray(
            bootstrap_led,
            dtype=np.int64,
        )[inactive]

    if cfd_enabled:
        _, selected_cfd_grid = _extract_grids(
            waves,
            times,
            anchors,
            reference_times,
            reference_threshold_mV=reference_threshold_mV,
            led_thresholds_mV=np.empty(0, dtype=np.float64),
            cfd_fractions=np.asarray(
                [selected_cfd_fraction],
                dtype=np.float64,
            ),
            before_ns=before_ns,
            after_ns=after_ns,
            chunk_size=chunk_size,
            indices=active,
        )

        selected_cfd = np.asarray(
            selected_cfd_grid[:, :, 0],
            dtype=np.int64,
        )

        cfd_fallback_events = np.zeros(
            selected_cfd.shape[0],
            dtype=bool,
        )
        cfd_fallback_events[active] = np.any(
            selected_cfd[active] == INVALID_TIME_FS,
            axis=1,
        )

        n_fallback = int(np.count_nonzero(cfd_fallback_events))

        if n_fallback:
            selected_cfd[cfd_fallback_events, :] = (
                selected_led[cfd_fallback_events, :]
            )
            logger.warning(
                "Adaptive standards | %s | CFD fallback to LED "
                "for %d active events",
                family,
                n_fallback,
            )

        selected_cfd[inactive] = selected_led[inactive]
    else:
        selected_cfd = selected_led.copy()

    selection = FamilySelection(
        family=family,
        cfd_enabled=bool(cfd_enabled),
        led_threshold_mV=float(led_fine[led_index]),
        cfd_fraction=selected_cfd_fraction,
        led_validation_sctr_ps=float(led_score),
        cfd_validation_sctr_ps=float(cfd_score),
        led_fold_sctr_ps=tuple(float(v) for v in led_folds),
        cfd_fold_sctr_ps=tuple(float(v) for v in cfd_folds),
        led_times_fs=selected_led,
        cfd_times_fs=selected_cfd,
        led_search_low_mV=float(low),
        led_search_high_mV=float(high),
        led_coarse_points=int(led_axis.size),
        led_refine_points=int(led_fine.size),
        cfd_coarse_points=int(cfd_axis.size),
        cfd_refine_points=int(cfd_fine.size),
    )

    logger.info(
        "Adaptive standards selected | %s | Tsearch %.6g mV | "
        "LED %.6g mV -> %.3f ps s-CTR",
        family,
        reference_threshold_mV,
        selection.led_threshold_mV,
        selection.led_validation_sctr_ps,
    )

    return selection

def apply_selections(
    config: dict[str, Any],
    dataset: PreparedDataset,
    selections: dict[str, FamilySelection],
) -> PreparedDataset:
    manifest = dict(dataset.manifest)
    manifest["adaptive_standard_methods"] = {
        family: selection.as_dict()
        for family, selection in selections.items()
    }
    manifest["adaptive_reference_edge_protocol"] = {
        family: {
            "analysis_crop_before_ns": _family_timing_config(config, family)[0],
            "analysis_crop_after_ns": _family_timing_config(config, family)[1],
            "reference_threshold_mV": _family_timing_config(config, family)[2],
            "led_edge_rule": "continuous rising excursion anchored to search threshold",
        }
        for family in selections
    }
    manifest["ml_window_alignment_source"] = "adaptive_selected_led_search_anchored"
    manifest["window_anchor_shift_factored"] = True

    kwargs: dict[str, Any] = {"manifest": manifest}
    energy = selections.get("energy")
    timing = selections.get("timing")

    if energy is not None:
        if dataset.energy_window_anchor_time_fs is None:
            raise ValueError("Energy preprocessing anchors are required for LED re-alignment")
        energy_shifts, energy_anchors = _alignment_shifts(
            energy.led_times_fs,
            dataset.energy_window_anchor_time_fs,
            dataset.relative_time_ps,
        )
        _check_shift_support(
            energy_shifts,
            dataset.relative_time_ps,
            config["windows_ns"],
            label="energy",
        )
        kwargs["windows_mV"] = ShiftedWaveformArray(
            dataset.windows_mV, energy_shifts
        )
        if dataset.denoised_windows_mV is not None:
            kwargs["denoised_windows_mV"] = ShiftedWaveformArray(
                dataset.denoised_windows_mV, energy_shifts
            )
        kwargs["energy_window_anchor_time_fs"] = energy_anchors
        kwargs["energy_led_time_fs"] = energy.led_times_fs
        kwargs["energy_cfd_time_fs"] = energy.cfd_times_fs
        kwargs["led_time_fs"] = energy.led_times_fs
        kwargs["cfd_time_fs"] = energy.cfd_times_fs
        kwargs["window_anchor_time_fs"] = energy_anchors

    if timing is not None:
        if dataset.timing_window_anchor_time_fs is None:
            raise ValueError("Timing preprocessing anchors are required for LED re-alignment")
        timing_shifts, timing_anchors = _alignment_shifts(
            timing.led_times_fs,
            dataset.timing_window_anchor_time_fs,
            dataset.timing_relative_time_ps,
        )
        _check_shift_support(
            timing_shifts,
            dataset.timing_relative_time_ps,
            config["windows_ns"],
            label="timing",
        )
        if dataset.timing_windows_mV is not None:
            kwargs["timing_windows_mV"] = ShiftedWaveformArray(
                dataset.timing_windows_mV, timing_shifts
            )
        if dataset.denoised_timing_windows_mV is not None:
            kwargs["denoised_timing_windows_mV"] = ShiftedWaveformArray(
                dataset.denoised_timing_windows_mV, timing_shifts
            )
        kwargs["timing_window_anchor_time_fs"] = timing_anchors
        kwargs["timing_led_time_fs"] = timing.led_times_fs
        kwargs["timing_cfd_time_fs"] = timing.cfd_times_fs

        if dataset.timing_aligned_energy_windows_mV is not None:
            if dataset.timing_aligned_energy_window_anchor_time_fs is None:
                raise ValueError(
                    "Timing-aligned energy preprocessing anchors are required"
                )
            aligned_shifts, aligned_anchors = _alignment_shifts(
                timing.led_times_fs,
                dataset.timing_aligned_energy_window_anchor_time_fs,
                dataset.relative_time_ps,
            )
            _check_shift_support(
                aligned_shifts,
                dataset.relative_time_ps,
                config["windows_ns"],
                label="timing-aligned energy",
            )
            kwargs["timing_aligned_energy_windows_mV"] = ShiftedWaveformArray(
                dataset.timing_aligned_energy_windows_mV,
                aligned_shifts,
            )
            if dataset.denoised_timing_aligned_energy_windows_mV is not None:
                kwargs["denoised_timing_aligned_energy_windows_mV"] = ShiftedWaveformArray(
                    dataset.denoised_timing_aligned_energy_windows_mV,
                    aligned_shifts,
                )
            kwargs["timing_aligned_energy_window_anchor_time_fs"] = aligned_anchors

    return replace(dataset, **kwargs)



def optimize_standard_methods(
    config: dict[str, Any],
    dataset: PreparedDataset,
    development: np.ndarray,
    splits: list[tuple[np.ndarray, np.ndarray]],
    *,
    families: Iterable[str],
    logger: Any,
    application_indices: np.ndarray | None = None,
) -> tuple[PreparedDataset, dict[str, FamilySelection]]:
    if not bool(
        (config.get("standard_methods", {}) or {}).get("enabled", True)
    ):
        return dataset, {}

    selections: dict[str, FamilySelection] = {}

    for family in sorted(set(str(v) for v in families)):
        selections[family] = optimize_family(
            config,
            dataset,
            development,
            splits,
            family=family,
            logger=logger,
            application_indices=application_indices,
        )

    return apply_selections(config, dataset, selections), selections

def parameter_payload(
    selections: dict[str, FamilySelection],
    mode: str,
    model: str,
) -> dict[str, Any]:
    family = family_for_mode(mode)
    selection = selections.get(family)
    if selection is None:
        return {}
    if str(model) == "led":
        return {
            "family": family,
            "led_threshold_mV": float(selection.led_threshold_mV),
            "selection_metric": "sctr",
            "validation_sctr_ps": float(selection.led_validation_sctr_ps),
        }
    if str(model) == "cfd":
        if not selection.cfd_enabled:
            return {}
        return {
            "family": family,
            "cfd_fraction": float(selection.cfd_fraction),
            "selection_metric": "sctr",
            "validation_sctr_ps": float(selection.cfd_validation_sctr_ps),
        }
    return {}

