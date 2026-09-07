from __future__ import annotations

from typing import Any

import numpy as np

from .dataset import PreparedDataset


def family_arrays(dataset: PreparedDataset, family: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if family == "energy":
        if dataset.energy_window_anchor_time_fs is None:
            raise ValueError("Energy waveform anchors are unavailable")
        return (
            dataset.windows_mV,
            np.asarray(dataset.relative_time_ps, dtype=np.float64),
            np.asarray(dataset.energy_window_anchor_time_fs, dtype=np.int64),
        )
    if family == "timing":
        if (
            dataset.timing_windows_mV is None
            or dataset.timing_relative_time_ps is None
            or dataset.timing_window_anchor_time_fs is None
        ):
            raise ValueError("Timing waveform family is unavailable")
        return (
            dataset.timing_windows_mV,
            np.asarray(dataset.timing_relative_time_ps, dtype=np.float64),
            np.asarray(dataset.timing_window_anchor_time_fs, dtype=np.int64),
        )
    raise ValueError(f"Unknown waveform family: {family}")


def family_timing_config(config: dict[str, Any], family: str) -> tuple[float, float, float]:
    preprocessing = config["preprocessing"]
    resolved = dict(preprocessing.get("common", {}) or {})
    resolved.update(dict(preprocessing.get(family, {}) or {}))
    crop = resolved.get("analysis_crop_ns", {"before": 5.0, "after": 60.0})
    return (
        float(crop["before"]),
        float(crop["after"]),
        float(resolved["search_trigger_threshold_mV"]),
    )


def _interpolate(t0: float, y0: float, t1: float, y1: float, level: float) -> float:
    if not all(np.isfinite(value) for value in (t0, y0, t1, y1, level)) or t1 <= t0 or y1 == y0:
        return float("nan")
    fraction = (level - y0) / (y1 - y0)
    if not 0.0 <= fraction <= 1.0:
        return float("nan")
    return float(t0 + fraction * (t1 - t0))


def _reference_edge(
    signal: np.ndarray,
    time_ps: np.ndarray,
    reference_mV: float,
    before_ns: float,
    after_ns: float,
) -> tuple[int, int, int] | None:
    low, high = -before_ns * 1000.0, after_ns * 1000.0
    selected = np.flatnonzero((time_ps >= low) & (time_ps <= high))
    if selected.size < 3:
        return None
    start, stop = int(selected[0]), int(selected[-1]) + 1
    y0, y1 = signal[start : stop - 1], signal[start + 1 : stop]
    crossing = np.flatnonzero(
        np.isfinite(y0)
        & np.isfinite(y1)
        & (y0 < reference_mV)
        & (y1 >= reference_mV)
    )
    if crossing.size == 0:
        return None
    lower_candidates = crossing + start
    # Prepared windows are anchored to this search edge. The closest crossing to
    # t=0 is therefore the canonical pulse edge if ringing produces more than one.
    lower = int(lower_candidates[np.argmin(np.abs(time_ps[lower_candidates + 1]))])
    post = signal[lower + 1 : stop]
    if post.size == 0 or not np.any(np.isfinite(post)):
        return None
    peak = lower + 1 + int(np.nanargmax(post))
    if peak <= lower:
        return None
    return start, lower, peak


def _same_edge_crossings(
    signal: np.ndarray,
    time_ps: np.ndarray,
    levels_mV: np.ndarray,
    *,
    reference_mV: float,
    before_ns: float,
    after_ns: float,
) -> np.ndarray:
    levels = np.asarray(levels_mV, dtype=np.float64).reshape(-1)
    result = np.full(levels.size, np.nan, dtype=np.float64)
    edge = _reference_edge(signal, time_ps, reference_mV, before_ns, after_ns)
    if edge is None:
        return result
    start, reference_lower, peak = edge

    below = np.flatnonzero(np.isfinite(levels) & (levels > 0.0) & (levels <= reference_mV))
    if below.size:
        minimum = float(np.min(levels[below]))
        # Start immediately before the same rising excursion instead of from the
        # full baseline, preventing an earlier noise crossing from being selected.
        candidates = np.flatnonzero(signal[start : reference_lower + 1] <= minimum)
        edge_start = start + int(candidates[-1]) if candidates.size else start
        segment = np.asarray(signal[edge_start : reference_lower + 2], dtype=np.float64)
        envelope = np.maximum.accumulate(np.where(np.isfinite(segment), segment, -np.inf))
        positions = np.searchsorted(envelope, levels[below], side="left")
        for target, position in zip(below, positions):
            if 0 < int(position) < segment.size:
                lower = edge_start + int(position) - 1
                result[target] = _interpolate(
                    time_ps[lower], signal[lower], time_ps[lower + 1], signal[lower + 1], levels[target]
                )

    above = np.flatnonzero(np.isfinite(levels) & (levels > reference_mV))
    if above.size:
        segment = np.asarray(signal[reference_lower : peak + 1], dtype=np.float64)
        envelope = np.maximum.accumulate(np.where(np.isfinite(segment), segment, -np.inf))
        positions = np.searchsorted(envelope, levels[above], side="left")
        for target, position in zip(above, positions):
            if 0 < int(position) < segment.size:
                lower = reference_lower + int(position) - 1
                result[target] = _interpolate(
                    time_ps[lower], signal[lower], time_ps[lower + 1], signal[lower + 1], levels[target]
                )
    return result


def leading_edge_grid(
    config: dict[str, Any],
    dataset: PreparedDataset,
    family: str,
    indices: np.ndarray,
    thresholds_mV: np.ndarray,
) -> np.ndarray:
    waves, time_ps, anchors_fs = family_arrays(dataset, family)
    before_ns, after_ns, reference_mV = family_timing_config(config, family)
    idx = np.asarray(indices, dtype=np.int64)
    thresholds = np.asarray(thresholds_mV, dtype=np.float64).reshape(-1)
    output = np.full((idx.size, 2, thresholds.size), np.nan, dtype=np.float64)
    for row, event in enumerate(idx):
        for detector in range(2):
            relative = _same_edge_crossings(
                np.asarray(waves[event, detector], dtype=np.float64),
                time_ps,
                thresholds,
                reference_mV=reference_mV,
                before_ns=before_ns,
                after_ns=after_ns,
            )
            output[row, detector] = relative + float(anchors_fs[event, detector]) / 1000.0
    return output


def cfd_grid(
    config: dict[str, Any],
    dataset: PreparedDataset,
    family: str,
    indices: np.ndarray,
    fractions: np.ndarray,
) -> np.ndarray:
    waves, time_ps, anchors_fs = family_arrays(dataset, family)
    before_ns, after_ns, reference_mV = family_timing_config(config, family)
    idx = np.asarray(indices, dtype=np.int64)
    fractions = np.asarray(fractions, dtype=np.float64).reshape(-1)
    if np.any((fractions <= 0.0) | (fractions > 1.0)):
        raise ValueError("CFD fractions must lie in (0, 1]")
    output = np.full((idx.size, 2, fractions.size), np.nan, dtype=np.float64)
    for row, event in enumerate(idx):
        for detector in range(2):
            signal = np.asarray(waves[event, detector], dtype=np.float64)
            edge = _reference_edge(signal, time_ps, reference_mV, before_ns, after_ns)
            if edge is None:
                continue
            _start, _lower, peak = edge
            peak_mV = float(signal[peak])
            if not np.isfinite(peak_mV) or peak_mV <= 0.0:
                continue
            levels = fractions * peak_mV
            relative = _same_edge_crossings(
                signal,
                time_ps,
                levels,
                reference_mV=reference_mV,
                before_ns=before_ns,
                after_ns=after_ns,
            )
            output[row, detector] = relative + float(anchors_fs[event, detector]) / 1000.0
    return output


def pair_delta(times_ps: np.ndarray) -> np.ndarray:
    values = np.asarray(times_ps, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 2:
        raise ValueError("Expected timing array with shape [event, 2]")
    return values[:, 0] - values[:, 1]
