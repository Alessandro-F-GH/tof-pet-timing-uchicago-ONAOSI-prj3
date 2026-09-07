from __future__ import annotations

import json
import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.lib.format import open_memmap

from .common import atomic_json, canonical_hash, source_signature
from .energy_io import iterate_energy_chunks
from .event_selection import SelectionData, decode_oriented

PREPROCESS_FORMAT_VERSION = 2


@dataclass(frozen=True)
class PreprocessedData:
    directory: Path
    manifest: dict[str, Any]
    event_index: np.ndarray
    split: np.ndarray
    bias_voltage_V: np.ndarray
    energy_windows_mV: np.ndarray | None
    timing_windows_mV: np.ndarray | None
    energy_window_start_time_s: np.ndarray | None
    timing_window_start_time_s: np.ndarray | None
    energy_sample_interval_s: np.ndarray | None
    timing_sample_interval_s: np.ndarray | None
    energy_rising_start: np.ndarray | None
    timing_rising_start: np.ndarray | None
    energy_rising_stop: np.ndarray | None
    timing_rising_stop: np.ndarray | None

    @property
    def n_events(self) -> int:
        return int(self.event_index.size)

    @property
    def development(self) -> np.ndarray:
        return np.flatnonzero(self.split == 0).astype(np.int64)

    @property
    def test(self) -> np.ndarray:
        return np.flatnonzero(self.split == 1).astype(np.int64)


def used_families(config: dict[str, Any]) -> tuple[str, ...]:
    modes = config["channel_modes"]
    out: list[str] = []
    if any("energy" in mode for mode in modes):
        out.append("energy")
    if any("timing" in mode for mode in modes):
        out.append("timing")
    return tuple(out)


def _limits(value: Any) -> np.ndarray:
    limits = np.asarray(value, dtype=float)
    if limits.shape == (2,):
        limits = np.repeat(limits[None, :], 2, axis=0)
    if limits.shape != (2, 2) or np.any(limits[:, 0] >= limits[:, 1]):
        raise ValueError(
            "vertical_scale_limit_mV must be [low, high] or two detector pairs"
        )
    return limits


def _family_arrays(chunk: Any, family: str):
    if family == "energy":
        return (
            chunk.samples,
            chunk.vertical_gain_v_per_count,
            chunk.vertical_offset_v,
            chunk.horizontal_interval_s,
            chunk.horizontal_offset_s,
        )
    return (
        chunk.timing_samples,
        chunk.timing_vertical_gain_v_per_count,
        chunk.timing_vertical_offset_v,
        chunk.timing_horizontal_interval_s,
        chunk.timing_horizontal_offset_s,
    )


def preprocessing_fingerprint(
    root: Path, selection: SelectionData, config: dict[str, Any]
) -> str:
    return canonical_hash(
        {
            "format_version": PREPROCESS_FORMAT_VERSION,
            "source": source_signature(root),
            "selection": selection.manifest["fingerprint"],
            "families": used_families(config),
            "materialized_window_ns": config["preprocessing"]["materialized_window_ns"],
            "family_config": {
                family: {
                    "vertical_scale_limit_mV": config["preprocessing"][family][
                        "vertical_scale_limit_mV"
                    ],
                    "rising_edge_before_trigger_ns": config["preprocessing"][family][
                        "rising_edge_before_trigger_ns"
                    ],
                }
                for family in used_families(config)
            },
        }
    )


def _optional(directory: Path, name: str) -> np.ndarray | None:
    path = directory / f"{name}.npy"
    return np.load(path, mmap_mode="r") if path.is_file() else None


def load_preprocessed(
    directory: Path,
    root: Path,
    selection: SelectionData,
    config: dict[str, Any],
) -> PreprocessedData:
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("fingerprint") != preprocessing_fingerprint(
        root, selection, config
    ):
        raise ValueError("Preprocessing fingerprint changed")
    return PreprocessedData(
        directory,
        manifest,
        np.load(directory / "event_index.npy", mmap_mode="r"),
        np.load(directory / "split.npy", mmap_mode="r"),
        np.load(directory / "bias_voltage_V.npy", mmap_mode="r"),
        _optional(directory, "energy_windows_mV"),
        _optional(directory, "timing_windows_mV"),
        _optional(directory, "energy_window_start_time_s"),
        _optional(directory, "timing_window_start_time_s"),
        _optional(directory, "energy_sample_interval_s"),
        _optional(directory, "timing_sample_interval_s"),
        _optional(directory, "energy_rising_start"),
        _optional(directory, "timing_rising_start"),
        _optional(directory, "energy_rising_stop"),
        _optional(directory, "timing_rising_stop"),
    )


def _close(array: np.memmap) -> None:
    array.flush()
    mmap = getattr(array, "_mmap", None)
    if mmap is not None:
        mmap.close()


def _window_bounds(
    trigger_index: int,
    trace_size: int,
    sample_interval_ns: float,
    before_ns: float,
    after_ns: float,
) -> tuple[int, int] | None:
    start = int(trigger_index) - int(np.ceil(float(before_ns) / sample_interval_ns))
    stop = int(trigger_index) + int(np.ceil(float(after_ns) / sample_interval_ns)) + 1
    if start < 0 or stop > int(trace_size):
        return None
    return start, stop


def _compact_npy(path: Path, keep: np.ndarray) -> None:
    source = np.load(path, mmap_mode="r")
    temporary = path.with_name(path.stem + ".compact.npy")
    np.save(temporary, np.asarray(source[keep]))
    mmap = getattr(source, "_mmap", None)
    if mmap is not None:
        mmap.close()
    os.replace(temporary, path)


def preprocess_selected(
    root_file: Path,
    selection: SelectionData,
    config: dict[str, Any],
    *,
    rebuild: bool,
    logger: Any,
) -> PreprocessedData:
    root_file = Path(root_file).resolve()
    base = Path(config["preprocessing"]["preprocessed_dir"]).resolve() / root_file.stem

    if base.is_dir() and not rebuild:
        try:
            return load_preprocessed(base, root_file, selection, config)
        except (FileNotFoundError, ValueError):
            pass

    if base.exists():
        shutil.rmtree(base)
    base.mkdir(parents=True)

    families = used_families(config)
    entries = np.asarray(selection.entry_index, dtype=np.int64)
    lookup = {int(entry): row for row, entry in enumerate(entries)}
    n_selected = int(entries.size)
    full_event_index = np.asarray(selection.event_index, dtype=np.int64)
    full_split = np.asarray(selection.split, dtype=np.int8)

    bias = open_memmap(
        base / "bias_voltage_V.npy",
        mode="w+",
        dtype=np.float64,
        shape=(n_selected,),
    )
    bias[:] = np.nan

    targets: dict[str, np.memmap] = {}
    starts: dict[str, np.memmap] = {}
    intervals: dict[str, np.memmap] = {}
    rise_a: dict[str, np.memmap] = {}
    rise_b: dict[str, np.memmap] = {}
    lengths: dict[str, int] = {}

    seen = np.zeros(n_selected, dtype=bool)
    keep = np.zeros(n_selected, dtype=bool)
    excluded_window = np.zeros(n_selected, dtype=bool)

    channels = config["data"]["channels"]
    polarities = {
        "energy": np.asarray(channels.get("polarities", [1, 1]), dtype=np.int8),
        "timing": np.asarray(
            channels.get("timing_polarities", [1, 1]), dtype=np.int8
        ),
    }
    triggers = {
        "energy": selection.main_trigger_energy,
        "timing": selection.main_trigger_timing,
    }
    stops = {
        "energy": selection.main_stop_energy,
        "timing": selection.main_stop_timing,
    }

    materialized = config["preprocessing"]["materialized_window_ns"]
    before_ns = float(materialized["before"])
    after_ns = float(materialized["after"])
    io = config["preprocessing"].get("io", {})
    max_events = int(io.get("max_events", 0))
    args = {
        "energy_channels_one_based": tuple(map(int, channels["energy"])),
        "timing_channels_one_based": (
            tuple(map(int, channels["timing"])) if "timing" in families else None
        ),
        "step_size": io.get("step_size", "128 MB"),
        "entry_stop": max_events if max_events > 0 else None,
    }

    entry = 0
    for chunk in iterate_energy_chunks(root_file, **args):
        for local in range(chunk.event_index.size):
            row = lookup.get(entry)
            entry += 1
            if row is None:
                continue

            seen[row] = True
            bias[row] = float(chunk.bias_voltage_V[local])
            event_payload: dict[
                str, tuple[list[np.ndarray], list[float], list[float], list[int], list[int]]
            ] = {}
            event_ok = True

            for family in families:
                raw, gain, offset, sample_interval, horizontal_offset = _family_arrays(
                    chunk, family
                )
                assert (
                    raw is not None
                    and gain is not None
                    and offset is not None
                    and sample_interval is not None
                    and horizontal_offset is not None
                    and triggers[family] is not None
                    and stops[family] is not None
                )

                vertical_limits = _limits(
                    config["preprocessing"][family]["vertical_scale_limit_mV"]
                )
                windows: list[np.ndarray] = []
                start_times: list[float] = []
                sample_intervals: list[float] = []
                rising_starts: list[int] = []
                rising_stops: list[int] = []
                rising_before_ns = float(
                    config["preprocessing"][family]["rising_edge_before_trigger_ns"]
                )

                for detector in range(2):
                    signal = decode_oriented(
                        np.asarray(raw[detector][local], dtype=np.int16),
                        gain[local, detector],
                        offset[local, detector],
                        int(polarities[family][detector]),
                    )
                    signal = np.clip(
                        signal,
                        vertical_limits[detector, 0],
                        vertical_limits[detector, 1],
                    )
                    interval_s = float(sample_interval[local, detector])
                    interval_ns = interval_s * 1e9
                    trigger = int(triggers[family][row, detector])
                    pulse_stop = int(stops[family][row, detector])

                    bounds = _window_bounds(
                        trigger,
                        signal.size,
                        interval_ns,
                        before_ns,
                        after_ns,
                    )
                    if bounds is None:
                        excluded_window[row] = True
                        event_ok = False
                        break
                    start, stop = bounds

                    window = np.asarray(signal[start:stop], dtype=np.float32)
                    search_start = max(
                        start,
                        trigger - int(np.ceil(rising_before_ns / interval_ns)),
                    )
                    presegment = signal[search_start : trigger + 1]
                    onset = (
                        search_start + int(np.nanargmin(presegment))
                        if presegment.size
                        else trigger
                    )
                    peak_stop = min(signal.size - 1, max(trigger, pulse_stop))
                    pulse = signal[trigger : peak_stop + 1]
                    peak = (
                        trigger + int(np.nanargmax(pulse))
                        if pulse.size
                        else trigger
                    )
                    if peak <= onset:
                        raise RuntimeError(
                            f"{root_file.name} event {selection.event_index[row]} "
                            f"{family} detector {detector + 1}: "
                            "unable to define rising edge interval"
                        )

                    windows.append(window)
                    start_times.append(
                        float(horizontal_offset[local, detector]) + start * interval_s
                    )
                    sample_intervals.append(interval_s)
                    rising_starts.append(onset - start)
                    rising_stops.append(peak - start)

                if not event_ok:
                    break

                event_payload[family] = (
                    windows,
                    start_times,
                    sample_intervals,
                    rising_starts,
                    rising_stops,
                )

            if not event_ok:
                continue

            for family, payload in event_payload.items():
                windows, start_times, sample_intervals, rising_starts, rising_stops = (
                    payload
                )
                if family not in targets:
                    length = int(windows[0].size)
                    if any(window.size != length for window in windows):
                        raise ValueError(f"{family} detector grids differ")
                    lengths[family] = length
                    targets[family] = open_memmap(
                        base / f"{family}_windows_mV.npy",
                        mode="w+",
                        dtype=np.float32,
                        shape=(n_selected, 2, length),
                    )
                    starts[family] = open_memmap(
                        base / f"{family}_window_start_time_s.npy",
                        mode="w+",
                        dtype=np.float64,
                        shape=(n_selected, 2),
                    )
                    intervals[family] = open_memmap(
                        base / f"{family}_sample_interval_s.npy",
                        mode="w+",
                        dtype=np.float64,
                        shape=(n_selected, 2),
                    )
                    rise_a[family] = open_memmap(
                        base / f"{family}_rising_start.npy",
                        mode="w+",
                        dtype=np.int32,
                        shape=(n_selected, 2),
                    )
                    rise_b[family] = open_memmap(
                        base / f"{family}_rising_stop.npy",
                        mode="w+",
                        dtype=np.int32,
                        shape=(n_selected, 2),
                    )

                if any(window.size != lengths[family] for window in windows):
                    raise ValueError(f"{family} sample interval changed")

                targets[family][row] = np.stack(windows)
                starts[family][row] = start_times
                intervals[family][row] = sample_intervals
                rise_a[family][row] = rising_starts
                rise_b[family][row] = rising_stops

            keep[row] = True

    if not np.all(seen):
        raise RuntimeError(
            f"Failed to inspect {np.count_nonzero(~seen)} selected events"
        )
    if not np.any(keep):
        raise RuntimeError("No selected events have a valid preprocessing window")

    _close(bias)
    for group in (targets, starts, intervals, rise_a, rise_b):
        for array in group.values():
            _close(array)

    kept_rows = np.flatnonzero(keep)
    if kept_rows.size != n_selected:
        _compact_npy(base / "bias_voltage_V.npy", kept_rows)
        for family in families:
            for suffix in (
                "windows_mV",
                "window_start_time_s",
                "sample_interval_s",
                "rising_start",
                "rising_stop",
            ):
                _compact_npy(base / f"{family}_{suffix}.npy", kept_rows)

    np.save(base / "event_index.npy", full_event_index[kept_rows])
    np.save(base / "split.npy", full_split[kept_rows])

    excluded_count = int(np.count_nonzero(excluded_window))
    manifest = {
        "format_version": PREPROCESS_FORMAT_VERSION,
        "fingerprint": preprocessing_fingerprint(root_file, selection, config),
        "source": str(root_file),
        "selection_dir": str(selection.directory),
        "n_input_selected": n_selected,
        "n_events": int(kept_rows.size),
        "excluded_window_exceeds_trace": excluded_count,
        "families": list(families),
        "materialized_window_ns": {"before": before_ns, "after": after_ns},
        "time_reference": "absolute_native_acquisition_time",
        "denoising": False,
    }
    atomic_json(base / "manifest.json", manifest)
    logger.info(
        "Preprocessing %s | n=%d/%d | excluded_window=%d | families=%s",
        root_file.name,
        kept_rows.size,
        n_selected,
        excluded_count,
        ",".join(families),
    )
    return load_preprocessed(base, root_file, selection, config)
