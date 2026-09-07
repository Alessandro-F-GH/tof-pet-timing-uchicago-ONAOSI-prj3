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

PREPROCESS_FORMAT_VERSION = 3


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


def preprocessing_fingerprint(root, selection, config):
    family_config = {}
    for family in used_families(config):
        values = {
            "vertical_scale_limit_mV": config["preprocessing"][family][
                "vertical_scale_limit_mV"
            ]
        }
        if family == "timing":
            values["rising_edge_before_trigger_ns"] = config["preprocessing"][family][
                "rising_edge_before_trigger_ns"
            ]
        family_config[family] = values
    return canonical_hash(
        {
            "format_version": PREPROCESS_FORMAT_VERSION,
            "source": source_signature(root),
            "selection": selection.manifest["fingerprint"],
            "families": used_families(config),
            "materialized_window_ns": config["preprocessing"]["materialized_window_ns"],
            "family_config": family_config,
        }
    )


def _optional(directory, name):
    p = directory / f"{name}.npy"
    return np.load(p, mmap_mode="r") if p.is_file() else None


def load_preprocessed(directory, root, selection, config):
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("fingerprint") != preprocessing_fingerprint(root, selection, config):
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


def _close(a):
    a.flush()
    m = getattr(a, "_mmap", None)
    if m is not None:
        m.close()


def _window_bounds(trigger_index, trace_size, sample_interval_ns, before_ns, after_ns):
    start = int(trigger_index) - int(np.ceil(float(before_ns) / sample_interval_ns))
    stop = int(trigger_index) + int(np.ceil(float(after_ns) / sample_interval_ns)) + 1
    if start < 0 or stop > int(trace_size):
        return None
    return start, stop


def _compact_npy(path, keep):
    source = np.load(path, mmap_mode="r")
    temporary = path.with_name(path.stem + ".compact.npy")
    np.save(temporary, np.asarray(source[keep]))
    mmap = getattr(source, "_mmap", None)
    if mmap is not None:
        mmap.close()
    os.replace(temporary, path)


def preprocess_selected(root_file, selection, config, *, rebuild, logger):
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
    lookup = {int(e): i for i, e in enumerate(entries)}
    n = entries.size
    full_event_index = np.asarray(selection.event_index, dtype=np.int64)
    full_split = np.asarray(selection.split, dtype=np.int8)

    bias = open_memmap(base / "bias_voltage_V.npy", mode="w+", dtype=np.float64, shape=(n,))
    bias[:] = np.nan
    targets = {}
    starts = {}
    intervals = {}
    rise_a = {}
    rise_b = {}
    lengths = {}
    seen = np.zeros(n, dtype=bool)
    keep = np.zeros(n, dtype=bool)
    excluded_window = np.zeros(n, dtype=bool)

    ch = config["data"]["channels"]
    pol = {
        "energy": np.asarray(ch.get("polarities", [1, 1]), dtype=np.int8),
        "timing": np.asarray(ch.get("timing_polarities", [1, 1]), dtype=np.int8),
    }
    triggers = {
        "energy": selection.main_trigger_energy,
        "timing": selection.main_trigger_timing,
    }
    stops = {
        "energy": selection.main_stop_energy,
        "timing": selection.main_stop_timing,
    }
    mat = config["preprocessing"]["materialized_window_ns"]
    before, after = float(mat["before"]), float(mat["after"])
    io = config["preprocessing"].get("io", {})
    max_events = int(io.get("max_events", 0))
    args = dict(
        energy_channels_one_based=tuple(map(int, ch["energy"])),
        timing_channels_one_based=tuple(map(int, ch["timing"])) if "timing" in families else None,
        step_size=io.get("step_size", "128 MB"),
        entry_stop=max_events if max_events > 0 else None,
    )

    entry = 0
    for chunk in iterate_energy_chunks(root_file, **args):
        for local in range(chunk.event_index.size):
            row = lookup.get(entry)
            entry += 1
            if row is None:
                continue
            seen[row] = True
            bias[row] = float(chunk.bias_voltage_V[local])
            event_payload = {}
            event_ok = True

            for f in families:
                raw, gain, off, dt, hoff = _family_arrays(chunk, f)
                assert raw is not None and gain is not None and off is not None and dt is not None and hoff is not None and triggers[f] is not None
                lim = _limits(config["preprocessing"][f]["vertical_scale_limit_mV"])
                windows, st, iv, ra, rb = [], [], [], [], []
                pre = float(config["preprocessing"]["timing"]["rising_edge_before_trigger_ns"]) if f == "timing" else None

                for d in range(2):
                    s = decode_oriented(
                        np.asarray(raw[d][local], dtype=np.int16),
                        gain[local, d], off[local, d], int(pol[f][d])
                    )
                    s = np.clip(s, lim[d, 0], lim[d, 1])
                    interval = float(dt[local, d])
                    dt_ns = interval * 1e9
                    trig = int(triggers[f][row, d])
                    bounds = _window_bounds(trig, s.size, dt_ns, before, after)
                    if bounds is None:
                        excluded_window[row] = True
                        event_ok = False
                        break
                    a, b = bounds
                    w = np.asarray(s[a:b], dtype=np.float32)

                    if f == "energy":
                        # Energy pulses are clean and long-lived. Search LED from the
                        # start of the materialized pre-trigger baseline up to the peak.
                        onset = a
                        pulse = s[trig:b]
                        peak = trig + int(np.nanargmax(pulse)) if pulse.size else trig
                    else:
                        assert stops[f] is not None and pre is not None
                        pulse_stop = int(stops[f][row, d])
                        search = max(a, trig - int(np.ceil(pre / dt_ns)))
                        preseg = s[search:trig + 1]
                        onset = search + int(np.nanargmin(preseg)) if preseg.size else trig
                        peak_stop = min(s.size - 1, max(trig, pulse_stop))
                        pulse = s[trig:peak_stop + 1]
                        peak = trig + int(np.nanargmax(pulse)) if pulse.size else trig

                    if peak <= onset:
                        raise RuntimeError(
                            f"{root_file.name} event {selection.event_index[row]} {f} detector {d+1}: unable to define rising edge interval"
                        )
                    windows.append(w)
                    st.append(float(hoff[local, d]) + a * interval)
                    iv.append(interval)
                    ra.append(onset - a)
                    rb.append(peak - a)

                if not event_ok:
                    break
                event_payload[f] = (windows, st, iv, ra, rb)

            if not event_ok:
                continue

            for f, payload in event_payload.items():
                windows, st, iv, ra, rb = payload
                if f not in targets:
                    L = windows[0].size
                    if any(x.size != L for x in windows):
                        raise ValueError(f"{f} detector grids differ")
                    lengths[f] = L
                    targets[f] = open_memmap(base / f"{f}_windows_mV.npy", mode="w+", dtype=np.float32, shape=(n, 2, L))
                    starts[f] = open_memmap(base / f"{f}_window_start_time_s.npy", mode="w+", dtype=np.float64, shape=(n, 2))
                    intervals[f] = open_memmap(base / f"{f}_sample_interval_s.npy", mode="w+", dtype=np.float64, shape=(n, 2))
                    rise_a[f] = open_memmap(base / f"{f}_rising_start.npy", mode="w+", dtype=np.int32, shape=(n, 2))
                    rise_b[f] = open_memmap(base / f"{f}_rising_stop.npy", mode="w+", dtype=np.int32, shape=(n, 2))
                if any(x.size != lengths[f] for x in windows):
                    raise ValueError(f"{f} sample interval changed")
                targets[f][row] = np.stack(windows)
                starts[f][row] = st
                intervals[f][row] = iv
                rise_a[f][row] = ra
                rise_b[f][row] = rb
            keep[row] = True

    if not np.all(seen):
        raise RuntimeError(f"Failed to inspect {np.count_nonzero(~seen)} selected events")
    if not np.any(keep):
        raise RuntimeError("No selected events have a valid preprocessing window")

    _close(bias)
    for group in (targets, starts, intervals, rise_a, rise_b):
        for a in group.values():
            _close(a)

    kept_rows = np.flatnonzero(keep)
    if kept_rows.size != n:
        _compact_npy(base / "bias_voltage_V.npy", kept_rows)
        for f in families:
            for suffix in ("windows_mV", "window_start_time_s", "sample_interval_s", "rising_start", "rising_stop"):
                _compact_npy(base / f"{f}_{suffix}.npy", kept_rows)

    np.save(base / "event_index.npy", full_event_index[kept_rows])
    np.save(base / "split.npy", full_split[kept_rows])
    excluded_count = int(np.count_nonzero(excluded_window))
    manifest = {
        "format_version": PREPROCESS_FORMAT_VERSION,
        "fingerprint": preprocessing_fingerprint(root_file, selection, config),
        "source": str(root_file),
        "selection_dir": str(selection.directory),
        "n_input_selected": int(n),
        "n_events": int(kept_rows.size),
        "excluded_window_exceeds_trace": excluded_count,
        "families": list(families),
        "materialized_window_ns": {"before": before, "after": after},
        "energy_rising_interval": "materialized_window_start_to_peak" if "energy" in families else None,
        "timing_rising_interval": "configured_pretrigger_minimum_to_selected_pulse_peak" if "timing" in families else None,
        "time_reference": "absolute_native_acquisition_time",
        "denoising": False,
    }
    atomic_json(base / "manifest.json", manifest)
    logger.info(
        "Preprocessing %s | n=%d/%d | excluded_window=%d | families=%s",
        root_file.name, kept_rows.size, n, excluded_count, ",".join(families)
    )
    return load_preprocessed(base, root_file, selection, config)
