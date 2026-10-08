from __future__ import annotations
import json, os, shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any
import numpy as np
from numpy.lib.format import open_memmap
from waveform_analysis.core.io import atomic_json, canonical_hash, source_signature
from waveform_analysis.data.root_io import iterate_energy_chunks
from waveform_analysis.engine.event_selection import (
    SelectionData as SelectionData,
    decode_oriented,
)

PREPROCESS_FORMAT_VERSION = 20


@dataclass(frozen=True)
class PreprocessedData:
    directory: Path
    manifest: dict[str, Any]
    event_index: np.ndarray
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


def _optional(d, n):
    p = d / f"{n}.npy"
    return np.load(p, mmap_mode="r") if p.is_file() else None


def _family(mode):
    return {"energy_to_energy": "energy", "timing_to_timing": "timing"}[str(mode)]


def preprocessing_fingerprint(root, selection, dataset, preprocessing, mode):
    f = _family(mode)
    return canonical_hash(
        {
            "format_version": PREPROCESS_FORMAT_VERSION,
            "source": source_signature(root),
            "selection": selection.manifest["fingerprint"],
            "family": f,
            "materialized_window_ns": preprocessing["materialized_window_ns"],
            "family_config": preprocessing[f],
        }
    )


def load_preprocessed(directory, root, selection, dataset, preprocessing, mode):
    m = json.loads((directory / "manifest.json").read_text())
    if m.get("fingerprint") != preprocessing_fingerprint(
        root, selection, dataset, preprocessing, mode
    ):
        raise ValueError("preprocessing cache stale")
    return PreprocessedData(
        directory,
        m,
        np.load(directory / "event_index.npy", mmap_mode="r"),
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


def _window_bounds(trigger, size, dt_ns, before, after):
    a = int(trigger) - int(np.ceil(before / dt_ns))
    b = int(trigger) + int(np.ceil(after / dt_ns)) + 1
    return None if a < 0 or b > int(size) else (a, b)


def preprocess_selected(
    root_file,
    selection,
    dataset,
    preprocessing,
    mode,
    *,
    cache_dir,
    rebuild=False,
    logger=None,
):
    root = Path(root_file).resolve()
    f = _family(mode)
    base = (
        Path(cache_dir).resolve()
        / canonical_hash(
            {"selection": selection.manifest["fingerprint"], "mode": mode}
        )[:16]
    )
    if base.is_dir() and not rebuild:
        try:
            return load_preprocessed(
                base, root, selection, dataset, preprocessing, mode
            )
        except (ValueError, FileNotFoundError):
            pass
    if base.exists():
        shutil.rmtree(base)
    base.mkdir(parents=True)
    entries = np.asarray(selection.entry_index, np.int64)
    lookup = {int(v): i for i, v in enumerate(entries)}
    n = entries.size
    ch = dataset["channels"]
    pol = np.asarray(
        ch.get("polarities", [1, 1])
        if f == "energy"
        else ch.get("timing_polarities", [1, 1]),
        np.int8,
    )
    io = preprocessing.get("io", {})
    mx = int(io.get("max_events", 0))
    args = dict(
        energy_channels_one_based=tuple(map(int, ch["energy"])),
        timing_channels_one_based=tuple(map(int, ch["timing"]))
        if f == "timing"
        else None,
        step_size=io.get("step_size", "128 MB"),
        entry_stop=mx if mx > 0 else None,
    )
    mat = preprocessing["materialized_window_ns"]
    before = float(mat["before"])
    after = float(mat["after"])
    bias = open_memmap(
        base / "bias_voltage_V.npy", mode="w+", dtype=np.float64, shape=(n,)
    )
    bias[:] = np.nan
    target = starts = intervals = ra = rb = None
    length = None
    keep = np.zeros(n, bool)
    seen = np.zeros(n, bool)
    entry = 0
    for chunk in iterate_energy_chunks(root, **args):
        for local in range(chunk.event_index.size):
            row = lookup.get(entry)
            entry += 1
            if row is None:
                continue
            seen[row] = True
            bias[row] = float(chunk.bias_voltage_V[local])
            raw = chunk.samples if f == "energy" else chunk.timing_samples
            gain = (
                chunk.vertical_gain_v_per_count
                if f == "energy"
                else chunk.timing_vertical_gain_v_per_count
            )
            off = (
                chunk.vertical_offset_v
                if f == "energy"
                else chunk.timing_vertical_offset_v
            )
            dt = (
                chunk.horizontal_interval_s
                if f == "energy"
                else chunk.timing_horizontal_interval_s
            )
            hoff = (
                chunk.horizontal_offset_s
                if f == "energy"
                else chunk.timing_horizontal_offset_s
            )
            assert (
                raw is not None
                and gain is not None
                and off is not None
                and dt is not None
                and hoff is not None
            )
            payload = []
            ok = True
            for d in range(2):
                s = decode_oriented(
                    raw[d][local], gain[local, d], off[local, d], int(pol[d])
                )
                interval = float(dt[local, d])
                dt_ns = interval * 1e9
                trig = int(selection.main_trigger[row, d])
                bounds = _window_bounds(trig, s.size, dt_ns, before, after)
                if bounds is None:
                    ok = False
                    break
                a, b = bounds
                w = np.asarray(s[a:b], np.float32)
                if f == "energy":
                    onset = a
                    peak = trig + int(np.nanargmax(s[trig:b]))
                else:
                    pre = float(
                        preprocessing["timing"]["rising_edge_before_trigger_ns"]
                    )
                    search = max(a, trig - int(np.ceil(pre / dt_ns)))
                    preseg = s[search : trig + 1]
                    onset = search + int(np.nanargmin(preseg)) if preseg.size else trig
                    pstop = min(s.size - 1, max(trig, int(selection.main_stop[row, d])))
                    pulse = s[trig : pstop + 1]
                    peak = trig + int(np.nanargmax(pulse))
                if peak <= onset:
                    ok = False
                    break
                payload.append(
                    (
                        w,
                        float(hoff[local, d]) + a * interval,
                        interval,
                        onset - a,
                        peak - a,
                    )
                )
            if not ok:
                continue
            if target is None:
                length = payload[0][0].size
                target = open_memmap(
                    base / f"{f}_windows_mV.npy",
                    mode="w+",
                    dtype=np.float32,
                    shape=(n, 2, length),
                )
                starts = open_memmap(
                    base / f"{f}_window_start_time_s.npy",
                    mode="w+",
                    dtype=np.float64,
                    shape=(n, 2),
                )
                intervals = open_memmap(
                    base / f"{f}_sample_interval_s.npy",
                    mode="w+",
                    dtype=np.float64,
                    shape=(n, 2),
                )
                ra = open_memmap(
                    base / f"{f}_rising_start.npy",
                    mode="w+",
                    dtype=np.int32,
                    shape=(n, 2),
                )
                rb = open_memmap(
                    base / f"{f}_rising_stop.npy",
                    mode="w+",
                    dtype=np.int32,
                    shape=(n, 2),
                )
            if any(p[0].size != length for p in payload):
                raise ValueError(f"{f} sample interval changed")
            target[row] = np.stack([p[0] for p in payload])
            starts[row] = [p[1] for p in payload]
            intervals[row] = [p[2] for p in payload]
            ra[row] = [p[3] for p in payload]
            rb[row] = [p[4] for p in payload]
            keep[row] = True
    if not np.all(seen):
        raise RuntimeError("failed to inspect every selected event")
    if not np.any(keep):
        raise RuntimeError("no selected events have a valid materialization window")
    _close(bias)
    for a in (target, starts, intervals, ra, rb):
        if a is not None:
            _close(a)
    rows = np.flatnonzero(keep)

    def compact(path):
        a = np.load(path, mmap_mode="r")
        tmp = path.with_name(path.stem + ".compact.npy")
        np.save(tmp, np.asarray(a[rows]))
        m = getattr(a, "_mmap", None)
        if m is not None:
            m.close()
        os.replace(tmp, path)

    if rows.size != n:
        compact(base / "bias_voltage_V.npy")
        for suffix in (
            "windows_mV",
            "window_start_time_s",
            "sample_interval_s",
            "rising_start",
            "rising_stop",
        ):
            compact(base / f"{f}_{suffix}.npy")
    np.save(base / "event_index.npy", np.asarray(selection.event_index)[rows])
    manifest = {
        "format_version": PREPROCESS_FORMAT_VERSION,
        "fingerprint": preprocessing_fingerprint(
            root, selection, dataset, preprocessing, mode
        ),
        "source": str(root),
        "mode": mode,
        "family": f,
        "selection_fingerprint": selection.manifest["fingerprint"],
        "n_events": int(rows.size),
        "materialized_window_ns": {"before": before, "after": after},
        "time_reference": "absolute_native_acquisition_time",
    }
    atomic_json(base / "manifest.json", manifest)
    if logger:
        logger.info(
            "Materialized preprocessing | %s | %s | n=%d", root.name, mode, rows.size
        )
    return load_preprocessed(base, root, selection, dataset, preprocessing, mode)


# Retain serialized identities while legacy imports resolve to this module.
from waveform_analysis.core.compat import (
    preserve_legacy_identity as _preserve_legacy_identity,
)

_preserve_legacy_identity(__name__, "waveform_analysis.ml_pipeline.data")
