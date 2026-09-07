from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .dataset import PreparedDataset

MODE_SOURCE = {
    "energy_to_energy": "energy",
    "energy_to_timing": "energy",
    "timing_to_timing": "timing",
}
MODE_TARGET = {
    "energy_to_energy": "energy",
    "energy_to_timing": "timing",
    "timing_to_timing": "timing",
}


@dataclass(frozen=True)
class WaveformView:
    source: np.ndarray
    indices: np.ndarray
    sample_slice: slice
    time_ps: np.ndarray
    family: str

    @property
    def n(self) -> int:
        return int(self.indices.size)

    def materialize(self, dtype=np.float32) -> np.ndarray:
        # One controlled copy at model fitting/prediction time. The source stays
        # memory-mapped and no per-model waveform cache is created.
        return np.asarray(self.source[self.indices, :, self.sample_slice], dtype=dtype)


def target_family(mode: str) -> str:
    try:
        return MODE_TARGET[str(mode)]
    except KeyError as exc:
        raise ValueError(f"Unsupported channel mode: {mode}") from exc


def source_family(mode: str) -> str:
    try:
        return MODE_SOURCE[str(mode)]
    except KeyError as exc:
        raise ValueError(f"Unsupported channel mode: {mode}") from exc


def _source_array(dataset: PreparedDataset, family: str, preprocessing: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    variants = preprocessing.get("input_variant_by_channel", {}) or {}
    variant = str(variants.get(family, "raw")).lower()
    if variant not in {"raw", "denoised"}:
        raise ValueError(f"Unknown {family} input variant: {variant}")
    if family == "energy":
        source = dataset.denoised_windows_mV if variant == "denoised" else dataset.windows_mV
        if source is None:
            raise ValueError("Denoised energy waveforms were requested but not materialized")
        return source, np.asarray(dataset.relative_time_ps, dtype=np.float64)
    if dataset.timing_windows_mV is None or dataset.timing_relative_time_ps is None:
        raise ValueError("Timing waveforms are unavailable in this prepared dataset")
    source = dataset.denoised_timing_windows_mV if variant == "denoised" else dataset.timing_windows_mV
    if source is None:
        raise ValueError("Denoised timing waveforms were requested but not materialized")
    return source, np.asarray(dataset.timing_relative_time_ps, dtype=np.float64)


def waveform_view(
    dataset: PreparedDataset,
    mode: str,
    indices: np.ndarray,
    *,
    start_ns: float,
    end_ns: float,
    subsampling: int,
    preprocessing: dict[str, Any],
) -> WaveformView:
    family = source_family(mode)
    source, time_ps = _source_array(dataset, family, preprocessing)
    factor = int(subsampling)
    if factor <= 0:
        raise ValueError("subsampling must be positive")
    low_ps, high_ps = float(start_ns) * 1000.0, float(end_ns) * 1000.0
    selected = np.flatnonzero((time_ps >= low_ps - 1e-9) & (time_ps <= high_ps + 1e-9))
    if selected.size < 2:
        raise ValueError(f"Window [{start_ns}, {end_ns}] ns contains fewer than two samples")
    first, last = int(selected[0]), int(selected[-1]) + 1
    if not np.array_equal(selected, np.arange(first, last)):
        raise ValueError("Requested waveform window is not contiguous")
    sample_slice = slice(first, last, factor)
    return WaveformView(
        source=source,
        indices=np.asarray(indices, dtype=np.int64),
        sample_slice=sample_slice,
        time_ps=np.asarray(time_ps[sample_slice], dtype=np.float64),
        family=family,
    )
