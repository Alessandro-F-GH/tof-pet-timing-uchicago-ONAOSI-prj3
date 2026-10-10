from __future__ import annotations
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from numpy.typing import ArrayLike, NDArray
import numpy as np

DATASET_FORMAT_VERSION = 34


@dataclass(frozen=True)
class InputTransform:
    minimum: np.ndarray
    maximum: np.ndarray

    def transform(self, values: ArrayLike) -> NDArray[np.float32]:
        x = np.asarray(values, dtype=np.float32)
        normalized = ((x - self.minimum) / (self.maximum - self.minimum)).astype(
            np.float32
        )
        # Clip at each detector's configured physical amplitude limits.
        return np.clip(normalized, 0.0, 1.0, out=normalized)

    def inverse(self, values: ArrayLike) -> NDArray[np.float32]:
        x = np.asarray(values, dtype=np.float32)
        return (x * (self.maximum - self.minimum) + self.minimum).astype(np.float32)


@dataclass(frozen=True)
class PreparedDataset:
    directory: Path
    manifest: dict[str, Any]
    event_index: np.ndarray
    bias_voltage_V: np.ndarray
    energy_windows: np.ndarray | None
    timing_windows: np.ndarray | None
    energy_time_ps: np.ndarray | None
    timing_time_ps: np.ndarray | None
    energy_transform: InputTransform | None
    timing_transform: InputTransform | None
    energy_target_ps: np.ndarray | None
    timing_target_ps: np.ndarray | None
    energy_led_time_ps: np.ndarray | None
    timing_led_time_ps: np.ndarray | None

    @property
    def n_events(self) -> int:
        return int(self.event_index.size)


def _optional(directory: Path, name: str, mmap: bool = True) -> np.ndarray | None:
    p = directory / f"{name}.npy"
    return np.load(p, mmap_mode="r" if mmap else None) if p.is_file() else None


def _transform(directory: Path, family: str) -> InputTransform | None:
    p = directory / f"{family}_transform.npz"
    if not p.is_file():
        return None
    with np.load(p) as d:
        return InputTransform(
            np.asarray(d["minimum"], np.float32), np.asarray(d["maximum"], np.float32)
        )


def load_prepared_dataset(directory: str | Path) -> PreparedDataset:
    directory = Path(directory).resolve()
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    if int(manifest.get("format_version", -1)) != DATASET_FORMAT_VERSION:
        raise ValueError(f"Prepared dataset format mismatch in {directory}")
    return PreparedDataset(
        directory,
        manifest,
        np.load(directory / "event_index.npy", mmap_mode="r"),
        np.load(directory / "bias_voltage_V.npy", mmap_mode="r"),
        _optional(directory, "energy_windows"),
        _optional(directory, "timing_windows"),
        _optional(directory, "energy_time_ps"),
        _optional(directory, "timing_time_ps"),
        _transform(directory, "energy"),
        _transform(directory, "timing"),
        _optional(directory, "energy_target_ps"),
        _optional(directory, "timing_target_ps"),
        _optional(directory, "energy_led_time_ps"),
        _optional(directory, "timing_led_time_ps"),
    )


# Retain serialized identities while legacy imports resolve to this module.
from waveform_analysis.core.compat import (
    preserve_legacy_identity as _preserve_legacy_identity,
)

_preserve_legacy_identity(__name__, "waveform_analysis.ml_pipeline.dataset")
