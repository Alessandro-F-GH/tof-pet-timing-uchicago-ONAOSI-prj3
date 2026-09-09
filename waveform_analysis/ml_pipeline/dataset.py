from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

DATASET_FORMAT_VERSION = 13


@dataclass(frozen=True)
class InputTransform:
    minimum: np.ndarray
    maximum: np.ndarray

    def transform(self, values: np.ndarray) -> np.ndarray:
        x = np.asarray(values, dtype=np.float32)
        return ((x - self.minimum) / (self.maximum - self.minimum)).astype(np.float32)

    def inverse(self, values: np.ndarray) -> np.ndarray:
        x = np.asarray(values, dtype=np.float32)
        return (x * (self.maximum - self.minimum) + self.minimum).astype(np.float32)


@dataclass(frozen=True)
class PreparedDataset:
    directory: Path
    manifest: dict[str, Any]
    event_index: np.ndarray
    bias_voltage_V: np.ndarray
    training: np.ndarray
    validation: np.ndarray
    test: np.ndarray
    energy_windows: np.ndarray | None
    timing_windows: np.ndarray | None
    energy_time_ps: np.ndarray | None
    timing_time_ps: np.ndarray | None
    energy_transform: InputTransform | None
    timing_transform: InputTransform | None
    energy_target_ps: np.ndarray | None
    timing_target_ps: np.ndarray | None
    energy_anchor_time_ps: np.ndarray | None
    timing_anchor_time_ps: np.ndarray | None
    energy_led_time_ps: np.ndarray | None
    timing_led_time_ps: np.ndarray | None
    energy_cfd_time_ps: np.ndarray | None
    timing_cfd_time_ps: np.ndarray | None
    energy_anchor_offset_ps: np.ndarray | None
    timing_anchor_offset_ps: np.ndarray | None

    @property
    def n_events(self) -> int:
        return int(self.event_index.size)

    @property
    def development(self) -> np.ndarray:
        return np.concatenate([self.training, self.validation])


def _optional(directory, name, mmap=True):
    path = directory / f"{name}.npy"
    return np.load(path, mmap_mode="r" if mmap else None) if path.is_file() else None


def _transform(directory, family):
    path = directory / f"{family}_transform.npz"
    if not path.is_file():
        return None
    with np.load(path) as data:
        return InputTransform(
            np.asarray(data["minimum"], dtype=np.float32),
            np.asarray(data["maximum"], dtype=np.float32),
        )


def load_prepared_dataset(directory: str | Path) -> PreparedDataset:
    directory = Path(directory).resolve()
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    if int(manifest.get("format_version", -1)) != DATASET_FORMAT_VERSION:
        raise ValueError(f"Prepared dataset format mismatch in {directory}")
    with np.load(directory / "splits.npz") as split:
        training = np.asarray(split["training"], dtype=np.int64)
        validation = np.asarray(split["validation"], dtype=np.int64)
        test = np.asarray(split["test"], dtype=np.int64)
    return PreparedDataset(
        directory,
        manifest,
        np.load(directory / "event_index.npy", mmap_mode="r"),
        np.load(directory / "bias_voltage_V.npy", mmap_mode="r"),
        training,
        validation,
        test,
        _optional(directory, "energy_windows"),
        _optional(directory, "timing_windows"),
        _optional(directory, "energy_time_ps"),
        _optional(directory, "timing_time_ps"),
        _transform(directory, "energy"),
        _transform(directory, "timing"),
        _optional(directory, "energy_target_ps"),
        _optional(directory, "timing_target_ps"),
        _optional(directory, "energy_anchor_time_ps"),
        _optional(directory, "timing_anchor_time_ps"),
        _optional(directory, "energy_led_time_ps"),
        _optional(directory, "timing_led_time_ps"),
        _optional(directory, "energy_cfd_time_ps"),
        _optional(directory, "timing_cfd_time_ps"),
        _optional(directory, "energy_anchor_offset_ps"),
        _optional(directory, "timing_anchor_offset_ps"),
    )
