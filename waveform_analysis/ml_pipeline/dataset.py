from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

DATASET_FORMAT_VERSION = 9


@dataclass(frozen=True)
class InputTransform:
    mean: np.ndarray
    scale: np.ndarray

    def transform(self, values: np.ndarray) -> np.ndarray:
        return ((np.asarray(values, dtype=np.float32) - self.mean) / self.scale).astype(np.float32)

    def inverse(self, values: np.ndarray) -> np.ndarray:
        return (np.asarray(values, dtype=np.float32) * self.scale + self.mean).astype(np.float32)


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
    energy_led_time_ps: np.ndarray | None
    timing_led_time_ps: np.ndarray | None
    energy_cfd_time_ps: np.ndarray | None
    timing_cfd_time_ps: np.ndarray | None
    energy_anchor_time_ps: np.ndarray | None
    timing_anchor_time_ps: np.ndarray | None
    energy_target_ps: np.ndarray | None
    timing_target_ps: np.ndarray | None

    @property
    def n_events(self) -> int:
        return int(self.event_index.size)

    @property
    def development(self) -> np.ndarray:
        return np.sort(np.concatenate([self.training, self.validation])).astype(np.int64)

    @property
    def true_tof_ps(self) -> float:
        return float(self.manifest["true_tof_ps"])


def _optional(directory: Path, name: str) -> np.ndarray | None:
    path = directory / f"{name}.npy"
    return np.load(path, mmap_mode="r") if path.is_file() else None


def _transform(directory: Path, family: str) -> InputTransform | None:
    path = directory / f"{family}_transform.npz"
    if not path.is_file():
        return None
    with np.load(path) as values:
        return InputTransform(mean=np.asarray(values["mean"], dtype=np.float32), scale=np.asarray(values["scale"], dtype=np.float32))


def load_prepared_dataset(directory: str | Path) -> PreparedDataset:
    directory = Path(directory).resolve()
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    if int(manifest.get("format_version", -1)) != DATASET_FORMAT_VERSION:
        raise ValueError(f"Prepared dataset must use format {DATASET_FORMAT_VERSION}")
    with np.load(directory / "splits.npz") as split:
        training = np.asarray(split["training"], dtype=np.int64)
        validation = np.asarray(split["validation"], dtype=np.int64)
        test = np.asarray(split["test"], dtype=np.int64)
    return PreparedDataset(
        directory=directory, manifest=manifest,
        event_index=np.load(directory / "event_index.npy", mmap_mode="r"),
        bias_voltage_V=np.load(directory / "bias_voltage_V.npy", mmap_mode="r"),
        training=training, validation=validation, test=test,
        energy_windows=_optional(directory, "energy_windows"), timing_windows=_optional(directory, "timing_windows"),
        energy_time_ps=_optional(directory, "energy_time_ps"), timing_time_ps=_optional(directory, "timing_time_ps"),
        energy_transform=_transform(directory, "energy"), timing_transform=_transform(directory, "timing"),
        energy_led_time_ps=_optional(directory, "energy_led_time_ps"), timing_led_time_ps=_optional(directory, "timing_led_time_ps"),
        energy_cfd_time_ps=_optional(directory, "energy_cfd_time_ps"), timing_cfd_time_ps=_optional(directory, "timing_cfd_time_ps"),
        energy_anchor_time_ps=_optional(directory, "energy_anchor_time_ps"), timing_anchor_time_ps=_optional(directory, "timing_anchor_time_ps"),
        energy_target_ps=_optional(directory, "energy_target_ps"), timing_target_ps=_optional(directory, "timing_target_ps"),
    )
