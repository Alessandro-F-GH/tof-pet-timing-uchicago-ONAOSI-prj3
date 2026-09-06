from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from .common import read_json

DATASET_FORMAT_VERSION = 7
_SUPPORTED_DATASET_FORMAT_VERSIONS = {1, 2, 3, 4, 5, 6, 7}


@dataclass(frozen=True)
class PreparedDataset:
    directory: Path
    manifest: dict[str, Any]
    event_id: np.ndarray
    event_index: np.ndarray
    source_file_id: np.ndarray
    source_run_index: np.ndarray
    bias_voltage_V: np.ndarray
    amplitude_mV: np.ndarray
    noise_rms_mV: np.ndarray
    trigger_index: np.ndarray
    led_time_fs: np.ndarray
    cfd_time_fs: np.ndarray
    windows_mV: np.ndarray
    relative_time_ps: np.ndarray
    energy_led_time_fs: np.ndarray | None = None
    timing_led_time_fs: np.ndarray | None = None
    energy_cfd_time_fs: np.ndarray | None = None
    timing_cfd_time_fs: np.ndarray | None = None
    energy_window_anchor_time_fs: np.ndarray | None = None
    timing_aligned_energy_window_anchor_time_fs: np.ndarray | None = None
    timing_window_anchor_time_fs: np.ndarray | None = None
    window_anchor_time_fs: np.ndarray | None = None
    timing_aligned_energy_windows_mV: np.ndarray | None = None
    timing_windows_mV: np.ndarray | None = None
    denoised_windows_mV: np.ndarray | None = None
    denoised_timing_aligned_energy_windows_mV: np.ndarray | None = None
    denoised_timing_windows_mV: np.ndarray | None = None
    timing_relative_time_ps: np.ndarray | None = None
    train: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=np.int64))
    validation: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=np.int64))
    test: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=np.int64))
    evaluation: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=np.int64))

    @property
    def input_length(self) -> int:
        return int(self.windows_mV.shape[2])

    @property
    def true_tof_ps(self) -> float:
        return float(self.manifest["true_tof_ps"])


def _load_array(directory: Path, name: str) -> np.ndarray:
    path = directory / f"{name}.npy"
    if not path.is_file():
        raise FileNotFoundError(f"Prepared dataset array not found: {path}")
    return np.load(path, mmap_mode="r")


def _load_optional_array(directory: Path, name: str) -> np.ndarray | None:
    path = directory / f"{name}.npy"
    return np.load(path, mmap_mode="r") if path.is_file() else None


def load_prepared_dataset(directory: str | Path) -> PreparedDataset:
    directory = Path(directory).resolve()
    manifest_path = directory / "manifest.json"
    split_path = directory / "splits.npz"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Not a prepared ML dataset: {directory}")

    manifest = read_json(manifest_path)
    version = int(manifest.get("format_version", -1))
    if version not in _SUPPORTED_DATASET_FORMAT_VERSIONS:
        raise ValueError(f"Unsupported prepared dataset version in {directory}")

    event_id = _load_array(directory, "event_id")
    if split_path.is_file():
        with np.load(split_path, allow_pickle=False) as splits:
            split_values = {
                name: splits[name].astype(np.int64)
                for name in ("train", "validation", "test", "evaluation")
            }
    else:
        all_indices = np.arange(event_id.size, dtype=np.int64)
        split_values = {
            "train": all_indices,
            "validation": np.empty(0, dtype=np.int64),
            "test": np.empty(0, dtype=np.int64),
            "evaluation": all_indices,
        }

    energy_led = _load_optional_array(directory, "energy_led_time_fs")
    timing_led = _load_optional_array(directory, "timing_led_time_fs")
    energy_cfd = _load_optional_array(directory, "energy_cfd_time_fs")
    timing_cfd = _load_optional_array(directory, "timing_cfd_time_fs")
    energy_anchor = _load_optional_array(directory, "energy_window_anchor_time_fs")
    timing_anchor = _load_optional_array(directory, "timing_window_anchor_time_fs")
    windows = _load_array(directory, "windows_mV")
    denoised = _load_optional_array(directory, "denoised_windows_mV")

    aligned_windows = _load_optional_array(directory, "timing_aligned_energy_windows_mV")
    aligned_anchor = _load_optional_array(directory, "timing_aligned_energy_window_anchor_time_fs")
    denoised_aligned = _load_optional_array(directory, "denoised_timing_aligned_energy_windows_mV")

    # Format v7 stores only one canonical energy window: the native sample window
    # referenced to the search-trigger threshold.  Downstream target-specific code
    # still expects the historical "timing_aligned_energy_*" fields.  Expose the
    # canonical arrays as zero-copy aliases so the public API remains compatible
    # without duplicating .npy waveform matrices on disk.
    if version >= 7 and aligned_windows is None:
        aligned_windows = windows
    if version >= 7 and aligned_anchor is None:
        aligned_anchor = energy_anchor
    if version >= 7 and denoised_aligned is None and denoised is not None:
        denoised_aligned = denoised

    generic_led = _load_optional_array(directory, "led_time_fs")
    if generic_led is None:
        generic_led = energy_led
    if generic_led is None:
        raise FileNotFoundError(
            f"Prepared dataset has neither led_time_fs.npy nor energy_led_time_fs.npy: {directory}"
        )

    generic_cfd = _load_optional_array(directory, "cfd_time_fs")
    if generic_cfd is None:
        generic_cfd = energy_cfd
    if generic_cfd is None:
        raise FileNotFoundError(
            f"Prepared dataset has neither cfd_time_fs.npy nor energy_cfd_time_fs.npy: {directory}"
        )

    return PreparedDataset(
        directory=directory,
        manifest=manifest,
        event_id=event_id,
        event_index=_load_array(directory, "event_index"),
        source_file_id=_load_array(directory, "source_file_id"),
        source_run_index=_load_array(directory, "source_run_index"),
        bias_voltage_V=_load_array(directory, "bias_voltage_V"),
        amplitude_mV=_load_array(directory, "amplitude_mV"),
        noise_rms_mV=_load_array(directory, "noise_rms_mV"),
        trigger_index=_load_array(directory, "trigger_index"),
        led_time_fs=generic_led,
        cfd_time_fs=generic_cfd,
        windows_mV=windows,
        relative_time_ps=_load_array(directory, "relative_time_ps"),
        energy_led_time_fs=energy_led,
        timing_led_time_fs=timing_led,
        energy_cfd_time_fs=energy_cfd,
        timing_cfd_time_fs=timing_cfd,
        energy_window_anchor_time_fs=energy_anchor,
        timing_aligned_energy_window_anchor_time_fs=aligned_anchor,
        timing_window_anchor_time_fs=timing_anchor,
        window_anchor_time_fs=energy_anchor,
        timing_aligned_energy_windows_mV=aligned_windows,
        timing_windows_mV=_load_optional_array(directory, "timing_windows_mV"),
        denoised_windows_mV=denoised,
        denoised_timing_aligned_energy_windows_mV=denoised_aligned,
        denoised_timing_windows_mV=_load_optional_array(directory, "denoised_timing_windows_mV"),
        timing_relative_time_ps=_load_optional_array(directory, "timing_relative_time_ps"),
        train=split_values["train"],
        validation=split_values["validation"],
        test=split_values["test"],
        evaluation=split_values["evaluation"],
    )


def load_prepared_dataset_spec(spec: str | Path | dict[str, Any]) -> PreparedDataset:
    """Load a canonical prepared dataset from a path or dataset object."""
    if isinstance(spec, (str, Path)):
        return load_prepared_dataset(spec)
    if not isinstance(spec, dict) or not str(spec.get("dataset", "")).strip():
        raise ValueError(
            "Dataset specification must be a path or an object containing 'dataset'"
        )
    return load_prepared_dataset(spec["dataset"])
