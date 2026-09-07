from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .common import read_json

DATASET_FORMAT_VERSION = 7


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
    windows_mV: np.ndarray
    relative_time_ps: np.ndarray
    energy_led_time_fs: np.ndarray | None = None
    timing_led_time_fs: np.ndarray | None = None
    energy_cfd_time_fs: np.ndarray | None = None
    timing_cfd_time_fs: np.ndarray | None = None
    energy_window_anchor_time_fs: np.ndarray | None = None
    timing_window_anchor_time_fs: np.ndarray | None = None
    timing_windows_mV: np.ndarray | None = None
    timing_relative_time_ps: np.ndarray | None = None
    denoised_windows_mV: np.ndarray | None = None
    denoised_timing_windows_mV: np.ndarray | None = None
    # Raw-cache-only optional arrays. Current permanent v7 datasets normally do
    # not materialize these duplicate energy representations.
    timing_aligned_energy_window_anchor_time_fs: np.ndarray | None = None
    timing_aligned_energy_windows_mV: np.ndarray | None = None
    denoised_timing_aligned_energy_windows_mV: np.ndarray | None = None
    # Generic aliases are kept only because the physical preparation module
    # writes/reads them while constructing the current v7 dataset.
    led_time_fs: np.ndarray | None = None
    cfd_time_fs: np.ndarray | None = None
    window_anchor_time_fs: np.ndarray | None = None

    @property
    def n_events(self) -> int:
        return int(self.event_id.size)

    @property
    def input_length(self) -> int:
        return int(self.windows_mV.shape[-1])

    @property
    def true_tof_ps(self) -> float:
        return float(self.manifest.get("true_tof_ps", 0.0))


def _load(directory: Path, name: str, *, required: bool = True) -> np.ndarray | None:
    path = directory / f"{name}.npy"
    if not path.is_file():
        if required:
            raise FileNotFoundError(f"Prepared dataset array not found: {path}")
        return None
    return np.load(path, mmap_mode="r")


def load_prepared_dataset(directory: str | Path) -> PreparedDataset:
    root = Path(directory).resolve()
    manifest_path = root / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Not a prepared waveform dataset: {root}")
    manifest = read_json(manifest_path)
    version = int(manifest.get("format_version", -1))
    if version != DATASET_FORMAT_VERSION:
        raise ValueError(
            f"Prepared dataset {root} uses format {version}; current format is "
            f"{DATASET_FORMAT_VERSION}. Rebuild preprocessing."
        )

    energy_led = _load(root, "energy_led_time_fs", required=False)
    energy_cfd = _load(root, "energy_cfd_time_fs", required=False)
    energy_anchor = _load(root, "energy_window_anchor_time_fs", required=False)
    return PreparedDataset(
        directory=root,
        manifest=manifest,
        event_id=_load(root, "event_id"),
        event_index=_load(root, "event_index"),
        source_file_id=_load(root, "source_file_id"),
        source_run_index=_load(root, "source_run_index"),
        bias_voltage_V=_load(root, "bias_voltage_V"),
        amplitude_mV=_load(root, "amplitude_mV"),
        noise_rms_mV=_load(root, "noise_rms_mV"),
        trigger_index=_load(root, "trigger_index"),
        windows_mV=_load(root, "windows_mV"),
        relative_time_ps=_load(root, "relative_time_ps"),
        energy_led_time_fs=energy_led,
        timing_led_time_fs=_load(root, "timing_led_time_fs", required=False),
        energy_cfd_time_fs=energy_cfd,
        timing_cfd_time_fs=_load(root, "timing_cfd_time_fs", required=False),
        energy_window_anchor_time_fs=energy_anchor,
        timing_window_anchor_time_fs=_load(root, "timing_window_anchor_time_fs", required=False),
        timing_windows_mV=_load(root, "timing_windows_mV", required=False),
        timing_relative_time_ps=_load(root, "timing_relative_time_ps", required=False),
        denoised_windows_mV=_load(root, "denoised_windows_mV", required=False),
        denoised_timing_windows_mV=_load(root, "denoised_timing_windows_mV", required=False),
        timing_aligned_energy_window_anchor_time_fs=_load(root, "timing_aligned_energy_window_anchor_time_fs", required=False),
        timing_aligned_energy_windows_mV=_load(root, "timing_aligned_energy_windows_mV", required=False),
        denoised_timing_aligned_energy_windows_mV=_load(root, "denoised_timing_aligned_energy_windows_mV", required=False),
        led_time_fs=energy_led,
        cfd_time_fs=energy_cfd,
        window_anchor_time_fs=energy_anchor,
    )
