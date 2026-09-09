from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from .dataset import PreparedDataset

MODE_SOURCE = {"energy_to_energy": "energy", "energy_to_timing": "energy", "timing_to_timing": "timing"}
MODE_TARGET = {"energy_to_energy": "energy", "energy_to_timing": "timing", "timing_to_timing": "timing"}


def source_family(mode: str) -> str:
    try:
        return MODE_SOURCE[str(mode)]
    except KeyError as exc:
        raise ValueError(f"Unsupported channel mode: {mode}") from exc


def target_family(mode: str) -> str:
    try:
        return MODE_TARGET[str(mode)]
    except KeyError as exc:
        raise ValueError(f"Unsupported channel mode: {mode}") from exc


@dataclass(frozen=True)
class WaveformView:
    pair: np.ndarray
    time_ps: np.ndarray
    family: str

    def materialize(self, dtype=np.float32) -> np.ndarray:
        return np.asarray(self.pair, dtype=dtype)


def waveform_view(dataset: PreparedDataset, mode: str, indices: np.ndarray) -> WaveformView:
    family = source_family(mode)
    waves, time_ps = (
        (dataset.energy_windows, dataset.energy_time_ps)
        if family == "energy"
        else (dataset.timing_windows, dataset.timing_time_ps)
    )
    if waves is None or time_ps is None:
        raise ValueError(f"{family} ML input is unavailable")
    return WaveformView(
        np.asarray(waves[np.asarray(indices, dtype=np.int64)], dtype=np.float32),
        np.asarray(time_ps, dtype=np.float64),
        family,
    )


def standard_delta(dataset: PreparedDataset, mode: str, method: str) -> np.ndarray:
    family = target_family(mode)
    if method == "led":
        values = dataset.energy_led_time_ps if family == "energy" else dataset.timing_led_time_ps
    elif method == "cfd":
        values = dataset.energy_cfd_time_ps if family == "energy" else dataset.timing_cfd_time_ps
    else:
        raise ValueError(f"Unknown standard method: {method}")
    if values is None:
        raise ValueError(f"{method.upper()} timing is unavailable for {family}")
    values = np.asarray(values, dtype=np.float64)
    return values[:, 0] - values[:, 1]


def calibrated_led(dataset: PreparedDataset, mode: str) -> np.ndarray:
    """Canonical supervised target: paired LED minus training-set LED mean."""
    family = target_family(mode)
    values = dataset.energy_target_ps if family == "energy" else dataset.timing_target_ps
    if values is None:
        raise ValueError(f"Calibrated LED target is unavailable for {family}")
    return np.asarray(values, dtype=np.float64)


def corrected_led_residual(calibrated_led_ps: np.ndarray, paired_prediction_ps: np.ndarray) -> np.ndarray:
    """Corrected timing used for CTR: calibrated LED minus paired model prediction."""
    led = np.asarray(calibrated_led_ps, dtype=np.float64)
    prediction = np.asarray(paired_prediction_ps, dtype=np.float64)
    if led.shape != prediction.shape:
        raise ValueError(f"Calibrated LED and paired prediction shapes differ: {led.shape} != {prediction.shape}")
    return led - prediction


def anchor_delta(dataset: PreparedDataset, mode: str) -> np.ndarray:
    """Native sample-anchor pair timing, for diagnostics only."""
    family = target_family(mode)
    values = dataset.energy_anchor_time_ps if family == "energy" else dataset.timing_anchor_time_ps
    if values is None:
        raise ValueError(f"{family} anchor timing is unavailable")
    values = np.asarray(values, dtype=np.float64)
    return values[:, 0] - values[:, 1]


def inverse_pair(dataset: PreparedDataset, mode: str, normalized_pair: np.ndarray) -> np.ndarray:
    family = source_family(mode)
    transform = dataset.energy_transform if family == "energy" else dataset.timing_transform
    if transform is None:
        raise ValueError(f"{family} input transform is unavailable")
    return transform.inverse(normalized_pair)
