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


def calibration_bias_ps(dataset: PreparedDataset, mode: str) -> float:
    """Training-only estimate C_hat_12 from the LED pair mean minus true TOF."""
    family = target_family(mode)
    try:
        mean_led = float(dataset.manifest["led_training_mean_ps"][family])
        true_tof = float(dataset.manifest["true_tof_ps"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"LED calibration is unavailable for {family}") from exc
    return mean_led - true_tof


def calibrated_led(dataset: PreparedDataset, mode: str) -> np.ndarray:
    """Slide-14 LED estimate after removing calibration and true TOF.

    This is Delta t_LED - C_hat_12 - TOF. It is the uncorrected reference
    residual used to compare LED against ML with the same CTR metric.
    """
    true_tof = float(dataset.manifest["true_tof_ps"])
    return standard_delta(dataset, mode, "led") - calibration_bias_ps(dataset, mode) - true_tof


def anchor_shift_delta(dataset: PreparedDataset, mode: str) -> np.ndarray:
    """Delta delta from slide 15, with delta_i = t_LED,i - t_a,i."""
    family = target_family(mode)
    values = (
        dataset.energy_anchor_offset_ps
        if family == "energy"
        else dataset.timing_anchor_offset_ps
    )
    if values is None:
        raise ValueError(f"{family} LED-to-anchor offsets are unavailable")
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 2:
        raise ValueError(f"Expected {family} anchor offsets shaped [event, detector]")
    return values[:, 0] - values[:, 1]


def model_target(dataset: PreparedDataset, mode: str) -> np.ndarray:
    """Canonical slide-15 target.

    y_target = Delta t_LED - Delta delta - TOF - C_hat_12.
    The Delta-delta term removes the discrete native-grid anchor shift from the
    supervised correction learned from windows centered on t_a.
    """
    return calibrated_led(dataset, mode) - anchor_shift_delta(dataset, mode)


def corrected_timing_residual(
    target_ps: np.ndarray,
    paired_prediction_ps: np.ndarray,
) -> np.ndarray:
    """CTR residual after slide-15 native-grid correction: y_target - y_theta."""
    target = np.asarray(target_ps, dtype=np.float64)
    prediction = np.asarray(paired_prediction_ps, dtype=np.float64)
    if target.shape != prediction.shape:
        raise ValueError(f"Target and paired prediction shapes differ: {target.shape} != {prediction.shape}")
    return target - prediction


def anchor_delta(dataset: PreparedDataset, mode: str) -> np.ndarray:
    """Native sample-anchor pair timing, retained for diagnostics."""
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
