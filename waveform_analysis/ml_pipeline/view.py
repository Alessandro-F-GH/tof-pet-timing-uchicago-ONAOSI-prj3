from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from .dataset import PreparedDataset

MODE_FAMILY = {"energy_to_energy": "energy", "timing_to_timing": "timing"}


def mode_family(mode: str) -> str:
    try:
        return MODE_FAMILY[str(mode)]
    except KeyError as exc:
        raise ValueError(f"Unsupported channel mode: {mode}") from exc


# These names remain semantic helpers for callers, but the two supported modes
# deliberately use the same waveform family for input and timing target.
def source_family(mode: str) -> str:
    return mode_family(mode)


def target_family(mode: str) -> str:
    return mode_family(mode)


@dataclass(frozen=True)
class WaveformView:
    pair: np.ndarray
    time_ps: np.ndarray
    family: str

    def materialize(self, dtype=np.float32) -> np.ndarray:
        return np.asarray(self.pair, dtype=dtype)


def waveform_view(dataset: PreparedDataset, mode: str, indices: np.ndarray) -> WaveformView:
    family = mode_family(mode)
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
    family = mode_family(mode)
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
    family = mode_family(mode)
    try:
        mean_led = float(dataset.manifest["led_training_mean_ps"][family])
        true_tof = float(dataset.manifest["true_tof_ps"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"LED calibration is unavailable for {family}") from exc
    return mean_led - true_tof


def calibrated_led(dataset: PreparedDataset, mode: str) -> np.ndarray:
    true_tof = float(dataset.manifest["true_tof_ps"])
    return standard_delta(dataset, mode, "led") - calibration_bias_ps(dataset, mode) - true_tof


def anchor_shift_delta(dataset: PreparedDataset, mode: str) -> np.ndarray:
    family = mode_family(mode)
    values = dataset.energy_anchor_offset_ps if family == "energy" else dataset.timing_anchor_offset_ps
    if values is None:
        raise ValueError(f"{family} LED-to-anchor offsets are unavailable")
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 2:
        raise ValueError(f"Expected {family} anchor offsets shaped [event, detector]")
    return values[:, 0] - values[:, 1]


def model_target(dataset: PreparedDataset, mode: str) -> np.ndarray:
    family = mode_family(mode)
    values = dataset.energy_target_ps if family == "energy" else dataset.timing_target_ps
    if values is None:
        raise ValueError(f"{family} slide-corrected ML target is unavailable")
    return np.asarray(values, dtype=np.float64)


def corrected_timing_residual(target_ps: np.ndarray, paired_prediction_ps: np.ndarray) -> np.ndarray:
    target = np.asarray(target_ps, dtype=np.float64)
    prediction = np.asarray(paired_prediction_ps, dtype=np.float64)
    if target.shape != prediction.shape:
        raise ValueError(f"Target and paired prediction shapes differ: {target.shape} != {prediction.shape}")
    return target - prediction


def anchor_delta(dataset: PreparedDataset, mode: str) -> np.ndarray:
    family = mode_family(mode)
    values = dataset.energy_anchor_time_ps if family == "energy" else dataset.timing_anchor_time_ps
    if values is None:
        raise ValueError(f"{family} anchor timing is unavailable")
    values = np.asarray(values, dtype=np.float64)
    return values[:, 0] - values[:, 1]


def inverse_pair(dataset: PreparedDataset, mode: str, normalized_pair: np.ndarray) -> np.ndarray:
    family = mode_family(mode)
    transform = dataset.energy_transform if family == "energy" else dataset.timing_transform
    if transform is None:
        raise ValueError(f"{family} input transform is unavailable")
    return transform.inverse(normalized_pair)
