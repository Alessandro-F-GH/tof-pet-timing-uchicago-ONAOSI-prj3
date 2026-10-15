from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from numpy.typing import ArrayLike, DTypeLike, NDArray
from waveform_analysis.data.dataset import PreparedDataset

MODE_FAMILY = {"energy_to_energy": "energy", "timing_to_timing": "timing"}


def mode_family(mode: str) -> str:
    try:
        return MODE_FAMILY[str(mode)]
    except KeyError as e:
        raise ValueError(f"Unsupported channel mode: {mode}") from e


source_family = mode_family
target_family = mode_family


@dataclass(frozen=True)
class WaveformView:
    pair: np.ndarray
    time_ps: np.ndarray
    family: str

    def materialize(self, dtype: DTypeLike = np.float32) -> np.ndarray:
        return np.asarray(self.pair, dtype=dtype)


def waveform_view(
    dataset: PreparedDataset, mode: str, indices: ArrayLike
) -> WaveformView:
    f = mode_family(mode)
    waves = dataset.energy_windows if f == "energy" else dataset.timing_windows
    time = dataset.energy_time_ps if f == "energy" else dataset.timing_time_ps
    if waves is None or time is None:
        raise ValueError(f"{f} ML input unavailable")
    return WaveformView(
        np.asarray(waves[np.asarray(indices, np.int64)], np.float32),
        np.asarray(time, np.float64),
        f,
    )


def model_sample_mask(dataset: PreparedDataset, mode: str) -> NDArray[np.bool_]:
    """Return the universal control mask; energy always retains the full grid."""
    family = mode_family(mode)
    time = getattr(dataset, f"{family}_time_ps")
    if mode == "energy_to_energy":
        return np.ones(len(time), dtype=bool)
    mask = getattr(dataset, "sample_mask", None)
    if mask is None or dataset.manifest.get("sample_mask_fit_role") != "control":
        raise ValueError("Timing inputs require a frozen control-derived sample mask")
    mask = np.asarray(mask, dtype=bool)
    if mask.shape != np.asarray(time).shape or not mask.any():
        raise ValueError("Timing sample mask does not match the prepared input grid")
    return mask


def standard_delta(
    dataset: PreparedDataset, mode: str, method: str = "led"
) -> NDArray[np.float64]:
    """Return the LED time difference with the frozen control offset removed."""
    if method != "led":
        raise ValueError("Only the frozen LED timing is part of the study dataset")
    f = mode_family(mode)
    v = dataset.energy_led_time_ps if f == "energy" else dataset.timing_led_time_ps
    if v is None:
        raise ValueError(f"LED timing unavailable for {f}")
    v = np.asarray(v, np.float64)
    return v[:, 0] - v[:, 1] - float(dataset.manifest["led_control_mean_ps"])


def model_target(dataset: PreparedDataset, mode: str) -> NDArray[np.float64]:
    f = mode_family(mode)
    v = dataset.energy_target_ps if f == "energy" else dataset.timing_target_ps
    if v is None:
        raise ValueError(f"{f} target unavailable")
    return np.asarray(v, np.float64)


def corrected_timing_residual(
    target_ps: ArrayLike, prediction_ps: ArrayLike
) -> NDArray[np.float64]:
    a = np.asarray(target_ps, np.float64)
    b = np.asarray(prediction_ps, np.float64)
    if a.shape != b.shape:
        raise ValueError("target/prediction shape mismatch")
    return a - b
