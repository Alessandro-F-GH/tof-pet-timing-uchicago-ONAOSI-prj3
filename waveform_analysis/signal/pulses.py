"""Pure ADC calibration, pulse finding and amplitude-selection kernels."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any
import numpy as np
from numpy.typing import ArrayLike, NDArray
from waveform_analysis.core.constants import MAD_NORMAL_SCALE, VOLTS_TO_MV


@dataclass(frozen=True)
class Hit:
    leading_index: int
    stop_index: int
    duration_ns: float


def robust_center_scale(values: ArrayLike) -> tuple[float, float]:
    """Finite median and original normal-scaled MAD for [event] values."""
    x = np.asarray(values, float)
    x = x[np.isfinite(x)]
    if not x.size:
        return float("nan"), float("nan")
    c = float(np.median(x))
    mad = float(np.median(np.abs(x - c)))
    return c, MAD_NORMAL_SCALE * mad


def decode_oriented(
    raw: ArrayLike, gain_v_per_count: float, offset_v: float, polarity: int
) -> NDArray[np.float64]:
    """Calibrate signed ADC samples [sample] to oriented mV, in original order."""
    if int(polarity) not in (-1, 1):
        raise ValueError("polarity must be +1 or -1")
    return float(polarity) * (
        (np.asarray(raw, float) * float(gain_v_per_count) - float(offset_v))
        * VOLTS_TO_MV
    )


def pulse_hits(
    signal_mV: ArrayLike, threshold_mV: float, sample_interval_s: float
) -> list[Hit]:
    """Return native-grid hits with the original leading/trailing interpolation."""
    y = np.asarray(signal_mV, float)
    threshold = float(threshold_mV)
    dt = float(sample_interval_s) * 1e9
    if y.size < 2 or threshold <= 0 or dt <= 0:
        return []
    y0, y1 = y[:-1], y[1:]
    finite = np.isfinite(y0) & np.isfinite(y1)
    rising = np.flatnonzero(finite & (y0 < threshold) & (y1 >= threshold))
    falling = np.flatnonzero(finite & (y0 >= threshold) & (y1 < threshold))

    def pos(i: int) -> float:
        d = float(y[i + 1] - y[i])
        return (
            float(i + 1)
            if not np.isfinite(d) or d == 0
            else float(i) + float(np.clip((threshold - y[i]) / d, 0, 1))
        )

    leads = np.asarray([pos(int(i)) for i in rising])
    trails = np.asarray([pos(int(i)) for i in falling])
    out = []
    for j, (lower, lead) in enumerate(zip(rising, leads)):
        next_lead = leads[j + 1] if j + 1 < leads.size else float(y.size - 1)
        cand = trails[(trails > lead) & (trails <= next_lead)]
        stop = float(cand[0]) if cand.size else next_lead
        out.append(
            Hit(
                int(lower) + 1,
                min(y.size - 1, int(np.ceil(stop))),
                max(0.0, float(stop - lead) * dt),
            )
        )
    return out


def photopeak_mask(amps: ArrayLike, rules: dict[str, Any]) -> NDArray[np.bool_]:
    """Apply the frozen inclusive detector amplitude intervals to [event, 2]."""
    m = np.all(np.isfinite(amps), axis=1)
    for d, (lo, hi) in enumerate(rules["photopeak_intervals_mV"]):
        m &= (amps[:, d] >= float(lo)) & (amps[:, d] <= float(hi))
    return m


# Existing pickles resolve Hit through the selection module, which re-exports it.
Hit.__module__ = "waveform_analysis.ml_pipeline.event_selection"
