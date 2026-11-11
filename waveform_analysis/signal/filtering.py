"""Optional waveform low-pass used consistently throughout ROOT preprocessing.

Filter COMPLETE native waveforms after ADC calibration, BEFORE feature
selection, threshold detection, and waveform cropping. Disabled by default.
"""
from __future__ import annotations
from functools import lru_cache
import numpy as np


@lru_cache(maxsize=32)
def _sos(order: int, cutoff_GHz: float, dt_s: float):
    from scipy.signal import butter
    fs_GHz = 1e-9 / dt_s
    if not 0 < cutoff_GHz < fs_GHz / 2:
        raise ValueError("lowpass cutoff must be below Nyquist")
    return butter(order, cutoff_GHz, btype="lowpass", fs=fs_GHz, output="sos")


def filter_native_waveform(signal_mV, interval_s, preprocessing):
    """Return original unchanged unless preprocessing.lowpass.enabled is true."""
    conf = preprocessing.get("lowpass", {})
    x = np.asarray(signal_mV, dtype=np.float64)
    if not conf.get("enabled", False):
        return x
    order = int(conf.get("order", 4))
    cutoff = float(conf.get("cutoff_GHz", 5.0))
    if order < 1 or not np.isfinite(interval_s) or interval_s <= 0:
        raise ValueError("Invalid lowpass settings / sampling period")
    from scipy.signal import sosfiltfilt
    return sosfiltfilt(_sos(order, cutoff, float(interval_s)), x)
