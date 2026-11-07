"""Streaming train-only per-time-sample normalization; no dataset copy."""
from __future__ import annotations

import numpy as np
import torch


def fit_time_sample_stats(waves, train_idx, *, chunk_size=256, scale_floor_ratio=0.05):
    """Fit mean/std [2,T] over training events using bounded-size chunks.

    Variance uses centered two-pass accumulation in float64 to avoid cancellation
    for nearly constant baseline signals. The std floor is relative to each
    detector's RMS temporal fluctuation; it prevents amplifying tiny baseline
    noise disproportionately. Neither validation nor holdout enters fitting.
    """
    if not 0 < scale_floor_ratio <= 1:
        raise ValueError("scale_floor_ratio must be in (0,1]")
    if chunk_size < 1 or len(train_idx) < 2:
        raise ValueError("Need positive chunk and at least two training rows")
    n = len(train_idx)
    total = None
    for left in range(0, n, chunk_size):
        batch = np.asarray(waves[train_idx[left:left + chunk_size]], dtype=np.float64)
        if batch.ndim != 3 or batch.shape[1] != 2:
            raise ValueError("Expected paired [events,2,T] waveform input")
        partial = batch.sum(axis=0)
        total = partial if total is None else total + partial
    mean = total / n
    squared = np.zeros_like(mean)
    for left in range(0, n, chunk_size):
        batch = np.asarray(waves[train_idx[left:left + chunk_size]], dtype=np.float64)
        deviations = batch - mean
        squared += np.sum(deviations * deviations, axis=0)
    standard_deviation = np.sqrt(squared / n)
    detector_rms = np.sqrt(np.mean(standard_deviation ** 2, axis=1))
    if np.any(~np.isfinite(detector_rms)) or np.any(detector_rms <= 0):
        raise ValueError("All training waveform values must have nonzero finite variability")
    floor = detector_rms[:, None] * scale_floor_ratio
    safe_scale = np.maximum(standard_deviation, floor)
    return mean.astype(np.float32), safe_scale.astype(np.float32), {
        "scale_floor_ratio": float(scale_floor_ratio),
        "detector_rms_std": detector_rms.tolist(),
        "detector_floor": floor[:, 0].tolist(),
        "min_unfloored_std": standard_deviation.min(axis=1).tolist(),
        "max_unfloored_std": standard_deviation.max(axis=1).tolist(),
        "fraction_floored": np.mean(standard_deviation < floor, axis=1).tolist(),
    }


def tensor_stats(mean, scale, *, device):
    return (torch.as_tensor(mean, device=device),
            torch.as_tensor(scale, device=device))


class FirstDifferenceView:
    """Lazy first-difference waveform view (no entire-dataset copy).

    d[...,0] = 0; d[...,t] = x[...,t] - x[...,t-1] for t >= 1.
    Differences are in waveform-amplitude units per sample, not volts/second.
    """

    def __init__(self, waves):
        self.waves = waves
        self.shape = waves.shape

    def __getitem__(self, indices):
        x = np.asarray(self.waves[indices], dtype=np.float32)
        result = np.empty_like(x)
        result[..., 0] = 0
        result[..., 1:] = x[..., 1:] - x[..., :-1]
        return result


def fit_first_difference_stats(waves, train_idx, *, chunk_size=256,
                               scale_floor_ratio=0.05):
    """Train-only, streaming statistics for per-detector temporal differences."""
    return fit_time_sample_stats(
        FirstDifferenceView(waves), train_idx, chunk_size=chunk_size,
        scale_floor_ratio=scale_floor_ratio)
