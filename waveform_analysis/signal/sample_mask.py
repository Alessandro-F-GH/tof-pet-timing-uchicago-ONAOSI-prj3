from __future__ import annotations

import numpy as np

SAMPLE_CONSTANT_FRACTION = 0.99


def apply_sample_mask(pair: np.ndarray, sample_mask: np.ndarray | None) -> np.ndarray:
    """Apply a full-length mask, while accepting input that is already masked."""
    x = np.asarray(pair, dtype=np.float32)
    if sample_mask is None:
        return x

    mask = np.asarray(sample_mask, dtype=bool).reshape(-1)
    kept = int(np.count_nonzero(mask))
    if x.shape[-1] == mask.size:
        return np.asarray(x[..., mask], dtype=np.float32)
    if x.shape[-1] == kept:
        return x
    raise ValueError(
        f"Waveform has {x.shape[-1]} samples but sample mask expects "
        f"{mask.size} full or {kept} retained samples"
    )


def apply_sample_mask_to_time(
    time_ps: np.ndarray,
    sample_mask: np.ndarray | None,
) -> np.ndarray:
    """Apply the same temporal mask to a waveform time axis."""
    time = np.asarray(time_ps, dtype=np.float64)
    if sample_mask is None:
        return time

    mask = np.asarray(sample_mask, dtype=bool).reshape(-1)
    if time.size == mask.size:
        return time[mask]
    if time.size == int(np.count_nonzero(mask)):
        return time
    raise ValueError(
        f"Time axis has {time.size} samples but sample mask expects "
        f"{mask.size} full entries"
    )



def control_clipping_sample_mask(
    control_x: np.ndarray, *, min_clipped_fraction: float = SAMPLE_CONSTANT_FRACTION,
    chunk_size: int = 4096,
) -> np.ndarray:
    """Keep temporal samples unless both detectors are consistently clipped.

    Parameters
    ----------
    control_x : ndarray, shape (n_control_events, 2, n_samples)
        Normalized control waveforms; clipping limits are exactly zero and one.
    min_clipped_fraction : float
        Fraction required at a single limit independently for each detector.
    chunk_size : int
        Event block size, bounding temporary memory when reading cached arrays.

    Returns
    -------
    ndarray, shape (n_samples,)
        One shared keep-mask. Constant values inside the limits are retained.
    """
    x = np.asarray(control_x, dtype=np.float32)
    if x.ndim != 3 or x.shape[1] != 2 or x.shape[0] < 1 or x.shape[2] < 1:
        raise ValueError("Control clipping mask requires [event>=1, detector=2, sample>=1]")
    if not 0.5 < float(min_clipped_fraction) <= 1.0 or int(chunk_size) < 1:
        raise ValueError("Clipping fraction must lie in (0.5, 1] and chunk_size must be positive")
    lower = np.zeros(x.shape[1:], dtype=np.int64)
    upper = np.zeros_like(lower)
    for start in range(0, len(x), int(chunk_size)):
        block = x[start:start + int(chunk_size)]
        lower += np.count_nonzero(block == 0.0, axis=0)
        upper += np.count_nonzero(block == 1.0, axis=0)
    required = int(np.ceil(float(min_clipped_fraction) * len(x) - 1e-12))
    clipped = np.maximum(lower, upper) >= required
    keep = ~np.all(clipped, axis=0)
    if not np.any(keep):
        raise ValueError("Control clipping mask would remove every time sample")
    return np.asarray(keep, dtype=bool)
