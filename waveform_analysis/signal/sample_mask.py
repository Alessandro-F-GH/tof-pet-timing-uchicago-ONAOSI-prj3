from __future__ import annotations

import numpy as np

SAMPLE_CONSTANT_FRACTION = 0.99


def training_sample_mask(
    train_x: np.ndarray,
    *,
    min_constant_fraction: float = SAMPLE_CONSTANT_FRACTION,
) -> np.ndarray:
    """Build one shared temporal keep-mask using training events only.

    A time sample is discarded when both detector channels independently have
    one exact normalized float32 value in at least min_constant_fraction of
    the training events. The returned binary mask is therefore identical for
    both input channels.
    """
    x = np.asarray(train_x, dtype=np.float32)
    if x.ndim != 3 or x.shape[1] != 2:
        raise ValueError(
            f"Sample mask expects [event, detector=2, sample], got {x.shape}"
        )
    if x.shape[0] < 1 or x.shape[2] < 1:
        raise ValueError(
            "Sample mask requires at least one training event and one time sample"
        )

    fraction = float(min_constant_fraction)
    if not 0.5 < fraction <= 1.0:
        raise ValueError("min_constant_fraction must lie in (0.5, 1]")

    required = int(np.ceil(fraction * x.shape[0] - 1e-12))

    # With a required fraction > 0.5, any value satisfying the threshold must
    # also be the median. Counting exact matches to the median avoids a Python
    # loop over time samples while preserving literal same-value semantics.
    reference = np.median(x, axis=0)
    constant_counts = np.count_nonzero(x == reference[None, :, :], axis=0)
    constant_by_channel = constant_counts >= required

    keep = ~np.all(constant_by_channel, axis=0)
    if not np.any(keep):
        raise ValueError(
            "Training-derived sample mask would remove every time sample; "
            "check the prepared waveform inputs"
        )
    return np.asarray(keep, dtype=bool)


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


# Retain serialized identities while legacy imports resolve to this module.
from waveform_analysis.core.compat import (
    preserve_legacy_identity as _preserve_legacy_identity,
)

_preserve_legacy_identity(__name__, "waveform_analysis.ml_pipeline.sample_mask")
