from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class RobustOutlierResult:
    mask: np.ndarray
    center: float
    robust_sigma: float
    max_distance: float
    rejected: int


def robust_mad_filter(
    values: np.ndarray,
    *,
    enabled: bool = True,
    zscore_limit: float = 4.0,
) -> RobustOutlierResult:
    # Mirrors waveform_analysis/ml_pipeline/prepared_data.py LED rejection.
    data = np.asarray(values, dtype=np.float64).reshape(-1)
    finite = np.isfinite(data)

    if not enabled:
        return RobustOutlierResult(
            mask=finite,
            center=float(np.median(data[finite])) if np.any(finite) else float("nan"),
            robust_sigma=float("nan"),
            max_distance=float("inf"),
            rejected=int(np.count_nonzero(~finite)),
        )

    if np.count_nonzero(finite) < 3:
        raise RuntimeError(
            "Too few valid LED timing pairs for 4-sigma outlier rejection"
        )

    core = data[finite]
    center = float(np.median(core))
    mad = float(np.median(np.abs(core - center)))
    sigma = 1.4826 * mad

    if not np.isfinite(sigma) or sigma <= 0.0:
        sigma = float(np.std(core, ddof=1)) if core.size > 1 else 0.0
    if not np.isfinite(sigma) or sigma <= 0.0:
        sigma = 1.0

    distance = float(zscore_limit) * sigma
    accepted = finite & (np.abs(data - center) <= distance)

    return RobustOutlierResult(
        mask=accepted,
        center=center,
        robust_sigma=sigma,
        max_distance=distance,
        rejected=int(np.count_nonzero(finite & ~accepted)),
    )
