from __future__ import annotations

from typing import Any

import numpy as np

from utils_fit import FitResult, fit_delta_times_ps


def gaussian_ctr(values_ps: np.ndarray, config: dict[str, Any] | None = None) -> FitResult:
    """Evaluate CTR with the repository-wide Gaussian fitter in ``utils_fit``."""
    values = np.asarray(values_ps, dtype=np.float64).reshape(-1)
    values = values[np.isfinite(values)]
    result = fit_delta_times_ps(values, method="ctr", config=config)
    if not result.success or not np.isfinite(result.ctr_ps):
        raise ValueError(result.message or "Gaussian CTR fit failed")
    return result


def bootstrap_ctr_uncertainty(
    values_ps: np.ndarray,
    samples: int,
    seed: int,
    config: dict[str, Any] | None = None,
) -> float:
    """Bootstrap the same Gaussian CTR estimator used for the central value."""
    values = np.asarray(values_ps, dtype=np.float64).reshape(-1)
    values = values[np.isfinite(values)]
    if int(samples) <= 1:
        return float("nan")
    rng = np.random.default_rng(int(seed))
    ctrs: list[float] = []
    for _ in range(int(samples)):
        try:
            ctrs.append(gaussian_ctr(rng.choice(values, values.size, replace=True), config).ctr_ps)
        except ValueError:
            continue
    return float(np.std(ctrs, ddof=1)) if len(ctrs) > 1 else float("nan")
