from __future__ import annotations

from typing import Any

import numpy as np

from .gaussian import FitResult, fit_delta_times_ps


def _fit_values(values_ps: np.ndarray, config: dict[str, Any] | None) -> np.ndarray:
    values = np.asarray(values_ps, dtype=np.float64).reshape(-1)
    values = values[np.isfinite(values)]
    limit = None if config is None else config.get("max_abs_ps")
    if limit is not None:
        limit = float(limit)
        if limit <= 0:
            raise ValueError("fit.max_abs_ps must be positive")
        values = values[np.abs(values) <= limit]
    return values


def fit_ctr_ps(values_ps: np.ndarray, config: dict[str, Any] | None = None) -> FitResult:
    """Canonical repository-wide Gaussian CTR estimator.

    When ``max_abs_ps`` is configured, only residuals inside that symmetric
    physical timing window are passed to the Gaussian fitter. CTR uncertainty
    is provided directly by ``FitResult.ctr_error_ps`` from the Gaussian fit.
    """
    values = _fit_values(values_ps, config)
    result = fit_delta_times_ps(values, method="ctr", config=config)
    if not result.success or not np.isfinite(result.ctr_ps):
        raise ValueError(result.message or "Gaussian CTR fit failed")
    return result
