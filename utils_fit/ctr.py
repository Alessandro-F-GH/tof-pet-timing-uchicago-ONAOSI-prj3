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
    physical timing window are passed to the Gaussian fitter.
    """
    values = _fit_values(values_ps, config)
    result = fit_delta_times_ps(values, method="ctr", config=config)
    if not result.success or not np.isfinite(result.ctr_ps):
        raise ValueError(result.message or "Gaussian CTR fit failed")
    return result


def bootstrap_ctr_ps(
    values_ps: np.ndarray,
    samples: int,
    seed: int,
    config: dict[str, Any] | None = None,
) -> tuple[FitResult, float]:
    """Return the Gaussian CTR fit and event-bootstrap CTR standard deviation."""
    values = _fit_values(values_ps, config)
    fit = fit_ctr_ps(values, config)
    if int(samples) <= 1:
        return fit, float("nan")
    rng = np.random.default_rng(int(seed))
    ctrs: list[float] = []
    for _ in range(int(samples)):
        try:
            ctrs.append(fit_ctr_ps(rng.choice(values, values.size, replace=True), config).ctr_ps)
        except ValueError:
            continue
    uncertainty = float(np.std(ctrs, ddof=1)) if len(ctrs) > 1 else float("nan")
    return fit, uncertainty
