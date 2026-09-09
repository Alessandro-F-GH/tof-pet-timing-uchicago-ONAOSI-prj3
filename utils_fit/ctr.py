from __future__ import annotations

from typing import Any

import numpy as np

from .histogram import (
    CTRResult,
    estimate_delta_times_integer_fs,
    estimate_delta_times_ps,
)


def fit_ctr_ps(
    values_ps: np.ndarray,
    config: dict[str, Any] | None = None,
    *,
    seed: int = 0,
    bootstrap: bool = True,
) -> CTRResult:
    """Canonical repository-wide CTR estimator with optional bootstrap."""
    result = estimate_delta_times_ps(
        np.asarray(values_ps, dtype=np.float64),
        method="ctr",
        config=config,
        seed=int(seed),
        bootstrap=bool(bootstrap),
    )
    if not result.success or not np.isfinite(result.ctr_ps):
        raise ValueError(result.message or "Histogram FWHM CTR estimation failed")
    return result


def fit_delta_times_ps(
    delta_ps: np.ndarray,
    *,
    method: str,
    parameter: float = 0.0,
    n_total: int | None = None,
    n_selected: int | None = None,
    config: dict[str, Any] | None = None,
    seed: int = 0,
    bootstrap: bool = False,
) -> CTRResult:
    """Generic fixed-bin FWHM extraction; bootstrap is opt-in for scan callers."""
    return estimate_delta_times_ps(
        delta_ps,
        method=method,
        parameter=parameter,
        n_total=n_total,
        n_selected=n_selected,
        config=config,
        seed=seed,
        bootstrap=bootstrap,
    )


def fit_delta_times_integer_fs(
    delta_fs: np.ndarray,
    *,
    method: str,
    parameter: float = 0.0,
    n_total: int | None = None,
    n_selected: int | None = None,
    config: dict[str, Any] | None = None,
    seed: int = 0,
    bootstrap: bool = False,
) -> CTRResult:
    """Integer-fs counterpart of :func:`fit_delta_times_ps`."""
    return estimate_delta_times_integer_fs(
        delta_fs,
        method=method,
        parameter=parameter,
        n_total=n_total,
        n_selected=n_selected,
        config=config,
        seed=seed,
        bootstrap=bootstrap,
    )
