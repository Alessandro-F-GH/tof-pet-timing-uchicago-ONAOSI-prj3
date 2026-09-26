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
    """Estimate CTR with the unique NEMA FWHM definition."""
    result = estimate_delta_times_ps(
        np.asarray(values_ps, dtype=np.float64),
        method="ctr",
        config=config,
        seed=int(seed),
        bootstrap=bool(bootstrap),
    )
    if not result.success or not np.isfinite(result.ctr_ps):
        raise ValueError(result.message or "CTR estimation failed")
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
    """NEMA timing-width extraction for floating-point residuals in ps."""
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
    """NEMA timing-width extraction for integer residuals in fs."""
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
