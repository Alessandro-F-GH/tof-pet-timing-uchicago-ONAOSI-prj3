from __future__ import annotations

from typing import Any

import numpy as np

from .binning import DEFAULT_HISTOGRAM_BIN_WIDTH_PS
from .histogram import (
    CTRResult,
    DEFAULT_CTR_DEFINITION,
    estimate_delta_times_integer_fs,
    estimate_delta_times_ps,
)


def fit_ctr_ps(
    values_ps: np.ndarray,
    config: dict[str, Any] | None = None,
    *,
    seed: int = 0,
    bootstrap: bool = True,
    definition: str = DEFAULT_CTR_DEFINITION,
    histogram_bin_width_ps: float = DEFAULT_HISTOGRAM_BIN_WIDTH_PS,
) -> CTRResult:
    """General CTR estimator with optional event-bootstrap uncertainty."""
    result = estimate_delta_times_ps(
        np.asarray(values_ps, dtype=np.float64),
        method="ctr",
        config=config,
        seed=int(seed),
        bootstrap=bool(bootstrap),
        definition=definition,
        histogram_bin_width_ps=histogram_bin_width_ps,
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
    definition: str = DEFAULT_CTR_DEFINITION,
    histogram_bin_width_ps: float = DEFAULT_HISTOGRAM_BIN_WIDTH_PS,
) -> CTRResult:
    """Generic timing-width extraction using one supported CTR definition."""
    return estimate_delta_times_ps(
        delta_ps,
        method=method,
        parameter=parameter,
        n_total=n_total,
        n_selected=n_selected,
        config=config,
        seed=seed,
        bootstrap=bootstrap,
        definition=definition,
        histogram_bin_width_ps=histogram_bin_width_ps,
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
    definition: str = DEFAULT_CTR_DEFINITION,
    histogram_bin_width_ps: float = DEFAULT_HISTOGRAM_BIN_WIDTH_PS,
) -> CTRResult:
    """Integer-fs counterpart of fit_delta_times_ps."""
    return estimate_delta_times_integer_fs(
        delta_fs,
        method=method,
        parameter=parameter,
        n_total=n_total,
        n_selected=n_selected,
        config=config,
        seed=seed,
        bootstrap=bootstrap,
        definition=definition,
        histogram_bin_width_ps=histogram_bin_width_ps,
    )
