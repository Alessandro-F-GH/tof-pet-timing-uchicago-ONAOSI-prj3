from __future__ import annotations

from typing import Any

import numpy as np

from .histogram import CTRResult, estimate_delta_times_ps


def fit_ctr_ps(
    values_ps: np.ndarray,
    config: dict[str, Any] | None = None,
    *,
    seed: int = 0,
    bootstrap: bool = True,
) -> CTRResult:
    """Canonical repository-wide CTR estimator.

    CTR is the direct full width at half maximum of a fixed-bin timing
    histogram. The histogram bin width is configured with ``fit.bin_width_ps``;
    uncertainty is the event-bootstrap standard deviation using
    ``fit.bootstrap_samples`` resamples. ``fit.max_abs_ps`` remains an optional
    symmetric physical timing window applied before histogramming.
    """
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
