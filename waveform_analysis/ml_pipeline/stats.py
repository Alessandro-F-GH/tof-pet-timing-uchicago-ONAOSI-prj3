from __future__ import annotations

from typing import Any

import numpy as np

from utils_fit import bootstrap_ctr_ps, fit_ctr_ps

# The waveform pipeline intentionally owns no CTR estimator. These names are
# thin imports for pipeline callers; all fitting and bootstrap logic lives in utils_fit.
gaussian_ctr = fit_ctr_ps


def bootstrap_ctr_uncertainty(
    values_ps: np.ndarray,
    samples: int,
    seed: int,
    config: dict[str, Any] | None = None,
) -> float:
    return bootstrap_ctr_ps(values_ps, samples, seed, config)[1]
