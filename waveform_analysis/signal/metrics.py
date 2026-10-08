"""Pure residual widths and errors using the established repository estimators.

CTR remains the direct F1 histogram FWHM. No Gaussian-sigma conversion or
single-detector sqrt(2) rescaling is introduced.
"""

from __future__ import annotations

import numpy as np
from utils_fit import fit_ctr_ps, fit_delta_times_ps
from utils_fit.direct_fwhm import direct_fwhm_from_histogram


def rmse_ps(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    values = values[np.isfinite(values)]
    if not values.size:
        return float("nan")
    return float(np.sqrt(np.mean(values**2)))


__all__ = ["rmse_ps", "fit_ctr_ps", "fit_delta_times_ps", "direct_fwhm_from_histogram"]
