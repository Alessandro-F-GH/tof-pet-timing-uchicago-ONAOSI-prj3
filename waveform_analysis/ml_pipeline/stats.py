from __future__ import annotations

import numpy as np

from utils_fit import fit_ctr_ps

# The waveform pipeline owns no CTR estimator. Direct fixed-bin histogram FWHM
# and bootstrap uncertainty live in utils_fit.
ctr_estimate = fit_ctr_ps


def residual_summary(values_ps: np.ndarray) -> dict[str, float | int]:
    """Compact diagnostics for timing residuals, used in metric-failure logs."""
    values = np.asarray(values_ps, dtype=np.float64).reshape(-1)
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return {
            "n_total": int(values.size),
            "n_finite": 0,
            "rmse_ps": float("nan"),
            "mean_ps": float("nan"),
            "std_ps": float("nan"),
            "min_ps": float("nan"),
            "max_ps": float("nan"),
            "q01_ps": float("nan"),
            "q99_ps": float("nan"),
        }
    return {
        "n_total": int(values.size),
        "n_finite": int(finite.size),
        "rmse_ps": float(np.sqrt(np.mean(finite**2))),
        "mean_ps": float(np.mean(finite)),
        "std_ps": float(np.std(finite)),
        "min_ps": float(np.min(finite)),
        "max_ps": float(np.max(finite)),
        "q01_ps": float(np.quantile(finite, 0.01)),
        "q99_ps": float(np.quantile(finite, 0.99)),
    }


def format_residual_summary(summary: dict[str, float | int]) -> str:
    return (
        f"n={summary['n_finite']}/{summary['n_total']} | "
        f"RMSE={summary['rmse_ps']:.3f} ps | mean={summary['mean_ps']:.3f} ps | "
        f"std={summary['std_ps']:.3f} ps | q01={summary['q01_ps']:.3f} ps | "
        f"q99={summary['q99_ps']:.3f} ps | min={summary['min_ps']:.3f} ps | "
        f"max={summary['max_ps']:.3f} ps"
    )
