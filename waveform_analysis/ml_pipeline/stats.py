from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from utils_fit import fit_ctr_ps

ctr_estimate = fit_ctr_ps


def rmse_ps(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    values = values[np.isfinite(values)]
    if not values.size:
        return float("nan")
    return float(np.sqrt(np.mean(values**2)))


@dataclass(frozen=True)
class PairedCTRImprovement:
    corrected_ctr_ps: float
    corrected_ctr_error_ps: float
    led_ctr_ps: float
    led_ctr_error_ps: float
    improvement_ps: float
    improvement_error_ps: float
    improvement_fraction: float
    improvement_percent: float
    improvement_ci_low_ps: float
    improvement_ci_high_ps: float
    corrected_rmse_ps: float
    corrected_rmse_error_ps: float
    led_rmse_ps: float
    led_rmse_error_ps: float
    rmse_improvement_ps: float
    rmse_improvement_error_ps: float
    rmse_improvement_fraction: float
    rmse_improvement_percent: float
    rmse_improvement_ci_low_ps: float
    rmse_improvement_ci_high_ps: float
    bootstrap_requested: int
    bootstrap_successful: int
    rmse_bootstrap_successful: int
    corrected_bootstrap_ctr_ps: np.ndarray
    led_bootstrap_ctr_ps: np.ndarray
    improvement_bootstrap_ps: np.ndarray
    corrected_bootstrap_rmse_ps: np.ndarray
    led_bootstrap_rmse_ps: np.ndarray
    rmse_improvement_bootstrap_ps: np.ndarray


def paired_ctr_improvement(
    corrected_ps: np.ndarray,
    led_ps: np.ndarray,
    fit_cfg: dict,
    *,
    seed: int,
) -> PairedCTRImprovement:
    """Evaluate LED and ML on one blind split without any inner resampling.

    Corrected and LED metrics use exactly the same event population.  The split is
    the statistical unit for the study-level paired comparison: this function
    therefore returns one point estimate per split and deliberately performs no
    event-level bootstrap.  Positive improvement means baseline metric -
    corrected metric > 0.
    """
    corrected = np.asarray(corrected_ps, dtype=np.float64).reshape(-1)
    led = np.asarray(led_ps, dtype=np.float64).reshape(-1)
    if corrected.shape != led.shape:
        raise ValueError("paired comparison requires equal-length residual arrays")
    finite = np.isfinite(corrected) & np.isfinite(led)
    corrected = corrected[finite]
    led = led[finite]
    if corrected.size < 5:
        raise ValueError("paired comparison requires at least 5 finite event pairs")

    corrected_point = ctr_estimate(corrected, fit_cfg, seed=int(seed), bootstrap=False)
    led_point = ctr_estimate(led, fit_cfg, seed=int(seed), bootstrap=False)
    ctr_improvement = float(led_point.ctr_ps - corrected_point.ctr_ps)
    ctr_fraction = ctr_improvement / float(led_point.ctr_ps) if led_point.ctr_ps != 0 else float("nan")

    corrected_rmse = rmse_ps(corrected)
    led_rmse = rmse_ps(led)
    rmse_improvement = float(led_rmse - corrected_rmse)
    rmse_fraction = rmse_improvement / led_rmse if led_rmse != 0 else float("nan")

    empty = np.empty(0, dtype=np.float64)
    nan = float("nan")
    return PairedCTRImprovement(
        corrected_ctr_ps=float(corrected_point.ctr_ps),
        corrected_ctr_error_ps=nan,
        led_ctr_ps=float(led_point.ctr_ps),
        led_ctr_error_ps=nan,
        improvement_ps=ctr_improvement,
        improvement_error_ps=nan,
        improvement_fraction=float(ctr_fraction),
        improvement_percent=float(100.0 * ctr_fraction),
        improvement_ci_low_ps=nan,
        improvement_ci_high_ps=nan,
        corrected_rmse_ps=corrected_rmse,
        corrected_rmse_error_ps=nan,
        led_rmse_ps=led_rmse,
        led_rmse_error_ps=nan,
        rmse_improvement_ps=rmse_improvement,
        rmse_improvement_error_ps=nan,
        rmse_improvement_fraction=float(rmse_fraction),
        rmse_improvement_percent=float(100.0 * rmse_fraction),
        rmse_improvement_ci_low_ps=nan,
        rmse_improvement_ci_high_ps=nan,
        bootstrap_requested=0,
        bootstrap_successful=0,
        rmse_bootstrap_successful=0,
        corrected_bootstrap_ctr_ps=empty.copy(),
        led_bootstrap_ctr_ps=empty.copy(),
        improvement_bootstrap_ps=empty.copy(),
        corrected_bootstrap_rmse_ps=empty.copy(),
        led_bootstrap_rmse_ps=empty.copy(),
        rmse_improvement_bootstrap_ps=empty.copy(),
    )


def residual_summary(values_ps: np.ndarray) -> dict[str, float | int]:
    values = np.asarray(values_ps, dtype=np.float64).reshape(-1)
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return {
            "n_total": int(values.size), "n_finite": 0, "rmse_ps": float("nan"),
            "mean_ps": float("nan"), "std_ps": float("nan"), "min_ps": float("nan"),
            "max_ps": float("nan"), "q01_ps": float("nan"), "q99_ps": float("nan"),
        }
    return {
        "n_total": int(values.size), "n_finite": int(finite.size), "rmse_ps": rmse_ps(finite),
        "mean_ps": float(np.mean(finite)), "std_ps": float(np.std(finite)),
        "min_ps": float(np.min(finite)), "max_ps": float(np.max(finite)),
        "q01_ps": float(np.quantile(finite, 0.01)), "q99_ps": float(np.quantile(finite, 0.99)),
    }


def format_residual_summary(summary: dict[str, float | int]) -> str:
    return (
        f"n={summary['n_finite']}/{summary['n_total']} | "
        f"RMSE={summary['rmse_ps']:.3f} ps | mean={summary['mean_ps']:.3f} ps | "
        f"std={summary['std_ps']:.3f} ps | q01={summary['q01_ps']:.3f} ps | "
        f"q99={summary['q99_ps']:.3f} ps | min={summary['min_ps']:.3f} ps | "
        f"max={summary['max_ps']:.3f} ps"
    )
