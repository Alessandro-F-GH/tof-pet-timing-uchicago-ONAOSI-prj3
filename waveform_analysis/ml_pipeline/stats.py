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
    """Paired event-bootstrap comparison of ML-corrected residuals against LED residuals.

    Corrected and LED metrics always use the same event indices. RMSE is retained
    for every valid resample independently of whether the CTR estimator succeeds.
    Positive improvement means baseline metric - corrected metric > 0.
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

    requested = int(fit_cfg.get("bootstrap_samples", 500))
    corrected_ctr_trials: list[float] = []
    led_ctr_trials: list[float] = []
    ctr_improvement_trials: list[float] = []
    corrected_rmse_trials: list[float] = []
    led_rmse_trials: list[float] = []
    rmse_improvement_trials: list[float] = []
    if requested > 1:
        rng = np.random.default_rng(int(seed))
        n = corrected.size
        for _ in range(requested):
            indices = rng.integers(0, n, size=n)
            corrected_sample = corrected[indices]
            led_sample = led[indices]

            corrected_trial_rmse = rmse_ps(corrected_sample)
            led_trial_rmse = rmse_ps(led_sample)
            if np.isfinite(corrected_trial_rmse) and np.isfinite(led_trial_rmse):
                corrected_rmse_trials.append(corrected_trial_rmse)
                led_rmse_trials.append(led_trial_rmse)
                rmse_improvement_trials.append(float(led_trial_rmse - corrected_trial_rmse))

            try:
                corrected_trial = ctr_estimate(corrected_sample, fit_cfg, bootstrap=False)
                led_trial = ctr_estimate(led_sample, fit_cfg, bootstrap=False)
            except ValueError:
                continue
            if not (np.isfinite(corrected_trial.ctr_ps) and np.isfinite(led_trial.ctr_ps)):
                continue
            corrected_ctr_trials.append(float(corrected_trial.ctr_ps))
            led_ctr_trials.append(float(led_trial.ctr_ps))
            ctr_improvement_trials.append(float(led_trial.ctr_ps - corrected_trial.ctr_ps))

    corrected_ctr_array = np.asarray(corrected_ctr_trials, dtype=np.float64)
    led_ctr_array = np.asarray(led_ctr_trials, dtype=np.float64)
    ctr_improvement_array = np.asarray(ctr_improvement_trials, dtype=np.float64)
    corrected_rmse_array = np.asarray(corrected_rmse_trials, dtype=np.float64)
    led_rmse_array = np.asarray(led_rmse_trials, dtype=np.float64)
    rmse_improvement_array = np.asarray(rmse_improvement_trials, dtype=np.float64)

    def _std(values):
        return float(np.std(values, ddof=1)) if values.size > 1 else float("nan")

    def _ci(values):
        if values.size:
            low, high = np.quantile(values, [0.025, 0.975])
            return float(low), float(high)
        return float("nan"), float("nan")

    ctr_ci_low, ctr_ci_high = _ci(ctr_improvement_array)
    rmse_ci_low, rmse_ci_high = _ci(rmse_improvement_array)

    return PairedCTRImprovement(
        corrected_ctr_ps=float(corrected_point.ctr_ps),
        corrected_ctr_error_ps=_std(corrected_ctr_array),
        led_ctr_ps=float(led_point.ctr_ps),
        led_ctr_error_ps=_std(led_ctr_array),
        improvement_ps=ctr_improvement,
        improvement_error_ps=_std(ctr_improvement_array),
        improvement_fraction=float(ctr_fraction),
        improvement_percent=float(100.0 * ctr_fraction),
        improvement_ci_low_ps=ctr_ci_low,
        improvement_ci_high_ps=ctr_ci_high,
        corrected_rmse_ps=corrected_rmse,
        corrected_rmse_error_ps=_std(corrected_rmse_array),
        led_rmse_ps=led_rmse,
        led_rmse_error_ps=_std(led_rmse_array),
        rmse_improvement_ps=rmse_improvement,
        rmse_improvement_error_ps=_std(rmse_improvement_array),
        rmse_improvement_fraction=float(rmse_fraction),
        rmse_improvement_percent=float(100.0 * rmse_fraction),
        rmse_improvement_ci_low_ps=rmse_ci_low,
        rmse_improvement_ci_high_ps=rmse_ci_high,
        bootstrap_requested=requested,
        bootstrap_successful=int(ctr_improvement_array.size),
        rmse_bootstrap_successful=int(rmse_improvement_array.size),
        corrected_bootstrap_ctr_ps=corrected_ctr_array,
        led_bootstrap_ctr_ps=led_ctr_array,
        improvement_bootstrap_ps=ctr_improvement_array,
        corrected_bootstrap_rmse_ps=corrected_rmse_array,
        led_bootstrap_rmse_ps=led_rmse_array,
        rmse_improvement_bootstrap_ps=rmse_improvement_array,
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
