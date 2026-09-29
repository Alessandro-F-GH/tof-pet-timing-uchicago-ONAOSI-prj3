from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from utils_fit import fit_ctr_ps

# The waveform pipeline owns no separate CTR estimator.
ctr_estimate = fit_ctr_ps


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
    bootstrap_requested: int
    bootstrap_successful: int
    corrected_bootstrap_ctr_ps: np.ndarray
    led_bootstrap_ctr_ps: np.ndarray
    improvement_bootstrap_ps: np.ndarray


def paired_ctr_improvement(
    corrected_ps: np.ndarray,
    led_ps: np.ndarray,
    fit_cfg: dict,
    *,
    seed: int,
) -> PairedCTRImprovement:
    """Paired event-bootstrap comparison of ML-corrected CTR against LED CTR.

    The same resampled event indices are used for corrected and LED residuals on
    every bootstrap trial. Positive improvement means CTR_LED - CTR_ML > 0.
    """
    corrected = np.asarray(corrected_ps, dtype=np.float64).reshape(-1)
    led = np.asarray(led_ps, dtype=np.float64).reshape(-1)
    if corrected.shape != led.shape:
        raise ValueError("paired CTR comparison requires equal-length residual arrays")
    finite = np.isfinite(corrected) & np.isfinite(led)
    corrected = corrected[finite]
    led = led[finite]
    if corrected.size < 5:
        raise ValueError("paired CTR comparison requires at least 5 finite event pairs")

    corrected_point = ctr_estimate(corrected, fit_cfg, seed=int(seed), bootstrap=False)
    led_point = ctr_estimate(led, fit_cfg, seed=int(seed), bootstrap=False)
    improvement = float(led_point.ctr_ps - corrected_point.ctr_ps)
    fraction = improvement / float(led_point.ctr_ps) if led_point.ctr_ps != 0 else float("nan")

    requested = int(fit_cfg.get("bootstrap_samples", 500))
    corrected_trials: list[float] = []
    led_trials: list[float] = []
    improvement_trials: list[float] = []
    if requested > 1:
        rng = np.random.default_rng(int(seed))
        n = corrected.size
        for _ in range(requested):
            indices = rng.integers(0, n, size=n)
            try:
                corrected_trial = ctr_estimate(corrected[indices], fit_cfg, bootstrap=False)
                led_trial = ctr_estimate(led[indices], fit_cfg, bootstrap=False)
            except ValueError:
                continue
            if not (np.isfinite(corrected_trial.ctr_ps) and np.isfinite(led_trial.ctr_ps)):
                continue
            corrected_trials.append(float(corrected_trial.ctr_ps))
            led_trials.append(float(led_trial.ctr_ps))
            improvement_trials.append(float(led_trial.ctr_ps - corrected_trial.ctr_ps))

    corrected_array = np.asarray(corrected_trials, dtype=np.float64)
    led_array = np.asarray(led_trials, dtype=np.float64)
    improvement_array = np.asarray(improvement_trials, dtype=np.float64)
    corrected_error = float(np.std(corrected_array, ddof=1)) if corrected_array.size > 1 else float("nan")
    led_error = float(np.std(led_array, ddof=1)) if led_array.size > 1 else float("nan")
    improvement_error = float(np.std(improvement_array, ddof=1)) if improvement_array.size > 1 else float("nan")
    if improvement_array.size:
        ci_low, ci_high = np.quantile(improvement_array, [0.025, 0.975])
    else:
        ci_low = ci_high = float("nan")

    return PairedCTRImprovement(
        corrected_ctr_ps=float(corrected_point.ctr_ps),
        corrected_ctr_error_ps=corrected_error,
        led_ctr_ps=float(led_point.ctr_ps),
        led_ctr_error_ps=led_error,
        improvement_ps=improvement,
        improvement_error_ps=improvement_error,
        improvement_fraction=float(fraction),
        improvement_percent=float(100.0 * fraction),
        improvement_ci_low_ps=float(ci_low),
        improvement_ci_high_ps=float(ci_high),
        bootstrap_requested=requested,
        bootstrap_successful=int(improvement_array.size),
        corrected_bootstrap_ctr_ps=corrected_array,
        led_bootstrap_ctr_ps=led_array,
        improvement_bootstrap_ps=improvement_array,
    )


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
