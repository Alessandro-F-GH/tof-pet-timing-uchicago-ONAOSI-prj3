from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from utils_fit import fit_direct_fwhm

from .reporting_fit import _reporting_ctr, _reporting_fit_config
from .splits import semantic_seed


def _threshold_label(value: float) -> str:
    return f"{float(value):g}".replace("-", "m").replace(".", "p") + "mV"


def _paired_reporting_improvement(
    led_residual: np.ndarray,
    model_residual: np.ndarray,
    *,
    histogram_bin_width_ps: float,
    bootstrap_samples: int,
    seed: int,
) -> tuple[float, float, int]:
    """Paired F1 improvement using bootstrap median as reporting central value."""
    led = np.asarray(led_residual, dtype=np.float64).reshape(-1)
    model = np.asarray(model_residual, dtype=np.float64).reshape(-1)
    if led.shape != model.shape:
        raise ValueError("Paired threshold-scan residual arrays are not aligned")

    finite = np.isfinite(led) & np.isfinite(model)
    led = led[finite]
    model = model[finite]
    if led.size < 5:
        raise ValueError("Paired threshold-scan reporting requires at least 5 events")

    rng = np.random.default_rng(int(seed))
    values: list[float] = []
    for _ in range(int(bootstrap_samples)):
        indices = rng.integers(0, led.size, size=led.size)
        try:
            led_ctr = fit_direct_fwhm(
                led[indices],
                histogram_bin_width_ps=histogram_bin_width_ps,
            ).ctr_ps
            model_ctr = fit_direct_fwhm(
                model[indices],
                histogram_bin_width_ps=histogram_bin_width_ps,
            ).ctr_ps
        except ValueError:
            continue
        if np.isfinite(led_ctr) and led_ctr > 0.0 and np.isfinite(model_ctr):
            values.append(100.0 * (led_ctr - model_ctr) / led_ctr)

    if not values:
        raise ValueError("No successful paired bootstrap replicate for threshold reporting")

    bootstrap = np.asarray(values, dtype=np.float64)
    central = float(np.median(bootstrap))
    uncertainty = (
        float(np.std(bootstrap, ddof=1))
        if bootstrap.size > 1
        else float("nan")
    )
    return central, uncertainty, int(bootstrap.size)


def recompute_threshold_scan_rows(
    run_dir: str | Path,
    rows: list[dict[str, Any]],
    *,
    histogram_bin_width_ps: float,
) -> list[dict[str, Any]]:
    """Recompute threshold-scan report values from persisted blind residuals.

    Stored CTR/improvement values are ignored. The threshold-scan CSV is used only
    for metadata and to locate the corresponding residual artifacts.
    """
    run = Path(run_dir).resolve()
    fit_config = _reporting_fit_config(histogram_bin_width_ps)
    bootstrap_samples = int(fit_config["bootstrap_samples"])
    recomputed: list[dict[str, Any]] = []

    for row in rows:
        updated = dict(row)
        dataset = str(row.get("dataset", ""))
        model = str(row.get("model", ""))
        threshold_raw = row.get("threshold_mV")
        if not dataset or not model or threshold_raw in {None, ""}:
            recomputed.append(updated)
            continue

        threshold = float(threshold_raw)
        artifact_dir = run / "artifacts" / dataset / _threshold_label(threshold)
        led_path = artifact_dir / "led_blind_residuals_ps.npy"
        model_path = artifact_dir / f"{model}_blind_residuals_ps.npy"
        if not led_path.is_file() or not model_path.is_file():
            missing = [
                str(path)
                for path in (led_path, model_path)
                if not path.is_file()
            ]
            raise FileNotFoundError(
                "Threshold-scan reporting requires persisted blind residuals; missing: "
                + ", ".join(missing)
            )

        led_residual = np.asarray(np.load(led_path), dtype=np.float64).reshape(-1)
        model_residual = np.asarray(np.load(model_path), dtype=np.float64).reshape(-1)

        led_fit = _reporting_ctr(
            led_residual,
            fit_config,
            seed=semantic_seed(
                0,
                dataset,
                "led",
                "threshold_report",
                f"{threshold:g}",
            ),
            bootstrap=True,
        )
        model_fit = _reporting_ctr(
            model_residual,
            fit_config,
            seed=semantic_seed(
                0,
                dataset,
                model,
                "threshold_report",
                f"{threshold:g}",
            ),
            bootstrap=True,
        )
        improvement, improvement_error, paired_successful = (
            _paired_reporting_improvement(
                led_residual,
                model_residual,
                histogram_bin_width_ps=float(histogram_bin_width_ps),
                bootstrap_samples=bootstrap_samples,
                seed=semantic_seed(
                    0,
                    dataset,
                    model,
                    "threshold_report_paired",
                    f"{threshold:g}",
                ),
            )
        )

        updated.update(
            {
                "led_blind_ctr_ps": float(led_fit.ctr_ps),
                "led_blind_ctr_uncertainty_ps": float(led_fit.ctr_error_ps),
                "model_blind_ctr_ps": float(model_fit.ctr_ps),
                "model_blind_ctr_uncertainty_ps": float(model_fit.ctr_error_ps),
                "relative_improvement_percent": float(improvement),
                "paired_bootstrap_uncertainty_percent": float(improvement_error),
                "paired_bootstrap_successful": int(paired_successful),
                "histogram_bin_width_ps": float(histogram_bin_width_ps),
                "bootstrap_samples": bootstrap_samples,
                "ctr_central_value": "bootstrap_median",
                "ctr_uncertainty_method": "bootstrap_standard_deviation",
                "improvement_central_value": "paired_bootstrap_median",
                "improvement_uncertainty_method": "paired_bootstrap_standard_deviation",
            }
        )
        recomputed.append(updated)

    return recomputed
