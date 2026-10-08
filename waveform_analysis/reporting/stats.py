from __future__ import annotations

import numpy as np

from utils_fit import fit_ctr_ps
from waveform_analysis.signal.metrics import rmse_ps

from waveform_analysis.data.splits import bootstrap_draw_indices, semantic_seed

ctr_estimate = fit_ctr_ps


def _finite_pair(first, second):
    first = np.asarray(first, dtype=np.float64).reshape(-1)
    second = np.asarray(second, dtype=np.float64).reshape(-1)
    if first.shape != second.shape:
        raise ValueError("paired arrays must have identical shape")
    finite = np.isfinite(first) & np.isfinite(second)
    return first[finite], second[finite]


def metric_values(residuals_ps, fit_cfg: dict, *, seed: int) -> dict[str, float]:
    residuals = np.asarray(residuals_ps, dtype=np.float64).reshape(-1)
    residuals = residuals[np.isfinite(residuals)]
    if residuals.size < 1:
        raise ValueError("metric calculation requires finite residuals")
    ctr = ctr_estimate(residuals, fit_cfg, seed=int(seed), bootstrap=False)
    return {"ctr_ps": float(ctr.ctr_ps), "rmse_ps": rmse_ps(residuals)}


def paired_central_metrics(corrected_ps, led_ps, fit_cfg: dict, *, seed: int) -> dict:
    corrected, led = _finite_pair(corrected_ps, led_ps)
    if corrected.size < 5:
        raise ValueError("paired blind evaluation requires at least five finite events")
    model = metric_values(corrected, fit_cfg, seed=semantic_seed(seed, "model"))
    reference = metric_values(led, fit_cfg, seed=semantic_seed(seed, "led"))
    ctr_improvement = reference["ctr_ps"] - model["ctr_ps"]
    rmse_improvement = reference["rmse_ps"] - model["rmse_ps"]
    return {
        "n_events": int(corrected.size),
        "ctr_ps": model["ctr_ps"],
        "rmse_ps": model["rmse_ps"],
        "led_ctr_ps": reference["ctr_ps"],
        "led_rmse_ps": reference["rmse_ps"],
        "ctr_improvement_ps": float(ctr_improvement),
        "rmse_improvement_ps": float(rmse_improvement),
        "ctr_improvement_percent": float(100.0 * ctr_improvement / reference["ctr_ps"])
        if reference["ctr_ps"]
        else float("nan"),
        "rmse_improvement_percent": float(
            100.0 * rmse_improvement / reference["rmse_ps"]
        )
        if reference["rmse_ps"]
        else float("nan"),
    }


def _sample_std(values) -> float:
    values = np.asarray(values, dtype=np.float64)
    return float(np.std(values, ddof=1)) if values.size >= 2 else 0.0


def blind_event_bootstrap(
    corrected_ps,
    led_ps,
    fit_cfg: dict,
    *,
    n_resamples: int,
    seed: int,
) -> tuple[dict, dict[str, np.ndarray]]:
    corrected, led = _finite_pair(corrected_ps, led_ps)
    if corrected.size < 5:
        raise ValueError("blind bootstrap requires at least five finite paired events")
    n_resamples = int(n_resamples)
    if n_resamples < 1:
        raise ValueError("n_resamples must be >= 1")

    central = paired_central_metrics(
        corrected, led, fit_cfg, seed=semantic_seed(seed, "central")
    )
    draws = {
        "ctr_ps": np.empty(n_resamples, dtype=np.float64),
        "led_ctr_ps": np.empty(n_resamples, dtype=np.float64),
        "ctr_improvement_ps": np.empty(n_resamples, dtype=np.float64),
        "rmse_ps": np.empty(n_resamples, dtype=np.float64),
        "led_rmse_ps": np.empty(n_resamples, dtype=np.float64),
        "rmse_improvement_ps": np.empty(n_resamples, dtype=np.float64),
    }
    rng = np.random.default_rng(int(seed))
    for index in range(n_resamples):
        sampled = bootstrap_draw_indices(corrected.size, rng)
        model = metric_values(
            corrected[sampled],
            fit_cfg,
            seed=semantic_seed(seed, "draw", index, "model"),
        )
        reference = metric_values(
            led[sampled], fit_cfg, seed=semantic_seed(seed, "draw", index, "led")
        )
        draws["ctr_ps"][index] = model["ctr_ps"]
        draws["led_ctr_ps"][index] = reference["ctr_ps"]
        draws["ctr_improvement_ps"][index] = reference["ctr_ps"] - model["ctr_ps"]
        draws["rmse_ps"][index] = model["rmse_ps"]
        draws["led_rmse_ps"][index] = reference["rmse_ps"]
        draws["rmse_improvement_ps"][index] = reference["rmse_ps"] - model["rmse_ps"]

    summary = dict(central)
    summary.update(
        {
            "n_resamples": n_resamples,
            "ctr_bootstrap_std_ps": _sample_std(draws["ctr_ps"]),
            "led_ctr_bootstrap_std_ps": _sample_std(draws["led_ctr_ps"]),
            "ctr_improvement_bootstrap_std_ps": _sample_std(
                draws["ctr_improvement_ps"]
            ),
            "rmse_bootstrap_std_ps": _sample_std(draws["rmse_ps"]),
            "led_rmse_bootstrap_std_ps": _sample_std(draws["led_rmse_ps"]),
            "rmse_improvement_bootstrap_std_ps": _sample_std(
                draws["rmse_improvement_ps"]
            ),
            "bootstrap_unit": "blind_event",
            "bootstrap_retrains_model": False,
            "central_value_source": "original_non_resampled_blind_distribution",
        }
    )
    return summary, draws


def align_by_event_id(event_id_a, values_a, event_id_b, values_b):
    ids_a = np.asarray(event_id_a, dtype=np.int64).reshape(-1)
    ids_b = np.asarray(event_id_b, dtype=np.int64).reshape(-1)
    values_a = np.asarray(values_a, dtype=np.float64).reshape(-1)
    values_b = np.asarray(values_b, dtype=np.float64).reshape(-1)
    if ids_a.size != values_a.size or ids_b.size != values_b.size:
        raise ValueError("event IDs and values must have matching lengths")
    if (
        len(set(map(int, ids_a))) != ids_a.size
        or len(set(map(int, ids_b))) != ids_b.size
    ):
        raise ValueError("event IDs must be unique within each model output")
    pos_b = {int(event_id): index for index, event_id in enumerate(ids_b)}
    common = [
        (int(event_id), index, pos_b[int(event_id)])
        for index, event_id in enumerate(ids_a)
        if int(event_id) in pos_b
    ]
    if not common:
        raise ValueError("model outputs have no common blind event IDs")
    common.sort(key=lambda item: item[0])
    ids = np.asarray([item[0] for item in common], dtype=np.int64)
    a = np.asarray([values_a[item[1]] for item in common], dtype=np.float64)
    b = np.asarray([values_b[item[2]] for item in common], dtype=np.float64)
    finite = np.isfinite(a) & np.isfinite(b)
    return ids[finite], a[finite], b[finite]


def paired_model_bootstrap(
    event_id_a,
    residual_a,
    event_id_b,
    residual_b,
    fit_cfg: dict,
    *,
    metric: str,
    n_resamples: int,
    seed: int,
) -> dict:
    ids, first, second = align_by_event_id(
        event_id_a, residual_a, event_id_b, residual_b
    )
    metric = str(metric).lower()
    if metric not in {"ctr", "rmse"}:
        raise ValueError("metric must be 'ctr' or 'rmse'")
    n_resamples = int(n_resamples)
    if n_resamples < 1:
        raise ValueError("n_resamples must be >= 1")

    if metric == "ctr":
        central_a = metric_values(
            first, fit_cfg, seed=semantic_seed(seed, "central", "a")
        )["ctr_ps"]
        central_b = metric_values(
            second, fit_cfg, seed=semantic_seed(seed, "central", "b")
        )["ctr_ps"]
    else:
        central_a = rmse_ps(first)
        central_b = rmse_ps(second)
    central = float(central_a - central_b)

    rng = np.random.default_rng(int(seed))
    differences = np.empty(n_resamples, dtype=np.float64)
    for index in range(n_resamples):
        sampled = bootstrap_draw_indices(ids.size, rng)
        if metric == "ctr":
            a = metric_values(
                first[sampled], fit_cfg, seed=semantic_seed(seed, index, "a")
            )["ctr_ps"]
            b = metric_values(
                second[sampled], fit_cfg, seed=semantic_seed(seed, index, "b")
            )["ctr_ps"]
        else:
            a = rmse_ps(first[sampled])
            b = rmse_ps(second[sampled])
        differences[index] = a - b
    return {
        "metric": metric,
        "n_matched": int(ids.size),
        "difference_ps": central,
        "bootstrap_std_ps": _sample_std(differences),
    }


def pearson_r(first, second) -> float:
    first, second = _finite_pair(first, second)
    if first.size < 2 or np.std(first) == 0 or np.std(second) == 0:
        return float("nan")
    return float(np.corrcoef(first, second)[0, 1])


def residual_summary(values_ps: np.ndarray) -> dict[str, float | int]:
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
        "rmse_ps": rmse_ps(finite),
        "mean_ps": float(np.mean(finite)),
        "std_ps": float(np.std(finite)),
        "min_ps": float(np.min(finite)),
        "max_ps": float(np.max(finite)),
        "q01_ps": float(np.quantile(finite, 0.01)),
        "q99_ps": float(np.quantile(finite, 0.99)),
    }


def format_residual_summary(summary: dict[str, float | int]) -> str:
    return (
        f"n={summary['n_finite']}/{summary['n_total']} | RMSE={summary['rmse_ps']:.3f} ps | mean={summary['mean_ps']:.3f} ps | "
        f"std={summary['std_ps']:.3f} ps | q01={summary['q01_ps']:.3f} ps | q99={summary['q99_ps']:.3f} ps | min={summary['min_ps']:.3f} ps | max={summary['max_ps']:.3f} ps"
    )


# Retain serialized identities while legacy imports resolve to this module.
from waveform_analysis.core.compat import (
    preserve_legacy_identity as _preserve_legacy_identity,
)

_preserve_legacy_identity(__name__, "waveform_analysis.ml_pipeline.stats")
