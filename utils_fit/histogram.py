from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .binning import (
    DEFAULT_HISTOGRAM_BIN_WIDTH_PS,
    fixed_width_histogram_edges,
    validate_histogram_bin_width_ps,
)
from .nema import fit_nema_fwhm

FS_PER_PS = 1000.0
DEFAULT_INVALID_TIME_FS = np.iinfo(np.int64).min
DEFAULT_FIT_CONFIG: dict[str, Any] = {
    "histogram_bin_width_ps": DEFAULT_HISTOGRAM_BIN_WIDTH_PS,
    "bootstrap_samples": 500,
}


@dataclass
class CTRResult:
    """NEMA FWHM CTR estimate and event-bootstrap uncertainty."""

    method: str
    parameter: float
    success: bool
    n_total: int
    n_selected: int
    n_valid: int
    crossing_efficiency: float
    ctr_ps: float
    ctr_error_ps: float
    center_ps: float
    peak_height: float
    half_max_events: float
    left_half_ps: float
    right_half_ps: float
    histogram_bin_width_ps: float
    histogram_bins: int
    fit_low_ps: float
    fit_high_ps: float
    bootstrap_samples: int
    bootstrap_successful: int
    message: str = ""
    edges_ps: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=np.float64))
    counts: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=np.int64))

    @property
    def n_fit(self) -> int:
        return self.n_valid

    @property
    def mean_ps(self) -> float:
        return self.center_ps

    def as_dict(self) -> dict[str, Any]:
        return {
            "method": self.method,
            "parameter": self.parameter,
            "success": self.success,
            "n_total": self.n_total,
            "n_selected": self.n_selected,
            "n_rejected": self.n_total - self.n_selected,
            "n_valid": self.n_valid,
            "n_fit": self.n_valid,
            "crossing_efficiency": self.crossing_efficiency,
            "ctr_ps": self.ctr_ps,
            "ctr_error_ps": self.ctr_error_ps,
            "center_ps": self.center_ps,
            "peak_height": self.peak_height,
            "half_max_events": self.half_max_events,
            "left_half_ps": self.left_half_ps,
            "right_half_ps": self.right_half_ps,
            "histogram_bin_width_ps": self.histogram_bin_width_ps,
            "histogram_bins": self.histogram_bins,
            "fit_low_ps": self.fit_low_ps,
            "fit_high_ps": self.fit_high_ps,
            "bootstrap_samples": self.bootstrap_samples,
            "bootstrap_successful": self.bootstrap_successful,
            "message": self.message,
        }


FitResult = CTRResult


def _config(config: dict[str, Any] | None) -> dict[str, Any]:
    cfg = dict(DEFAULT_FIT_CONFIG)
    if config:
        cfg.update(config)
    unknown = set(cfg) - set(DEFAULT_FIT_CONFIG)
    if unknown:
        raise ValueError(f"Unknown CTR option(s): {sorted(unknown)}")

    width = validate_histogram_bin_width_ps(cfg["histogram_bin_width_ps"])
    samples_raw = cfg["bootstrap_samples"]
    if isinstance(samples_raw, bool):
        raise ValueError("fit.bootstrap_samples must be a non-negative integer")
    samples = int(samples_raw)
    if samples != samples_raw or samples < 0:
        raise ValueError("fit.bootstrap_samples must be a non-negative integer")

    cfg["histogram_bin_width_ps"] = width
    cfg["bootstrap_samples"] = samples
    return cfg


def _failure(
    *,
    method: str,
    parameter: float,
    n_total: int,
    n_selected: int,
    n_valid: int,
    histogram_bin_width_ps: float,
    bootstrap_samples: int,
    message: str,
) -> CTRResult:
    return CTRResult(
        method=method,
        parameter=float(parameter),
        success=False,
        n_total=int(n_total),
        n_selected=int(n_selected),
        n_valid=int(n_valid),
        crossing_efficiency=n_valid / n_selected if n_selected else 0.0,
        ctr_ps=np.nan,
        ctr_error_ps=np.nan,
        center_ps=np.nan,
        peak_height=np.nan,
        half_max_events=np.nan,
        left_half_ps=np.nan,
        right_half_ps=np.nan,
        histogram_bin_width_ps=float(histogram_bin_width_ps),
        histogram_bins=0,
        fit_low_ps=np.nan,
        fit_high_ps=np.nan,
        bootstrap_samples=int(bootstrap_samples),
        bootstrap_successful=0,
        message=message,
    )


def _estimate_values(
    values_ps: np.ndarray,
    *,
    method: str,
    parameter: float,
    n_total: int,
    n_selected: int,
    config: dict[str, Any] | None,
    seed: int,
    bootstrap: bool,
) -> CTRResult:
    cfg = _config(config)
    bin_width = float(cfg["histogram_bin_width_ps"])
    requested = int(cfg["bootstrap_samples"]) if bootstrap else 0

    values = np.asarray(values_ps, dtype=np.float64).reshape(-1)
    finite = values[np.isfinite(values)]
    n_valid = int(finite.size)
    if n_valid < 5:
        return _failure(
            method=method,
            parameter=parameter,
            n_total=n_total,
            n_selected=n_selected,
            n_valid=n_valid,
            histogram_bin_width_ps=bin_width,
            bootstrap_samples=requested,
            message="NEMA CTR requires at least 5 finite residuals",
        )

    try:
        point = fit_nema_fwhm(
            finite,
            histogram_bin_width_ps=bin_width,
        )
    except ValueError as exc:
        return _failure(
            method=method,
            parameter=parameter,
            n_total=n_total,
            n_selected=n_selected,
            n_valid=n_valid,
            histogram_bin_width_ps=bin_width,
            bootstrap_samples=requested,
            message=str(exc),
        )

    bootstrap_ctrs: list[float] = []
    if requested > 1:
        rng = np.random.default_rng(int(seed))
        for _ in range(requested):
            sample = finite[rng.integers(0, n_valid, size=n_valid)]
            try:
                trial = fit_nema_fwhm(
                    sample,
                    histogram_bin_width_ps=bin_width,
                )
            except ValueError:
                continue
            if np.isfinite(trial.ctr_ps):
                bootstrap_ctrs.append(float(trial.ctr_ps))

    error = (
        float(np.std(bootstrap_ctrs, ddof=1))
        if len(bootstrap_ctrs) > 1
        else float("nan")
    )
    edges = fixed_width_histogram_edges(finite, bin_width)
    counts, _ = np.histogram(finite, bins=edges)

    return CTRResult(
        method=method,
        parameter=float(parameter),
        success=True,
        n_total=int(n_total),
        n_selected=int(n_selected),
        n_valid=n_valid,
        crossing_efficiency=n_valid / n_selected if n_selected else 0.0,
        ctr_ps=float(point.ctr_ps),
        ctr_error_ps=error,
        center_ps=float(point.center_ps),
        peak_height=float(point.peak_height),
        half_max_events=0.5 * float(point.peak_height),
        left_half_ps=float(point.half_max_left_ps),
        right_half_ps=float(point.half_max_right_ps),
        histogram_bin_width_ps=float(point.histogram_bin_width_ps),
        histogram_bins=int(point.histogram_bins),
        fit_low_ps=float(point.fit_low_ps),
        fit_high_ps=float(point.fit_high_ps),
        bootstrap_samples=requested,
        bootstrap_successful=len(bootstrap_ctrs),
        edges_ps=np.asarray(edges, dtype=np.float64),
        counts=np.asarray(counts, dtype=np.int64),
    )


def estimate_delta_times_ps(
    delta_ps: np.ndarray,
    *,
    method: str,
    parameter: float = 0.0,
    n_total: int | None = None,
    n_selected: int | None = None,
    config: dict[str, Any] | None = None,
    seed: int = 0,
    bootstrap: bool = True,
) -> CTRResult:
    values = np.asarray(delta_ps, dtype=np.float64).reshape(-1)
    total = values.size if n_total is None else int(n_total)
    selected = values.size if n_selected is None else int(n_selected)
    return _estimate_values(
        values,
        method=method,
        parameter=parameter,
        n_total=total,
        n_selected=selected,
        config=config,
        seed=seed,
        bootstrap=bootstrap,
    )


def estimate_delta_times_integer_fs(
    delta_fs: np.ndarray,
    *,
    method: str,
    parameter: float = 0.0,
    n_total: int | None = None,
    n_selected: int | None = None,
    config: dict[str, Any] | None = None,
    seed: int = 0,
    bootstrap: bool = True,
) -> CTRResult:
    raw = np.asarray(delta_fs)
    if raw.ndim != 1:
        raw = raw.reshape(-1)
    if not np.issubdtype(raw.dtype, np.integer):
        raise TypeError("delta_fs must be an integer array")
    values = raw.astype(np.float64, copy=False) / FS_PER_PS
    total = values.size if n_total is None else int(n_total)
    selected = values.size if n_selected is None else int(n_selected)
    return _estimate_values(
        values,
        method=method,
        parameter=parameter,
        n_total=total,
        n_selected=selected,
        config=config,
        seed=seed,
        bootstrap=bootstrap,
    )


fit_delta_times_ps = estimate_delta_times_ps
fit_delta_times_integer_fs = estimate_delta_times_integer_fs


def scan_timing_grid(
    times_a_fs: np.ndarray,
    times_b_fs: np.ndarray,
    selected: np.ndarray,
    parameters: np.ndarray,
    *,
    method: str,
    config: dict[str, Any],
    invalid_time_fs: int = DEFAULT_INVALID_TIME_FS,
) -> list[CTRResult]:
    a_grid = np.asarray(times_a_fs)
    b_grid = np.asarray(times_b_fs)
    selected_mask = np.asarray(selected, dtype=bool)
    parameters = np.asarray(parameters, dtype=np.float64)
    if a_grid.shape != b_grid.shape:
        raise ValueError(f"{method} channel timing grids have different shapes")
    if a_grid.ndim != 2 or a_grid.shape[1] != parameters.size:
        raise ValueError(f"{method} timing-grid shape does not match parameter grid")
    if selected_mask.shape != (a_grid.shape[0],):
        raise ValueError("selection mask shape does not match timing arrays")
    n_total = int(selected_mask.size)
    n_selected = int(np.count_nonzero(selected_mask))
    results: list[CTRResult] = []
    for index, parameter in enumerate(parameters):
        a = a_grid[selected_mask, index].astype(np.int64, copy=False)
        b = b_grid[selected_mask, index].astype(np.int64, copy=False)
        valid = (a != int(invalid_time_fs)) & (b != int(invalid_time_fs))
        results.append(
            estimate_delta_times_integer_fs(
                a[valid] - b[valid],
                method=method,
                parameter=float(parameter),
                n_total=n_total,
                n_selected=n_selected,
                config=config,
                bootstrap=False,
            )
        )
    return results


def choose_best(results: list[CTRResult]) -> CTRResult | None:
    successful = [item for item in results if item.success and np.isfinite(item.ctr_ps)]
    return min(successful, key=lambda item: item.ctr_ps) if successful else None
