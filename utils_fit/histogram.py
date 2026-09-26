from __future__ import annotations

from dataclasses import dataclass
import math
from statistics import NormalDist
from typing import Any

import numpy as np

from .double_gaussian import DoubleGaussianFit, fit_double_gaussian_fwhm
from .nema import fit_nema_fwhm

FS_PER_PS = 1000.0
DEFAULT_INVALID_TIME_FS = np.iinfo(np.int64).min
FWHM_SIGMA = 2.0 * math.sqrt(2.0 * math.log(2.0))
CTR_DEFINITIONS = ("shortest_interval", "double_gaussian", "nema")
DEFAULT_CTR_DEFINITION = "shortest_interval"
DEFAULT_HISTOGRAM_BINS = 22
_HISTOGRAM_CTR_DEFINITIONS = {"double_gaussian", "nema"}
DEFAULT_FIT_CONFIG: dict[str, Any] = {
    "coverage_fraction": 0.90,
    "bootstrap_samples": 500,
}


@dataclass
class CTRResult:
    """CTR estimate and uncertainty from one supported timing-width definition."""

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
    coverage_fraction: float
    interval_events: int
    interval_low_ps: float
    interval_high_ps: float
    interval_width_ps: float
    gaussian_equivalent_scale: float
    bootstrap_samples: int
    bootstrap_successful: int
    definition: str = DEFAULT_CTR_DEFINITION
    histogram_bins: int = 0
    sigma_narrow_ps: float = float("nan")
    sigma_wide_ps: float = float("nan")
    narrow_fraction: float = float("nan")
    message: str = ""

    @property
    def n_fit(self) -> int:
        return self.n_valid

    @property
    def mean_ps(self) -> float:
        return self.center_ps

    @property
    def mean_error_ps(self) -> float:
        return float("nan")

    @property
    def sigma_ps(self) -> float:
        return float("nan")

    @property
    def sigma_error_ps(self) -> float:
        return float("nan")

    @property
    def chi2(self) -> float:
        return float("nan")

    @property
    def ndof(self) -> int:
        return 0

    @property
    def chi2_ndof(self) -> float:
        return float("nan")

    @property
    def fit_low_ps(self) -> float:
        return self.interval_low_ps

    @property
    def fit_high_ps(self) -> float:
        return self.interval_high_ps

    @property
    def iterations(self) -> int:
        return 0

    @property
    def bin_phase_ps(self) -> float:
        return 0.0

    @property
    def phase_ctr_std_ps(self) -> float:
        return 0.0

    @property
    def expected(self) -> np.ndarray:
        return np.empty(0, dtype=np.float64)

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
            "coverage_fraction": self.coverage_fraction,
            "interval_events": self.interval_events,
            "interval_low_ps": self.interval_low_ps,
            "interval_high_ps": self.interval_high_ps,
            "interval_width_ps": self.interval_width_ps,
            "gaussian_equivalent_scale": self.gaussian_equivalent_scale,
            "bootstrap_samples": self.bootstrap_samples,
            "bootstrap_successful": self.bootstrap_successful,
            "definition": self.definition,
            "histogram_bins": self.histogram_bins,
            "sigma_narrow_ps": self.sigma_narrow_ps,
            "sigma_wide_ps": self.sigma_wide_ps,
            "narrow_fraction": self.narrow_fraction,
            "message": self.message,
        }


FitResult = CTRResult


@dataclass(frozen=True)
class _PointEstimate:
    ctr_ps: float
    center_ps: float
    interval_events: int = 0
    interval_low_ps: float = float("nan")
    interval_high_ps: float = float("nan")
    interval_width_ps: float = float("nan")
    gaussian_equivalent_scale: float = float("nan")
    histogram_bins: int = 0
    sigma_narrow_ps: float = float("nan")
    sigma_wide_ps: float = float("nan")
    narrow_fraction: float = float("nan")
    double_gaussian_fit: DoubleGaussianFit | None = None


def _gaussian_equivalent_scale(coverage_fraction: float) -> float:
    """Scale a central Gaussian coverage width to Gaussian FWHM."""
    p = float(coverage_fraction)
    z = NormalDist().inv_cdf(0.5 * (1.0 + p))
    return float(FWHM_SIGMA / (2.0 * z))


def _normalize_definition(definition: str) -> str:
    value = str(definition).strip().lower()
    if value not in CTR_DEFINITIONS:
        raise ValueError(
            f"Unknown CTR definition {definition!r}; expected one of {CTR_DEFINITIONS}"
        )
    return value


def _validate_histogram_bins(histogram_bins: int) -> int:
    if isinstance(histogram_bins, bool):
        raise ValueError("histogram_bins must be a positive integer")
    bins = int(histogram_bins)
    if bins != histogram_bins or bins <= 0:
        raise ValueError("histogram_bins must be a positive integer")
    return bins


def _failure(
    *,
    method: str,
    parameter: float,
    n_total: int,
    n_selected: int,
    n_valid: int,
    coverage_fraction: float,
    bootstrap_samples: int,
    definition: str,
    histogram_bins: int,
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
        coverage_fraction=(
            float(coverage_fraction) if definition == "shortest_interval" else np.nan
        ),
        interval_events=0,
        interval_low_ps=np.nan,
        interval_high_ps=np.nan,
        interval_width_ps=np.nan,
        gaussian_equivalent_scale=(
            _gaussian_equivalent_scale(coverage_fraction)
            if definition == "shortest_interval"
            else np.nan
        ),
        bootstrap_samples=int(bootstrap_samples),
        bootstrap_successful=0,
        definition=definition,
        histogram_bins=(
            histogram_bins if definition in _HISTOGRAM_CTR_DEFINITIONS else 0
        ),
        message=message,
    )


def _config(config: dict[str, Any] | None) -> dict[str, Any]:
    cfg = dict(DEFAULT_FIT_CONFIG)
    if config:
        cfg.update(config)
    unknown = set(cfg) - set(DEFAULT_FIT_CONFIG)
    if unknown:
        raise ValueError(f"Unknown CTR option(s): {sorted(unknown)}")
    coverage = float(cfg.get("coverage_fraction", 0.90))
    if not np.isfinite(coverage) or not 0.0 < coverage < 1.0:
        raise ValueError("fit.coverage_fraction must be in (0, 1)")
    samples = int(cfg.get("bootstrap_samples", 500))
    if samples < 0:
        raise ValueError("fit.bootstrap_samples must be >= 0")
    cfg["coverage_fraction"] = coverage
    cfg["bootstrap_samples"] = samples
    return cfg


def _shortest_interval(
    values_ps: np.ndarray,
    coverage_fraction: float,
) -> tuple[float, float, float, float, float, int]:
    """Return CTR and the shortest empirical interval containing the requested coverage."""
    values = np.sort(np.asarray(values_ps, dtype=np.float64).reshape(-1))
    n = int(values.size)
    if n < 2:
        raise ValueError("At least two values are required for an interval width")
    count = min(n, max(2, int(math.ceil(float(coverage_fraction) * n))))
    widths = values[count - 1 :] - values[: n - count + 1]
    if not widths.size:
        raise ValueError("Unable to construct shortest coverage interval")
    minimum = float(np.min(widths))
    tolerance = max(1e-12, abs(minimum) * 1e-12)
    candidates = np.flatnonzero(np.abs(widths - minimum) <= tolerance)
    if candidates.size > 1:
        median = float(np.median(values))
        centers = 0.5 * (values[candidates] + values[candidates + count - 1])
        index = int(candidates[np.argmin(np.abs(centers - median))])
    else:
        index = int(candidates[0])
    low = float(values[index])
    high = float(values[index + count - 1])
    width = float(high - low)
    center = 0.5 * (low + high)
    ctr = float(_gaussian_equivalent_scale(coverage_fraction) * width)
    return ctr, center, low, high, width, count


def _point_estimate(
    values_ps: np.ndarray,
    *,
    definition: str,
    coverage_fraction: float,
    histogram_bins: int,
    initial: _PointEstimate | None = None,
) -> _PointEstimate:
    if definition == "shortest_interval":
        ctr, center, low, high, width, count = _shortest_interval(
            values_ps,
            coverage_fraction,
        )
        return _PointEstimate(
            ctr_ps=ctr,
            center_ps=center,
            interval_events=count,
            interval_low_ps=low,
            interval_high_ps=high,
            interval_width_ps=width,
            gaussian_equivalent_scale=_gaussian_equivalent_scale(coverage_fraction),
        )

    if definition == "double_gaussian":
        seed_fit = initial.double_gaussian_fit if initial is not None else None
        fit = fit_double_gaussian_fwhm(
            values_ps,
            histogram_bins=histogram_bins,
            initial=seed_fit,
        )
        return _PointEstimate(
            ctr_ps=fit.ctr_ps,
            center_ps=fit.center_ps,
            histogram_bins=fit.histogram_bins,
            sigma_narrow_ps=fit.sigma_narrow_ps,
            sigma_wide_ps=fit.sigma_wide_ps,
            narrow_fraction=fit.narrow_fraction,
            double_gaussian_fit=fit,
        )

    fit = fit_nema_fwhm(
        values_ps,
        histogram_bins=histogram_bins,
    )
    return _PointEstimate(
        ctr_ps=fit.ctr_ps,
        center_ps=fit.center_ps,
        histogram_bins=fit.histogram_bins,
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
    definition: str,
    histogram_bins: int,
) -> CTRResult:
    cfg = _config(config)
    definition = _normalize_definition(definition)
    bins = _validate_histogram_bins(histogram_bins)
    values = np.asarray(values_ps, dtype=np.float64).reshape(-1)
    finite = values[np.isfinite(values)]
    n_valid = int(finite.size)
    coverage = float(cfg["coverage_fraction"])
    requested = int(cfg["bootstrap_samples"]) if bootstrap else 0
    minimum_events = {
        "shortest_interval": 2,
        "double_gaussian": 10,
        "nema": 5,
    }[definition]
    if n_valid < minimum_events:
        return _failure(
            method=method,
            parameter=parameter,
            n_total=n_total,
            n_selected=n_selected,
            n_valid=n_valid,
            coverage_fraction=coverage,
            bootstrap_samples=requested,
            definition=definition,
            histogram_bins=bins,
            message=(
                f"At least {minimum_events} finite residuals are required for "
                f"CTR definition {definition}"
            ),
        )

    try:
        point = _point_estimate(
            finite,
            definition=definition,
            coverage_fraction=coverage,
            histogram_bins=bins,
        )
    except ValueError as exc:
        return _failure(
            method=method,
            parameter=parameter,
            n_total=n_total,
            n_selected=n_selected,
            n_valid=n_valid,
            coverage_fraction=coverage,
            bootstrap_samples=requested,
            definition=definition,
            histogram_bins=bins,
            message=str(exc),
        )

    bootstrap_ctrs: list[float] = []
    if requested > 1:
        rng = np.random.default_rng(int(seed))
        for _ in range(requested):
            sample = finite[rng.integers(0, n_valid, size=n_valid)]
            try:
                trial = _point_estimate(
                    sample,
                    definition=definition,
                    coverage_fraction=coverage,
                    histogram_bins=bins,
                    initial=point,
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
        coverage_fraction=(coverage if definition == "shortest_interval" else np.nan),
        interval_events=int(point.interval_events),
        interval_low_ps=float(point.interval_low_ps),
        interval_high_ps=float(point.interval_high_ps),
        interval_width_ps=float(point.interval_width_ps),
        gaussian_equivalent_scale=float(point.gaussian_equivalent_scale),
        bootstrap_samples=requested,
        bootstrap_successful=len(bootstrap_ctrs),
        definition=definition,
        histogram_bins=(
            point.histogram_bins if definition in _HISTOGRAM_CTR_DEFINITIONS else 0
        ),
        sigma_narrow_ps=float(point.sigma_narrow_ps),
        sigma_wide_ps=float(point.sigma_wide_ps),
        narrow_fraction=float(point.narrow_fraction),
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
    definition: str = DEFAULT_CTR_DEFINITION,
    histogram_bins: int = DEFAULT_HISTOGRAM_BINS,
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
        definition=definition,
        histogram_bins=histogram_bins,
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
    definition: str = DEFAULT_CTR_DEFINITION,
    histogram_bins: int = DEFAULT_HISTOGRAM_BINS,
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
        definition=definition,
        histogram_bins=histogram_bins,
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
