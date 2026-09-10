from __future__ import annotations

from dataclasses import dataclass, field
import math
from statistics import NormalDist
from typing import Any

import numpy as np

FS_PER_PS = 1000.0
DEFAULT_INVALID_TIME_FS = np.iinfo(np.int64).min
FWHM_SIGMA = 2.0 * math.sqrt(2.0 * math.log(2.0))
DEFAULT_FIT_CONFIG: dict[str, Any] = {
    "min_events": 100,
    "coverage_fraction": 0.90,
    "bin_width_ps": 5.0,
    "bootstrap_samples": 500,
}


@dataclass
class CTRResult:
    """Robust timing-resolution result with a secondary core-peak FWHM diagnostic.

    ``ctr_ps`` is the Gaussian-equivalent shortest interval containing the
    configured fraction of all finite residuals. For the default 90% coverage,
    CTR = 0.715814... * W90. The histogram FWHM of the dominant local peak is
    retained separately as ``core_fwhm_ps`` and is never used as the canonical
    CTR value.
    """

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
    core_fwhm_ps: float
    core_fwhm_error_ps: float
    core_fraction: float
    left_half_ps: float
    right_half_ps: float
    half_max_events: float
    bin_width_ps: float
    bootstrap_samples: int
    bootstrap_successful: int
    core_bootstrap_successful: int
    message: str = ""
    edges_ps: np.ndarray = field(default_factory=lambda: np.empty(0), repr=False)
    counts: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=np.int64), repr=False)

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
            "core_fwhm_ps": self.core_fwhm_ps,
            "core_fwhm_error_ps": self.core_fwhm_error_ps,
            "core_fraction": self.core_fraction,
            "left_half_ps": self.left_half_ps,
            "right_half_ps": self.right_half_ps,
            "half_max_events": self.half_max_events,
            "bin_width_ps": self.bin_width_ps,
            "bootstrap_samples": self.bootstrap_samples,
            "bootstrap_successful": self.bootstrap_successful,
            "core_bootstrap_successful": self.core_bootstrap_successful,
            "message": self.message,
        }


FitResult = CTRResult


def _gaussian_equivalent_scale(coverage_fraction: float) -> float:
    """Scale a central Gaussian coverage width to the Gaussian FWHM."""
    p = float(coverage_fraction)
    z = NormalDist().inv_cdf(0.5 * (1.0 + p))
    return float(FWHM_SIGMA / (2.0 * z))


def _failure(
    *,
    method: str,
    parameter: float,
    n_total: int,
    n_selected: int,
    n_valid: int,
    coverage_fraction: float,
    bin_width_ps: float,
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
        coverage_fraction=float(coverage_fraction),
        interval_events=0,
        interval_low_ps=np.nan,
        interval_high_ps=np.nan,
        interval_width_ps=np.nan,
        gaussian_equivalent_scale=_gaussian_equivalent_scale(coverage_fraction),
        core_fwhm_ps=np.nan,
        core_fwhm_error_ps=np.nan,
        core_fraction=np.nan,
        left_half_ps=np.nan,
        right_half_ps=np.nan,
        half_max_events=np.nan,
        bin_width_ps=float(bin_width_ps),
        bootstrap_samples=int(bootstrap_samples),
        bootstrap_successful=0,
        core_bootstrap_successful=0,
        message=message,
    )


def _config(config: dict[str, Any] | None) -> dict[str, Any]:
    cfg = dict(DEFAULT_FIT_CONFIG)
    if config:
        cfg.update(config)
    if "max_abs_ps" in cfg:
        raise ValueError(
            "fit.max_abs_ps is obsolete: robust CTR uses every finite residual. "
            "Apply any physical/event rejection explicitly before CTR evaluation."
        )
    width = float(cfg.get("bin_width_ps", 5.0))
    if not np.isfinite(width) or width <= 0.0:
        raise ValueError("fit.bin_width_ps must be positive")
    coverage = float(cfg.get("coverage_fraction", 0.90))
    if not np.isfinite(coverage) or not 0.5 < coverage < 1.0:
        raise ValueError("fit.coverage_fraction must be in (0.5, 1.0)")
    samples = int(cfg.get("bootstrap_samples", 500))
    if samples < 0:
        raise ValueError("fit.bootstrap_samples must be >= 0")
    minimum = int(cfg.get("min_events", 100))
    if minimum < 3:
        raise ValueError("fit.min_events must be >= 3")
    cfg["bin_width_ps"] = width
    cfg["coverage_fraction"] = coverage
    cfg["bootstrap_samples"] = samples
    cfg["min_events"] = minimum
    return cfg


def _shortest_interval(
    values_ps: np.ndarray,
    coverage_fraction: float,
) -> tuple[float, float, float, float, float, int]:
    """Return Gaussian-equivalent CTR and the shortest empirical coverage interval."""
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
    scale = _gaussian_equivalent_scale(coverage_fraction)
    ctr = float(scale * width)
    return ctr, center, low, high, width, count


def _fixed_edges(values_ps: np.ndarray, width_ps: float) -> np.ndarray:
    """Fixed-width histogram grid used only for the secondary core FWHM."""
    values = np.asarray(values_ps, dtype=np.float64)
    if values.size == 0:
        raise ValueError("Cannot build histogram edges from an empty sample")
    width = float(width_ps)
    median = float(np.median(values))
    low = float(np.min(values))
    high = float(np.max(values))
    if high <= low:
        low -= width
        high += width
    median_bin_left = median - 0.5 * width
    steps_left = max(0, int(np.ceil((median_bin_left - low) / width)))
    start = median_bin_left - steps_left * width
    n_bins = max(3, int(np.ceil((high - start) / width)))
    # The pipeline normally has physically bounded residuals. Avoid allocating a
    # pathological histogram if a standalone caller supplies an extreme value.
    if n_bins > 1_000_000:
        raise ValueError("Residual range is too large for the configured core-FWHM bin width")
    edges = start + np.arange(n_bins + 1, dtype=np.float64) * width
    if edges[-1] < high - 1e-12:
        edges = np.append(edges, edges[-1] + width)
    return edges


def _interpolate_crossing(x0: float, y0: float, x1: float, y1: float, level: float) -> float:
    if y1 == y0:
        return 0.5 * (float(x0) + float(x1))
    fraction = (float(level) - float(y0)) / (float(y1) - float(y0))
    return float(x0) + float(np.clip(fraction, 0.0, 1.0)) * (float(x1) - float(x0))


def _measure_histogram(
    values_ps: np.ndarray,
    edges_ps: np.ndarray,
) -> tuple[float, float, float, float, np.ndarray] | None:
    values = np.asarray(values_ps, dtype=np.float64)
    counts, edges = np.histogram(values, bins=np.asarray(edges_ps, dtype=np.float64))
    if counts.size < 3 or int(np.max(counts)) <= 0:
        return None
    centers = 0.5 * (edges[:-1] + edges[1:])
    peak = int(np.argmax(counts))
    half = 0.5 * float(counts[peak])

    left = None
    for i in range(peak - 1, -1, -1):
        if float(counts[i]) < half <= float(counts[i + 1]):
            left = _interpolate_crossing(centers[i], counts[i], centers[i + 1], counts[i + 1], half)
            break
        if float(counts[i]) == half:
            left = float(centers[i])
            break

    right = None
    for i in range(peak, counts.size - 1):
        if float(counts[i]) >= half > float(counts[i + 1]):
            right = _interpolate_crossing(centers[i], counts[i], centers[i + 1], counts[i + 1], half)
            break
        if i > peak and float(counts[i]) == half:
            right = float(centers[i])
            break

    if left is None or right is None or not np.isfinite(left) or not np.isfinite(right) or right <= left:
        return None
    fwhm = float(right - left)
    center = 0.5 * float(left + right)
    return fwhm, center, float(left), float(right), counts.astype(np.int64, copy=False)


def _core_measurement(values_ps: np.ndarray, width_ps: float):
    try:
        edges = _fixed_edges(values_ps, width_ps)
    except ValueError:
        return None, np.empty(0, dtype=np.float64)
    return _measure_histogram(values_ps, edges), edges


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
    values = np.asarray(values_ps, dtype=np.float64).reshape(-1)
    finite = values[np.isfinite(values)]
    n_valid = int(finite.size)
    coverage = float(cfg["coverage_fraction"])
    width = float(cfg["bin_width_ps"])
    requested = int(cfg["bootstrap_samples"]) if bootstrap else 0
    if n_valid < int(cfg["min_events"]):
        return _failure(
            method=method,
            parameter=parameter,
            n_total=n_total,
            n_selected=n_selected,
            n_valid=n_valid,
            coverage_fraction=coverage,
            bin_width_ps=width,
            bootstrap_samples=requested,
            message=f"Only {n_valid} finite events; need {cfg['min_events']}",
        )

    ctr, center, interval_low, interval_high, interval_width, interval_events = _shortest_interval(finite, coverage)
    core, edges = _core_measurement(finite, width)
    if core is None:
        core_fwhm = core_center = left = right = half_max = float("nan")
        counts = np.empty(0, dtype=np.int64)
        core_fraction = float("nan")
    else:
        core_fwhm, core_center, left, right, counts = core
        half_max = 0.5 * float(np.max(counts))
        core_fraction = float(np.mean((finite >= left) & (finite <= right)))

    bootstrap_ctrs: list[float] = []
    bootstrap_core: list[float] = []
    if requested > 1:
        rng = np.random.default_rng(int(seed))
        for _ in range(requested):
            sample = finite[rng.integers(0, n_valid, size=n_valid)]
            try:
                trial_ctr = _shortest_interval(sample, coverage)[0]
            except ValueError:
                continue
            if np.isfinite(trial_ctr):
                bootstrap_ctrs.append(float(trial_ctr))
            trial_core, _trial_edges = _core_measurement(sample, width)
            if trial_core is not None and np.isfinite(trial_core[0]):
                bootstrap_core.append(float(trial_core[0]))

    error = float(np.std(bootstrap_ctrs, ddof=1)) if len(bootstrap_ctrs) > 1 else float("nan")
    core_error = float(np.std(bootstrap_core, ddof=1)) if len(bootstrap_core) > 1 else float("nan")

    return CTRResult(
        method=method,
        parameter=float(parameter),
        success=True,
        n_total=int(n_total),
        n_selected=int(n_selected),
        n_valid=n_valid,
        crossing_efficiency=n_valid / n_selected if n_selected else 0.0,
        ctr_ps=float(ctr),
        ctr_error_ps=error,
        center_ps=float(center),
        coverage_fraction=coverage,
        interval_events=int(interval_events),
        interval_low_ps=float(interval_low),
        interval_high_ps=float(interval_high),
        interval_width_ps=float(interval_width),
        gaussian_equivalent_scale=_gaussian_equivalent_scale(coverage),
        core_fwhm_ps=float(core_fwhm),
        core_fwhm_error_ps=core_error,
        core_fraction=float(core_fraction),
        left_half_ps=float(left),
        right_half_ps=float(right),
        half_max_events=float(half_max),
        bin_width_ps=width,
        bootstrap_samples=requested,
        bootstrap_successful=len(bootstrap_ctrs),
        core_bootstrap_successful=len(bootstrap_core),
        message=("" if core is not None else "Canonical robust CTR succeeded; secondary core FWHM unavailable"),
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
