from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

FS_PER_PS = 1000.0
DEFAULT_INVALID_TIME_FS = np.iinfo(np.int64).min
DEFAULT_FIT_CONFIG: dict[str, Any] = {
    "min_events": 100,
    "bin_width_ps": 5.0,
    "bootstrap_samples": 100,
}


@dataclass
class CTRResult:
    """Direct fixed-bin histogram FWHM timing-resolution result."""

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
    left_half_ps: float
    right_half_ps: float
    half_max_events: float
    bin_width_ps: float
    bootstrap_samples: int
    bootstrap_successful: int
    message: str = ""
    edges_ps: np.ndarray = field(default_factory=lambda: np.empty(0), repr=False)
    counts: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=np.int64), repr=False)

    @property
    def n_fit(self) -> int:
        return self.n_valid

    # Generic aliases retained for consumers that only need a central timing value.
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
        return self.left_half_ps

    @property
    def fit_high_ps(self) -> float:
        return self.right_half_ps

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
            "left_half_ps": self.left_half_ps,
            "right_half_ps": self.right_half_ps,
            "half_max_events": self.half_max_events,
            "bin_width_ps": self.bin_width_ps,
            "bootstrap_samples": self.bootstrap_samples,
            "bootstrap_successful": self.bootstrap_successful,
            "message": self.message,
        }


# Generic name used by existing CSV/storage helpers.
FitResult = CTRResult


def _failure(
    *,
    method: str,
    parameter: float,
    n_total: int,
    n_selected: int,
    n_valid: int,
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
        left_half_ps=np.nan,
        right_half_ps=np.nan,
        half_max_events=np.nan,
        bin_width_ps=float(bin_width_ps),
        bootstrap_samples=int(bootstrap_samples),
        bootstrap_successful=0,
        message=message,
    )


def _config(config: dict[str, Any] | None) -> dict[str, Any]:
    cfg = dict(DEFAULT_FIT_CONFIG)
    if config:
        cfg.update(config)
    width = float(cfg.get("bin_width_ps", 5.0))
    if not np.isfinite(width) or width <= 0.0:
        raise ValueError("fit.bin_width_ps must be positive")
    samples = int(cfg.get("bootstrap_samples", 100))
    if samples < 0:
        raise ValueError("fit.bootstrap_samples must be >= 0")
    minimum = int(cfg.get("min_events", 100))
    if minimum < 3:
        raise ValueError("fit.min_events must be >= 3")
    cfg["bin_width_ps"] = width
    cfg["bootstrap_samples"] = samples
    cfg["min_events"] = minimum
    return cfg


def _fixed_edges(values_ps: np.ndarray, width_ps: float, max_abs_ps: float | None) -> np.ndarray:
    """Build fixed-width bins with the sample median at the center of a bin.

    The bin width is fixed, but the absolute phase is not tied to zero. Centering
    a bin on the median makes the estimator translation-invariant and avoids a
    small displacement of an otherwise identical peak changing the measured
    FWHM simply because it falls across different bin boundaries.
    """
    values = np.asarray(values_ps, dtype=np.float64)
    if values.size == 0:
        raise ValueError("Cannot build histogram edges from an empty sample")
    width = float(width_ps)
    median = float(np.median(values))

    if max_abs_ps is not None:
        limit = float(max_abs_ps)
        if not np.isfinite(limit) or limit <= 0.0:
            raise ValueError("fit.max_abs_ps must be positive")
        low = -limit
        high = limit
    else:
        low = float(np.min(values))
        high = float(np.max(values))
        if high <= low:
            low -= width
            high += width

    # One bin is centered exactly on the median. Extend that same fixed-width
    # grid until it covers the requested physical range.
    median_bin_left = median - 0.5 * width
    steps_left = max(0, int(np.ceil((median_bin_left - low) / width)))
    start = median_bin_left - steps_left * width
    n_bins = max(3, int(np.ceil((high - start) / width)))
    edges = start + np.arange(n_bins + 1, dtype=np.float64) * width
    if edges[-1] < high - 1e-12:
        edges = np.append(edges, edges[-1] + width)
    return edges


def _interpolate_crossing(x0: float, y0: float, x1: float, y1: float, level: float) -> float:
    if y1 == y0:
        return 0.5 * (float(x0) + float(x1))
    fraction = (float(level) - float(y0)) / (float(y1) - float(y0))
    return float(x0) + float(np.clip(fraction, 0.0, 1.0)) * (float(x1) - float(x0))


def _measure_histogram(values_ps: np.ndarray, edges_ps: np.ndarray) -> tuple[float, float, float, float, np.ndarray] | None:
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
    ctr = float(right - left)
    center = 0.5 * float(left + right)
    return ctr, center, float(left), float(right), counts.astype(np.int64, copy=False)


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
    limit = cfg.get("max_abs_ps")
    if limit is not None:
        finite = finite[np.abs(finite) <= float(limit)]
    n_valid = int(finite.size)
    if n_valid < int(cfg["min_events"]):
        return _failure(
            method=method,
            parameter=parameter,
            n_total=n_total,
            n_selected=n_selected,
            n_valid=n_valid,
            bin_width_ps=cfg["bin_width_ps"],
            bootstrap_samples=cfg["bootstrap_samples"] if bootstrap else 0,
            message=f"Only {n_valid} valid events; need {cfg['min_events']}",
        )

    max_abs = None if limit is None else float(limit)
    width = float(cfg["bin_width_ps"])
    edges = _fixed_edges(finite, width, max_abs)
    measured = _measure_histogram(finite, edges)
    if measured is None:
        return _failure(
            method=method,
            parameter=parameter,
            n_total=n_total,
            n_selected=n_selected,
            n_valid=n_valid,
            bin_width_ps=cfg["bin_width_ps"],
            bootstrap_samples=cfg["bootstrap_samples"] if bootstrap else 0,
            message="Direct histogram FWHM could not find both half-maximum crossings",
        )
    ctr, center, left, right, counts = measured

    requested = int(cfg["bootstrap_samples"]) if bootstrap else 0
    bootstrap_ctrs: list[float] = []
    if requested > 1:
        rng = np.random.default_rng(int(seed))
        for _ in range(requested):
            sample = finite[rng.integers(0, n_valid, size=n_valid)]
            # The estimator definition includes median-based bin alignment, so
            # each bootstrap resample gets its own median-centered fixed-width grid.
            trial_edges = _fixed_edges(sample, width, max_abs)
            trial = _measure_histogram(sample, trial_edges)
            if trial is not None and np.isfinite(trial[0]):
                bootstrap_ctrs.append(float(trial[0]))
    error = float(np.std(bootstrap_ctrs, ddof=1)) if len(bootstrap_ctrs) > 1 else float("nan")

    return CTRResult(
        method=method,
        parameter=float(parameter),
        success=True,
        n_total=int(n_total),
        n_selected=int(n_selected),
        n_valid=n_valid,
        crossing_efficiency=n_valid / n_selected if n_selected else 0.0,
        ctr_ps=ctr,
        ctr_error_ps=error,
        center_ps=center,
        left_half_ps=left,
        right_half_ps=right,
        half_max_events=0.5 * float(np.max(counts)),
        bin_width_ps=width,
        bootstrap_samples=requested,
        bootstrap_successful=len(bootstrap_ctrs),
        message="",
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


# Generic function names retained because they describe timing-width extraction,
# not a particular parametric model.
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
