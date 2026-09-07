from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from scipy.ndimage import gaussian_filter1d


FWHM_PER_SIGMA = 2.0 * np.sqrt(2.0 * np.log(2.0))


@dataclass
class FWHMResult:
    """Non-parametric timing-width result based on the KDE half maximum.

    ``ctr_ps`` is the full width at half maximum (FWHM) of the connected KDE
    peak containing the global maximum.  No Gaussian distribution is fitted.

    A few legacy FitResult-like fields are intentionally retained so existing
    study/reporting code can migrate without requiring an immediate rewrite.
    In particular, ``mean_ps`` aliases the KDE peak position and ``sigma_ps``
    is only the Gaussian-equivalent width ``FWHM / 2.355``; neither comes from
    a Gaussian fit.
    """

    method: str
    parameter: float
    success: bool
    n_total: int
    n_selected: int
    n_valid: int
    n_fit: int
    crossing_efficiency: float

    ctr_ps: float
    ctr_error_ps: float
    peak_ps: float
    peak_density: float
    left_halfmax_ps: float
    right_halfmax_ps: float
    bandwidth_ps: float
    n_halfmax_regions: int
    multimodal_halfmax: bool

    message: str = ""
    edges_ps: np.ndarray = field(default_factory=lambda: np.empty(0), repr=False)
    counts: np.ndarray = field(default_factory=lambda: np.empty(0), repr=False)
    expected: np.ndarray = field(default_factory=lambda: np.empty(0), repr=False)
    bin_width_ps: float = float("nan")
    bin_phase_ps: float = 0.0
    phase_ctr_std_ps: float = 0.0
    iterations: int = 0

    @property
    def mean_ps(self) -> float:
        """Legacy compatibility alias: KDE mode, not a Gaussian mean."""
        return self.peak_ps

    @property
    def mean_error_ps(self) -> float:
        """No local fit-covariance error exists for the KDE mode."""
        return float("nan")

    @property
    def sigma_ps(self) -> float:
        """Gaussian-equivalent width only; no Gaussian fit is performed."""
        return self.ctr_ps / FWHM_PER_SIGMA if self.success else float("nan")

    @property
    def sigma_error_ps(self) -> float:
        if not self.success or not np.isfinite(self.ctr_error_ps):
            return float("nan")
        return self.ctr_error_ps / FWHM_PER_SIGMA

    @property
    def fit_low_ps(self) -> float:
        """Legacy compatibility alias for the left half-maximum crossing."""
        return self.left_halfmax_ps

    @property
    def fit_high_ps(self) -> float:
        """Legacy compatibility alias for the right half-maximum crossing."""
        return self.right_halfmax_ps

    @property
    def chi2(self) -> float:
        """Not defined because no parametric fit is performed."""
        return float("nan")

    @property
    def ndof(self) -> int:
        return 0

    @property
    def chi2_ndof(self) -> float:
        return float("nan")


@dataclass
class FWHMBootstrapSummary:
    """Bootstrap summary for the non-parametric KDE-FWHM estimator."""

    success: bool
    n_requested: int
    n_successful: int
    success_fraction: float
    std_ps: float
    median_ps: float
    p16_ps: float
    p84_ps: float
    message: str = ""
    draws_ps: np.ndarray = field(default_factory=lambda: np.empty(0), repr=False)


def _robust_location_scale(values: np.ndarray) -> tuple[float, float]:
    """Return median and a robust scale, with safe fallbacks."""
    center = float(np.median(values))
    mad = float(np.median(np.abs(values - center)))
    mad_sigma = 1.4826 * mad

    q25, q75 = np.quantile(values, [0.25, 0.75])
    iqr_sigma = float(q75 - q25) / 1.3489795003921634
    std = float(np.std(values, ddof=1)) if values.size > 1 else float("nan")

    # Prefer a robust core scale.  This prevents a few distant timing outliers
    # from setting the KDE bandwidth or stretching the FWHM search grid.
    candidates = [
        x
        for x in (mad_sigma, iqr_sigma)
        if np.isfinite(x) and x > np.finfo(float).eps
    ]
    if candidates:
        scale = min(candidates)
    elif np.isfinite(std) and std > np.finfo(float).eps:
        scale = std
    else:
        scale = float("nan")
    return center, float(scale)


def _positive_lattice_spacing(values: np.ndarray) -> float:
    """Estimate a quantization step when the timing values lie on a lattice."""
    unique = np.unique(values)
    if unique.size < 2:
        return float("nan")
    differences = np.diff(unique)
    differences = differences[np.isfinite(differences) & (differences > 1.0e-12)]
    if differences.size == 0:
        return float("nan")
    return float(np.median(differences))


def _kde_bandwidth_ps(
    values: np.ndarray,
    robust_scale_ps: float,
    config: dict[str, Any],
) -> float:
    """Choose an absolute KDE bandwidth in ps.

    Default is a robust Scott-style rule

        h = robust_scale * N**(-1/5)

    rather than scipy's raw standard-deviation Scott bandwidth.  This is
    deliberately less sensitive to far-out timing tails.
    """
    kde_cfg = config.get("kde", {}) or {}

    configured = kde_cfg.get("bandwidth_ps", config.get("kde_bandwidth_ps", np.nan))
    try:
        configured = float(configured)
    except (TypeError, ValueError):
        configured = float("nan")

    if np.isfinite(configured) and configured > 0.0:
        bandwidth = configured
    else:
        multiplier = kde_cfg.get("bandwidth_multiplier", 1.0)
        try:
            multiplier = float(multiplier)
        except (TypeError, ValueError):
            multiplier = 1.0
        if not np.isfinite(multiplier) or multiplier <= 0.0:
            multiplier = 1.0
        bandwidth = multiplier * robust_scale_ps * float(values.size) ** (-0.2)

    # If samples are visibly quantized, avoid a KDE bandwidth much narrower
    # than the measurement lattice, which would create artificial micro-peaks.
    lattice = _positive_lattice_spacing(values)
    if np.isfinite(lattice) and lattice > 0.0:
        lattice_floor = kde_cfg.get("lattice_bandwidth_fraction", 0.75)
        try:
            lattice_floor = float(lattice_floor)
        except (TypeError, ValueError):
            lattice_floor = 0.75
        if not np.isfinite(lattice_floor) or lattice_floor <= 0.0:
            lattice_floor = 0.75
        bandwidth = max(bandwidth, lattice_floor * lattice)

    minimum = kde_cfg.get("min_bandwidth_ps", 0.05)
    maximum = kde_cfg.get("max_bandwidth_ps", np.inf)
    try:
        minimum = float(minimum)
    except (TypeError, ValueError):
        minimum = 0.05
    try:
        maximum = float(maximum)
    except (TypeError, ValueError):
        maximum = float("inf")

    if not np.isfinite(minimum) or minimum <= 0.0:
        minimum = 0.05
    if not np.isfinite(maximum) or maximum < minimum:
        maximum = float("inf")

    return float(np.clip(bandwidth, minimum, maximum))


def _linear_crossing(
    x0: float,
    y0: float,
    x1: float,
    y1: float,
    target: float,
) -> float:
    """Linearly interpolate x where y crosses target between two grid points."""
    dy = float(y1 - y0)
    if not np.isfinite(dy) or abs(dy) <= np.finfo(float).eps:
        return float(0.5 * (x0 + x1))
    fraction = float((target - y0) / dy)
    return float(x0 + np.clip(fraction, 0.0, 1.0) * (x1 - x0))


def _count_true_regions(mask: np.ndarray) -> int:
    """Count connected True runs in a 1-D boolean mask."""
    mask = np.asarray(mask, dtype=bool)
    if mask.size == 0:
        return 0
    return int(mask[0]) + int(np.count_nonzero((~mask[:-1]) & mask[1:]))


def _smoothed_histogram_density(
    values: np.ndarray,
    *,
    center_ps: float,
    robust_scale_ps: float,
    bandwidth_ps: float,
    config: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
    """Fast binned approximation to a 1-D Gaussian-kernel KDE.

    The raw event histogram is evaluated on a fine grid and convolved with a
    Gaussian kernel.  This is much faster than evaluating ``gaussian_kde`` at
    thousands of points for every bootstrap replica, while preserving the same
    non-parametric half-maximum idea.
    """
    kde_cfg = config.get("kde", {}) or {}

    half_width_scale = kde_cfg.get("search_half_width_scale", 10.0)
    try:
        half_width_scale = float(half_width_scale)
    except (TypeError, ValueError):
        half_width_scale = 10.0
    if not np.isfinite(half_width_scale) or half_width_scale < 3.0:
        half_width_scale = 10.0

    # Robust search range: distant low-density outliers cannot stretch the grid
    # and therefore cannot reduce the half-maximum resolution.  Four bandwidths
    # of padding are enough for gaussian_filter1d's default kernel truncation.
    core_half = max(half_width_scale * robust_scale_ps, 8.0 * bandwidth_ps)
    grid_low = float(center_ps - core_half - 4.0 * bandwidth_ps)
    grid_high = float(center_ps + core_half + 4.0 * bandwidth_ps)
    span = float(grid_high - grid_low)
    if not np.isfinite(span) or span <= 0.0:
        raise RuntimeError("invalid KDE histogram range")

    explicit_bins = kde_cfg.get("grid_points", config.get("kde_grid_points", None))
    if explicit_bins is not None:
        try:
            n_bins = int(explicit_bins)
        except (TypeError, ValueError):
            n_bins = 0
    else:
        n_bins = 0

    if n_bins <= 0:
        samples_per_bandwidth = kde_cfg.get("samples_per_bandwidth", 6.0)
        try:
            samples_per_bandwidth = float(samples_per_bandwidth)
        except (TypeError, ValueError):
            samples_per_bandwidth = 6.0
        if not np.isfinite(samples_per_bandwidth) or samples_per_bandwidth < 3.0:
            samples_per_bandwidth = 6.0
        target_step = bandwidth_ps / samples_per_bandwidth
        n_bins = int(np.ceil(span / target_step))

    min_bins = kde_cfg.get("min_grid_points", 512)
    max_bins = kde_cfg.get("max_grid_points", 8192)
    try:
        min_bins = int(min_bins)
    except (TypeError, ValueError):
        min_bins = 512
    try:
        max_bins = int(max_bins)
    except (TypeError, ValueError):
        max_bins = 8192
    min_bins = max(128, min_bins)
    max_bins = max(min_bins, max_bins)
    n_bins = int(np.clip(n_bins, min_bins, max_bins))

    edges = np.linspace(grid_low, grid_high, n_bins + 1, dtype=np.float64)
    counts, edges = np.histogram(values, bins=edges)
    bin_width = float(edges[1] - edges[0])
    centers = 0.5 * (edges[:-1] + edges[1:])

    sigma_bins = float(bandwidth_ps / bin_width)
    if not np.isfinite(sigma_bins) or sigma_bins <= 0.0:
        raise RuntimeError("invalid smoothing width for binned KDE")

    smoothed_counts = gaussian_filter1d(
        counts.astype(np.float64),
        sigma=sigma_bins,
        mode="constant",
        cval=0.0,
        truncate=4.0,
    )
    density = smoothed_counts / (max(values.size, 1) * bin_width)
    return (
        np.asarray(centers, dtype=np.float64),
        np.asarray(density, dtype=np.float64),
        np.asarray(edges, dtype=np.float64),
        np.asarray(counts, dtype=np.int64),
        bin_width,
    )


def empirical_fwhm_ps(
    values_ps: np.ndarray,
    *,
    method: str = "timing",
    config: dict[str, Any] | None = None,
) -> FWHMResult:
    """Measure timing CTR as a non-parametric full width at half maximum.

    The density estimate is a fast binned Gaussian-kernel KDE approximation:

      1. choose a robust bandwidth from the core timing scale;
      2. histogram events on a fine, robustly bounded time grid;
      3. smooth the histogram with a Gaussian kernel (no distribution fit);
      4. locate the dominant density maximum;
      5. search outward to the nearest left/right half-height crossings;
      6. report ``right - left`` as CTR.

    The Gaussian kernel is only a local smoothing kernel.  The timing
    distribution itself is not assumed or fitted to be Gaussian.
    """
    cfg = config or {}
    values = np.asarray(values_ps, dtype=np.float64).reshape(-1)
    values = values[np.isfinite(values)]

    min_events = int(cfg.get("min_events", 20))
    if values.size < min_events:
        raise RuntimeError(
            f"{method}: only {values.size} finite events are available; "
            f"need at least {min_events} for KDE-FWHM"
        )

    center, robust_scale = _robust_location_scale(values)
    if not np.isfinite(robust_scale) or robust_scale <= 0.0:
        raise RuntimeError(f"{method}: cannot determine a non-zero robust timing scale")

    bandwidth = _kde_bandwidth_ps(values, robust_scale, cfg)
    grid, density, edges, counts, bin_width = _smoothed_histogram_density(
        values,
        center_ps=center,
        robust_scale_ps=robust_scale,
        bandwidth_ps=bandwidth,
        config=cfg,
    )
    if density.size != grid.size or density.size < 3 or np.any(~np.isfinite(density)):
        raise RuntimeError(f"{method}: KDE returned invalid density values")

    peak_index = int(np.argmax(density))
    peak_density = float(density[peak_index])
    peak_ps = float(grid[peak_index])
    if not np.isfinite(peak_density) or peak_density <= 0.0:
        raise RuntimeError(f"{method}: KDE peak is invalid")

    half_max = 0.5 * peak_density
    above = density >= half_max
    n_regions = _count_true_regions(above)

    # Search outward from the dominant peak.  Only the connected half-maximum
    # component containing that peak defines the reported CTR.
    left_below = np.flatnonzero(~above[:peak_index])
    if left_below.size == 0:
        raise RuntimeError(
            f"{method}: left half-maximum crossing lies outside KDE search range"
        )
    i0 = int(left_below[-1])
    i1 = i0 + 1
    left = _linear_crossing(
        float(grid[i0]), float(density[i0]),
        float(grid[i1]), float(density[i1]),
        half_max,
    )

    right_relative = np.flatnonzero(~above[peak_index + 1 :])
    if right_relative.size == 0:
        raise RuntimeError(
            f"{method}: right half-maximum crossing lies outside KDE search range"
        )
    j1 = int(peak_index + 1 + right_relative[0])
    j0 = j1 - 1
    right = _linear_crossing(
        float(grid[j0]), float(density[j0]),
        float(grid[j1]), float(density[j1]),
        half_max,
    )

    ctr = float(right - left)
    if not np.isfinite(ctr) or ctr <= 0.0:
        raise RuntimeError(f"{method}: invalid KDE-FWHM width {ctr}")

    n_fit = int(np.count_nonzero((values >= left) & (values <= right)))

    # ``expected`` remains available for legacy plotting code and now contains
    # the smoothed non-parametric expected counts at each histogram-bin center.
    expected = density * values.size * bin_width

    message = "KDE-FWHM; no Gaussian distribution fit"
    if n_regions > 1:
        message += f"; warning: {n_regions} disconnected half-maximum regions"

    return FWHMResult(
        method=method,
        parameter=0.0,
        success=True,
        n_total=int(values.size),
        n_selected=int(values.size),
        n_valid=int(values.size),
        n_fit=n_fit,
        crossing_efficiency=1.0,
        ctr_ps=ctr,
        ctr_error_ps=float("nan"),
        peak_ps=peak_ps,
        peak_density=peak_density,
        left_halfmax_ps=float(left),
        right_halfmax_ps=float(right),
        bandwidth_ps=float(bandwidth),
        n_halfmax_regions=n_regions,
        multimodal_halfmax=bool(n_regions > 1),
        message=message,
        edges_ps=edges,
        counts=counts,
        expected=np.asarray(expected, dtype=np.float64),
        bin_width_ps=bin_width,
        bin_phase_ps=0.0,
        phase_ctr_std_ps=0.0,
        iterations=0,
    )


def fit_times_ps(values_ps: np.ndarray, method: str, fit_config: dict[str, Any]):
    """Primary timing-width estimator used by the study.

    Despite the historical function name, this no longer performs a Gaussian
    fit.  It returns the non-parametric KDE FWHM of the dominant timing peak.

    Keeping the function name minimizes changes required in study.py and
    reporting.py: any existing bootstrap loop that repeatedly calls
    ``fit_times_ps`` will now automatically bootstrap the complete KDE-FWHM
    estimator instead of repeatedly fitting Gaussians.
    """
    return empirical_fwhm_ps(values_ps, method=method, config=fit_config)


def distribution_metrics(
    values_ps: np.ndarray,
    *,
    true_value_ps: float,
    fit: Any,
) -> dict[str, Any]:
    values = np.asarray(values_ps, dtype=np.float64).reshape(-1)
    values = values[np.isfinite(values)]

    ctr_error = getattr(fit, "ctr_error_ps", getattr(fit, "ctr_err_ps", float("nan")))
    peak_ps = getattr(fit, "peak_ps", getattr(fit, "mean_ps", float("nan")))

    return {
        "event_count": int(values.size),
        "true_value_ps": float(true_value_ps),
        "ctr_definition": "kde_fwhm",
        "ctr_ps": float(fit.ctr_ps) if fit.success else float("nan"),
        "ctr_error_ps": float(ctr_error) if fit.success else float("nan"),
        "peak_ps": float(peak_ps) if fit.success else float("nan"),
        "peak_bias_ps": float(peak_ps - true_value_ps) if fit.success else float("nan"),
        "left_halfmax_ps": float(getattr(fit, "left_halfmax_ps", np.nan)) if fit.success else float("nan"),
        "right_halfmax_ps": float(getattr(fit, "right_halfmax_ps", np.nan)) if fit.success else float("nan"),
        "kde_bandwidth_ps": float(getattr(fit, "bandwidth_ps", np.nan)) if fit.success else float("nan"),
        "n_halfmax_regions": int(getattr(fit, "n_halfmax_regions", 0)) if fit.success else 0,
        "multimodal_halfmax": bool(getattr(fit, "multimodal_halfmax", False)) if fit.success else False,

        # Retain the old output columns so existing CSV/reporting readers do not
        # fail, but deliberately do not populate them with KDE quantities under
        # misleading Gaussian names.
        "gaussian_mean_ps": float("nan"),
        "gaussian_mean_error_ps": float("nan"),
        "gaussian_bias_ps": float("nan"),
        "chi2_ndof": float("nan"),

        "arithmetic_mean_ps": float(np.mean(values)) if values.size else float("nan"),
        "arithmetic_bias_ps": float(np.mean(values) - true_value_ps) if values.size else float("nan"),
        "standard_deviation_ps": float(np.std(values, ddof=0)) if values.size else float("nan"),
        "fit_success": bool(fit.success),
        "fit_message": str(getattr(fit, "message", "")),
    }


def residual_metrics(values_ps: np.ndarray) -> dict[str, Any]:
    """All-event arithmetic diagnostics plus non-parametric FWHM CTR.

    ``std_ps`` remains an ordinary sample standard deviation diagnostic, while
    ``ctr_ps`` is now the same KDE-FWHM definition used by the main study.
    """
    values = np.asarray(values_ps, dtype=np.float64).reshape(-1)
    values = values[np.isfinite(values)]
    n = int(values.size)
    if n == 0:
        return {
            "n": 0,
            "mean_ps": float("nan"),
            "std_ps": float("nan"),
            "ctr_ps": float("nan"),
            "ctr_definition": "kde_fwhm",
            "rmse_ps": float("nan"),
            "bias_ps": float("nan"),
        }

    mean = float(np.mean(values))
    rmse = float(np.sqrt(np.mean(values * values)))
    std = float(np.std(values, ddof=1)) if n >= 2 else float("nan")

    ctr = float("nan")
    if n >= 3:
        try:
            result = empirical_fwhm_ps(
                values,
                method="residual_metrics",
                config={"min_events": 3},
            )
            ctr = float(result.ctr_ps)
        except Exception:
            ctr = float("nan")

    return {
        "n": n,
        "mean_ps": mean,
        "std_ps": std,
        "ctr_ps": ctr,
        "ctr_definition": "kde_fwhm",
        "rmse_ps": rmse,
        "bias_ps": mean,
    }


def ctr_bootstrap_summary(
    values_ps: np.ndarray,
    n_bootstrap: int = 1000,
    seed: int = 12345,
    *,
    fit_config: dict[str, Any] | None = None,
    method: str = "bootstrap",
) -> FWHMBootstrapSummary:
    """Bootstrap the complete KDE-FWHM extraction.

    Every bootstrap draw resamples events with replacement and reruns
    ``empirical_fwhm_ps``.  Failed draws are discarded.  By default at least
    80% of the requested draws must succeed for the summary to be accepted.
    """
    values = np.asarray(values_ps, dtype=np.float64).reshape(-1)
    values = values[np.isfinite(values)]
    cfg = dict(fit_config or {})

    requested = int(n_bootstrap)
    if values.size < 3 or requested <= 1:
        return FWHMBootstrapSummary(
            success=False,
            n_requested=max(requested, 0),
            n_successful=0,
            success_fraction=0.0,
            std_ps=float("nan"),
            median_ps=float("nan"),
            p16_ps=float("nan"),
            p84_ps=float("nan"),
            message="not enough events/bootstrap draws",
        )

    # A bootstrap sample cannot contain more unique information than the input;
    # allow small legacy callers while preserving any stricter configured limit.
    cfg["min_events"] = min(int(cfg.get("min_events", 20)), int(values.size))

    rng = np.random.default_rng(int(seed))
    successful: list[float] = []
    for _ in range(requested):
        sample = rng.choice(values, size=values.size, replace=True)
        try:
            result = empirical_fwhm_ps(sample, method=method, config=cfg)
        except Exception:
            continue
        if result.success and np.isfinite(result.ctr_ps) and result.ctr_ps > 0.0:
            successful.append(float(result.ctr_ps))

    draws = np.asarray(successful, dtype=np.float64)
    fraction = float(draws.size / requested)

    minimum_fraction = (cfg.get("kde", {}) or {}).get("bootstrap_min_success_fraction", 0.80)
    try:
        minimum_fraction = float(minimum_fraction)
    except (TypeError, ValueError):
        minimum_fraction = 0.80
    minimum_fraction = float(np.clip(minimum_fraction, 0.0, 1.0))

    if draws.size < 2 or fraction < minimum_fraction:
        return FWHMBootstrapSummary(
            success=False,
            n_requested=requested,
            n_successful=int(draws.size),
            success_fraction=fraction,
            std_ps=float("nan"),
            median_ps=float("nan"),
            p16_ps=float("nan"),
            p84_ps=float("nan"),
            message=(
                f"only {draws.size}/{requested} KDE-FWHM bootstrap draws succeeded "
                f"({fraction:.1%}); required {minimum_fraction:.1%}"
            ),
            draws_ps=draws,
        )

    p16, median, p84 = np.percentile(draws, [16.0, 50.0, 84.0])
    return FWHMBootstrapSummary(
        success=True,
        n_requested=requested,
        n_successful=int(draws.size),
        success_fraction=fraction,
        std_ps=float(np.std(draws, ddof=1)),
        median_ps=float(median),
        p16_ps=float(p16),
        p84_ps=float(p84),
        message="bootstrap of complete KDE-FWHM estimator",
        draws_ps=draws,
    )


def ctr_bootstrap_uncertainty(
    values_ps: np.ndarray,
    n_bootstrap: int = 1000,
    seed: int = 12345,
    *,
    fit_config: dict[str, Any] | None = None,
) -> float:
    """Return the standard deviation of bootstrap KDE-FWHM estimates.

    This replaces the old ``2.355 * sample_std`` bootstrap.  It now resamples
    events and repeats the same non-parametric FWHM extraction used for the
    nominal CTR.
    """
    summary = ctr_bootstrap_summary(
        values_ps,
        n_bootstrap=n_bootstrap,
        seed=seed,
        fit_config=fit_config,
    )
    return float(summary.std_ps) if summary.success else float("nan")
