from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
from scipy.ndimage import gaussian_filter1d


@dataclass(frozen=True)
class CTRResult:
    ctr_ps: float
    center_ps: float
    left_ps: float
    right_ps: float
    n: int
    bin_width_ps: float
    bandwidth_ps: float
    uncertainty_ps: float = float("nan")

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def _robust_scale(values: np.ndarray) -> tuple[float, float]:
    center = float(np.median(values))
    mad = float(np.median(np.abs(values - center)))
    q25, q75 = np.quantile(values, [0.25, 0.75])
    scales = [1.4826 * mad, float(q75 - q25) / 1.3489795003921634]
    valid = [value for value in scales if np.isfinite(value) and value > np.finfo(float).eps]
    if valid:
        return center, float(min(valid))
    std = float(np.std(values, ddof=1)) if values.size > 1 else float("nan")
    return center, std


def _crossing(x0: float, y0: float, x1: float, y1: float, target: float) -> float:
    if y1 == y0:
        return float((x0 + x1) / 2.0)
    fraction = float(np.clip((target - y0) / (y1 - y0), 0.0, 1.0))
    return float(x0 + fraction * (x1 - x0))


def ctr_fwhm(values_ps: np.ndarray, config: dict[str, Any] | None = None) -> CTRResult:
    """Direct FWHM of the dominant peak of a smoothed event histogram.

    The Gaussian kernel is only a local histogram smoother. No distribution is
    fitted and no Gaussian width is inferred.
    """
    cfg = config or {}
    values = np.asarray(values_ps, dtype=np.float64).reshape(-1)
    values = values[np.isfinite(values)]
    min_events = int(cfg.get("min_events", 20))
    if values.size < min_events:
        raise ValueError(f"Direct FWHM needs at least {min_events} finite events")

    center, scale = _robust_scale(values)
    if not np.isfinite(scale) or scale <= 0.0:
        raise ValueError("Timing distribution has no measurable width")

    smooth_cfg = cfg.get("fwhm", {}) if isinstance(cfg.get("fwhm"), dict) else {}
    bandwidth = smooth_cfg.get("bandwidth_ps")
    if bandwidth is None:
        multiplier = float(smooth_cfg.get("bandwidth_multiplier", 1.0))
        bandwidth = multiplier * scale * float(values.size) ** (-0.2)
    bandwidth = max(float(smooth_cfg.get("min_bandwidth_ps", 0.05)), float(bandwidth))

    half_width = max(float(smooth_cfg.get("search_half_width_scale", 10.0)) * scale, 8.0 * bandwidth)
    low, high = center - half_width, center + half_width
    samples_per_bandwidth = max(3.0, float(smooth_cfg.get("samples_per_bandwidth", 6.0)))
    target_step = bandwidth / samples_per_bandwidth
    n_bins = int(np.ceil((high - low) / target_step))
    n_bins = int(np.clip(n_bins, int(smooth_cfg.get("min_bins", 512)), int(smooth_cfg.get("max_bins", 8192))))
    edges = np.linspace(low, high, n_bins + 1, dtype=np.float64)
    counts, edges = np.histogram(values, bins=edges)
    bin_width = float(edges[1] - edges[0])
    centers = 0.5 * (edges[:-1] + edges[1:])
    smoothed = gaussian_filter1d(
        counts.astype(np.float64),
        sigma=float(bandwidth / bin_width),
        mode="constant",
        cval=0.0,
        truncate=4.0,
    )
    peak = int(np.argmax(smoothed))
    peak_height = float(smoothed[peak])
    if not np.isfinite(peak_height) or peak_height <= 0.0:
        raise ValueError("Timing histogram has no valid peak")
    half = peak_height / 2.0
    above = smoothed >= half

    left_below = np.flatnonzero(~above[:peak])
    right_below = np.flatnonzero(~above[peak + 1 :])
    if left_below.size == 0 or right_below.size == 0:
        raise ValueError("Half-maximum crossing lies outside the robust histogram range")
    i0 = int(left_below[-1]); i1 = i0 + 1
    j1 = int(peak + 1 + right_below[0]); j0 = j1 - 1
    left = _crossing(centers[i0], smoothed[i0], centers[i1], smoothed[i1], half)
    right = _crossing(centers[j0], smoothed[j0], centers[j1], smoothed[j1], half)
    width = float(right - left)
    if not np.isfinite(width) or width <= 0.0:
        raise ValueError("Invalid direct FWHM width")
    return CTRResult(width, float(centers[peak]), left, right, int(values.size), bin_width, float(bandwidth))


def bootstrap_ctr(
    values_ps: np.ndarray,
    samples: int,
    seed: int,
    config: dict[str, Any] | None = None,
) -> CTRResult:
    base = ctr_fwhm(values_ps, config)
    if int(samples) <= 1:
        return base
    values = np.asarray(values_ps, dtype=np.float64).reshape(-1)
    values = values[np.isfinite(values)]
    rng = np.random.default_rng(int(seed))
    widths: list[float] = []
    for _ in range(int(samples)):
        try:
            widths.append(ctr_fwhm(rng.choice(values, values.size, replace=True), config).ctr_ps)
        except ValueError:
            continue
    uncertainty = float(np.std(widths, ddof=1)) if len(widths) > 1 else float("nan")
    return CTRResult(**{**base.as_dict(), "uncertainty_ps": uncertainty})
