from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .peak import fit_histogram_peak


@dataclass
class PhotopeakResult:
    channel: int
    success: bool
    mean_mV: float
    sigma_mV: float
    mean_error_mV: float
    sigma_error_mV: float
    selection_low_mV: float
    selection_high_mV: float
    chi2: float
    ndof: int
    iterations: int
    message: str = ""
    edges_mV: np.ndarray = field(default_factory=lambda: np.empty(0), repr=False)
    counts: np.ndarray = field(default_factory=lambda: np.empty(0), repr=False)
    expected: np.ndarray = field(default_factory=lambda: np.empty(0), repr=False)
    fit_low_mV: float = np.nan
    fit_high_mV: float = np.nan

    @property
    def chi2_ndof(self) -> float:
        return self.chi2 / self.ndof if self.ndof > 0 else np.nan

    def as_dict(self) -> dict[str, Any]:
        return {
            "channel": self.channel,
            "success": self.success,
            "mean_mV": self.mean_mV,
            "sigma_mV": self.sigma_mV,
            "mean_error_mV": self.mean_error_mV,
            "sigma_error_mV": self.sigma_error_mV,
            "selection_low_mV": self.selection_low_mV,
            "selection_high_mV": self.selection_high_mV,
            "chi2": self.chi2,
            "ndof": self.ndof,
            "chi2_ndof": self.chi2_ndof,
            "iterations": self.iterations,
            "fit_low_mV": self.fit_low_mV,
            "fit_high_mV": self.fit_high_mV,
            "message": self.message,
        }


def fit_photopeak(
    amplitudes_mV: np.ndarray,
    *,
    channel: int,
    config: dict[str, Any],
) -> PhotopeakResult:
    """Photopeak API preserved; numerical peak logic is shared with ToT selection."""
    fit = fit_histogram_peak(
        amplitudes_mV,
        config=config,
        unit_suffix="mV",
        fit_name="Photopeak",
        value_name="amplitudes",
    )
    return PhotopeakResult(
        channel=channel,
        success=fit.success,
        mean_mV=fit.mean,
        sigma_mV=fit.sigma,
        mean_error_mV=fit.mean_error,
        sigma_error_mV=fit.sigma_error,
        selection_low_mV=fit.selection_low,
        selection_high_mV=fit.selection_high,
        chi2=fit.chi2,
        ndof=fit.ndof,
        iterations=fit.iterations,
        message=fit.message,
        edges_mV=fit.edges,
        counts=fit.counts,
        expected=fit.expected,
        fit_low_mV=fit.fit_low,
        fit_high_mV=fit.fit_high,
    )


def photopeak_mask(values_mV: np.ndarray, result: PhotopeakResult) -> np.ndarray:
    values = np.asarray(values_mV, dtype=np.float64)
    if not result.success:
        return np.zeros(values.shape, dtype=bool)
    return (
        np.isfinite(values)
        & (values >= result.selection_low_mV)
        & (values <= result.selection_high_mV)
    )
