"""Configuration for sklearn RidgeCV, independent of the outer search engine."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

LINEAR_RIDGE_MODELS = frozenset({"direct_linear_ridge", "shared_linear_ridge"})
RIDGE_CV_VERSION = 1
DEFAULT_ALPHAS = tuple(float(value) for value in np.geomspace(1e-8, 1e3, 23))


@dataclass(frozen=True)
class RidgeCVConfig:
    """Lambda grid and internal CV; None uses efficient leave-one-out CV.

    MSE is measured on the unmodified estimator prediction, in ps squared.
    The independent blind dataset never participates in alpha selection.
    """

    alphas: tuple[float, ...] = DEFAULT_ALPHAS
    cv: int | None = None
    gcv_mode: str | None = None
    scoring: str = "neg_mean_squared_error"

    def as_dict(self) -> dict[str, Any]:
        return {
            "alphas": list(self.alphas),
            "cv": self.cv,
            "gcv_mode": self.gcv_mode,
            "scoring": self.scoring,
        }


def ridge_cv_config(space: dict[str, Any]) -> RidgeCVConfig:
    """Validate new settings, or migrate a legacy finite/ranged alpha space."""
    raw = space.get("ridge_cv")
    if raw is None:
        legacy = (space.get("parameters") or {}).get("ridge_alpha")
        raw = {}
        if isinstance(legacy, dict):
            kind = legacy.get("type")
            if kind == "fixed":
                raw["alphas"] = [legacy["value"]]
            elif kind == "categorical":
                raw["alphas"] = legacy["choices"]
            elif kind == "float":
                low, high = float(legacy["low"]), float(legacy["high"])
                if (
                    not np.isfinite(low)
                    or not np.isfinite(high)
                    or low <= 0
                    or high < low
                ):
                    raise ValueError(
                        "ridge_alpha range must be finite, positive and increasing"
                    )
                raw["alphas"] = (
                    np.geomspace(low, high, 23)
                    if legacy.get("log", False)
                    else np.linspace(low, high, 23)
                ).tolist()
            else:
                raise ValueError(
                    "Legacy ridge_alpha requires fixed, categorical or float type"
                )
        elif legacy is not None:
            raw["alphas"] = (
                legacy if isinstance(legacy, (list, tuple, np.ndarray)) else [legacy]
            )
    if not isinstance(raw, dict):
        raise ValueError("ridge_cv must be an object")
    extra = set(raw) - {"alphas", "cv", "gcv_mode", "scoring"}
    if extra:
        raise ValueError(f"Unsupported ridge_cv fields: {sorted(extra)}")
    alphas = np.asarray(raw.get("alphas", DEFAULT_ALPHAS), dtype=np.float64)
    if (
        alphas.ndim != 1
        or not alphas.size
        or np.any(~np.isfinite(alphas))
        or np.any(alphas <= 0)
    ):
        raise ValueError(
            "ridge_cv.alphas must be a nonempty list of finite positive values"
        )
    cv = raw.get("cv")
    if cv is not None and (isinstance(cv, bool) or not isinstance(cv, int) or cv < 2):
        raise ValueError("ridge_cv.cv must be null (leave-one-out) or an integer >= 2")
    mode = raw.get("gcv_mode")
    if mode not in {None, "auto", "svd", "eigen"}:
        raise ValueError("ridge_cv.gcv_mode must be null, auto, svd or eigen")
    scoring = raw.get("scoring", "neg_mean_squared_error")
    if scoring != "neg_mean_squared_error":
        raise ValueError("ridge_cv.scoring must be neg_mean_squared_error")
    return RidgeCVConfig(tuple(map(float, alphas)), cv, mode, scoring)


def normalize_ridge_space(space: dict[str, Any], name: str) -> dict[str, Any]:
    """Remove obsolete outer-search/solver settings from effective configuration."""
    return {
        "model": name,
        "ridge_cv": ridge_cv_config(space).as_dict(),
        "verbose": bool(space.get("verbose", False)),
    }
