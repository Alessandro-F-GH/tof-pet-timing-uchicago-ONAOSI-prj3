"""Pointwise conditional-mean heatmap features and an additive Ridge surrogate.

The heatmap provides *associations* with saved model predictions, not causal
attributions. Ridge contributions sum to the surrogate prediction minus its
intercept, not necessarily to the frozen teacher's prediction.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class ConditionalMap:
    positions: np.ndarray
    time_ps: np.ndarray
    edges: np.ndarray
    mean: np.ndarray
    scale: np.ndarray
    target_mean: np.ndarray
    counts: np.ndarray
    baseline_ps: float
    representation: str
    prediction_type: str


def load_conditional_map(path):
    """Read a saved *single-model*, original-waveform feature heatmap."""
    with np.load(Path(path), allow_pickle=False) as a:
        keys = ("positions", "time_ps", "z_edges", "target_mean_ps",
                "support_counts", "normalization_mean", "normalization_scale",
                "training_prediction_mean_ps", "representation",
                "prediction_type", "feature_names")
        missing = set(keys) - set(a.files)
        if missing:
            raise ValueError(f"Heatmap lacks fields: {sorted(missing)}")
        features = [str(v) for v in a["feature_names"]]
        if not features or features[0] != "waveform":
            raise ValueError("Expected a waveform conditional heatmap")
        prediction_type = str(a["prediction_type"].item())
        if prediction_type != "saved_reference":
            raise ValueError("Surrogate requires saved_reference, not a model-difference heatmap")
        representation = str(a["representation"].item())
        if representation not in ("original", "baseline_subtracted", "amplitude_normalized"):
            raise ValueError(f"Unsupported representation: {representation}")
        m = ConditionalMap(
            positions=np.array(a["positions"], dtype=np.int64),
            time_ps=np.array(a["time_ps"], dtype=np.float64),
            edges=np.array(a["z_edges"], dtype=np.float64),
            mean=np.array(a["normalization_mean"], dtype=np.float64),
            scale=np.array(a["normalization_scale"], dtype=np.float64),
            target_mean=np.array(a["target_mean_ps"][0, :2], dtype=np.float64),
            counts=np.array(a["support_counts"][0, :2], dtype=np.int64),
            baseline_ps=float(a["training_prediction_mean_ps"]),
            representation=representation,
            prediction_type=prediction_type,
        )
    p = len(m.positions)
    bins = len(m.edges) - 1
    if (p < 2 or bins < 2 or m.positions[0] < 0
            or np.any(np.diff(m.positions) <= 0)
            or np.any(np.diff(m.time_ps) <= 0)
            or m.time_ps.shape != (p,)
            or m.mean.ndim != 2 or m.mean.shape[0] != 2
            or m.scale.shape != m.mean.shape
            or m.positions[-1] >= m.mean.shape[1]
            or m.target_mean.shape != (2, bins, p)
            or m.counts.shape != m.target_mean.shape
            or not np.all(np.diff(m.edges) > 0)
            or not np.isfinite(m.edges).all()
            or not np.isfinite(m.mean).all()
            or not np.isfinite(m.scale).all()
            or np.any(m.scale <= 0)
            or not np.isfinite(m.baseline_ps)
            or np.any(m.counts < 0)):
        raise ValueError("Invalid conditional heatmap shapes or statistics")
    return m


def split_development(n, seed=42, calibration_count=500, validation_count=500):
    """Reproduce the historical heatmap split: first excluded=calibration."""
    if min(calibration_count, validation_count) < 1 or n <= calibration_count + validation_count:
        raise ValueError("Need positive calibration/validation sizes and nonempty map reference")
    excluded = np.random.default_rng(seed).choice(
        n, calibration_count + validation_count, replace=False)
    keep = np.ones(n, dtype=bool)
    keep[excluded] = False
    return (np.flatnonzero(keep), excluded[:calibration_count],
            excluded[calibration_count:])


def heatmap_pointwise_features(waves, rows, heatmap, *,
                               full_time_ps, baseline_positions=None,
                               retained=None, min_count=25, chunk_size=128):
    """Map prepared paired waveforms to train-only heatmap lookups, without inference.

    Returns [event, detector, plotted sample] deviations in ps and coverage mask.
    Missing/out-of-range/sparse cells are imputed as ZERO DEVIATION only for
    the surrogate matrix, with a separate validity mask preserved for plots.
    """
    from waveform_analysis.scripts.plot_conditional_target_heatmaps import (
        WaveformRepresentation,
    )
    t = np.asarray(full_time_ps, dtype=np.float64)
    rows = np.asarray(rows, dtype=np.int64)
    if t.ndim != 1 or t.size != waves.shape[-1]:
        raise ValueError("Prepared waveform time grid mismatch")
    if not np.allclose(t[heatmap.positions], heatmap.time_ps, atol=1e-6, rtol=0):
        raise ValueError("Heatmap grid and prepared waveform grid differ")
    if (rows.ndim != 1 or (rows.size and
        (rows.min() < 0 or rows.max() >= waves.shape[0]))):
        raise ValueError("Invalid event rows")
    if min_count < 1 or chunk_size < 1:
        raise ValueError("min_count and chunk_size must be positive")
    if retained is None:
        retained = np.ones_like(t, dtype=bool)
    retained = np.asarray(retained, dtype=bool)
    if retained.shape != t.shape:
        raise ValueError("Invalid timing sample mask")
    if heatmap.representation == "original":
        view = waves
    else:
        if baseline_positions is None:
            raise ValueError("Transformed heatmap needs the original baseline indices")
        view = WaveformRepresentation(waves, heatmap.representation, baseline_positions)
    n, p = len(rows), len(heatmap.positions)
    deviations = np.zeros((n, 2, p), dtype=np.float64)
    valid = np.zeros((n, 2, p), dtype=bool)
    mu = heatmap.mean[:, heatmap.positions]
    scale = heatmap.scale[:, heatmap.positions]
    allowed = retained[heatmap.positions]
    for left in range(0, n, chunk_size):
        indices = rows[left:left + chunk_size]
        wave = np.asarray(view[indices], dtype=np.float64)
        z = (wave[:, :, heatmap.positions] - mu[None]) / scale[None]
        for d in range(2):
            x = z[:, d, :]
            ix = np.searchsorted(heatmap.edges, x, side="right") - 1
            # The last bin includes the right endpoint, as in accumulate_heatmaps.
            ix = np.where(x == heatmap.edges[-1], len(heatmap.edges) - 2, ix)
            inside = (np.isfinite(x) & (x >= heatmap.edges[0])
                      & (x <= heatmap.edges[-1]))
            safe_ix = np.clip(ix, 0, len(heatmap.edges) - 2)
            time_ix = np.arange(p)[None, :]
            value = heatmap.target_mean[d, safe_ix, time_ix]
            count = heatmap.counts[d, safe_ix, time_ix]
            good = inside & (count >= min_count) & np.isfinite(value) & allowed[None]
            valid[left:left + len(indices), d] = good
            deviations[left:left + len(indices), d] = np.where(
                good, value - heatmap.baseline_ps, 0.)
    return deviations, valid


def _ridge_fit(features, predictions, alpha):
    from sklearn.linear_model import Ridge
    x = np.asarray(features, dtype=np.float64)
    y = np.asarray(predictions, dtype=np.float64)
    if (x.ndim != 2 or y.shape != (len(x),) or len(x) < 3
            or not np.isfinite(x).all() or not np.isfinite(y).all()):
        raise ValueError("Need finite reference features and teacher predictions")
    return Ridge(alpha=float(alpha), fit_intercept=True).fit(x, y)


def calibrate_ridge(features, predictions, *,
                    alphas=(0.1, 1., 10., 100., 1000., 10000.),
                    folds=5, seed=42):
    """Select L2 via calibration-only CV; no held-out or blind leakage."""
    from sklearn.model_selection import KFold
    x = np.asarray(features, dtype=np.float64)
    y = np.asarray(predictions, dtype=np.float64)
    if x.ndim != 2 or y.shape != (len(x),) or len(x) < folds or folds < 2:
        raise ValueError("Insufficient calibration events")
    grid = np.asarray(alphas, dtype=np.float64)
    if grid.ndim != 1 or not len(grid) or not np.all(np.isfinite(grid)) or np.any(grid <= 0):
        raise ValueError("Ridge alpha grid must contain positive finite values")
    splits = list(KFold(n_splits=folds, shuffle=True, random_state=seed).split(x))
    scores = []
    for alpha in grid:
        mse = []
        for train, test in splits:
            model = _ridge_fit(x[train], y[train], alpha)
            mse.append(float(np.mean((model.predict(x[test]) - y[test]) ** 2)))
        scores.append(float(np.mean(mse)))
    best = int(np.argmin(scores))
    return _ridge_fit(x, y, grid[best]), {
        "alpha": float(grid[best]), "alphas": grid.tolist(),
        "cv_mse_ps2": scores, "cv_rmse_ps": float(np.sqrt(scores[best])),
        "cv_folds": folds, "seed": seed,
    }


def surrogate_components(model, deviations):
    """Per-event additive terms [N,2,P]; sum plus intercept is Ridge output."""
    a = np.asarray(deviations, dtype=np.float64)
    if a.ndim != 3 or a.shape[1] != 2 or model.coef_.size != a.shape[1] * a.shape[2]:
        raise ValueError("Incompatible feature grid and fitted Ridge")
    c = a * np.asarray(model.coef_).reshape(1, 2, -1)
    output = float(model.intercept_) + c.sum(axis=(1, 2))
    return c, output


def fidelity_metrics(predicted, teacher, baseline):
    """All scores are fidelity to saved ML outputs, never true physical labels."""
    a = np.asarray(predicted, dtype=np.float64)
    y = np.asarray(teacher, dtype=np.float64)
    if a.shape != y.shape or a.ndim != 1 or len(y) < 2:
        raise ValueError("Need equally sized nontrivial prediction vectors")
    error = a - y
    var = np.sum((y - y.mean()) ** 2)
    corr = (float(np.corrcoef(a, y)[0, 1])
            if np.std(a) > 0 and np.std(y) > 0 else None)
    return {
        "n": int(len(y)), "rmse_ps": float(np.sqrt(np.mean(error ** 2))),
        "mae_ps": float(np.mean(np.abs(error))),
        "bias_ps": float(np.mean(error)),
        "r2": float(1 - np.sum(error ** 2) / var) if var > 0 else None,
        "pearson_r": corr,
        "constant_mean_baseline_rmse_ps": float(
            np.sqrt(np.mean((y - float(baseline)) ** 2))),
        "teacher_mean_ps": float(y.mean()), "surrogate_mean_ps": float(a.mean()),
        "teacher_std_ps": float(y.std()), "surrogate_std_ps": float(a.std()),
    }
