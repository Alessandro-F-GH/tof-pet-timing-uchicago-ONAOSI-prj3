"""Bounded-RAM LSQR on frozen memmapped feature blocks.

Equivalent Ridge objective to the existing in-memory LSQR fit; the RDST block
is reparameterized to implement independent penalties on the two blocks.
No normal-equation matrix or full concatenated training matrix is created.
"""
from __future__ import annotations

import time
from contextlib import nullcontext
import numpy as np
from scipy.sparse.linalg import LinearOperator, lsqr
from sklearn.linear_model import Ridge


def fit_streamed_composite_ridge(features, target, parameters, *,
                                 shared=False, rows_per_chunk=128,
                                 logger=None):
    from waveform_analysis.models.kernel.multirocket_common import MultiRocketArtifact

    if features.block_names != ("multirocket", "rdst") or len(features.feature_blocks) != 2:
        raise ValueError("Streaming Ridge requires frozen MultiRocket and RDST feature blocks")
    a, b = features.feature_blocks
    n, p = a.shape
    if b.shape[0] != n or n < 2 or p < 1 or b.shape[1] < 1:
        raise ValueError("Incompatible feature block shapes")
    y = np.asarray(target, dtype=np.float64)
    if y.shape != (n,) or not np.all(np.isfinite(y)):
        raise ValueError("Invalid streamed Ridge target")
    if rows_per_chunk < 1:
        raise ValueError("rows_per_chunk must be positive")
    alpha_a = float(parameters["multirocket_alpha"])
    alpha_b = float(parameters["rdst_alpha"])
    if not (np.isfinite(alpha_a) and alpha_a > 0 and
            np.isfinite(alpha_b) and alpha_b > 0):
        raise ValueError("Ridge penalties must be finite positive numbers")

    # Same reparameterization as multirocket_rdst_common.fit().
    scale = np.sqrt(alpha_a / alpha_b)
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("Invalid RDST scaling ratio")
    chunk = int(rows_per_chunk)
    weights = (1., float(scale))
    arrays = (a, b)
    widths = (p, b.shape[1])
    offsets = (0, p)
    means = np.zeros(p + b.shape[1], dtype=np.float64)
    started = time.monotonic()

    # For direct Ridge, sklearn centers X and y for fit_intercept=True.
    # Control-standardized features use scaler(with_mean=False), so the
    # development-set mean must be computed here, without materializing X.
    if not shared:
        for array, width, offset, multiplier in zip(arrays, widths, offsets, weights):
            accum = np.zeros(width, dtype=np.float64)
            for begin in range(0, n, chunk):
                end = min(begin + chunk, n)
                accum += np.sum(array[begin:end], axis=0, dtype=np.float64)
            means[offset:offset + width] = (accum / n) * multiplier
    y_mean = 0. if shared else float(np.mean(y))
    rhs = y if shared else y - y_mean

    def matvec(coef):
        coef = np.asarray(coef, dtype=np.float64)
        out = np.zeros(n, dtype=np.float64)
        for array, width, offset, multiplier in zip(arrays, widths, offsets, weights):
            part = coef[offset:offset + width]
            for begin in range(0, n, chunk):
                end = min(begin + chunk, n)
                values = np.asarray(array[begin:end], dtype=np.float64)
                out[begin:end] += multiplier * (values @ part)
        if not shared:
            out -= float(means @ coef)
        return out

    def rmatvec(vec):
        vec = np.asarray(vec, dtype=np.float64)
        out = np.zeros(p + b.shape[1], dtype=np.float64)
        for array, width, offset, multiplier in zip(arrays, widths, offsets, weights):
            part = out[offset:offset + width]
            for begin in range(0, n, chunk):
                end = min(begin + chunk, n)
                values = np.asarray(array[begin:end], dtype=np.float64)
                part += multiplier * (values.T @ vec[begin:end])
        if not shared:
            out -= means * float(np.sum(vec))
        return out

    operator = LinearOperator((n, p + b.shape[1]), matvec=matvec,
                              rmatvec=rmatvec, dtype=np.float64)
    if logger:
        logger.info(
            "STREAMED RIDGE START | events=%d | features=%d+%d | chunk=%d "
            "| alphas=(%.6g, %.6g) | memory=bounded | solver=lsqr",
            n, p, b.shape[1], chunk, alpha_a, alpha_b,
        )

    # scipy LSQR's damp=sqrt(alpha) solves the same penalized objective
    # used by sklearn Ridge(solver='lsqr', tol=1e-4). The operator performs
    # matvec / rmatvec on small rows from the original cached memmaps.
    solved = lsqr(operator, rhs, damp=np.sqrt(alpha_a),
                  atol=1e-4, btol=1e-4)
    coef_scaled = np.asarray(solved[0], dtype=np.float64)
    if solved[1] not in (1, 2):
        raise RuntimeError(
            f"Streamed LSQR did not converge: istop={solved[1]}, "
            f"iterations={solved[2]}, rnorm={solved[3]:.6g}"
        )
    intercept = 0. if shared else y_mean - float(means @ coef_scaled)
    coef_scaled[p:] *= scale  # model predicts on original unscaled feature blocks

    regressor = Ridge(alpha=alpha_a, fit_intercept=not shared,
                      solver="lsqr", tol=1e-4)
    regressor.coef_ = coef_scaled
    regressor.intercept_ = intercept
    regressor.n_features_in_ = p + b.shape[1]
    meta = {
        "ridge_alpha": alpha_a,
        "training_events": n,
        "feature_count": p + b.shape[1],
        "fit_intercept": not shared,
        "detector_swap_antisymmetry_enforced": shared,
        "multirocket_alpha": alpha_a,
        "rdst_alpha": alpha_b,
        "rdst_fit_scale": float(scale),
        "feature_blocks": {"multirocket": [0, p], "rdst": [p, p + b.shape[1]]},
        "coefficient_coordinates": "control_standardized_features",
        "solver": "streamed_lsqr",
        "lsqr_istop": int(solved[1]),
        "lsqr_iterations": int(solved[2]),
        "lsqr_rnorm": float(solved[3]),
        "streamed_rows_per_chunk": chunk,
    }
    if logger:
        logger.info(
            "STREAMED RIDGE DONE | iterations=%d | stop=%d | residual=%.6g "
            "| elapsed=%.1f s", solved[2], solved[1], solved[3],
            time.monotonic()-started,
        )
    return MultiRocketArtifact(regressor, meta)
