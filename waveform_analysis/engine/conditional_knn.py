"""Model-agnostic conditional-mean KNN over already prepared paired waveforms.

No model, feature transform, inference method, or training pipeline is imported.
The *only* regression labels are previously materialized model predictions.
"""
from __future__ import annotations

import numpy as np


def tshap_windows(n_samples, width, stride, retained_mask=None):
    """Reproduce the interior, fully retained temporal windows used by TSHAP."""
    if width < 1 or stride < 1 or width > n_samples:
        raise ValueError("Require 1 <= width <= samples and stride >= 1")
    keep = (np.ones(n_samples, dtype=bool) if retained_mask is None
            else np.asarray(retained_mask, dtype=bool))
    if keep.shape != (n_samples,) or not np.any(keep):
        raise ValueError("Invalid temporal sample mask")
    candidates = list(range(0, min(n_samples, n_samples - width + stride), stride))
    starts = [s for s in candidates[1:-1]
              if s + width <= n_samples and np.all(keep[s:s + width])]
    if not starts:
        raise ValueError("No interior TSHAP windows fully inside the retained samples")
    if int(keep.sum()) <= width:
        raise ValueError("Complement has no retained samples")
    return np.asarray(starts, dtype=np.int64), keep


def _weighted_prefix(values, normalized_distances, k_values):
    """Inverse-distance estimates; exact zero-distance neighbors dominate."""
    zeros = normalized_distances <= 1e-12
    nzeros = np.cumsum(zeros.astype(np.int64))
    sum_zeros = np.cumsum(np.where(zeros, values, 0.0))
    safe = np.maximum(normalized_distances, 1e-12)
    weights = 1.0 / safe
    weighted = np.cumsum(weights * values) / np.cumsum(weights)
    indices = np.asarray(k_values, dtype=np.int64) - 1
    return np.where(nzeros[indices] > 0,
                    sum_zeros[indices] / np.maximum(nzeros[indices], 1),
                    weighted[indices])


class ConditionalKNN:
    """Exact Euclidean top-k for any number of reference events.

    Work is O(N * channels * samples) per query: a squared-distance pass
    followed by prefix sums yields every TSHAP window AND its complement.
    Distance normalization does not affect neighbor ordering. No full
    query-by-reference-by-window tensor is retained.
    """

    def __init__(self, reference, predictions, *, starts, width, keep,
                 k_values, weighted=False, reference_batch=2048):
        self.reference = np.asarray(reference)
        self.labels = np.asarray(predictions, dtype=np.float64)
        if self.reference.ndim != 3 or self.reference.shape[1] != 2:
            raise ValueError("Expected prepared [events, two detectors, samples] waveforms")
        n, _, t = self.reference.shape
        if self.labels.shape != (n,) or not np.all(np.isfinite(self.labels)):
            raise ValueError("Prediction labels must be finite and aligned to the reference events")
        if self.reference.dtype.kind != "f":
            raise ValueError("Prepared waveform values must be floating point")
        self.keep = np.asarray(keep, dtype=bool)
        self.starts = np.asarray(starts, dtype=np.int64)
        self.width = int(width)
        self.k_values = np.unique(np.asarray(k_values, dtype=np.int64))
        self.weighted = bool(weighted)
        self.batch = int(reference_batch)
        if self.keep.shape != (t,) or not self.keep.any():
            raise ValueError("Invalid retained sample mask")
        if self.starts.ndim != 1 or not self.starts.size or np.any(self.starts < 0) or np.any(self.starts + width > t):
            raise ValueError("Invalid temporal window starts")
        if any(not self.keep[s:s + width].all() for s in self.starts):
            raise ValueError("All attributed windows must be fully retained")
        if self.k_values.size == 0 or self.k_values[0] < 1 or self.k_values[-1] >= n:
            raise ValueError("k must satisfy 1 <= k < number of reference events (LOO)")
        if self.batch < 1 or self.keep.sum() <= width:
            raise ValueError("Invalid reference batch or empty complement")
        self.kmax = int(self.k_values[-1])
        self.counts = np.asarray([2 * self.width, 2 * (int(self.keep.sum()) - self.width)], dtype=np.float64)

    def estimate(self, query_index):
        """Return (window, [W,R], k, [uniform,weighted]) means and neighbor radii.

        The reference event with query_index is forcibly excluded (leave-one-out).
        """
        idx = int(query_index)
        n, _, t = self.reference.shape
        if not 0 <= idx < n:
            raise IndexError("Query must belong to the reference population for LOO")
        query = np.asarray(self.reference[idx], dtype=np.float64)
        nwin = len(self.starts)
        per_window = np.empty((n, nwin), dtype=np.float64)
        full = np.empty(n, dtype=np.float64)
        ends = self.starts + self.width - 1
        before = np.maximum(self.starts - 1, 0)
        for first in range(0, n, self.batch):
            last = min(first + self.batch, n)
            chunk = np.asarray(self.reference[first:last], dtype=np.float64)
            delta = chunk - query[None, :, :]
            sq = np.einsum("nct,nct->nt", delta, delta, optimize=True)
            sq[:, ~self.keep] = 0.0
            prefix = np.cumsum(sq, axis=1)
            right = prefix[:, ends]
            left = prefix[:, before].copy()
            left[:, self.starts == 0] = 0.0
            per_window[first:last] = np.maximum(right - left, 0.0)
            full[first:last] = prefix[:, -1]
        if not np.all(np.isfinite(full)) or not np.all(np.isfinite(per_window)):
            raise ValueError("Non-finite waveform distances")

        nweight = 2 if self.weighted else 1
        estimates = np.empty((nwin, 2, self.k_values.size, nweight), dtype=np.float64)
        radii = np.empty((nwin, 2), dtype=np.float64)
        nearest = np.empty((nwin, 2), dtype=np.float64)
        for w in range(nwin):
            complement = full - per_window[:, w]
            # A tiny negative can arise from cancellation in float64.
            if np.min(complement) < -1e-8 * max(1.0, float(np.max(full))):
                raise ArithmeticError("Numerical instability in complement distances")
            for side, sq_dist in enumerate((per_window[:, w], np.maximum(complement, 0.0))):
                squared = sq_dist.copy()
                squared[idx] = np.inf  # critical: never return the query itself
                top = np.argpartition(squared, self.kmax - 1)[:self.kmax]
                top = top[np.lexsort((top, squared[top]))]
                if not np.all(np.isfinite(squared[top])):
                    raise ValueError("Not enough finite reference neighbors")
                pred = self.labels[top]
                distance = np.sqrt(squared[top] / self.counts[side])
                uniform = np.cumsum(pred)[self.k_values - 1] / self.k_values
                estimates[w, side, :, 0] = uniform
                if self.weighted:
                    estimates[w, side, :, 1] = _weighted_prefix(pred, distance, self.k_values)
                nearest[w, side] = distance[0]
                radii[w, side] = distance[-1]
        return estimates, nearest, radii
