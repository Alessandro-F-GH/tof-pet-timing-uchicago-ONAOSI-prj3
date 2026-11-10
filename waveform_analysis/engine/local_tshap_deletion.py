"""Small, paired-detector TSHAP + deletion kernels with frozen imputers.

A temporal window always acts on BOTH detectors. Each imputer is fitted or
prepared once on Development waveforms. Deletion changes only the observed
mask; the original query and frozen reference population never change.

The point-imputed value function is f(E_hat[X | X_observed]); it is not
generally E[f(X) | X_observed] for a nonlinear f.
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from waveform_analysis.models.ppca_imputer import ProbabilisticPCA

METHODS = ("knn", "ppca", "mean")
DIRECTIONS = ("positive", "negative")


def validate_windows(retained, starts, width):
    keep = np.asarray(retained, dtype=bool)
    starts = np.asarray(starts, dtype=np.int64)
    if (keep.ndim != 1 or starts.ndim != 1 or starts.size == 0 or
            width < 1 or starts.min() < 0 or
            (starts + width).max() > keep.size or
            np.any(np.diff(starts) < width) or
            any(not keep[s:s + width].all() for s in starts)):
        raise ValueError("Windows must be disjoint, within the retained sample mask")
    return keep, starts


@dataclass
class KNNQuery:
    full_cost: np.ndarray
    window_cost: np.ndarray


class FrozenImputationBank:
    """Joint PPCA, standardized KNN and timewise mean on one reference bank."""

    def __init__(self, reference, retained, *, knn_k=20, weighting="distance",
                 knn_block=256, scale_floor=0.05, ppca_components=32,
                 ppca_state=None):
        ref = np.asarray(reference, dtype=np.float32)
        retained = np.asarray(retained, dtype=bool)
        if (ref.ndim != 3 or ref.shape[1] != 2 or ref.shape[0] < 3 or
                retained.shape != (ref.shape[-1],) or not retained.any() or
                not np.all(np.isfinite(ref)) or not 1 <= knn_k <= len(ref) or
                weighting not in ("distance", "uniform") or knn_block < 1 or
                not 0 < scale_floor <= 1):
            raise ValueError("Invalid frozen imputation reference or settings")
        self.reference = np.ascontiguousarray(ref)
        self.retained = retained.copy()
        self.mean = np.mean(ref, axis=0, dtype=np.float64).astype(np.float32)
        self.knn_k = int(knn_k)
        self.weighting = weighting
        self.knn_block = int(knn_block)
        sample_std = np.std(ref.astype(np.float64), axis=0)
        rms = np.sqrt(np.mean(sample_std[:, retained] ** 2, axis=1))
        if np.any(rms <= 0) or not np.all(np.isfinite(rms)):
            raise ValueError("Reference has no usable detector variance")
        self.scale = np.maximum(sample_std, rms[:, None] * scale_floor).astype(np.float32)
        self.reference_z = (self.reference - self.mean) / self.scale
        self.ppca = ProbabilisticPCA(ppca_components)
        self._posterior_cholesky = {}
        if ppca_state is None:
            fitted = self.reference[:, :, retained].reshape(len(ref), -1)
            self.ppca.fit(fitted)
        else:
            mean, W, sigma2 = ppca_state
            self.ppca.mean = np.asarray(mean, dtype=np.float64)
            self.ppca.W = np.asarray(W, dtype=np.float64)
            self.ppca.sigma2 = float(sigma2)
            if (self.ppca.mean.shape != (2 * retained.sum(),) or
                    self.ppca.W.shape != (2 * retained.sum(), ppca_components) or
                    not np.isfinite(self.ppca.mean).all() or
                    not np.isfinite(self.ppca.W).all() or
                    not np.isfinite(self.ppca.sigma2) or self.ppca.sigma2 <= 0):
                raise ValueError("Cached PPCA state is incompatible with reference")
        self.ppca_state = (self.ppca.mean, self.ppca.W, self.ppca.sigma2)

    def knn_context(self, query, starts, width):
        """One reference pass; prefix sums give ALL windows and complements."""
        keep, starts = validate_windows(self.retained, starts, width)
        x = np.asarray(query, dtype=np.float32)
        if x.shape != self.mean.shape or not np.isfinite(x).all():
            raise ValueError("Invalid query waveform")
        query_z = (x - self.mean) / self.scale
        nref = len(self.reference)
        full = np.empty(nref, dtype=np.float64)
        window = np.empty((nref, len(starts)), dtype=np.float64)
        for first in range(0, nref, self.knn_block):
            last = min(first + self.knn_block, nref)
            diff = self.reference_z[first:last].astype(np.float64) - query_z
            contributions = np.einsum("nct,nct->nt", diff, diff)
            contributions[:, ~keep] = 0.0
            prefixes = np.pad(np.cumsum(contributions, axis=1), ((0, 0), (1, 0)))
            full[first:last] = prefixes[:, -1]
            window[first:last] = (
                prefixes[:, starts + width] - prefixes[:, starts])
        return KNNQuery(full_cost=full, window_cost=window)

    def _knn_impute(self, x, observed, costs):
        missing = self.retained & ~observed
        output = np.asarray(x, dtype=np.float32).copy()
        if not missing.any():
            return output
        if not observed.any():
            output[:, missing] = self.mean[:, missing]
            return output
        scores = np.maximum(np.asarray(costs, dtype=np.float64), 0.)
        k = self.knn_k
        # Tie-break by reference index; every run uses the same frozen order.
        chosen = np.argpartition(scores, k - 1)[:k]
        chosen = chosen[np.lexsort((chosen, scores[chosen]))]
        distance = np.sqrt(scores[chosen] / (2 * int(observed.sum())))
        if self.weighting == "uniform":
            weights = np.full(k, 1. / k)
        else:
            zero = distance <= 1e-12
            weights = zero.astype(np.float64) if zero.any() else (
                1. / np.maximum(distance, 1e-12))
            weights /= weights.sum()
        output[:, missing] = np.einsum(
            "k,kct->ct", weights, self.reference[chosen][:, :, missing],
            optimize=True).astype(np.float32)
        return output

    def _ppca_impute(self, x, observed):
        missing = self.retained & ~observed
        output = np.asarray(x, dtype=np.float32).copy()
        if not missing.any():
            return output
        obs = np.tile(observed[self.retained], 2)
        flat = np.asarray(x[:, self.retained].reshape(-1), dtype=np.float64)
        mu, W, sigma2 = self.ppca_state
        if not obs.any():
            flat[~obs] = mu[~obs]
        else:
            key = obs.tobytes()
            factor = self._posterior_cholesky.get(key)
            if factor is None:
                wo = W[obs]
                precision = np.eye(W.shape[1]) + (wo.T @ wo) / sigma2
                factor = np.linalg.cholesky(precision)
                self._posterior_cholesky[key] = factor
            rhs = W[obs].T @ (flat[obs] - mu[obs]) / sigma2
            latent = np.linalg.solve(factor.T, np.linalg.solve(factor, rhs))
            flat[~obs] = mu[~obs] + W[~obs] @ latent
        # Frozen ML inputs are min-max transformed and clipped to [0,1].
        # Keep observed values untouched; clamp only PPCA imputations.
        flat[~obs] = np.clip(flat[~obs], 0., 1.)
        output[:, self.retained] = flat.reshape(2, -1).astype(np.float32)
        return output

    def impute(self, method, query, observed, *, knn_cost=None):
        if method not in METHODS:
            raise ValueError(f"Unknown imputer {method}")
        x = np.asarray(query, dtype=np.float32)
        observed = np.asarray(observed, dtype=bool)
        if x.shape != self.mean.shape or observed.shape != self.retained.shape:
            raise ValueError("Query and temporal observed-mask shapes differ")
        observed = observed & self.retained
        if method == "mean":
            output = x.copy()
            missing = self.retained & ~observed
            output[:, missing] = self.mean[:, missing]
            return output
        if method == "ppca":
            return self._ppca_impute(x, observed)
        if knn_cost is None:
            raise ValueError("KNN requires frozen observed-coordinate distances")
        return self._knn_impute(x, observed, knn_cost)


def paired_shapley(bank, query, starts, width, predict, original_prediction,
                   *, method, knn_context=None):
    """Two-player TSHAP by deterministic point-conditional mean imputation."""
    _, starts = validate_windows(bank.retained, starts, width)
    empty = bank.impute(method, query,
                        np.zeros(len(bank.retained), bool),
                        knn_cost=np.zeros(len(bank.reference)))
    baseline = float(predict(empty[None])[0])
    modified = []
    for j, start in enumerate(starts):
        inside = np.zeros(len(bank.retained), bool)
        inside[start:start + width] = True
        observed_window = inside & bank.retained
        observed_complement = ~inside & bank.retained
        if method == "knn":
            if knn_context is None:
                raise ValueError("KNN query context is required")
            window_cost = knn_context.window_cost[:, j]
            complement_cost = knn_context.full_cost - window_cost
        else:
            window_cost = complement_cost = None
        modified.append(bank.impute(
            method, query, observed_window, knn_cost=window_cost))
        modified.append(bank.impute(
            method, query, observed_complement, knn_cost=complement_cost))
    predicted = np.asarray(predict(np.stack(modified)), dtype=np.float64)
    if predicted.shape != (2 * len(starts),) or not np.isfinite(predicted).all():
        raise ValueError("Model returned invalid coalition predictions")
    v_window = predicted[0::2]
    v_complement = predicted[1::2]
    scores = .5 * ((v_window - baseline) +
                   (float(original_prediction) - v_complement))
    return {
        "scores": scores, "v_empty": baseline,
        "v_window": v_window, "v_complement": v_complement,
    }


def signed_window_order(scores, direction):
    """Only windows with the requested sign; deterministic tie-breaking."""
    values = np.asarray(scores, dtype=np.float64)
    if values.ndim != 1 or not np.isfinite(values).all():
        raise ValueError("Signed window scores must be finite 1D")
    if direction == "positive":
        members = np.flatnonzero(values > 0)
        return members[np.lexsort((members, -values[members]))]
    if direction == "negative":
        members = np.flatnonzero(values < 0)
        return members[np.lexsort((members, values[members]))]
    raise ValueError("Unknown deletion direction")


def deletion_curve(bank, query, starts, width, scores, predict, original_prediction,
                   *, method, direction, knn_context=None):
    """Reconstruct the entire missing union from the ORIGINAL observed samples.

    Never reuse previous imputed values as observations. PPCA posterior and KNN
    neighbor search are conditioned afresh on the remaining original samples.
    """
    _, starts = validate_windows(bank.retained, starts, width)
    order = signed_window_order(scores, direction)
    remaining = bank.retained.copy()
    modified = []
    for j in order:
        start = int(starts[j])
        remaining[start:start + width] = False
        if method == "knn":
            if knn_context is None:
                raise ValueError("KNN query context is required")
            removed = np.flatnonzero(~remaining[starts])  # includes only fully removed windows
            cost = knn_context.full_cost - np.sum(
                knn_context.window_cost[:, removed], axis=1)
        else:
            cost = None
        modified.append(bank.impute(method, query, remaining, knn_cost=cost))
    if modified:
        output = np.asarray(predict(np.stack(modified)), dtype=np.float64)
        if output.shape != (len(order),) or not np.isfinite(output).all():
            raise ValueError("Invalid deletion model predictions")
        predictions = np.r_[float(original_prediction), output]
    else:
        predictions = np.array([float(original_prediction)], dtype=np.float64)
    n = np.arange(len(predictions))
    # A local explanation evaluates model-output fidelity, not noisy event labels.
    # Step zero is the untouched model prediction, so its deviation is zero.
    deviation = predictions - float(original_prediction)
    return {
        "removed_window_indices": order,
        "step": n,
        "fraction_all_windows": n / len(starts),
        "prediction_ps": predictions,
        "signed_prediction_change_ps": deviation,
        "absolute_prediction_deviation_ps": np.abs(deviation),
    }


DELETION_SEQUENCES = ("positive_negative", "negative_positive")


def signed_full_order(scores, sequence):
    """Rank every explained window once, with a documented sign transition.

    Positive: highest to lowest; negative: most negative to least negative.
    Exactly zero values follow all nonzero windows (no signed attribution).
    """
    values = np.asarray(scores, dtype=np.float64)
    if values.ndim != 1 or not np.isfinite(values).all():
        raise ValueError("Attribution scores must be a finite 1-D array")
    if sequence not in DELETION_SEQUENCES:
        raise ValueError("Unknown deletion sequence")
    positive = signed_window_order(values, "positive")
    negative = signed_window_order(values, "negative")
    zero = np.flatnonzero(values == 0.)
    first, second = ((positive, negative) if sequence == "positive_negative"
                     else (negative, positive))
    return np.concatenate((first, second, zero)), len(first), len(second)


def deletion_sequence_curve(bank, query, starts, width, scores, predict,
                            original_prediction, *, method, sequence,
                            knn_context=None, expected_endpoint=None):
    """Full-window deletion, with an explicit final unranked-sample cleanup.

    All imputations condition on original, still-observed values. The terminal
    waveform is always the frozen Development mean for ALL model-retained
    coordinates, including excluded boundary windows or sample-mask gaps.
    """
    keep, starts = validate_windows(bank.retained, starts, width)
    order, first_count, second_count = signed_full_order(scores, sequence)
    if len(order) != len(starts):
        raise AssertionError("Deletion order does not cover each ranked window once")
    remaining = keep.copy()
    modified = []
    removed_window_count = []
    removed_sample_count = []
    stage = []
    for step, j in enumerate(order, start=1):
        start = int(starts[j])
        remaining[start:start + width] = False
        if method == "knn":
            if knn_context is None:
                raise ValueError("Missing precomputed KNN query context")
            removed = order[:step]
            distance = np.maximum(
                knn_context.full_cost -
                np.sum(knn_context.window_cost[:, removed], axis=1),
                0.)
        else:
            distance = None
        modified.append(bank.impute(method, query, remaining, knn_cost=distance))
        removed_window_count.append(step)
        removed_sample_count.append(int(keep.sum() - remaining.sum()))
        stage.append("ranked")

    has_unranked_tail = bool(remaining.any())
    # Enforce the same mean-imputed endpoint for ALL methods even if PPCA's
    # unconditional mean differs from the arithmetic Development mean.
    terminal = np.asarray(query, dtype=np.float32).copy()
    terminal[:, keep] = bank.mean[:, keep]
    if has_unranked_tail:
        remaining[:] = False
        modified.append(terminal)
        removed_window_count.append(len(order))
        removed_sample_count.append(int(keep.sum()))
        stage.append("unranked_cleanup")
    else:
        modified[-1] = terminal
        stage[-1] = "mean_endpoint"
    if not modified:
        raise ValueError("No deletion steps available")
    values = np.asarray(predict(np.stack(modified)), dtype=np.float64)
    if values.shape != (len(modified),) or not np.isfinite(values).all():
        raise ValueError("Invalid full-sequence model predictions")
    if expected_endpoint is not None and not np.isclose(
            values[-1], expected_endpoint, atol=1e-5, rtol=1e-6):
        raise AssertionError("Full deletion does not converge to shared mean prediction")
    predictions = np.r_[float(original_prediction), values]
    removed_samples = np.r_[0, removed_sample_count]
    steps = np.arange(len(predictions))
    if removed_samples[-1] != int(keep.sum()):
        raise AssertionError("Deletion must remove every model-retained sample")
    return {
        "removed_window_indices": order,
        "first_phase_count": int(first_count),
        "second_phase_count": int(second_count),
        "zero_attribution_count": int(len(order) - first_count - second_count),
        "terminal_cleanup": has_unranked_tail,
        "step": steps,
        "removed_ranked_windows": np.r_[0, removed_window_count],
        "removed_samples": removed_samples,
        "fraction_deleted": removed_samples / int(keep.sum()),
        "stage": np.asarray(["original", *stage], dtype="U"),
        "prediction_ps": predictions,
        "signed_prediction_change_ps": predictions - float(original_prediction),
    }
