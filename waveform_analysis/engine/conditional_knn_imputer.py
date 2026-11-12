"""Deterministic conditional-mean KNN over paired PET waveforms.

A reference-block distance pass serves every temporal window and complement.
All distances use the frozen model-input normalization, with both detectors.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import numpy as np

WEIGHT_NAMES = ("uniform", "distance")
IMPUTER_VERSION = 1


def digest(a):
    return hashlib.sha256(np.ascontiguousarray(a).view(np.uint8)).hexdigest()


def reference_validation_split(n, validation_events, reference_events, seed):
    if n < 3 or not 1 <= validation_events < n:
        raise ValueError("KNN requires disjoint, nonempty development reference and validation populations")
    order = np.random.default_rng(int(seed)).permutation(n)
    validation = np.sort(order[:validation_events])
    reference = order[validation_events:]
    if reference_events:
        if reference_events < 1:
            raise ValueError("Invalid KNN reference size")
        reference = reference[:reference_events]
    return np.sort(reference), validation


def neighbor_weights(distances, kind):
    distances = np.asarray(distances, dtype=np.float64)
    if kind == "uniform":
        return np.full(len(distances), 1.0 / len(distances))
    if kind != "distance":
        raise ValueError("Invalid KNN weighting")
    zero = distances <= 1e-12
    weights = zero.astype(float) if zero.any() else 1.0 / np.maximum(distances, 1e-12)
    return weights / weights.sum()


@dataclass
class KNNSelection:
    ks: np.ndarray
    selected_k: np.ndarray
    selected_weight: np.ndarray
    mse: np.ndarray
    mae: np.ndarray
    mse_mv2: np.ndarray
    mae_mv: np.ndarray

    def summary(self, engine):
        result = []
        for mask in range(engine.n_masks):
            weighting = int(self.selected_weight[mask])
            k = int(self.selected_k[mask])
            j = int(np.flatnonzero(self.ks == k)[0])
            result.append({
                "observed": "window" if mask < engine.n_windows else "complement",
                "window_index": mask % engine.n_windows,
                "start_sample": int(engine.starts[mask % engine.n_windows]),
                "end_sample": int(engine.ends[mask % engine.n_windows]),
                "k": k,
                "weighting": WEIGHT_NAMES[weighting],
                "mse": float(self.mse[mask, weighting, j]),
                "rmse": float(np.sqrt(self.mse[mask, weighting, j])),
                "mae": float(self.mae[mask, weighting, j]),
                "mse_mv2": float(self.mse_mv2[mask, weighting, j]),
                "rmse_mv": float(np.sqrt(self.mse_mv2[mask, weighting, j])),
                "mae_mv": float(self.mae_mv[mask, weighting, j]),
            })
        return result


class PairedWindowKNN:
    """Both detectors share each temporal mask; only model-retained samples count."""

    def __init__(self, reference, starts, width, retained, ks,
                 *, scale_mv, reference_block=512):
        reference = np.asarray(reference, dtype=np.float32)
        self.starts = np.asarray(starts, dtype=np.int64)
        self.ends = self.starts + int(width)
        self.retained = np.asarray(retained, dtype=bool)
        self.ks = np.asarray(sorted(set(int(k) for k in ks)), dtype=np.int64)
        if (reference.ndim != 3 or reference.shape[1] != 2 or
                not np.all(np.isfinite(reference))):
            raise ValueError("Expected finite reference waveforms [events,2,samples]")
        if self.retained.shape != (reference.shape[-1],) or not self.retained.any():
            raise ValueError("KNN retained mask incompatible with waveforms")
        if (self.starts.ndim != 1 or not len(self.starts) or int(width) < 1 or
                np.any(self.starts < 0) or np.any(self.ends > reference.shape[-1]) or
                any(not self.retained[s:e].all() for s,e in zip(self.starts,self.ends))):
            raise ValueError("Invalid KNN windows")
        if not len(self.ks) or self.ks[0] < 1 or self.ks[-1] > len(reference):
            raise ValueError("KNN candidate k exceeds reference count")
        if reference_block < 1:
            raise ValueError("KNN reference block must be positive")
        self.reference = np.ascontiguousarray(reference)
        self.n_windows = len(self.starts)
        self.n_masks = 2 * self.n_windows
        self.kmax = int(self.ks[-1])
        self.reference_block = int(reference_block)
        self.scale_mv = np.broadcast_to(np.asarray(scale_mv, dtype=np.float64),
                                         reference.shape[1:]).copy()
        if not np.isfinite(self.scale_mv).all() or (self.scale_mv <= 0).any():
            raise ValueError("Invalid frozen input-transform mV scales")
        observed = []
        for side in range(2):
            for start,end in zip(self.starts,self.ends):
                mask = np.zeros(reference.shape[-1], dtype=bool)
                mask[start:end] = True
                if side:
                    mask = ~mask
                mask &= self.retained
                if not mask.any() or not (self.retained & ~mask).any():
                    raise ValueError("KNN coalitions must have nonempty observed and hidden samples")
                observed.append(mask)
        self.observed = np.asarray(observed)
        self.hidden = self.retained[None,:] & ~self.observed

    def search(self, query):
        query = np.asarray(query,dtype=np.float32)
        if query.shape != self.reference.shape[1:] or not np.isfinite(query).all():
            raise ValueError("Invalid KNN query waveform")
        distances = np.full((self.n_masks,self.kmax),np.inf)
        ids = np.full((self.n_masks,self.kmax),-1,dtype=np.int64)
        for first in range(0,len(self.reference),self.reference_block):
            stop = min(first+self.reference_block,len(self.reference))
            difference = self.reference[first:stop].astype(np.float64)-query.astype(np.float64)
            contributions = np.sum(difference*difference,axis=1)
            contributions[:,~self.retained] = 0
            prefixes = np.pad(np.cumsum(contributions,axis=1),((0,0),(1,0)))
            window = (prefixes[:,self.ends]-prefixes[:,self.starts]).T
            complement = np.maximum(prefixes[:,-1][None,:]-window,0)
            scores = np.concatenate((window,complement),axis=0)
            candidates = np.concatenate((distances,scores),axis=1)
            candidate_ids = np.concatenate((
                ids,np.broadcast_to(np.arange(first,stop)[None,:],scores.shape)
            ),axis=1)
            for m in range(self.n_masks):
                order = np.lexsort((candidate_ids[m],candidates[m]))[:self.kmax]
                distances[m] = candidates[m,order]
                ids[m] = candidate_ids[m,order]
        if (ids < 0).any() or not np.isfinite(distances).all():
            raise RuntimeError("KNN returned incomplete neighbors")
        return ids,np.sqrt(distances)

    def reconstruct(self, query, mask, indices, distances, k, weighting):
        picked = np.asarray(indices[:k],dtype=np.int64)
        weights = neighbor_weights(distances[:k],weighting)
        output = np.asarray(query,dtype=np.float32).copy()
        missing = self.hidden[mask]
        output[:,missing] = np.einsum(
            "k,kct->ct",weights,self.reference[picked][:,:,missing],optimize=True
        ).astype(np.float32)
        return output,weights

    def reconstruct_selected(self, query, indices, distances, tuning, mask):
        k = int(tuning.selected_k[mask])
        weighting = WEIGHT_NAMES[int(tuning.selected_weight[mask])]
        waveform,weights = self.reconstruct(
            query,mask,indices[mask],distances[mask],k,weighting
        )
        return waveform,indices[mask,:k],weights

    def tune(self, validation, logger=None):
        validation = np.asarray(validation,dtype=np.float32)
        if (validation.ndim != 3 or validation.shape[1:] != self.reference.shape[1:]
                or not len(validation) or not np.isfinite(validation).all()):
            raise ValueError("Invalid validation waveforms")
        shape = (self.n_masks,2,len(self.ks))
        mse,mae,mse_mv2,mae_mv = [np.zeros(shape) for _ in range(4)]
        for n,query in enumerate(validation,1):
            neighbors,distances = self.search(query)
            for mask in range(self.n_masks):
                hidden = self.hidden[mask]
                error = (self.reference[neighbors[mask]][:,:,hidden].astype(np.float64)
                         -query[None,:,hidden].astype(np.float64)).reshape(self.kmax,-1)
                scale = self.scale_mv[:,hidden].reshape(-1)
                cumulative = np.cumsum(error,axis=0)
                zeros = distances[mask] <= 1e-12
                raw = zeros.astype(float) if zeros.any() else 1/np.maximum(distances[mask],1e-12)
                weighted = np.cumsum(error*raw[:,None],axis=0)
                summed_weights = np.cumsum(raw)
                for j,k in enumerate(self.ks):
                    for w,residual in enumerate((cumulative[k-1]/k,
                                                  weighted[k-1]/summed_weights[k-1])):
                        mse[mask,w,j] += np.mean(residual**2)
                        mae[mask,w,j] += np.mean(np.abs(residual))
                        mv = residual*scale
                        mse_mv2[mask,w,j] += np.mean(mv**2)
                        mae_mv[mask,w,j] += np.mean(np.abs(mv))
            if logger and (n == 1 or n%25 == 0 or n == len(validation)):
                logger.info("KNN TUNING | events=%d/%d | masks=%d | reference=%d",
                            n,len(validation),self.n_masks,len(self.reference))
        for metric in (mse,mae,mse_mv2,mae_mv):
            metric /= len(validation)
        best_k = np.empty(self.n_masks,dtype=np.int64)
        best_weight = np.empty(self.n_masks,dtype=np.int8)
        for mask in range(self.n_masks):
            w,j = np.unravel_index(np.argmin(mse[mask]),(2,len(self.ks)))
            best_k[mask],best_weight[mask] = self.ks[j],w
        tuning = KNNSelection(self.ks.copy(),best_k,best_weight,mse,mae,mse_mv2,mae_mv)
        if logger:
            for item in tuning.summary(self):
                logger.debug("KNN SELECT | observed=%s | window=%d | start=%d | end=%d | k=%d | "
                            "weight=%s | MSE=%.8g | RMSE=%.6g | MAE=%.6g | "
                            "MSE_mV2=%.8g | RMSE_mV=%.6g | MAE_mV=%.6g",
                            item["observed"],item["window_index"],item["start_sample"],
                            item["end_sample"],item["k"],item["weighting"],item["mse"],
                            item["rmse"],item["mae"],item["mse_mv2"],item["rmse_mv"],item["mae_mv"])
        return tuning
