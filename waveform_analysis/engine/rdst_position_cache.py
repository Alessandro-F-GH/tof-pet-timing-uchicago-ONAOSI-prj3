"""Persistent, interruption-resumable frozen RDST best-match position cache.

Existing shared RDST feature files store only phi(s1)-phi(s2); the two
argmin positions cannot be inverted from their difference. Compute them
ONCE per blind event with the saved transform, then reuse via event-indexed
memory maps. No training, fit, or waveform perturbation is performed.
"""
from __future__ import annotations

from contextlib import contextmanager
import fcntl
import hashlib
import json
from pathlib import Path

import numpy as np

from waveform_analysis.core.io import atomic_json, canonical_hash
from waveform_analysis.data.view import waveform_view
from waveform_analysis.signal.sample_mask import apply_sample_mask

CACHE_VERSION = 1


def _array_digest(data):
    contiguous = np.ascontiguousarray(data)
    return hashlib.sha256(contiguous.view(np.uint8)).hexdigest()


def _identity(fitted, dataset, time_ps):
    artifact = fitted.feature_transform.artifact
    from waveform_analysis.models.kernel.rdst_common import export_shapelets
    shapelets = export_shapelets(artifact.transformer)
    if not artifact.shared:
        raise ValueError("Detector position caching applies to shared RDST only")
    mask = np.asarray(fitted.sample_mask, dtype=bool)
    time = np.asarray(time_ps, dtype=np.float64)
    indices = np.asarray(dataset.event_index)
    if time.ndim != 1 or int(np.count_nonzero(mask)) != len(time):
        raise ValueError("Cached argmin grid must match the saved model mask")
    transform_id = str(fitted.feature_transform.identity)
    if not transform_id or transform_id == "reloaded":
        # Without an identifiable transform, avoid accidental cross-run reuse.
        raise ValueError("Shared RDST position cache requires the original frozen transform identity")
    return {
        "version": CACHE_VERSION,
        "transform_identity": transform_id,
        "dataset_identity": str(dataset.manifest["analysis_protocol_identity"]),
        "n_events": int(dataset.n_events),
        "n_shapelets": int(len(shapelets["shapelet_id"])),
        "n_samples": int(len(time)),
        "mask_sha256": _array_digest(mask),
        "time_sha256": _array_digest(time),
        "event_ids_sha256": _array_digest(indices),
    }


@contextmanager
def _locked_cache(directory, identity):
    directory.mkdir(parents=True, exist_ok=True)
    lockfile = directory / "cache.lock"
    with lockfile.open("a+b") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        positions = done = None
        try:
            meta = directory / "identity.json"
            locs = directory / "positions.npy"
            valid = directory / "valid.npy"
            n = identity["n_events"]
            j = identity["n_shapelets"]
            dtype = np.uint16 if identity["n_samples"] < 65536 else np.uint32
            if meta.exists():
                stored = json.loads(meta.read_text(encoding="utf-8"))
                if stored != identity:
                    raise ValueError("Incompatible RDST temporal position cache metadata")
                if not locs.is_file() or not valid.is_file():
                    raise ValueError("RDST position cache is incomplete: remove its directory")
                positions = np.load(locs, mmap_mode="r+")
                done = np.load(valid, mmap_mode="r+")
                if (positions.shape != (n, 2, j) or positions.dtype != dtype
                        or done.shape != (n,) or done.dtype != np.uint8):
                    raise ValueError("RDST temporal position cache arrays have invalid shape/dtype")
            else:
                # Stale partial initialization is safely rebuilt under the lock.
                positions = np.lib.format.open_memmap(locs, mode="w+", dtype=dtype,
                                                       shape=(n, 2, j))
                done = np.lib.format.open_memmap(valid, mode="w+", dtype=np.uint8,
                                                  shape=(n,))
                done[:] = 0
                positions.flush()
                done.flush()
                atomic_json(meta, identity)
            yield positions, done
        finally:
            if positions is not None:
                positions.flush()
                if isinstance(positions, np.memmap):
                    positions._mmap.close()
            if done is not None:
                done.flush()
                if isinstance(done, np.memmap):
                    done._mmap.close()
            fcntl.flock(lock.fileno(), fcntl.LOCK_UN)


def shared_positions_for_events(*, cache_root, fitted, dataset, mode, event_indices,
                                frozen_features, time_ps, logger=None):
    """Return best-match positions [event, detector=2, shapelet].

    Disk cache is indexed by full blind population positions, not a particular
    sampling size. Previously computed events are reused across XAI runs.
    Uncomputed positions use one batched aeon.transform on both detectors,
    and are checked against the saved DIFFERENCED frozen features.
    """
    ids = np.asarray(event_indices, dtype=np.int64)
    if ids.ndim != 1 or np.any(ids < 0) or np.any(ids >= dataset.n_events):
        raise ValueError("Invalid requested blind event indices")
    if not len(ids):
        return np.empty((0, 2, len(fitted.feature_transform.artifact.scaler.scale_) // 3),
                        dtype=np.int32)
    if len(np.unique(ids)) != len(ids):
        raise ValueError("Duplicate blind event indices")
    identity = _identity(fitted, dataset, time_ps)
    directory = Path(cache_root) / "rdst_temporal_positions" / canonical_hash(identity)[:20]
    artifact = fitted.feature_transform.artifact
    from waveform_analysis.engine.rdst_temporal import recover_argmin
    lengths, dilations = (np.asarray(artifact.transformer.shapelets_[i]) for i in (2, 3))
    scale = np.asarray(artifact.scaler.scale_)
    with _locked_cache(directory, identity) as (positions, done):
        have = np.asarray(done[ids], dtype=bool)
        missing = ids[~have]
        if logger:
            logger.info("RDST position cache | hit=%d missing=%d | %s",
                        int(have.sum()), int(len(missing)), directory)
        if len(missing):
            # Bounded computation and per-batch durable checkpoints. This is
            # particularly important for wide waveforms and expensive aeon JIT.
            for offset in range(0, len(missing), 64):
                subset = missing[offset:offset + 64]
                pair = apply_sample_mask(waveform_view(dataset, mode, subset).materialize(),
                                         fitted.sample_mask)
                k = len(subset)
                pooled = np.concatenate((pair[:, :1, :], pair[:, 1:, :]), axis=0)
                raw = np.asarray(artifact.transformer.transform(pooled), dtype=np.float32)
                if raw.shape != (2*k, 3*identity["n_shapelets"]):
                    raise ValueError("Unexpected original shared RDST transformation output")
                z = np.asarray(artifact.scaler.transform(raw), dtype=np.float32)
                stored = np.asarray(frozen_features[subset], dtype=np.float32)
                if not np.allclose(z[:k] - z[k:], stored, atol=2e-4, rtol=2e-4):
                    raise ValueError("Individual detector transforms do not match frozen shared features")
                p1 = recover_argmin(z[:k], scale, lengths, dilations, identity["n_samples"])
                p2 = recover_argmin(z[k:], scale, lengths, dilations, identity["n_samples"])
                positions[subset, 0, :] = p1
                positions[subset, 1, :] = p2
                positions.flush()
                done[subset] = 1
                done.flush()
                if logger:
                    logger.info("RDST position cache | computed=%d/%d | saved",
                                min(offset + 64, len(missing)), len(missing))
        result = np.asarray(positions[ids], dtype=np.int32)
    return result
