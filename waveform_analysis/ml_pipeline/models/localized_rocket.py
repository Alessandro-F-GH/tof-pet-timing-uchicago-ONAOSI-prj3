from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any
import joblib
import numpy as np
import torch
from sklearn.linear_model import Ridge
from torch.nn import functional as F

from .spec import ModelSpec


@dataclass(frozen=True)
class RocketKernel:
    weights: np.ndarray
    dilation: int
    bias: float


@dataclass
class LocalizedRocketArtifact:
    model: Ridge
    kernels: tuple[RocketKernel, ...]
    n_time_bins: int
    statistics: tuple[str, ...]
    feature_scale: np.ndarray
    signature: tuple[Any, ...]
    chunk_size: int
    device: str
    metadata: dict[str, Any]


_FEATURE_CACHE: OrderedDict[tuple[Any, ...], np.ndarray] = OrderedDict()
_CACHE_ENTRIES = 2


def candidates(config: dict[str, Any]) -> list[dict[str, Any]]:
    values = (config.get("parameters", {}) or {}).get("alpha", [1e-3, 1e-2, 1e-1, 1.0, 10.0])
    return [{"alpha": float(alpha)} for alpha in values]


def _transform_config(config: dict[str, Any]) -> dict[str, Any]:
    return config.get("transform", {}) or {}


def _device(value: str) -> torch.device:
    requested = str(value or "auto").lower()
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("localized_rocket requested CUDA but CUDA is unavailable")
    return device


def _signature(config: dict[str, Any], n_samples: int) -> tuple[Any, ...]:
    transform = _transform_config(config)
    return (
        "torch_vectorized_v1",
        int(n_samples),
        int(transform.get("n_kernels", 128)),
        tuple(int(v) for v in transform.get("kernel_sizes", [7, 9, 11])),
        int(transform.get("max_dilation", 32)),
        int(transform.get("n_time_bins", 64)),
        tuple(str(v).lower() for v in transform.get("features", ["ppv", "max"])),
        int(transform.get("seed", 20260815)),
    )


def _kernel_bank(config: dict[str, Any], n_samples: int) -> tuple[RocketKernel, ...]:
    transform = _transform_config(config); rng = np.random.default_rng(int(transform.get("seed", 20260815)))
    n_kernels = int(transform.get("n_kernels", 128)); sizes = [int(v) for v in transform.get("kernel_sizes", [7, 9, 11])]; max_dilation = int(transform.get("max_dilation", 32))
    if n_kernels < 1 or not sizes or any(size < 3 or size % 2 == 0 for size in sizes):
        raise ValueError("localized_rocket requires positive n_kernels and odd kernel sizes >= 3")
    kernels = []
    for _ in range(n_kernels):
        size = int(rng.choice(sizes)); allowed_max = max(1, min(max_dilation, (int(n_samples) - 1) // max(1, size - 1))); dilations = [1]
        while dilations[-1] * 2 <= allowed_max: dilations.append(dilations[-1] * 2)
        dilation = int(rng.choice(dilations)); weights = rng.normal(size=size).astype(np.float32); weights -= np.mean(weights); norm = float(np.linalg.norm(weights)); weights = weights / (norm if norm > 1e-12 else 1.0); bias = float(rng.uniform(-1.0, 1.0)); kernels.append(RocketKernel(weights, dilation, bias))
    return tuple(kernels)


def _bin_geometry(n_samples: int, n_bins: int) -> tuple[np.ndarray, np.ndarray]:
    n_bins = max(1, min(int(n_bins), int(n_samples))); edges = np.linspace(0, int(n_samples), n_bins + 1, dtype=np.int64); return edges[:-1], np.diff(edges)


def _kernel_groups(kernels: tuple[RocketKernel, ...], device: torch.device):
    grouped: dict[tuple[int, int], list[tuple[int, RocketKernel]]] = {}
    for index, kernel in enumerate(kernels): grouped.setdefault((int(kernel.weights.size), int(kernel.dilation)), []).append((index, kernel))
    output = []
    for (size, dilation), members in grouped.items():
        indices = torch.tensor([index for index, _ in members], dtype=torch.long, device=device)
        # torch.conv1d is correlation; flip weights to preserve scipy convolve1d semantics.
        weights = torch.as_tensor(np.stack([kernel.weights[::-1].copy() for _, kernel in members]), dtype=torch.float32, device=device)[:, None, :]
        bias = torch.tensor([kernel.bias for _, kernel in members], dtype=torch.float32, device=device)
        output.append((indices, weights, bias, dilation, dilation * (size - 1) // 2))
    return output


def _pool_bins(response: torch.Tensor, bin_index: torch.Tensor, widths: torch.Tensor, statistics: tuple[str, ...]) -> list[torch.Tensor]:
    # response: [event, detector, kernel, sample]
    events, detectors, kernels, samples = response.shape; flat = response.reshape(events * detectors, kernels, samples); bins = int(widths.numel()); index = bin_index.view(1, 1, samples).expand(flat.shape[0], kernels, samples); pooled = []
    for statistic in statistics:
        if statistic == "ppv":
            values = torch.zeros((flat.shape[0], kernels, bins), dtype=flat.dtype, device=flat.device); values.scatter_add_(2, index, (flat > 0.0).to(flat.dtype)); values /= widths.view(1, 1, bins)
        elif statistic == "max":
            values = torch.full((flat.shape[0], kernels, bins), -torch.inf, dtype=flat.dtype, device=flat.device); values.scatter_reduce_(2, index, flat, reduce="amax", include_self=True)
        else:
            raise ValueError(f"Unsupported localized_rocket feature: {statistic}")
        pooled.append(values.reshape(events, detectors, kernels, bins))
    return pooled


def _pair_features(pair: np.ndarray, kernels: tuple[RocketKernel, ...], n_bins: int, statistics: tuple[str, ...], chunk_size: int, device_name: str) -> np.ndarray:
    x = np.asarray(pair, dtype=np.float32)
    if x.ndim != 3 or x.shape[1] != 2: raise ValueError("localized_rocket expects [event, detector, sample]")
    unknown = set(statistics) - {"ppv", "max"}
    if unknown or not statistics: raise ValueError(f"Unsupported localized_rocket features: {sorted(unknown)}")
    device = _device(device_name); starts, widths_np = _bin_geometry(x.shape[-1], n_bins); bins = starts.size; output = np.empty((x.shape[0], len(kernels), len(statistics), bins), dtype=np.float32); chunk_size = max(1, int(chunk_size)); groups = _kernel_groups(kernels, device); widths = torch.tensor(widths_np, dtype=torch.float32, device=device); bin_index = torch.repeat_interleave(torch.arange(bins, device=device), torch.tensor(widths_np, dtype=torch.long, device=device))
    with torch.inference_mode():
        for first in range(0, x.shape[0], chunk_size):
            last = min(x.shape[0], first + chunk_size); events = last - first; signal = torch.as_tensor(np.ascontiguousarray(x[first:last]), dtype=torch.float32, device=device).reshape(events * 2, 1, x.shape[-1]); chunk_features = torch.empty((events, len(kernels), len(statistics), bins), dtype=torch.float32, device=device)
            for indices, weights, bias, dilation, padding in groups:
                response = F.conv1d(signal, weights, bias=bias, stride=1, padding=padding, dilation=dilation).reshape(events, 2, indices.numel(), x.shape[-1]); pooled = _pool_bins(response, bin_index, widths, statistics)
                for statistic_index, values in enumerate(pooled): chunk_features[:, indices, statistic_index, :] = values[:, 0, :, :] - values[:, 1, :, :]
            output[first:last] = chunk_features.reshape(events, -1).cpu().numpy().reshape(events, len(kernels), len(statistics), bins)
    return output.reshape(x.shape[0], -1)


def _array_key(values: np.ndarray) -> tuple[Any, ...]:
    x = np.asarray(values); return (int(x.__array_interface__["data"][0]), tuple(x.shape), tuple(x.strides), str(x.dtype))


def _cached_features(pair: np.ndarray, kernels: tuple[RocketKernel, ...], n_bins: int, statistics: tuple[str, ...], signature: tuple[Any, ...], chunk_size: int, device_name: str) -> np.ndarray:
    key = (_array_key(pair), signature)
    cached = _FEATURE_CACHE.get(key)
    if cached is not None:
        _FEATURE_CACHE.move_to_end(key); return cached
    features = _pair_features(pair, kernels, n_bins, statistics, chunk_size, device_name); _FEATURE_CACHE[key] = features; _FEATURE_CACHE.move_to_end(key)
    while len(_FEATURE_CACHE) > _CACHE_ENTRIES: _FEATURE_CACHE.popitem(last=False)
    return features


def fit(params, train_x, train_target, *, seed, config, validation_x=None, validation_target=None, final_epochs=None):
    del seed, validation_x, validation_target, final_epochs
    x = np.asarray(train_x, dtype=np.float32); transform = _transform_config(config); signature = _signature(config, x.shape[-1]); kernels = _kernel_bank(config, x.shape[-1]); n_bins = int(transform.get("n_time_bins", 64)); statistics = tuple(str(v).lower() for v in transform.get("features", ["ppv", "max"])); chunk_size = int(transform.get("chunk_size", 512)); device_name = str(transform.get("device", "auto")); features = _cached_features(x, kernels, n_bins, statistics, signature, chunk_size, device_name)
    scale = np.std(features, axis=0, dtype=np.float64); scale = np.where(scale > 1e-6, scale, 1.0).astype(np.float32)
    ridge_config = config.get("ridge", {}) or {}; model = Ridge(alpha=float(params["alpha"]), fit_intercept=False, solver=str(ridge_config.get("solver", "lsqr")), tol=float(ridge_config.get("tolerance", 1e-4)), max_iter=int(ridge_config.get("max_iterations", 2000))); model.fit(features / scale[None, :], np.asarray(train_target, dtype=np.float64))
    resolved_device = str(_device(device_name)); metadata = {"n_kernels": len(kernels), "n_time_bins": min(n_bins, x.shape[-1]), "n_features": int(features.shape[1]), "statistics": list(statistics), "fixed_transform": True, "vectorized": True, "chunk_size": chunk_size, "device": resolved_device}
    return LocalizedRocketArtifact(model, kernels, n_bins, statistics, scale, signature, chunk_size, resolved_device, metadata)


def predict(artifact: LocalizedRocketArtifact, normalized_pair: np.ndarray) -> np.ndarray:
    x = np.asarray(normalized_pair, dtype=np.float32); device_name = artifact.device if artifact.device != "cuda" or torch.cuda.is_available() else "cpu"; features = _cached_features(x, artifact.kernels, artifact.n_time_bins, artifact.statistics, artifact.signature, artifact.chunk_size, device_name); return np.asarray(artifact.model.predict(features / artifact.feature_scale[None, :]), dtype=np.float64)


def save(artifact: LocalizedRocketArtifact, path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True); joblib.dump(artifact, path / "model.joblib")


def explain(artifact: LocalizedRocketArtifact, normalized_pair: np.ndarray) -> np.ndarray:
    n_samples = int(np.asarray(normalized_pair).shape[-1]); starts, widths = _bin_geometry(n_samples, artifact.n_time_bins); bins = starts.size; effective = np.abs(np.asarray(artifact.model.coef_, dtype=np.float64) / np.asarray(artifact.feature_scale, dtype=np.float64)); importance_bins = effective.reshape(len(artifact.kernels), len(artifact.statistics), bins).sum(axis=(0, 1)); importance = np.zeros(n_samples, dtype=np.float64)
    for index, start in enumerate(starts): importance[int(start):int(start + widths[index])] = importance_bins[index]
    return importance


MODEL_SPEC = ModelSpec(name="localized_rocket", candidates=candidates, fit=fit, predict=predict, save=save, explain=explain)
