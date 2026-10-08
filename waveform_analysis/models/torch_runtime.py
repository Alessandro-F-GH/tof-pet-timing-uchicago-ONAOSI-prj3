from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Protocol
from collections.abc import Mapping

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from waveform_analysis.data.torch_dataset import WaveformDataset
from numpy.typing import ArrayLike, NDArray


def configure_reproducibility(seed: int) -> int:
    value = int(seed)
    np.random.seed(value)
    torch.manual_seed(value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(value)
    torch.use_deterministic_algorithms(True)
    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    return value


def device_from_config(config: Mapping[str, Any]) -> torch.device:
    requested = str(config.get("training", {}).get("device", "auto")).lower()
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(requested)


def make_loader(
    x: ArrayLike, y: ArrayLike, batch: int, *, shuffle: bool, seed: int
) -> DataLoader:
    """Build the original float32 tensor dataset and seeded shuffle generator."""
    generator = torch.Generator().manual_seed(int(seed))
    return DataLoader(
        WaveformDataset(x, y),
        batch_size=int(batch),
        shuffle=bool(shuffle),
        generator=generator,
    )


def predict_tensor(
    model: nn.Module, x: ArrayLike, device: torch.device, batch: int
) -> NDArray[np.float64]:
    """Predict [event] in original batch order from [event, 2, sample] inputs."""
    loader = DataLoader(
        torch.from_numpy(np.ascontiguousarray(x, dtype=np.float32)),
        batch_size=int(batch),
        shuffle=False,
    )
    values = []
    model.eval()
    with torch.no_grad():
        for pair in loader:
            values.append(model(pair.to(device)).detach().cpu().numpy())
    return np.concatenate(values).astype(np.float64, copy=False)


def rmse(residual: ArrayLike) -> float:
    values = np.asarray(residual, dtype=np.float64)
    return float(np.sqrt(np.mean(values**2)))


def rmse_loss(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return torch.sqrt(torch.mean((prediction - target) ** 2))


def gradient_norm(model: nn.Module) -> float:
    total = 0.0
    for parameter in model.parameters():
        if parameter.grad is not None:
            value = parameter.grad.detach().norm(2).item()
            total += value * value
    return float(total**0.5)


def internal_early_stopping_split(
    x: ArrayLike,
    y: ArrayLike,
    fraction: float,
    seed: int,
) -> tuple[
    NDArray[np.float32], NDArray[np.float64], NDArray[np.float32], NDArray[np.float64]
]:
    """Retain the original permutation, rounding and fit/holdout slice order."""
    x = np.asarray(x, dtype=np.float32)
    y = np.asarray(y, dtype=np.float64)
    if x.shape[0] != y.shape[0]:
        raise ValueError(
            "Training inputs and targets must contain the same number of events"
        )
    n = int(y.shape[0])
    fraction = float(fraction)
    if not 0.0 < fraction < 1.0:
        raise ValueError("training.early_stopping_fraction must lie in (0, 1)")
    if n < 2:
        raise ValueError("Neural-network training requires at least two events")
    n_early = max(1, min(n - 1, int(round(n * fraction))))
    order = np.random.default_rng(int(seed)).permutation(n)
    early_idx = order[:n_early]
    fit_idx = order[n_early:]
    return x[fit_idx], y[fit_idx], x[early_idx], y[early_idx]


class TorchArtifact(Protocol):
    """Structural interface shared by the existing neural artifact dataclasses."""

    model: nn.Module
    device: str
    metadata: dict[str, Any]


def save_torch_artifact(artifact: TorchArtifact, path: Path) -> None:
    """Save model.pt with exactly the existing state_dict/metadata schema."""
    path.mkdir(parents=True, exist_ok=True)
    torch.save(
        {"state_dict": artifact.model.state_dict(), "metadata": artifact.metadata},
        path / "model.pt",
    )


def input_gradient_importance(
    artifact: TorchArtifact,
    normalized_pair: ArrayLike,
    *,
    missing_gradient_message: str,
) -> NDArray[np.float64]:
    """Compute the existing mean absolute gradient over event/detector axes.

    Input: [event, detector=2, sample]. Output: [sample]. Preserve model.eval(),
    zero_grad, summed backward pass and float32 reduction before float64 casting.
    This is distinct from the engine's grouped temporal occlusion procedure.
    """
    device = torch.device(artifact.device)
    pair = torch.tensor(
        np.asarray(normalized_pair, dtype=np.float32), device=device, requires_grad=True
    )
    artifact.model.eval()
    artifact.model.zero_grad(set_to_none=True)
    artifact.model(pair).sum().backward()
    if pair.grad is None:
        raise RuntimeError(missing_gradient_message)
    return pair.grad.detach().abs().mean(dim=(0, 1)).cpu().numpy().astype(np.float64)


# Retain serialized identities while legacy imports resolve to this module.
from waveform_analysis.core.compat import (
    preserve_legacy_identity as _preserve_legacy_identity,
)

_preserve_legacy_identity(
    __name__, "waveform_analysis.ml_pipeline.models._torch_common"
)
