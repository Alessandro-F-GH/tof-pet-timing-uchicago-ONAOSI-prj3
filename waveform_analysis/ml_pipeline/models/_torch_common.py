from __future__ import annotations

import os

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


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


def device_from_config(config) -> torch.device:
    requested = str(config.get("training", {}).get("device", "auto")).lower()
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(requested)


def make_loader(x, y, batch, *, shuffle: bool, seed: int) -> DataLoader:
    generator = torch.Generator().manual_seed(int(seed))
    return DataLoader(
        TensorDataset(
            torch.from_numpy(np.ascontiguousarray(x, dtype=np.float32)),
            torch.from_numpy(np.asarray(y, dtype=np.float32)),
        ),
        batch_size=int(batch),
        shuffle=bool(shuffle),
        generator=generator,
    )


def predict_tensor(model, x, device: torch.device, batch: int) -> np.ndarray:
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


def rmse(residual) -> float:
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


def internal_early_stopping_split(x, y, fraction: float, seed: int):
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
