from __future__ import annotations

import copy
import itertools
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from ..stats import ctr_fwhm
from .spec import ModelSpec


class SharedScorerCNN(nn.Module):
    """One detector scorer shared by both channels; pair output is antisymmetric."""

    def __init__(self, architecture: dict[str, Any]) -> None:
        super().__init__()
        channels = [int(v) for v in architecture.get("channels", [16, 32, 64])]
        kernels = [int(v) for v in architecture.get("kernels", [33, 15, 13])]
        strides = [int(v) for v in architecture.get("strides", [2, 4, 4])]
        dilations = [int(v) for v in architecture.get("dilations", [1, 1, 2])]
        if not (len(channels) == len(kernels) == len(strides) == len(dilations)):
            raise ValueError("CNN channels/kernels/strides/dilations must have equal length")
        layers: list[nn.Module] = []
        incoming = 1
        for outgoing, kernel, stride, dilation in zip(channels, kernels, strides, dilations):
            padding = dilation * (kernel - 1) // 2
            layers.extend(
                [
                    nn.Conv1d(incoming, outgoing, kernel, stride=stride, dilation=dilation, padding=padding),
                    nn.BatchNorm1d(outgoing),
                    nn.SiLU(),
                ]
            )
            incoming = outgoing
        dense = [int(v) for v in architecture.get("dense_units", [32])]
        head: list[nn.Module] = [nn.AdaptiveAvgPool1d(1), nn.Flatten()]
        for width in dense:
            head.extend([nn.Linear(incoming, width), nn.SiLU()])
            incoming = width
        head.append(nn.Linear(incoming, 1))
        self.features = nn.Sequential(*layers)
        self.head = nn.Sequential(*head)

    def score(self, waveform: torch.Tensor) -> torch.Tensor:
        return self.head(self.features(waveform[:, None, :])).squeeze(1)

    def forward(self, pair: torch.Tensor) -> torch.Tensor:
        return self.score(pair[:, 0, :]) - self.score(pair[:, 1, :])


@dataclass
class CNNArtifact:
    model: SharedScorerCNN
    device: str
    metadata: dict[str, Any]


def candidates(config: dict[str, Any]) -> list[dict[str, Any]]:
    parameters = config.get("parameters", {})
    learning_rates = parameters.get("learning_rate", [1e-3])
    weight_decays = parameters.get("weight_decay", [1e-5])
    return [
        {"learning_rate": float(lr), "weight_decay": float(wd)}
        for lr, wd in itertools.product(learning_rates, weight_decays)
    ]


def _device(config: dict[str, Any]) -> torch.device:
    requested = str(config.get("training", {}).get("device", "auto")).lower()
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(requested)


def _loader(x: np.ndarray, y: np.ndarray, batch: int, *, shuffle: bool, seed: int) -> DataLoader:
    generator = torch.Generator().manual_seed(int(seed))
    return DataLoader(
        TensorDataset(torch.from_numpy(np.ascontiguousarray(x, dtype=np.float32)), torch.from_numpy(np.asarray(y, dtype=np.float32))),
        batch_size=int(batch),
        shuffle=shuffle,
        generator=generator,
    )


def _predict_tensor(model: nn.Module, x: np.ndarray, device: torch.device, batch: int) -> np.ndarray:
    loader = DataLoader(torch.from_numpy(np.ascontiguousarray(x, dtype=np.float32)), batch_size=int(batch), shuffle=False)
    values: list[np.ndarray] = []
    model.eval()
    with torch.no_grad():
        for pair in loader:
            values.append(model(pair.to(device)).detach().cpu().numpy())
    return np.concatenate(values).astype(np.float64, copy=False)


def fit(
    params: dict[str, Any],
    train_x: np.ndarray,
    train_target: np.ndarray,
    *,
    seed: int,
    config: dict[str, Any],
    validation_x: np.ndarray | None = None,
    validation_target: np.ndarray | None = None,
    final_epochs: int | None = None,
) -> CNNArtifact:
    torch.manual_seed(int(seed)); np.random.seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))
    training = config.get("training", {})
    device = _device(config)
    model = SharedScorerCNN(config.get("architecture", {})).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(params["learning_rate"]),
        weight_decay=float(params["weight_decay"]),
    )
    loss_fn = nn.MSELoss()
    batch = int(training.get("batch_size", 64))
    max_epochs = int(final_epochs or training.get("epochs", 350))
    patience = int(training.get("patience", 30))
    min_delta = float(training.get("min_delta_ps", 0.05))
    loader = _loader(train_x, train_target, batch, shuffle=True, seed=seed)

    best_score = float("inf")
    best_epoch = max_epochs
    best_state: dict[str, torch.Tensor] | None = None
    stale = 0
    for epoch in range(1, max_epochs + 1):
        model.train()
        for pair, target in loader:
            pair = pair.to(device); target = target.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model(pair), target)
            loss.backward()
            clip = float(training.get("gradient_clip_norm", 10.0))
            if clip > 0:
                nn.utils.clip_grad_norm_(model.parameters(), clip)
            optimizer.step()

        if validation_x is None or validation_target is None or final_epochs is not None:
            continue
        prediction = _predict_tensor(model, validation_x, device, batch)
        # target = -baseline residual, hence corrected residual = prediction-target.
        score = float(ctr_fwhm(prediction - np.asarray(validation_target), config.get("fit")).ctr_ps)
        if score < best_score - min_delta:
            best_score = score
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            stale = 0
        else:
            stale += 1
            if stale >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return CNNArtifact(
        model=model,
        device=str(device),
        metadata={"best_epoch": int(best_epoch), "best_validation_ctr_ps": float(best_score)},
    )


def predict(artifact: CNNArtifact, normalized_pair: np.ndarray) -> np.ndarray:
    device = torch.device(artifact.device)
    batch = 512
    return _predict_tensor(artifact.model, normalized_pair, device, batch)


def save(artifact: CNNArtifact, path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    torch.save(
        {"state_dict": artifact.model.state_dict(), "metadata": artifact.metadata},
        path / "model.pt",
    )


def explain(artifact: CNNArtifact, normalized_pair: np.ndarray) -> np.ndarray:
    device = torch.device(artifact.device)
    pair = torch.tensor(np.asarray(normalized_pair, dtype=np.float32), device=device, requires_grad=True)
    artifact.model.zero_grad(set_to_none=True)
    artifact.model(pair).sum().backward()
    return pair.grad.detach().abs().mean(dim=(0, 1)).cpu().numpy().astype(np.float64)


MODEL_SPEC = ModelSpec(
    name="cnn",
    normalization="global",
    candidates=candidates,
    fit=fit,
    predict=predict,
    save=save,
    explain=explain,
)
