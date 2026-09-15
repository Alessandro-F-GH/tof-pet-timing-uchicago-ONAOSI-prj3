from __future__ import annotations

import itertools
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch
from torch import nn

from .cnn import (
    _configure_reproducibility,
    _device,
    _gradient_norm,
    _loader,
    _predict_tensor,
    _rmse,
)
from .spec import ModelSpec


class JointPairCNN2D(nn.Module):
    """Paired waveform CNN following the Onishi-style TOF correction architecture."""

    def __init__(self, architecture: dict[str, Any]):
        super().__init__()
        channels = [int(v) for v in architecture.get("channels", [32, 64, 64])]
        kernels = [int(v) for v in architecture.get("kernels", [5, 3, 3])]
        dense_units = int(architecture.get("dense_units", 256))
        pool_size = int(architecture.get("pool_size", 3))

        if channels != [32, 64, 64]:
            raise ValueError("cnn_2d Onishi architecture requires channels=[32, 64, 64]")
        if kernels != [5, 3, 3]:
            raise ValueError("cnn_2d Onishi architecture requires kernels=[5, 3, 3]")
        if dense_units != 256:
            raise ValueError("cnn_2d Onishi architecture requires dense_units=256")
        if pool_size != 3:
            raise ValueError("cnn_2d Onishi architecture requires pool_size=3")

        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=(2, 5)),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(1, 3), stride=(1, 3)),
            nn.Conv2d(32, 64, kernel_size=(1, 3)),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(1, 3), stride=(1, 3)),
            nn.Conv2d(64, 64, kernel_size=(1, 3)),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(1, 3), stride=(1, 3)),
        )
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.LazyLinear(256),
            nn.ReLU(),
            nn.Linear(256, 1),
        )

    def forward(self, pair: torch.Tensor) -> torch.Tensor:
        if pair.ndim != 3 or pair.shape[1] != 2:
            raise ValueError(
                f"cnn_2d expects [event, detector=2, time], got {tuple(pair.shape)}"
            )
        features = self.features(pair[:, None, :, :])
        if features.shape[2] != 1:
            raise RuntimeError(
                f"cnn_2d first convolution must fuse detector height 2 -> 1, got {tuple(features.shape)}"
            )
        return self.head(features).squeeze(1)


@dataclass
class CNN2DArtifact:
    model: JointPairCNN2D
    device: str
    metadata: dict[str, Any]


def candidates(config):
    p = config.get("parameters", {})
    training = config.get("training", {})
    return [
        {"learning_rate": float(lr), "batch_size": int(batch)}
        for lr, batch in itertools.product(
            p.get("learning_rate", [1e-4]),
            p.get("batch_size", [training.get("batch_size", 32)]),
        )
    ]


def _mse_loss(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return torch.mean((prediction - target) ** 2)


def fit(
    params,
    train_x,
    train_target,
    *,
    seed,
    config,
    validation_x=None,
    validation_target=None,
):
    training_seed = _configure_reproducibility(seed)
    training = config.get("training", {})
    verbose = bool(config.get("verbose", False))
    logger = config.get("_logger")
    device = _device(config)

    batch = int(params.get("batch_size", training.get("batch_size", 32)))
    epochs = int(training.get("epochs", 100))
    learning_rate = float(params.get("learning_rate", 1e-4))
    decay_epochs = [int(v) for v in training.get("lr_decay_epochs", [30, 60])]
    decay_factor = float(training.get("lr_decay_factor", 0.1))

    x = np.asarray(train_x, dtype=np.float32)
    target = np.asarray(train_target, dtype=np.float64)
    if x.shape[0] != target.shape[0]:
        raise ValueError("cnn_2d train_x and train_target must contain the same number of events")
    if x.shape[0] < 1:
        raise ValueError("cnn_2d training requires at least one event")

    model = JointPairCNN2D(config.get("architecture", {})).to(device)
    with torch.no_grad():
        model(torch.from_numpy(np.ascontiguousarray(x[:1])).to(device))

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer,
        milestones=decay_epochs,
        gamma=decay_factor,
    )
    loader = _loader(x, target, batch, shuffle=True, seed=training_seed)
    output_limit = config.get("_prediction_max_abs_ps")

    if verbose and logger is not None:
        logger.info(
            "cnn_2d training | Onishi-style | loss=MSE | optimizer=Adam | "
            "lr=%.6g | batch=%d | epochs=%d | lr_decay_epochs=%s | "
            "lr_decay_factor=%.6g | train=%d | device=%s",
            learning_rate,
            batch,
            epochs,
            decay_epochs,
            decay_factor,
            target.size,
            device,
        )

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_gradient_norms = []
        for pair, batch_target in loader:
            pair = pair.to(device)
            batch_target = batch_target.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = _mse_loss(model(pair), batch_target)
            loss.backward()
            if verbose and logger is not None:
                epoch_gradient_norms.append(_gradient_norm(model))
            optimizer.step()
        scheduler.step()

        if verbose and logger is not None:
            prediction = _predict_tensor(model, x, device, batch)
            if output_limit is not None:
                prediction = np.clip(
                    prediction, -float(output_limit), float(output_limit)
                )
            train_rmse = _rmse(prediction - target)
            logger.info(
                "cnn_2d epoch %d/%d | train RMSE=%.4f ps | grad norm=%.6g | "
                "lr=%.6g | pred mean=%.4f ps | pred std=%.4f ps | "
                "pred min=%.4f ps | pred max=%.4f ps",
                epoch,
                epochs,
                train_rmse,
                float(np.mean(epoch_gradient_norms))
                if epoch_gradient_norms
                else float("nan"),
                float(optimizer.param_groups[0]["lr"]),
                float(np.mean(prediction)),
                float(np.std(prediction)),
                float(np.min(prediction)),
                float(np.max(prediction)),
            )

    return CNN2DArtifact(
        model=model,
        device=str(device),
        metadata={
            "training_loss": "mse",
            "optimizer": "adam",
            "epochs": epochs,
            "training_events": int(target.size),
            "training_uses_full_split": True,
            "refit_on_full_training_split": False,
            "external_validation_used_for_training": False,
            "learning_rate": learning_rate,
            "lr_decay_epochs": decay_epochs,
            "lr_decay_factor": decay_factor,
            "batch_size": batch,
            "output_max_abs_ps": None
            if output_limit is None
            else float(output_limit),
            "training_seed": training_seed,
            "deterministic_algorithms": True,
            "architecture_reference": "Onishi et al., Phys Med Biol 67 (2022) 04NT01",
            "input_definition": "paired normalized detector waveforms stacked as [2,time]",
            "prediction_definition": "joint CNN correction f_theta([s1;s2]) [ps]",
            "detector_axis_policy": "first 2x5 convolution fuses the two detector rows immediately",
            "detector_swap_antisymmetry_enforced": False,
            "parameter_count": int(
                sum(parameter.numel() for parameter in model.parameters())
            ),
        },
    )


def predict(artifact: CNN2DArtifact, normalized_pair: np.ndarray) -> np.ndarray:
    return _predict_tensor(
        artifact.model,
        normalized_pair,
        torch.device(artifact.device),
        512,
    )


def save(artifact: CNN2DArtifact, path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": artifact.model.state_dict(),
            "metadata": artifact.metadata,
        },
        path / "model.pt",
    )


def explain(artifact: CNN2DArtifact, normalized_pair: np.ndarray) -> np.ndarray:
    """Mean absolute input gradient, aggregated over events and detector rows."""
    device = torch.device(artifact.device)
    pair = torch.tensor(
        np.asarray(normalized_pair, dtype=np.float32),
        device=device,
        requires_grad=True,
    )
    artifact.model.eval()
    artifact.model.zero_grad(set_to_none=True)
    artifact.model(pair).sum().backward()
    if pair.grad is None:
        raise RuntimeError("cnn_2d XAI gradient is unavailable")
    importance = pair.grad.detach().abs().mean(dim=(0, 1))
    return importance.cpu().numpy().astype(np.float64)


MODEL_SPEC = ModelSpec(
    name="cnn_2d",
    candidates=candidates,
    fit=fit,
    predict=predict,
    save=save,
    explain=explain,
)
