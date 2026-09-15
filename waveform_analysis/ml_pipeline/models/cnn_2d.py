from __future__ import annotations

import copy
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch
from torch import nn

from .cnn import _configure_reproducibility, _device, _gradient_norm, _internal_early_stopping_split, _loader, _predict_tensor, _rmse, candidates
from .spec import ModelSpec


class JointPairCNN2D(nn.Module):
    """Joint CNN: one 2-D detector-fusion convolution followed by 1-D temporal convolutions."""

    def __init__(self, architecture: dict[str, Any]):
        super().__init__()
        channels = [int(v) for v in architecture.get("channels", [16, 32, 64])]
        kernels = [int(v) for v in architecture.get("kernels", [9, 7, 5])]
        strides = [int(v) for v in architecture.get("strides", [2, 2, 2])]
        dilations = [int(v) for v in architecture.get("dilations", [1, 1, 1])]
        if not channels or not (len(channels) == len(kernels) == len(strides) == len(dilations)):
            raise ValueError("cnn_2d channels/kernels/strides/dilations must be non-empty and have equal length")
        if any(kernel < 1 for kernel in kernels) or any(stride < 1 for stride in strides) or any(dilation < 1 for dilation in dilations):
            raise ValueError("cnn_2d kernels/strides/dilations must be positive")

        pool_length = int(architecture.get("adaptive_pool_length", 128))
        pooling = str(architecture.get("pooling", "avg_max")).lower()
        if pool_length < 1:
            raise ValueError("cnn_2d adaptive_pool_length must be >= 1")
        if pooling not in {"avg", "max", "avg_max"}:
            raise ValueError("cnn_2d pooling must be avg, max, or avg_max")

        first_padding = dilations[0] * (kernels[0] - 1) // 2
        self.fusion = nn.Sequential(
            nn.Conv2d(
                1,
                channels[0],
                kernel_size=(2, kernels[0]),
                stride=(1, strides[0]),
                dilation=(1, dilations[0]),
                padding=(0, first_padding),
            ),
            nn.BatchNorm2d(channels[0]),
            nn.SiLU(),
        )

        temporal_layers: list[nn.Module] = []
        incoming = channels[0]
        for outgoing, kernel, stride, dilation in zip(
            channels[1:], kernels[1:], strides[1:], dilations[1:]
        ):
            temporal_padding = dilation * (kernel - 1) // 2
            temporal_layers.extend(
                [
                    nn.Conv1d(
                        incoming,
                        outgoing,
                        kernel_size=kernel,
                        stride=stride,
                        dilation=dilation,
                        padding=temporal_padding,
                    ),
                    nn.BatchNorm1d(outgoing),
                    nn.SiLU(),
                ]
            )
            incoming = outgoing
        self.features = nn.Sequential(*temporal_layers)

        self.avg_pool = (
            nn.AdaptiveAvgPool1d(pool_length)
            if pooling in {"avg", "avg_max"}
            else None
        )
        self.max_pool = (
            nn.AdaptiveMaxPool1d(pool_length)
            if pooling in {"max", "avg_max"}
            else None
        )

        incoming *= pool_length * (2 if pooling == "avg_max" else 1)
        head: list[nn.Module] = [nn.Flatten()]
        for width in [int(v) for v in architecture.get("dense_units", [32])]:
            head.extend([nn.Linear(incoming, width), nn.SiLU()])
            incoming = width
        head.append(nn.Linear(incoming, 1))
        self.head = nn.Sequential(*head)

    def forward(self, pair: torch.Tensor) -> torch.Tensor:
        if pair.ndim != 3 or pair.shape[1] != 2:
            raise ValueError(f"cnn_2d expects [event, detector=2, time], got {tuple(pair.shape)}")
        features_2d = self.fusion(pair[:, None, :, :])
        if features_2d.shape[2] != 1:
            raise RuntimeError(
                f"cnn_2d fusion must reduce detector height to 1, got {tuple(features_2d.shape)}"
            )
        features = self.features(features_2d.squeeze(2))
        pooled = []
        if self.avg_pool is not None:
            pooled.append(self.avg_pool(features))
        if self.max_pool is not None:
            pooled.append(self.max_pool(features))
        return self.head(torch.cat(pooled, dim=1) if len(pooled) > 1 else pooled[0]).squeeze(1)


@dataclass
class CNN2DArtifact:
    model: JointPairCNN2D
    device: str
    metadata: dict[str, Any]


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
    batch = int(params.get("batch_size", training.get("batch_size", 64)))
    max_epochs = int(training.get("epochs", 350))
    patience = int(training.get("patience", 30))
    min_delta = float(training.get("min_delta", 0.05))
    early_fraction = float(training.get("early_stopping_fraction", 0.20))
    split_seed = int(config.get("_early_stopping_seed", seed))
    fit_x, fit_target, early_x, early_target = _internal_early_stopping_split(
        train_x, train_target, early_fraction, split_seed
    )
    model = JointPairCNN2D(config.get("architecture", {})).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(params["learning_rate"]),
        weight_decay=float(params["weight_decay"]),
    )
    loss_fn = nn.MSELoss()
    loader = _loader(fit_x, fit_target, batch, shuffle=True, seed=training_seed)
    output_limit = config.get("_prediction_max_abs_ps")
    if verbose and logger is not None:
        logger.info(
            "cnn_2d training | lr=%.6g | weight_decay=%.6g | batch=%d | epochs=%d | patience=%d | min_delta=%.6g | early_stop_fraction=%.3f | fit=%d | early_stop=%d | device=%s",
            float(params["learning_rate"]),
            float(params["weight_decay"]),
            batch,
            max_epochs,
            patience,
            min_delta,
            early_fraction,
            fit_target.size,
            early_target.size,
            device,
        )

    best_score = float("inf")
    best_epoch = 0
    best_state = None
    stale = 0
    for epoch in range(1, max_epochs + 1):
        model.train()
        epoch_gradient_norms = []
        for pair, target in loader:
            pair = pair.to(device)
            target = target.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model(pair), target)
            loss.backward()
            if verbose and logger is not None:
                epoch_gradient_norms.append(_gradient_norm(model))
            clip = float(training.get("gradient_clip_norm", 10.0))
            if clip > 0:
                nn.utils.clip_grad_norm_(model.parameters(), clip)
            optimizer.step()

        prediction = _predict_tensor(model, early_x, device, batch)
        if output_limit is not None:
            prediction = np.clip(prediction, -float(output_limit), float(output_limit))
        score = _rmse(prediction - early_target)
        if verbose and logger is not None:
            train_prediction = _predict_tensor(model, fit_x, device, batch)
            if output_limit is not None:
                train_prediction = np.clip(train_prediction, -float(output_limit), float(output_limit))
            train_score = _rmse(train_prediction - np.asarray(fit_target, dtype=np.float64))
            logger.info(
                "cnn_2d epoch %d/%d | fit RMSE=%.4f ps | early-stop RMSE=%.4f ps | grad norm=%.6g | pred mean=%.4f ps | pred std=%.4f ps | pred min=%.4f ps | pred max=%.4f ps",
                epoch,
                max_epochs,
                train_score,
                score,
                float(np.mean(epoch_gradient_norms)) if epoch_gradient_norms else float("nan"),
                float(np.mean(prediction)),
                float(np.std(prediction)),
                float(np.min(prediction)),
                float(np.max(prediction)),
            )
        if score < best_score - min_delta:
            best_score = score
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            stale = 0
        else:
            stale += 1
            if stale >= patience:
                break

    if best_state is None:
        raise RuntimeError("cnn_2d early stopping did not produce a valid checkpoint")
    model.load_state_dict(best_state)

    return CNN2DArtifact(
        model=model,
        device=str(device),
        metadata={
            "best_epoch": int(best_epoch),
            "best_early_stopping_rmse_ps": float(best_score),
            "early_stopping_metric": "internal_train_holdout_rmse",
            "early_stopping_fraction": early_fraction,
            "early_stopping_events": int(early_target.size),
            "optimizer_training_events": int(fit_target.size),
            "training_events_available": int(len(train_target)),
            "training_uses_full_split": False,
            "refit_on_full_training_split": False,
            "external_validation_used_for_early_stopping": False,
            "early_stopping_split_seed": split_seed,
            "learning_rate": float(params["learning_rate"]),
            "weight_decay": float(params["weight_decay"]),
            "batch_size": batch,
            "output_max_abs_ps": None if output_limit is None else float(output_limit),
            "training_seed": training_seed,
            "deterministic_algorithms": True,
            "input_definition": "normalized detector pair stacked as one [2,time] input",
            "prediction_definition": "single joint CNN f_theta([s1;s2]) [ps] with immediate 2-D detector fusion followed by 1-D temporal convolutions",
            "detector_axis_policy": "first convolution spans both detector rows (2-D -> 1-D), all subsequent convolutions are temporal Conv1d layers",
            "detector_swap_antisymmetry_enforced": False,
            "parameter_count": int(sum(parameter.numel() for parameter in model.parameters())),
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
