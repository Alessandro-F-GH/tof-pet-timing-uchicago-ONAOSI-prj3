from __future__ import annotations

import copy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from .cnn import _device, _loader, _predict_tensor, _rmse, candidates
from .spec import ModelSpec


class JointPairCNN2D(nn.Module):
    """Joint CNN with delayed fusion of the stacked detector pair [2, time]."""

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

        default_fusion_layer = 1 if len(channels) > 1 else 0
        detector_fusion_layer = int(architecture.get("detector_fusion_layer", default_fusion_layer))
        if detector_fusion_layer < 0 or detector_fusion_layer >= len(channels):
            raise ValueError(
                f"cnn_2d detector_fusion_layer must lie in [0, {len(channels) - 1}]"
            )
        self.detector_fusion_layer = detector_fusion_layer

        pool_length = int(architecture.get("adaptive_pool_length", 128))
        pooling = str(architecture.get("pooling", "avg_max")).lower()
        if pool_length < 1:
            raise ValueError("cnn_2d adaptive_pool_length must be >= 1")
        if pooling not in {"avg", "max", "avg_max"}:
            raise ValueError("cnn_2d pooling must be avg, max, or avg_max")

        layers: list[nn.Module] = []
        detector_kernel_heights: list[int] = []
        incoming = 1
        for layer_index, (outgoing, kernel, stride, dilation) in enumerate(
            zip(channels, kernels, strides, dilations)
        ):
            # Before fusion, height-1 kernels preserve the two detector rows and
            # apply the same temporal filters to each row. The configured fusion
            # layer then spans both rows once, reducing detector height 2 -> 1.
            detector_height = 2 if layer_index == detector_fusion_layer else 1
            detector_kernel_heights.append(detector_height)
            temporal_padding = dilation * (kernel - 1) // 2
            layers.extend(
                [
                    nn.Conv2d(
                        incoming,
                        outgoing,
                        kernel_size=(detector_height, kernel),
                        stride=(1, stride),
                        dilation=(1, dilation),
                        padding=(0, temporal_padding),
                    ),
                    nn.BatchNorm2d(outgoing),
                    nn.SiLU(),
                ]
            )
            incoming = outgoing

        self.detector_kernel_heights = tuple(detector_kernel_heights)
        self.features = nn.Sequential(*layers)
        self.avg_pool = (
            nn.AdaptiveAvgPool2d((1, pool_length))
            if pooling in {"avg", "avg_max"}
            else None
        )
        self.max_pool = (
            nn.AdaptiveMaxPool2d((1, pool_length))
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
        features = self.features(pair[:, None, :, :])
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
    if validation_x is None or validation_target is None:
        raise ValueError("cnn_2d training requires a validation set for early stopping")

    torch.manual_seed(int(seed))
    np.random.seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))

    training = config.get("training", {})
    device = _device(config)
    model = JointPairCNN2D(config.get("architecture", {})).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(params["learning_rate"]),
        weight_decay=float(params["weight_decay"]),
    )
    loss_fn = nn.MSELoss()
    batch = int(params.get("batch_size", training.get("batch_size", 64)))
    max_epochs = int(training.get("epochs", 350))
    patience = int(training.get("patience", 30))
    min_delta = float(training.get("min_delta", 0.05))
    loader = _loader(train_x, train_target, batch, shuffle=True, seed=seed)
    output_limit = config.get("_prediction_max_abs_ps")

    best_score = float("inf")
    best_epoch = 0
    best_state = None
    stale = 0
    validation_target = np.asarray(validation_target, dtype=np.float64)

    for epoch in range(1, max_epochs + 1):
        model.train()
        for pair, target in loader:
            pair = pair.to(device)
            target = target.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model(pair), target)
            loss.backward()
            clip = float(training.get("gradient_clip_norm", 10.0))
            if clip > 0:
                nn.utils.clip_grad_norm_(model.parameters(), clip)
            optimizer.step()

        prediction = _predict_tensor(model, validation_x, device, batch)
        if output_limit is not None:
            prediction = np.clip(prediction, -float(output_limit), float(output_limit))
        score = _rmse(prediction - validation_target)
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
            "best_validation_rmse_ps": float(best_score),
            "early_stopping_metric": "validation_rmse",
            "batch_size": batch,
            "output_max_abs_ps": None if output_limit is None else float(output_limit),
            "input_definition": "normalized detector pair stacked as one [2,time] input",
            "prediction_definition": "single joint CNN f_theta([s1;s2]) [ps] with delayed detector fusion",
            "detector_fusion_layer": int(model.detector_fusion_layer),
            "detector_kernel_heights": list(model.detector_kernel_heights),
            "detector_axis_policy": "preserve height 2 before fusion, fuse once with a height-2 kernel, then continue at height 1",
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
