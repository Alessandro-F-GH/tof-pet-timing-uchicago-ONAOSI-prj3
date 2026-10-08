from __future__ import annotations
from waveform_analysis.models.torch_runtime import (
    save_torch_artifact,
    input_gradient_importance,
)
from waveform_analysis.models.base import BaseTorchModel

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch
from torch import nn

from waveform_analysis.models.torch_runtime import (
    configure_reproducibility as _configure_reproducibility,  # noqa: F401 - legacy helper re-export
    device_from_config as _device,  # noqa: F401 - legacy helper re-export
    gradient_norm as _gradient_norm,  # noqa: F401 - legacy helper re-export
    make_loader as _loader,  # noqa: F401 - legacy helper re-export
    predict_tensor as _predict_tensor,
    rmse as _rmse,  # noqa: F401 - legacy helper re-export
)
from waveform_analysis.models.spec import ModelSpec


class OnishiPairedCNN(BaseTorchModel):
    """Paired waveform CNN following the Onishi-style TOF correction architecture."""

    def __init__(self, architecture: dict[str, Any]):
        super().__init__()
        channels = [int(v) for v in architecture.get("channels", [32, 32, 64])]
        kernels = [int(v) for v in architecture.get("kernels", [5, 3, 3])]
        dense_units = int(architecture.get("dense_units", 256))

        if not channels:
            raise ValueError(
                "onishi_cnn architecture requires at least one convolutional layer"
            )
        if len(channels) != len(kernels):
            raise ValueError(
                "onishi_cnn channels and kernels must have the same length"
            )
        if any(value < 1 for value in channels):
            raise ValueError("onishi_cnn convolution channels must be positive")
        if any(value < 1 for value in kernels):
            raise ValueError("onishi_cnn convolution kernels must be positive")
        if dense_units < 1:
            raise ValueError("onishi_cnn dense_units must be positive")

        layers = []
        in_channels = 1
        for index, (out_channels, kernel) in enumerate(zip(channels, kernels)):
            kernel_height = 2 if index == 0 else 1
            layers.extend(
                [
                    nn.Conv2d(
                        in_channels,
                        out_channels,
                        kernel_size=(kernel_height, kernel),
                    ),
                    nn.ReLU(),
                ]
            )
            in_channels = out_channels
        self.features = nn.Sequential(*layers)
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.LazyLinear(dense_units),
            nn.ReLU(),
            nn.Linear(dense_units, 1),
        )

    def forward(self, pair: torch.Tensor) -> torch.Tensor:
        if pair.ndim != 3 or pair.shape[1] != 2:
            raise ValueError(
                f"onishi_cnn expects [event, detector=2, time], got {tuple(pair.shape)}"
            )
        features = self.features(pair[:, None, :, :])
        if features.shape[2] != 1:
            raise RuntimeError(
                f"onishi_cnn first convolution must fuse detector height 2 -> 1, got {tuple(features.shape)}"
            )
        return self.head(features).squeeze(1)


@dataclass
class OnishiCNNArtifact:
    model: OnishiPairedCNN
    device: str
    metadata: dict[str, Any]


def candidates(config):
    """Legacy candidate factory; current studies use the generic search-space engine."""
    parameters = config.get("parameters", {})
    training = config.get("training", {})

    def values(name, default):
        raw = parameters.get(name)
        if raw is None:
            return [default]
        if isinstance(raw, dict):
            kind = str(raw.get("type", "")).strip().lower()
            if kind == "fixed":
                return [raw.get("value")]
            if kind == "categorical":
                return list(raw.get("choices", []))
            return [default]
        if isinstance(raw, (list, tuple)):
            return list(raw)
        return [raw]

    learning_rates = values("learning_rate", training.get("learning_rate", 1e-3))
    batch_sizes = values("batch_size", training.get("batch_size", 128))
    return [
        {
            "learning_rate": float(learning_rate),
            "batch_size": int(batch_size),
        }
        for learning_rate in learning_rates
        for batch_size in batch_sizes
    ]


def _mse_loss(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return torch.mean((prediction - target) ** 2)


def fit(
    params: dict[str, Any],
    train_x: np.ndarray,
    train_target: np.ndarray,
    *,
    seed: int,
    config: dict[str, Any],
) -> OnishiCNNArtifact:
    """Delegate to the unchanged Onishi full-split/MSE training algorithm."""
    from waveform_analysis.engine.onishi_training import fit_onishi

    return fit_onishi(params, train_x, train_target, seed=seed, config=config)


def predict(artifact: OnishiCNNArtifact, normalized_pair: np.ndarray) -> np.ndarray:
    return _predict_tensor(
        artifact.model,
        normalized_pair,
        torch.device(artifact.device),
        512,
    )


def save(artifact: OnishiCNNArtifact, path: Path) -> None:
    """Persist the unchanged neural checkpoint schema."""
    save_torch_artifact(artifact, path)


def explain(artifact: OnishiCNNArtifact, normalized_pair: np.ndarray) -> np.ndarray:
    """Return mean absolute input gradients [sample]."""
    return input_gradient_importance(
        artifact,
        normalized_pair,
        missing_gradient_message="onishi_cnn XAI gradient is unavailable",
    )


MODEL_SPEC = ModelSpec(
    name="onishi_cnn",
    candidates=candidates,
    fit=fit,
    predict=predict,
    save=save,
    explain=explain,
)


# Preserve serialized identities and legacy import behavior.
from waveform_analysis.core.compat import (
    preserve_legacy_identity as _preserve_legacy_identity,
)

_preserve_legacy_identity(__name__, "waveform_analysis.ml_pipeline.models.onishi_cnn")
