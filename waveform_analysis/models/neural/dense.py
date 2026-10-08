"""Dense scorer architecture, fitted artifact and candidate inspection."""

from __future__ import annotations

import itertools
from dataclasses import dataclass
from typing import Any, Callable
import torch
from torch import nn

_ACTIVATIONS: dict[str, Callable[[], nn.Module]] = {
    "relu": nn.ReLU,
    "silu": nn.SiLU,
    "gelu": nn.GELU,
    "tanh": nn.Tanh,
    "elu": nn.ELU,
}


def _activation(name: str) -> nn.Module:
    key = str(name).strip().lower()
    try:
        return _ACTIVATIONS[key]()
    except KeyError as exc:
        raise ValueError(
            f"Unsupported MLP activation {name!r}; available: {sorted(_ACTIVATIONS)}"
        ) from exc


def _validate_architecture(architecture: list[int] | tuple[int, ...]) -> list[int]:
    if not isinstance(architecture, (list, tuple)) or not architecture:
        raise ValueError(
            "MLP architecture must be a non-empty list of hidden-layer widths"
        )
    widths = [int(value) for value in architecture]
    if any(width <= 0 for width in widths):
        raise ValueError("MLP architecture widths must all be positive")
    return widths


class DenseStack(nn.Module):
    def __init__(
        self, input_dim: int, architecture: list[int] | tuple[int, ...], activation: str
    ) -> None:
        super().__init__()
        widths = _validate_architecture(architecture)
        layers: list[nn.Module] = []
        incoming = int(input_dim)
        for width in widths:
            layers.extend([nn.Linear(incoming, width), _activation(activation)])
            incoming = width
        layers.append(nn.Linear(incoming, 1))
        self.network = nn.Sequential(*layers)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.network(values).squeeze(1)


@dataclass
class MLPArtifact:
    model: nn.Module
    device: str
    metadata: dict[str, Any]


def candidates(config: dict[str, Any]) -> list[dict[str, Any]]:
    parameters = config.get("parameters", {})
    training = config.get("training", {})
    architectures = parameters.get("architecture", [[128, 64]])
    activations = parameters.get("activation", ["silu"])
    learning_rates = parameters.get("learning_rate", [1e-3])
    batch_sizes = parameters.get("batch_size", [training.get("batch_size", 128)])
    return [
        {
            "architecture": _validate_architecture(architecture),
            "activation": str(activation).strip().lower(),
            "learning_rate": float(learning_rate),
            "batch_size": int(batch_size),
        }
        for architecture, activation, learning_rate, batch_size in itertools.product(
            architectures,
            activations,
            learning_rates,
            batch_sizes,
        )
    ]


# Keep old checkpoint/feature-artifact class and function import identities.
for _definition in (
    DenseStack,
    MLPArtifact,
    candidates,
    _activation,
    _validate_architecture,
):
    _definition.__module__ = "waveform_analysis.ml_pipeline.models._mlp_common"
