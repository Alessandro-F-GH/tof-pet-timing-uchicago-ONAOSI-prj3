from __future__ import annotations

import itertools

import torch
from torch import nn

from ._mlp_common import (
    DenseStack,
    MLPArtifact,
    _activation,
    _validate_architecture,
    explain,
    fit_mlp,
    predict,
    save,
)
from .spec import ModelSpec


class LocallyConnected1D(nn.Module):
    """One unshared scalar node per overlapping temporal receptive field."""

    def __init__(
        self,
        input_samples: int,
        receptive_field_samples: int,
        overlap_samples: int,
    ):
        super().__init__()
        input_samples = int(input_samples)
        width = int(receptive_field_samples)
        overlap = int(overlap_samples)
        if width < 1:
            raise ValueError("receptive_field_samples must be >= 1")
        if width > input_samples:
            raise ValueError(
                "receptive_field_samples cannot exceed the waveform length"
            )
        if overlap < 0 or overlap >= width:
            raise ValueError(
                "overlap_samples must satisfy 0 <= overlap_samples < receptive_field_samples"
            )
        stride = width - overlap
        fields = 1 + (input_samples - width) // stride
        if fields < 1:
            raise ValueError("Locally connected layer has no valid receptive fields")

        self.input_samples = input_samples
        self.receptive_field_samples = width
        self.overlap_samples = overlap
        self.stride_samples = stride
        self.n_fields = fields

        self.weight = nn.Parameter(torch.empty(fields, width))
        self.bias = nn.Parameter(torch.empty(fields))
        nn.init.kaiming_uniform_(self.weight, a=5**0.5)
        bound = 1.0 / width**0.5
        nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        if values.ndim != 2 or values.shape[1] != self.input_samples:
            raise ValueError(
                "LocallyConnected1D expects [event, time] with "
                f"{self.input_samples} samples, got {tuple(values.shape)}"
            )
        windows = values.unfold(
            dimension=1,
            size=self.receptive_field_samples,
            step=self.stride_samples,
        )
        return (windows * self.weight.unsqueeze(0)).sum(dim=-1) + self.bias


class SharedLocallyConnectedScorer(nn.Module):
    """Shared detector scorer with local unshared first layer and exact antisymmetry."""

    def __init__(
        self,
        input_samples: int,
        architecture,
        activation: str,
        *,
        receptive_field_samples: int,
        overlap_samples: int,
    ):
        super().__init__()
        self.local = LocallyConnected1D(
            input_samples,
            receptive_field_samples,
            overlap_samples,
        )
        self.local_activation = _activation(activation)
        self.scorer = DenseStack(
            self.local.n_fields,
            architecture,
            activation,
        )

    def detector_score(self, values: torch.Tensor) -> torch.Tensor:
        local = self.local_activation(self.local(values))
        return self.scorer(local)

    def forward(self, pair: torch.Tensor) -> torch.Tensor:
        if pair.ndim != 3 or pair.shape[1] != 2:
            raise ValueError(
                "locally_connected_mlp expects [event, detector=2, time], "
                f"got {tuple(pair.shape)}"
            )
        return self.detector_score(pair[:, 0, :]) - self.detector_score(
            pair[:, 1, :]
        )


def candidates(config):
    parameters = config.get("parameters", {})
    training = config.get("training", {})
    architectures = parameters.get("architecture", [[64]])
    activations = parameters.get("activation", ["silu"])
    learning_rates = parameters.get("learning_rate", [1e-3])
    batch_sizes = parameters.get(
        "batch_size", [training.get("batch_size", 128)]
    )
    widths = parameters.get("receptive_field_samples", [16])
    overlaps = parameters.get("overlap_samples", [8])

    rows = []
    for architecture, activation, learning_rate, batch_size, width, overlap in itertools.product(
        architectures,
        activations,
        learning_rates,
        batch_sizes,
        widths,
        overlaps,
    ):
        width = int(width)
        overlap = int(overlap)
        if width < 1:
            raise ValueError("receptive_field_samples must be >= 1")
        if overlap < 0 or overlap >= width:
            raise ValueError(
                "overlap_samples must satisfy 0 <= overlap_samples < receptive_field_samples"
            )
        rows.append(
            {
                "architecture": _validate_architecture(architecture),
                "activation": str(activation).strip().lower(),
                "learning_rate": float(learning_rate),
                "batch_size": int(batch_size),
                "receptive_field_samples": width,
                "overlap_samples": overlap,
            }
        )
    return rows


def fit(
    params,
    train_x,
    train_target,
    *,
    seed,
    config,
):
    width = int(params["receptive_field_samples"])
    overlap = int(params["overlap_samples"])
    stride = width - overlap
    fields = 1 + (int(train_x.shape[-1]) - width) // stride
    return fit_mlp(
        model_name="locally_connected_mlp",
        model_factory=SharedLocallyConnectedScorer,
        params=params,
        train_x=train_x,
        train_target=train_target,
        seed=seed,
        config=config,
        model_factory_kwargs={
            "receptive_field_samples": width,
            "overlap_samples": overlap,
        },
        metadata_extra={
            "input_definition": (
                "two normalized detector waveforms scored independently by one "
                "shared locally connected MLP"
            ),
            "prediction_definition": (
                "shared locally connected correction g_theta(s1)-g_theta(s2) [ps]"
            ),
            "detector_swap_antisymmetry_enforced": True,
            "local_connectivity": {
                "receptive_field_samples": width,
                "overlap_samples": overlap,
                "stride_samples": stride,
                "local_nodes": fields,
                "nodes_per_receptive_field": 1,
                "weight_sharing": False,
                "padding": "valid",
            },
        },
    )


MODEL_SPEC = ModelSpec(
    name="locally_connected_mlp",
    candidates=candidates,
    fit=fit,
    predict=predict,
    save=save,
    explain=explain,
    preserve_temporal_grid=True,
)
