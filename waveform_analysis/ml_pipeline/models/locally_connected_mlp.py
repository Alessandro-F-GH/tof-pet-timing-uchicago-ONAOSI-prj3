from __future__ import annotations

import itertools
import math

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
    """Conv1d-like scalar local layer with position-specific, unshared kernels."""

    def __init__(
        self,
        input_positions: int,
        kernel_size: int,
        stride: int,
    ):
        super().__init__()
        input_positions = int(input_positions)
        kernel_size = int(kernel_size)
        stride = int(stride)
        if input_positions < 1:
            raise ValueError("input_positions must be >= 1")
        if kernel_size < 1 or kernel_size > input_positions:
            raise ValueError(
                "kernel_size must satisfy 1 <= kernel_size <= input_positions"
            )
        if stride < 1:
            raise ValueError("stride must be >= 1")

        self.input_positions = input_positions
        self.kernel_size = kernel_size
        self.stride = stride
        self.output_positions = 1 + (input_positions - kernel_size) // stride

        self.weight = nn.Parameter(
            torch.empty(self.output_positions, kernel_size)
        )
        self.bias = nn.Parameter(torch.empty(self.output_positions))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        bound = 1.0 / math.sqrt(self.kernel_size)
        nn.init.uniform_(self.weight, -bound, bound)
        nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        if values.ndim != 2 or values.shape[1] != self.input_positions:
            raise ValueError(
                "LocallyConnected1D expects [event, position] = "
                f"[N, {self.input_positions}], got {tuple(values.shape)}"
            )
        windows = values.unfold(
            dimension=-1,
            size=self.kernel_size,
            step=self.stride,
        )
        return (
            windows * self.weight.unsqueeze(0)
        ).sum(dim=-1) + self.bias.unsqueeze(0)


class SharedLocallyConnectedScorer(nn.Module):
    """Hierarchical local scorer without temporal weight sharing.

    Two scalar locally connected layers reproduce the kernel/stride hierarchy
    of a 1D CNN while learning an independent kernel at every temporal
    position. Detector scoring remains shared, so the paired correction is
    exactly antisymmetric under detector exchange.
    """

    def __init__(
        self,
        input_samples: int,
        architecture,
        activation: str,
        *,
        layer1_kernel_samples: int,
        layer1_stride_samples: int,
        layer2_kernel_positions: int,
        layer2_stride_positions: int,
        max_correction_ps: float,
    ):
        super().__init__()
        self.max_correction_ps = float(max_correction_ps)
        if (
            not math.isfinite(self.max_correction_ps)
            or self.max_correction_ps <= 0.0
        ):
            raise ValueError("max_correction_ps must be positive and finite")

        self.local1 = LocallyConnected1D(
            input_positions=int(input_samples),
            kernel_size=int(layer1_kernel_samples),
            stride=int(layer1_stride_samples),
        )
        self.local1_activation = _activation(activation)

        self.local2 = LocallyConnected1D(
            input_positions=self.local1.output_positions,
            kernel_size=int(layer2_kernel_positions),
            stride=int(layer2_stride_positions),
        )
        self.local2_activation = _activation(activation)

        self.scorer = DenseStack(
            self.local2.output_positions,
            architecture,
            activation,
        )

    def detector_score(self, values: torch.Tensor) -> torch.Tensor:
        local = self.local1_activation(self.local1(values))
        local = self.local2_activation(self.local2(local))
        return self.scorer(local)

    def forward(self, pair: torch.Tensor) -> torch.Tensor:
        if pair.ndim != 3 or pair.shape[1] != 2:
            raise ValueError(
                "locally_connected_mlp expects [event, detector=2, time], "
                f"got {tuple(pair.shape)}"
            )
        raw = self.detector_score(pair[:, 0, :]) - self.detector_score(
            pair[:, 1, :]
        )
        limit = self.max_correction_ps
        return limit * torch.tanh(raw / limit)


def candidates(config):
    parameters = config.get("parameters", {})
    training = config.get("training", {})
    architectures = parameters.get("architecture", [[64]])
    activations = parameters.get("activation", ["silu"])
    learning_rates = parameters.get("learning_rate", [1e-3])
    batch_sizes = parameters.get(
        "batch_size", [training.get("batch_size", 128)]
    )
    layer1_kernels = parameters.get("layer1_kernel_samples", [16])
    layer1_strides = parameters.get("layer1_stride_samples", [4])
    layer2_kernels = parameters.get("layer2_kernel_positions", [3])
    layer2_strides = parameters.get("layer2_stride_positions", [1])
    max_corrections = parameters.get("max_correction_ps", [250.0])

    rows = []
    for (
        architecture,
        activation,
        learning_rate,
        batch_size,
        kernel1,
        stride1,
        kernel2,
        stride2,
        max_correction,
    ) in itertools.product(
        architectures,
        activations,
        learning_rates,
        batch_sizes,
        layer1_kernels,
        layer1_strides,
        layer2_kernels,
        layer2_strides,
        max_corrections,
    ):
        kernel1 = int(kernel1)
        stride1 = int(stride1)
        kernel2 = int(kernel2)
        stride2 = int(stride2)
        max_correction = float(max_correction)

        if min(kernel1, stride1, kernel2, stride2) < 1:
            raise ValueError("local kernels and strides must be >= 1")
        if not math.isfinite(max_correction) or max_correction <= 0.0:
            raise ValueError("max_correction_ps must be positive and finite")

        rows.append(
            {
                "architecture": _validate_architecture(architecture),
                "activation": str(activation).strip().lower(),
                "learning_rate": float(learning_rate),
                "batch_size": int(batch_size),
                "layer1_kernel_samples": kernel1,
                "layer1_stride_samples": stride1,
                "layer2_kernel_positions": kernel2,
                "layer2_stride_positions": stride2,
                "max_correction_ps": max_correction,
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
    input_samples = int(train_x.shape[-1])
    kernel1 = int(params["layer1_kernel_samples"])
    stride1 = int(params["layer1_stride_samples"])
    kernel2 = int(params["layer2_kernel_positions"])
    stride2 = int(params["layer2_stride_positions"])
    max_correction = float(params["max_correction_ps"])

    if kernel1 > input_samples:
        raise ValueError(
            "layer1_kernel_samples exceeds the available waveform samples"
        )
    first_positions = 1 + (input_samples - kernel1) // stride1
    if kernel2 > first_positions:
        raise ValueError(
            "layer2_kernel_positions exceeds positions produced by layer 1"
        )
    second_positions = 1 + (first_positions - kernel2) // stride2

    return fit_mlp(
        model_name="locally_connected_mlp",
        model_factory=SharedLocallyConnectedScorer,
        params=params,
        train_x=train_x,
        train_target=train_target,
        seed=seed,
        config=config,
        model_factory_kwargs={
            "layer1_kernel_samples": kernel1,
            "layer1_stride_samples": stride1,
            "layer2_kernel_positions": kernel2,
            "layer2_stride_positions": stride2,
            "max_correction_ps": max_correction,
        },
        metadata_extra={
            "input_definition": (
                "two normalized detector waveforms scored independently by one "
                "shared hierarchical locally connected network"
            ),
            "prediction_definition": (
                "bounded correction C*tanh((g_theta(s1)-g_theta(s2))/C) [ps]"
            ),
            "detector_swap_antisymmetry_enforced": True,
            "max_correction_ps": max_correction,
            "output_bounding": "smooth_tanh",
            "local_connectivity": {
                "layers": [
                    {
                        "channels": 1,
                        "kernel_samples": kernel1,
                        "stride_samples": stride1,
                        "output_positions": first_positions,
                    },
                    {
                        "channels": 1,
                        "kernel_positions": kernel2,
                        "stride_positions": stride2,
                        "output_positions": second_positions,
                    },
                ],
                "flattened_local_features": second_positions,
                "weight_sharing": False,
                "edge_policy": "valid_no_padding",
                "hierarchical_temporal_ordering": True,
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
