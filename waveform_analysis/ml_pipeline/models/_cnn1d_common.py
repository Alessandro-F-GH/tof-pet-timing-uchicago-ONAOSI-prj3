from __future__ import annotations

import itertools

import torch
from torch import nn

from ._mlp_common import DenseStack, _activation, _validate_architecture, fit_mlp, predict, save


def candidates(config):
    p = config.get("parameters", {})
    training = config.get("training", {})
    architectures = p.get("architecture", [[64]])
    activations = p.get("activation", ["silu"])
    learning_rates = p.get("learning_rate", [1e-3])
    batch_sizes = p.get("batch_size", [training.get("batch_size", 128)])
    conv_channels = p.get("conv_channels", [[16, 32]])
    kernels = p.get("kernel_samples", [[5, 3]])
    rows = []
    for architecture, activation, lr, batch, channels, kernel in itertools.product(
        architectures, activations, learning_rates, batch_sizes, conv_channels, kernels
    ):
        channels = [int(v) for v in channels]
        kernel = [int(v) for v in kernel]
        if not channels or len(channels) != len(kernel):
            raise ValueError("conv_channels and kernel_samples must be non-empty lists of equal length")
        if any(v < 1 for v in channels + kernel):
            raise ValueError("CNN channels and kernels must be positive")
        rows.append({
            "architecture": _validate_architecture(architecture),
            "activation": str(activation).strip().lower(),
            "learning_rate": float(lr),
            "batch_size": int(batch),
            "conv_channels": channels,
            "kernel_samples": kernel,
        })
    return rows


class _TemporalBackbone(nn.Module):
    def __init__(self, input_channels: int, input_samples: int, conv_channels, kernel_samples, activation: str):
        super().__init__()
        layers = []
        incoming = int(input_channels)
        length = int(input_samples)
        for channels, kernel in zip(conv_channels, kernel_samples):
            channels = int(channels); kernel = int(kernel)
            if kernel > length:
                raise ValueError("CNN kernel exceeds available temporal samples")
            layers.extend([nn.Conv1d(incoming, channels, kernel_size=kernel), _activation(activation)])
            incoming = channels
            length = length - kernel + 1
        self.network = nn.Sequential(*layers)
        self.output_dim = incoming * length

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.network(values).flatten(start_dim=1)


class SharedCNN1D(nn.Module):
    """One shared temporal CNN scorer per detector, combined antisymmetrically."""
    def __init__(self, input_samples: int, architecture, activation: str, *, conv_channels, kernel_samples):
        super().__init__()
        self.backbone = _TemporalBackbone(1, input_samples, conv_channels, kernel_samples, activation)
        self.head = DenseStack(self.backbone.output_dim, architecture, activation)

    def detector_score(self, waveform: torch.Tensor) -> torch.Tensor:
        return self.head(self.backbone(waveform[:, None, :]))

    def forward(self, pair: torch.Tensor) -> torch.Tensor:
        if pair.ndim != 3 or pair.shape[1] != 2:
            raise ValueError(f"shared_cnn1d expects [event, detector=2, time], got {tuple(pair.shape)}")
        return self.detector_score(pair[:, 0, :]) - self.detector_score(pair[:, 1, :])


class DirectCNN1D(nn.Module):
    """Direct temporal CNN with the two detectors represented as input channels."""
    def __init__(self, input_samples: int, architecture, activation: str, *, conv_channels, kernel_samples):
        super().__init__()
        self.backbone = _TemporalBackbone(2, input_samples, conv_channels, kernel_samples, activation)
        self.head = DenseStack(self.backbone.output_dim, architecture, activation)

    def forward(self, pair: torch.Tensor) -> torch.Tensor:
        if pair.ndim != 3 or pair.shape[1] != 2:
            raise ValueError(f"direct_cnn1d expects [event, detector=2, time], got {tuple(pair.shape)}")
        return self.head(self.backbone(pair))


def fit_cnn(*, model_name, model_factory, params, train_x, train_target, seed, config, metadata_extra):
    return fit_mlp(
        model_name=model_name,
        model_factory=model_factory,
        params=params,
        train_x=train_x,
        train_target=train_target,
        seed=seed,
        config=config,
        model_factory_kwargs={
            "conv_channels": list(params["conv_channels"]),
            "kernel_samples": list(params["kernel_samples"]),
        },
        metadata_extra={
            **metadata_extra,
            "conv_channels": list(params["conv_channels"]),
            "kernel_samples": list(params["kernel_samples"]),
            "temporal_weight_sharing": True,
            "edge_policy": "valid_no_padding",
        },
    )


__all__ = ["SharedCNN1D", "DirectCNN1D", "candidates", "fit_cnn", "predict", "save"]
