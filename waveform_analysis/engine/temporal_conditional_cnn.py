"""Time-aware convolutional surrogate with NN-Olsen masked-regression objective.

The original model is NEVER queried. Inputs are the identical [B,4*T]
zero-filled waveform and explicit observed-mask representation generated for
NN-Olsen. All temporal processing is learned on the surrogate side only.
"""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


def _groups(channels: int) -> int:
    # GroupNorm is independent of minibatch coalition composition (W vs R).
    return next(g for g in (8, 4, 2, 1) if channels % g == 0)


class TemporalResidualBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, *, dilation: int = 1,
                 stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=7, stride=stride,
                               padding=3 * dilation, dilation=dilation)
        self.norm1 = nn.GroupNorm(_groups(out_channels), out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=5,
                               padding=2 * dilation, dilation=dilation)
        self.norm2 = nn.GroupNorm(_groups(out_channels), out_channels)
        self.shortcut = (nn.Identity()
                         if in_channels == out_channels and stride == 1
                         else nn.Conv1d(in_channels, out_channels, kernel_size=1,
                                        stride=stride))

    def forward(self, x):
        residual = self.shortcut(x)
        x = F.gelu(self.norm1(self.conv1(x)))
        x = self.norm2(self.conv2(x))
        return F.gelu(x + residual)


class TemporalConditionalCNN(nn.Module):
    """Multi-resolution 1D CNN surrogate with explicit time coordinate.

    Does not change the two-detector waveform layout or original prediction
    targets. The mask is already included in the 4T NN-Olsen input.
    """

    def __init__(self, n_samples: int, width: int = 64, depth: int = 3,
                 bins: int = 8, use_derivative: bool = False):
        super().__init__()
        if n_samples < 8 or width < 8 or depth < 1 or bins < 1:
            raise ValueError("Invalid temporal CNN architecture")
        self.n_samples = int(n_samples)
        self.width = int(width)
        self.depth = int(depth)
        self.bins = int(bins)
        self.use_derivative = bool(use_derivative)
        stem_width = max(16, width // 2)
        self.stem = nn.Sequential(
            nn.Conv1d(7 if self.use_derivative else 5, stem_width, kernel_size=9, stride=4, padding=4),
            nn.GroupNorm(_groups(stem_width), stem_width),
            nn.GELU(),
        )
        # Downsample gradually for long waveform sequences, while residual
        # and dilated convolutions encode both local shapes and wider context.
        stages = []
        channels = stem_width
        for stage in range(depth):
            output_channels = min(width * 2, width if stage == 0 else width * 2)
            stages.append(TemporalResidualBlock(channels, output_channels,
                                                stride=2, dilation=1))
            stages.append(TemporalResidualBlock(output_channels, output_channels,
                                                dilation=2 ** min(stage + 1, 4)))
            channels = output_channels
        self.stages = nn.Sequential(*stages)
        self.regressor = nn.Sequential(
            nn.Linear(channels * bins * 2, width * 4),
            nn.GELU(),
            nn.Linear(width * 4, width * 2),
            nn.GELU(),
            nn.Linear(width * 2, 1),
        )
        self.register_buffer(
            "time_coordinate", torch.linspace(-1., 1., n_samples).reshape(1, 1, n_samples),
            persistent=False,
        )

    def forward(self, augmented):
        expected_channels = 6 if self.use_derivative else 4
        if augmented.ndim != 2 or augmented.shape[1] != expected_channels * self.n_samples:
            raise ValueError(f"Expected [B,{expected_channels}*T] two-detector inputs")
        x = augmented.reshape(-1, expected_channels, self.n_samples)
        position = self.time_coordinate.expand(x.shape[0], -1, -1)
        x = torch.cat((x, position), dim=1)
        x = self.stages(self.stem(x))
        pooled = torch.cat((F.adaptive_avg_pool1d(x, self.bins),
                            F.adaptive_max_pool1d(x, self.bins)), dim=1)
        return self.regressor(pooled.flatten(start_dim=1)).squeeze(-1)
