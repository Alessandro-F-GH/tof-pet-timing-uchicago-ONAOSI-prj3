"""Generic InceptionTime-style 1D backbone for black-box time-series distillation.

Based on the InceptionTime architectural family: multiscale convolution branches,
max-pooling branch, bottleneck 1x1 projections, BN/ReLU, skip connections every
three modules, and global average pooling. This implementation is an independent
PyTorch regression adaptation, not a reproduction of a particular trained model.
It uses ONLY the two input waveforms and their explicit observation masks.
No teacher features, transforms or model code are accessed.
"""
from __future__ import annotations

import torch
from torch import nn


class InceptionModule1D(nn.Module):
    def __init__(self, in_channels: int, filters: int, bottleneck: int,
                 kernel_size: int = 39):
        super().__init__()
        if in_channels < 1 or filters < 1 or bottleneck < 1 or kernel_size < 3:
            raise ValueError("Invalid Inception module parameters")
        # Odd kernel sizes permit exactly aligned same-length convolutions.
        kernels = [kernel_size, max(3, kernel_size // 2 | 1),
                   max(3, kernel_size // 4 | 1)]
        self.bottleneck = nn.Conv1d(in_channels, bottleneck, 1, bias=False)
        self.branches = nn.ModuleList(
            nn.Conv1d(bottleneck, filters, k, padding=k // 2, bias=False)
            for k in kernels
        )
        self.pool = nn.MaxPool1d(kernel_size=3, stride=1, padding=1)
        self.pool_projection = nn.Conv1d(in_channels, filters, 1, bias=False)
        self.norm = nn.BatchNorm1d(4 * filters)
        self.activation = nn.ReLU(inplace=False)

    def forward(self, x):
        projected = self.bottleneck(x)
        values = [conv(projected) for conv in self.branches]
        values.append(self.pool_projection(self.pool(x)))
        return self.activation(self.norm(torch.cat(values, dim=1)))


class InceptionTimeSurrogate(nn.Module):
    """InceptionTime regression model accepting NN-Olsen's [B,4*T] layout.

    In full-signal experiments every observation-mask entry is 1; the same
    channel layout is retained for later conditional-surrogate experiments.
    """

    def __init__(self, n_samples: int, filters: int = 16, bottleneck: int = 32,
                 depth: int = 6, kernel_size: int = 39):
        super().__init__()
        if n_samples < 8 or depth < 1:
            raise ValueError("Invalid InceptionTime configuration")
        self.n_samples = int(n_samples)
        self.filters = int(filters)
        self.bottleneck_size = int(bottleneck)
        self.depth = int(depth)
        self.kernel_size = int(kernel_size)
        modules = []
        residuals = nn.ModuleDict()
        current = 4
        block_start = 4
        for index in range(depth):
            modules.append(InceptionModule1D(current, filters, bottleneck,
                                              kernel_size))
            current = 4 * filters
            if (index + 1) % 3 == 0:
                residuals[str(index)] = nn.Sequential(
                    nn.Conv1d(block_start, current, kernel_size=1, bias=False),
                    nn.BatchNorm1d(current),
                )
                block_start = current
        self.modules_list = nn.ModuleList(modules)
        self.residuals = residuals
        self.output = nn.Linear(current, 1)

    def forward(self, augmented):
        if augmented.ndim != 2 or augmented.shape[1] != 4 * self.n_samples:
            raise ValueError("Expected [B,4*T] masked two-channel inputs")
        x = augmented.reshape(-1, 4, self.n_samples)
        shortcut = x
        for index, module in enumerate(self.modules_list):
            x = module(x)
            if str(index) in self.residuals:
                x = torch.relu(x + self.residuals[str(index)](shortcut))
                shortcut = x
        x = x.mean(dim=-1)  # Standard InceptionTime global average pooling.
        return self.output(x).squeeze(-1)
