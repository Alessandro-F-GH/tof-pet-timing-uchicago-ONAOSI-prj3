"""NN-Olsen-style conditional expectation surrogate for paired time-series.

Based on Olsen et al. (2024): MLP with batch normalization, ELU residual
blocks, explicitly concatenated missingness mask, zero-filled missing values,
and paired complementary mask training. Here the mask distribution is restricted
to exactly the W/R temporal coalitions used by paired TSHAP.

This module NEVER imports or invokes the original predictive model.
"""
from __future__ import annotations

import numpy as np
import torch
from torch import nn


class ResidualBlock(nn.Module):
    def __init__(self, width: int):
        super().__init__()
        self.linear = nn.Linear(width, width)
        self.norm = nn.BatchNorm1d(width)
        self.activation = nn.ELU()

    def forward(self, x):
        return x + self.activation(self.norm(self.linear(x)))


class NNOlsenSurrogate(nn.Module):
    """Fully connected 3-block ELU/BN residual surrogate (default paper settings)."""

    def __init__(self, n_samples: int, width: int = 64, depth: int = 3):
        super().__init__()
        if n_samples < 1 or width < 1 or depth < 1:
            raise ValueError("Invalid surrogate architecture")
        self.n_samples = int(n_samples)
        self.input = nn.Linear(4 * n_samples, width)
        self.input_activation = nn.ELU()
        self.blocks = nn.Sequential(*(ResidualBlock(width) for _ in range(depth)))
        self.output = nn.Linear(width, 1)

    def forward(self, augmented):
        if augmented.ndim != 2 or augmented.shape[1] != 4 * self.n_samples:
            raise ValueError("Expected concatenated [masked two-channel waveform, mask] input")
        x = self.input_activation(self.input(augmented))
        return self.output(self.blocks(x)).squeeze(-1)


def coalitions(starts, width, retained, device="cpu"):
    """Return [n_windows, 2, T] temporal masks: W and complementary R."""
    retained = np.asarray(retained, dtype=bool)
    starts = np.asarray(starts, dtype=np.int64)
    t = retained.size
    mask = np.zeros((len(starts), 2, t), dtype=np.float32)
    for i, start in enumerate(starts):
        if start < 0 or start + width > t or not retained[start:start+width].all():
            raise ValueError("Invalid window or sample mask")
        mask[i, 0, start:start+width] = 1.
        mask[i, 1] = retained.astype(np.float32) - mask[i, 0]
    return torch.as_tensor(mask, device=device)


def augment(waves, masks):
    """Zero-fill unobserved raw prepared samples and append binary mask.

    waves [B,2,T], masks [B,T] -> [B,4T]. No waveform transforms.
    """
    if waves.ndim != 3 or waves.shape[1] != 2 or masks.shape != (waves.shape[0], waves.shape[2]):
        raise ValueError("Incompatible prepared waveform and temporal mask shapes")
    both = masks[:, None, :].expand(-1, 2, -1)
    return torch.cat((waves * both, both), dim=1).reshape(waves.shape[0], -1)


def paired_batch(waves, targets, window_index, masks):
    """Return W/R pairs for each event with identical saved-prediction targets."""
    if waves.ndim != 3 or waves.shape[1] != 2 or targets.shape != (len(waves),):
        raise ValueError("Invalid paired training batch")
    if window_index.shape != (len(waves),):
        raise ValueError("Each event must have one sampled window")
    chosen = masks[window_index]  # [B, 2 coalitions, T]
    pair_wave = waves[:, None, :, :].expand(-1, 2, -1, -1)
    pair_in = augment(pair_wave.reshape(-1, 2, waves.shape[-1]),
                      chosen.reshape(-1, waves.shape[-1]))
    return pair_in, targets.repeat_interleave(2)


@torch.no_grad()
def predict_coalitions(model, waveforms, starts, width, retained, *,
                       batch_size=1024, device="cpu"):
    """Compute [events, windows, W/R] conditional estimates (surrogate only)."""
    model.eval()
    masks = coalitions(starts, width, retained, device=device)
    result = np.empty((len(waveforms), len(starts), 2), dtype=np.float64)
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    for left in range(0, len(waveforms), batch_size):
        right = min(left + batch_size, len(waveforms))
        wave = torch.as_tensor(np.asarray(waveforms[left:right], dtype=np.float32), device=device)
        for j in range(len(starts)):
            ms = masks[j].unsqueeze(0).expand(len(wave), -1, -1)
            pair = wave[:, None, :, :].expand(-1, 2, -1, -1).reshape(-1, 2, wave.shape[-1])
            features = augment(pair, ms.reshape(-1, wave.shape[-1]))
            result[left:right, j, :] = (model(features).reshape(-1, 2)
                                        .detach().cpu().numpy())
    return result
