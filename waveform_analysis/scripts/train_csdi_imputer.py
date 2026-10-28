#!/usr/bin/env python3
"""Train a CSDI-style conditional diffusion imputer for paired wide waveforms.

This is an efficient adaptation of CSDI for long (thousands-of-samples) signals:
the conditional DDPM noise-prediction objective and reverse sampler follow CSDI,
while the quadratic full-sequence Transformer denoiser is replaced by a
dilated depthwise-separable temporal convolutional denoiser.

Input data are paired detector waveforms [events, 2, samples]. A contiguous
fixed-width interval is hidden simultaneously in both detectors; the remaining
samples from both detectors condition the joint probabilistic reconstruction.
Training uses development events only and random-uniform mask positions per
event/epoch. At evaluation, CSDI samples are compared with the repository's
compact detector-specific MLP checkpoints (if compatible) and interpolation.

No TSHAP integration is performed by this script.

From repository root:
    python -m waveform_analysis.scripts.train_csdi_imputer \
        --results waveform_analysis/results/UC/test_ridge_48V_R1_R2 \
        --channel energy --max-events 5000

A compatible compact MLP baseline is searched for under:
    waveform_analysis/imputation_experiments/<results-name>/<channel>/detector_specific_mlp/
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import random
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if __package__ in (None, "") and str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

from waveform_analysis.scripts.train_wide_unet_imputer_detector_specific import (
    MaskedMLP,
    load_wide_development,
    root_relative,
)


def make_starts(length: int, mask_len: int, stride: int) -> list[int]:
    if not (0 < mask_len < length):
        raise ValueError("Mask length must be between 1 and waveform length - 1")
    if stride < 1:
        raise ValueError("Mask stride must be positive")
    starts = list(range(0, length - mask_len + 1, stride))
    final_start = length - mask_len
    if starts[-1] != final_start:
        starts.append(final_start)
    return starts


class EventMaskDataset(Dataset):
    """Return complete paired waveforms and one contiguous mask start per item."""

    def __init__(
        self,
        waves,
        events,
        starts: list[int],
        *,
        random_masks: bool,
        masks_per_event: int = 1,
    ):
        self.waves = waves
        self.events = np.asarray(events, dtype=np.int64)
        self.starts = list(starts)
        self.random_masks = bool(random_masks)
        self.masks_per_event = int(masks_per_event)
        if self.masks_per_event < 1:
            raise ValueError("masks_per_event must be positive")
        if len(self.events) < 1 or len(self.starts) < 1:
            raise ValueError("Dataset needs at least one event and one mask position")

    def __len__(self):
        return len(self.events) * self.masks_per_event

    def __getitem__(self, index):
        event_slot = int(index // self.masks_per_event)
        event_index = int(self.events[event_slot])
        if self.random_masks:
            start = self.starts[int(np.random.randint(len(self.starts)))]
        else:
            # Deterministic, approximately uniform position coverage over validation events.
            pos = min(len(self.starts) - 1,
                      (event_slot * len(self.starts)) // len(self.events))
            start = self.starts[pos]
        waveform = np.array(self.waves[event_index], dtype=np.float32, copy=True, order="C")
        if waveform.ndim != 2 or waveform.shape[0] != 2:
            raise ValueError(f"Expected a paired waveform [2,T], got {waveform.shape}")
        if not np.isfinite(waveform).all():
            raise ValueError(f"Non-finite samples in event {event_index}")
        return torch.from_numpy(waveform), torch.tensor(start, dtype=torch.long)


def target_mask_from_starts(starts, length: int, mask_len: int, device):
    """Return 1 inside each target interval and 0 elsewhere, shape [B,2,T]."""
    starts = torch.as_tensor(starts, dtype=torch.long, device=device).reshape(-1)
    positions = torch.arange(length, device=device).unsqueeze(0)
    mask = (positions >= starts.unsqueeze(1)) & (
        positions < (starts + mask_len).unsqueeze(1)
    )
    return mask.unsqueeze(1).expand(-1, 2, -1).to(dtype=torch.float32)


def sinusoidal_position_features(length: int, dim: int) -> torch.Tensor:
    """Fixed sinusoidal sample-position features, shape [1,dim,T]."""
    half = dim // 2
    position = torch.arange(length, dtype=torch.float32).unsqueeze(1)
    denominator = max(half - 1, 1)
    frequencies = torch.exp(
        -math.log(10000.0) * torch.arange(half, dtype=torch.float32) / denominator
    )
    angles = position * frequencies.unsqueeze(0)
    features = torch.cat((torch.sin(angles), torch.cos(angles)), dim=1)
    if features.shape[1] < dim:
        features = torch.nn.functional.pad(features, (0, dim - features.shape[1]))
    return features[:, :dim].transpose(0, 1).unsqueeze(0).contiguous()


def sinusoidal_step_embedding(steps: torch.Tensor, dim: int) -> torch.Tensor:
    half = dim // 2
    denominator = max(half - 1, 1)
    frequencies = torch.exp(
        -math.log(10000.0)
        * torch.arange(half, device=steps.device, dtype=torch.float32)
        / denominator
    )
    angles = steps.float().unsqueeze(1) * frequencies.unsqueeze(0)
    features = torch.cat((torch.sin(angles), torch.cos(angles)), dim=1)
    if features.shape[1] < dim:
        features = torch.nn.functional.pad(features, (0, dim - features.shape[1]))
    return features[:, :dim]


class DiffusionStepEmbedding(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = int(dim)
        self.projection = nn.Sequential(
            nn.Linear(self.dim, self.dim),
            nn.SiLU(),
            nn.Linear(self.dim, self.dim),
            nn.SiLU(),
        )

    def forward(self, steps):
        return self.projection(sinusoidal_step_embedding(steps, self.dim))


class TemporalResidualBlock(nn.Module):
    """Gated dilated residual block with diffusion and sample-position embeddings."""

    def __init__(self, channels: int, dilation: int, diffusion_dim: int, time_dim: int):
        super().__init__()
        self.diffusion_projection = nn.Linear(diffusion_dim, channels)
        self.depthwise = nn.Conv1d(
            channels, channels, kernel_size=3, padding=dilation,
            dilation=dilation, groups=channels,
        )
        self.channel_mix = nn.Conv1d(channels, 2 * channels, kernel_size=1)
        self.time_projection = nn.Conv1d(time_dim, 2 * channels, kernel_size=1)
        self.output_projection = nn.Conv1d(channels, 2 * channels, kernel_size=1)

    def forward(self, x, time_features, diffusion_embedding):
        h = x + self.diffusion_projection(diffusion_embedding).unsqueeze(-1)
        h = self.channel_mix(self.depthwise(h)) + self.time_projection(time_features)
        gate, filt = h.chunk(2, dim=1)
        h = torch.sigmoid(gate) * torch.tanh(filt)
        output = self.output_projection(h)
        residual, skip = output.chunk(2, dim=1)
        return (x + residual) / math.sqrt(2.0), skip


class CSDIStyleDenoiser(nn.Module):
    """Efficient long-waveform denoiser; detector identity is encoded by channel order."""

    def __init__(
        self,
        length: int,
        hidden_channels: int = 32,
        residual_layers: int = 11,
        time_dim: int = 32,
        diffusion_dim: int = 128,
    ):
        super().__init__()
        self.length = int(length)
        self.hidden_channels = int(hidden_channels)
        self.input_projection = nn.Conv1d(6, hidden_channels, kernel_size=1)
        self.diffusion_embedding = DiffusionStepEmbedding(diffusion_dim)
        self.register_buffer(
            "time_features", sinusoidal_position_features(length, time_dim),
            persistent=False,
        )
        # One cycle of powers-of-two dilation gives a receptive field of about 4095 samples
        # at the default 11 layers, covering nearly the complete 4160-sample waveform.
        dilations = [2 ** min(i, 10) for i in range(residual_layers)]
        self.residual_layers = nn.ModuleList([
            TemporalResidualBlock(hidden_channels, dilation, diffusion_dim, time_dim)
            for dilation in dilations
        ])
        self.output_projection1 = nn.Conv1d(hidden_channels, hidden_channels, kernel_size=1)
        self.output_projection2 = nn.Conv1d(hidden_channels, 2, kernel_size=1)
        nn.init.zeros_(self.output_projection2.weight)
        nn.init.zeros_(self.output_projection2.bias)

    def forward(self, observed_data, cond_mask, noisy_data, diffusion_step):
        # Inputs have shape [B,2,T]. The target interval is represented by cond_mask=0.
        target_mask = 1.0 - cond_mask
        conditioned_values = cond_mask * observed_data
        noisy_target = target_mask * noisy_data
        network_input = torch.cat((conditioned_values, noisy_target, cond_mask), dim=1)
        hidden = self.input_projection(network_input)
        diffusion_embedding = self.diffusion_embedding(diffusion_step)
        skip_connections = []
        for layer in self.residual_layers:
            hidden, skip = layer(hidden, self.time_features, diffusion_embedding)
            skip_connections.append(skip)
        hidden = torch.stack(skip_connections, dim=0).sum(dim=0)
        hidden = hidden / math.sqrt(len(skip_connections))
        hidden = torch.nn.functional.silu(self.output_projection1(hidden))
        return self.output_projection2(hidden)


class CSDIImputer(nn.Module):
    """Conditional DDPM trained to predict noise only in the masked interval."""

    def __init__(
        self,
        length: int,
        hidden_channels: int,
        residual_layers: int,
        diffusion_steps: int,
        beta_start: float,
        beta_end: float,
        beta_schedule: str,
        diffusion_dim: int = 128,
        time_dim: int = 32,
    ):
        super().__init__()
        if diffusion_steps < 2:
            raise ValueError("At least two diffusion steps are required")
        if not (0 < beta_start < beta_end < 1):
            raise ValueError("Require 0 < beta-start < beta-end < 1")
        if beta_schedule == "quad":
            betas = np.linspace(
                math.sqrt(beta_start), math.sqrt(beta_end), diffusion_steps
            ) ** 2
        elif beta_schedule == "linear":
            betas = np.linspace(beta_start, beta_end, diffusion_steps)
        else:
            raise ValueError(f"Unsupported beta schedule: {beta_schedule}")
        betas = torch.as_tensor(betas, dtype=torch.float32)
        alphas = 1.0 - betas
        alpha_bar = torch.cumprod(alphas, dim=0)
        alpha_bar_previous = torch.cat((torch.ones(1), alpha_bar[:-1]))
        self.register_buffer("beta", betas)
        self.register_buffer("alpha", alphas)
        self.register_buffer("alpha_bar", alpha_bar)
        self.register_buffer("alpha_bar_previous", alpha_bar_previous)
        self.length = int(length)
        self.diffusion_steps = int(diffusion_steps)
        self.denoiser = CSDIStyleDenoiser(
            length, hidden_channels, residual_layers, time_dim, diffusion_dim
        )

    def noise_prediction_loss(self, data, cond_mask, steps=None, noise=None, generator=None):
        batch_size = data.shape[0]
        if steps is None:
            steps = torch.randint(
                self.diffusion_steps, (batch_size,), device=data.device, generator=generator
            )
        if noise is None:
            noise = torch.randn(
                data.shape, dtype=data.dtype, device=data.device, generator=generator
            )
        alpha_bar = self.alpha_bar[steps].view(batch_size, 1, 1)
        noisy_data = alpha_bar.sqrt() * data + (1.0 - alpha_bar).sqrt() * noise
        predicted_noise = self.denoiser(data, cond_mask, noisy_data, steps)
        target_mask = 1.0 - cond_mask
        denominator = target_mask.sum().clamp_min(1.0)
        return ((noise - predicted_noise).square() * target_mask).sum() / denominator

    @torch.no_grad()
    def sample(self, observed_data, cond_mask, n_samples: int):
        """Draw conditional samples; output shape [B,S,2,T], observed samples are preserved."""
        if n_samples < 1:
            raise ValueError("n_samples must be positive")
        batch_size, features, length = observed_data.shape
        if features != 2 or length != self.length:
            raise ValueError(f"Expected [B,2,{self.length}], got {tuple(observed_data.shape)}")
        repeated_data = observed_data.repeat_interleave(n_samples, dim=0)
        repeated_mask = cond_mask.repeat_interleave(n_samples, dim=0)
        target_mask = 1.0 - repeated_mask
        current = torch.randn_like(repeated_data)
        for step in reversed(range(self.diffusion_steps)):
            step_batch = torch.full(
                (repeated_data.shape[0],), step, dtype=torch.long, device=observed_data.device
            )
            predicted_noise = self.denoiser(
                repeated_data, repeated_mask, current, step_batch
            )
            beta_t = self.beta[step]
            alpha_t = self.alpha[step]
            alpha_bar_t = self.alpha_bar[step]
            mean = (
                current - beta_t * predicted_noise / (1.0 - alpha_bar_t).sqrt()
            ) / alpha_t.sqrt()
            if step > 0:
                variance = beta_t * (1.0 - self.alpha_bar_previous[step]) / (
                    1.0 - alpha_bar_t
                )
                current = mean + variance.clamp_min(0.0).sqrt() * torch.randn_like(current)
            else:
                current = mean
            # Clamp the conditioning context to the original observations at every step.
            current = target_mask * current + repeated_mask * repeated_data
        return current.reshape(batch_size, n_samples, features, length)


def validation_noise_loss(model, loader, mask_len, device, seed):
    model.eval()
    generator = torch.Generator(device=device)
    generator.manual_seed(int(seed))
    total, count = 0.0, 0
    with torch.no_grad():
        for data, starts in loader:
            data = data.to(device)
            target_mask = target_mask_from_starts(starts, data.shape[-1], mask_len, device)
            cond_mask = 1.0 - target_mask
            loss = model.noise_prediction_loss(
                data, cond_mask, generator=generator
            )
            points = int(target_mask.sum().item())
            total += float(loss.item()) * points
            count += points
    return total / max(count, 1)


def load_mlp_baselines(mlp_root, length, mask_len, starts, dataset_identity, device):
    """Load only checkpoints from the compact masked-output MLP formulation."""
    models = {}
    status = {}
    if mlp_root is None:
        return models, status
    for detector in range(2):
        label = f"detector_{detector + 1}"
        checkpoint_path = mlp_root / label / "best.pt"
        if not checkpoint_path.is_file():
            status[label] = f"not found: {checkpoint_path}"
            continue
        try:
            checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
            if checkpoint.get("architecture") != "mlp_relu_128_128_masked_output":
                raise ValueError("checkpoint is not from the compact masked-output MLP")
            if int(checkpoint.get("length", -1)) != length:
                raise ValueError("waveform length mismatch")
            if int(checkpoint.get("mask_samples", -1)) != mask_len:
                raise ValueError("mask length mismatch")
            if int(checkpoint.get("mask_positions", -1)) != len(starts):
                raise ValueError("mask-position count mismatch")
            saved_identity = checkpoint.get("dataset_identity")
            if dataset_identity and saved_identity and saved_identity != dataset_identity:
                raise ValueError("prepared dataset identity mismatch")
            model = MaskedMLP(length - mask_len, mask_len, len(starts))
            model.load_state_dict(checkpoint["state_dict"])
            model.to(device).eval()
            models[detector] = model
            status[label] = f"loaded: {checkpoint_path}"
        except Exception as exc:
            status[label] = f"skipped ({exc})"
    return models, status


def mlp_reconstruction(model, signal, start, starts, mask_len, device):
    end = start + mask_len
    context = np.concatenate((signal[:start], signal[end:])).astype(np.float32, copy=False)
    position_code = np.zeros(len(starts), dtype=np.float32)
    position_code[starts.index(start)] = 1.0
    model_input = np.concatenate((context, position_code))
    prediction = model(torch.from_numpy(model_input[None]).to(device))[0].cpu().numpy()
    reconstructed = signal.copy()
    reconstructed[start:end] = prediction
    return reconstructed


def region_mask_starts(time_ps, mask_len, starts):
    length = len(time_ps)
    regions = [
        ("leading_edge", 0.0),
        ("late_20ns", 20000.0),
        ("final_window", float(time_ps[-1])),
    ]
    output = []
    starts_array = np.asarray(starts, dtype=np.int64)
    for name, center in regions:
        desired = int(np.clip(
            np.argmin(np.abs(time_ps - center)) - mask_len // 2,
            0, length - mask_len,
        ))
        start = int(starts_array[np.argmin(np.abs(starts_array - desired))])
        output.append((name, start))
    return output


@torch.no_grad()
def evaluate_reconstructions(
    model, waves, eval_events, time_ps, starts, mask_len, device,
    n_samples, eval_batch_size, mlp_models, output_dir,
):
    """Evaluate sampled conditional means against MLP and interpolation on identical masks."""
    model.eval()
    length = len(time_ps)
    time_ns = time_ps / 1000.0
    all_metrics = []
    cached_regions = {}
    region_specs = region_mask_starts(time_ps, mask_len, starts)
    eval_events = np.asarray(eval_events, dtype=np.int64)
    if len(eval_events) == 0:
        raise ValueError("No validation events available for reconstruction evaluation")

    for region_name, start in region_specs:
        end = start + mask_len
        data = np.stack([
            np.array(waves[int(event)], dtype=np.float32, copy=True, order="C")
            for event in eval_events
        ])
        sampled_batches = []
        for offset in range(0, len(data), eval_batch_size):
            batch_array = data[offset:offset + eval_batch_size]
            batch = torch.from_numpy(batch_array).to(device)
            cond_mask = torch.ones_like(batch)
            cond_mask[:, :, start:end] = 0.0
            draws = model.sample(batch, cond_mask, n_samples)
            sampled_batches.append(draws.cpu().numpy())
        samples = np.concatenate(sampled_batches, axis=0)  # [events, draws, 2, T]
        mlp_full = np.full_like(data, np.nan, dtype=np.float32)
        interpolation_full = np.empty_like(data, dtype=np.float32)
        has_mlp = all(detector in mlp_models for detector in range(2))
        known = np.ones(length, dtype=bool)
        known[start:end] = False
        sample_target = samples[:, :, :, start:end]
        truth_target = data[:, :, start:end]
        conditional_mean = sample_target.mean(axis=1)
        sample_std = sample_target.std(axis=1, ddof=1 if n_samples > 1 else 0)

        for event_idx in range(len(data)):
            for detector in range(2):
                signal = data[event_idx, detector]
                interpolation_full[event_idx, detector] = np.interp(
                    np.arange(length), np.flatnonzero(known), signal[known]
                )
                if detector in mlp_models:
                    mlp_full[event_idx, detector] = mlp_reconstruction(
                        mlp_models[detector], signal, start, starts, mask_len, device
                    )

        for detector in range(2):
            truth = truth_target[:, detector, :]
            pred_mean = conditional_mean[:, detector, :]
            draws = sample_target[:, :, detector, :]
            row = dict(
                region=region_name,
                detector=detector + 1,
                mask_start_sample=start,
                mask_start_ns=float(time_ps[start] / 1000.0),
                mask_samples=mask_len,
                evaluation_events=len(data),
                conditional_samples=n_samples,
                csdi_conditional_mean_rmse=float(np.sqrt(np.mean((pred_mean - truth) ** 2))),
                csdi_mean_draw_rmse=float(np.sqrt(np.mean((draws - truth[:, None, :]) ** 2))),
                csdi_mean_predictive_std=float(np.mean(sample_std[:, detector, :])),
                mlp_rmse=None,
                linear_interp_rmse=float(np.sqrt(np.mean(
                    (interpolation_full[:, detector, start:end] - truth) ** 2
                ))),
            )
            if detector in mlp_models:
                row["mlp_rmse"] = float(np.sqrt(np.mean(
                    (mlp_full[:, detector, start:end] - truth) ** 2
                )))
            all_metrics.append(row)

        cached_regions[region_name] = dict(
            start=start, data=data, samples=samples, mlp_full=mlp_full,
            interpolation_full=interpolation_full,
        )

    # Save compact numeric comparison table.
    metrics_path = output_dir / "comparison_metrics.csv"
    fieldnames = [
        "region", "detector", "mask_start_sample", "mask_start_ns", "mask_samples",
        "evaluation_events", "conditional_samples", "csdi_conditional_mean_rmse",
        "csdi_mean_draw_rmse", "csdi_mean_predictive_std", "mlp_rmse",
        "linear_interp_rmse",
    ]
    with metrics_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_metrics)

    # Plot one held-out event for each region; show uncertainty only in the imputed interval.
    fig, axes = plt.subplots(2, 3, figsize=(15, 7), sharey="row", constrained_layout=True)
    for col, (region_name, _) in enumerate(region_specs):
        region = cached_regions[region_name]
        start = region["start"]
        end = start + mask_len
        lo = max(0, start - 2 * mask_len)
        hi = min(length, end + 2 * mask_len)
        data = region["data"]
        samples = region["samples"]
        for detector in range(2):
            ax = axes[detector, col]
            truth = data[0, detector]
            mean_wave = samples[0, :, detector, :].mean(axis=0)
            ax.plot(time_ns[lo:hi], truth[lo:hi], label="True", lw=1.2, color="black")
            ax.plot(time_ns[lo:hi], mean_wave[lo:hi], label="CSDI sample mean", lw=1.1)
            target_times = time_ns[start:end]
            target_draws = samples[0, :, detector, start:end]
            ax.fill_between(
                target_times,
                np.quantile(target_draws, 0.1, axis=0),
                np.quantile(target_draws, 0.9, axis=0),
                alpha=0.22, label="CSDI 10-90%",
            )
            if detector in mlp_models:
                ax.plot(
                    time_ns[lo:hi], region["mlp_full"][0, detector, lo:hi],
                    "--", label="MLP", lw=1.0,
                )
            ax.plot(
                time_ns[lo:hi], region["interpolation_full"][0, detector, lo:hi],
                ":", label="Interpolation", lw=1.0,
            )
            ax.axvspan(time_ns[start], time_ns[end - 1], color="grey", alpha=0.15)
            ax.set_title(f"{region_name} | detector {detector + 1}")
            ax.set_xlabel("Time (ns)")
            ax.set_ylabel("Normalized signal")
            ax.legend(fontsize=7, loc="best")
    fig.savefig(output_dir / "reconstruction_examples.png", dpi=180)
    fig.savefig(output_dir / "reconstruction_examples.pdf")
    plt.close(fig)
    return all_metrics



@torch.no_grad()
def evaluate_position_scan(model, waves, eval_events, time_ps, starts, mask_len,
                           device, n_samples, eval_batch_size, mlp_models, output_dir):
    """Complete mask-position comparison with bounded memory use.

    All methods use identical events and masks. Aggregate over events and
    masked time samples; coverage is an empirical marginal predictive-interval
    diagnostic and is not a guarantee of joint waveform calibration.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    model.eval()
    eval_events = np.asarray(eval_events, dtype=np.int64)
    results = []
    for index, start in enumerate(starts):
        stop = start + mask_len
        errors = np.zeros((2, 4), dtype=np.float64)  # mean, draw, MLP, interpolation
        coverage = np.zeros((2, 2), dtype=np.float64)  # 50%, 90%
        widths = np.zeros((2, 2), dtype=np.float64)
        mlp_count = np.zeros(2, dtype=np.int64)
        num_points = len(eval_events) * mask_len
        for offset in range(0, len(eval_events), eval_batch_size):
            event_ids = eval_events[offset:offset + eval_batch_size]
            data = np.stack([
                np.asarray(waves[int(e)], dtype=np.float32) for e in event_ids
            ])
            batch = torch.from_numpy(data.copy()).to(device)
            condition = torch.ones_like(batch)
            condition[:, :, start:stop] = 0.0
            samples = model.sample(batch, condition, n_samples)
            truth = data[:, :, start:stop]
            draws = samples[:, :, :, start:stop].cpu().numpy()
            mean = draws.mean(axis=1)
            errors[:, 0] += ((mean - truth) ** 2).sum(axis=(0, 2))
            errors[:, 1] += ((draws - truth[:, None, :, :]) ** 2).sum(axis=(0, 1, 3)) / n_samples
            for j, q in enumerate((0.5, 0.9)):
                low = np.quantile(draws, (1 - q) / 2, axis=1)
                high = np.quantile(draws, (1 + q) / 2, axis=1)
                coverage[:, j] += ((truth >= low) & (truth <= high)).sum(axis=(0, 2))
                widths[:, j] += (high - low).sum(axis=(0, 2))
            # np.interp uses constant endpoint extension for boundary masks.
            known = np.ones(data.shape[-1], dtype=bool)
            known[start:stop] = False
            for detector in range(2):
                for b in range(len(data)):
                    signal = data[b, detector]
                    interp = np.interp(
                        np.arange(len(signal)), np.flatnonzero(known), signal[known]
                    )[start:stop]
                    errors[detector, 3] += np.square(interp - truth[b, detector]).sum()
                if detector in mlp_models:
                    context = np.concatenate(
                        (data[:, detector, :start], data[:, detector, stop:]), axis=-1
                    )
                    one_hot = np.zeros((len(data), len(starts)), dtype=np.float32)
                    one_hot[:, index] = 1.0
                    model_input = torch.from_numpy(
                        np.concatenate((context, one_hot), axis=-1)
                    ).to(device)
                    prediction = mlp_models[detector](model_input).cpu().numpy()
                    errors[detector, 2] += np.square(prediction - truth[:, detector]).sum()
                    mlp_count[detector] += len(data) * mask_len
        for detector in range(2):
            results.append(dict(
                detector=detector + 1, mask_position=index,
                mask_start_sample=int(start),
                mask_start_ns=float(time_ps[start] / 1000),
                n_events=len(eval_events), n_samples=n_samples,
                csdi_mean_rmse=float(np.sqrt(errors[detector, 0] / num_points)),
                csdi_draw_rmse=float(np.sqrt(errors[detector, 1] / num_points)),
                mlp_rmse=(float(np.sqrt(errors[detector, 2] / mlp_count[detector]))
                          if mlp_count[detector] else None),
                interpolation_rmse=float(np.sqrt(errors[detector, 3] / num_points)),
                coverage_50=float(coverage[detector, 0] / num_points),
                coverage_90=float(coverage[detector, 1] / num_points),
                interval_width_50=float(widths[detector, 0] / num_points),
                interval_width_90=float(widths[detector, 1] / num_points),
            ))
        print(
            f"Position {index + 1}/{len(starts)} at {float(time_ps[start] / 1000):.2f} ns done",
            flush=True,
        )

    fieldnames = list(results[0])
    with (output_dir / "position_metrics.csv").open(
        "w", newline="", encoding="utf-8"
    ) as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)
    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    for detector in range(2):
        ax = axes[detector]
        rows = [r for r in results if r["detector"] == detector + 1]
        x = [r["mask_start_ns"] for r in rows]
        for key, label in (
            ("csdi_mean_rmse", "CSDI conditional mean"),
            ("mlp_rmse", "MLP"),
            ("interpolation_rmse", "Interpolation"),
        ):
            if all(r[key] is not None for r in rows):
                ax.plot(x, [r[key] for r in rows], label=label, linewidth=1.2)
        ax.set_ylabel(f"Detector {detector + 1} RMSE")
        ax.grid(alpha=0.25)
        ax.legend()
    axes[1].set_xlabel("Mask start time (ns)")
    fig.tight_layout()
    fig.savefig(output_dir / "rmse_by_position.png", dpi=180)
    fig.savefig(output_dir / "rmse_by_position.pdf")
    plt.close(fig)
    summary = dict(
        n_events=len(eval_events), n_positions=len(starts), n_samples=n_samples,
        metric_note="RMSE averaged over held-out events and masked samples at each position",
        calibration_note="50%/90% marginal quantile coverage estimated from finite diffusion samples",
        baseline_note="Interpolation has constant endpoint extension at boundary masks",
        position_metrics=results,
    )
    (output_dir / "evaluation_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    return results


def train(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")
    results = root_relative(args.results)
    waves, time_ps, source_run, manifest = load_wide_development(
        results, args.channel, root_relative(args.cache_root)
    )
    num_events, detectors, length = waves.shape
    if detectors != 2:
        raise ValueError(f"Expected exactly two detectors, got {detectors}")
    time_ps = np.asarray(time_ps, dtype=np.float64)
    steps = np.diff(time_ps)
    dt_ps = float(np.median(steps))
    if dt_ps <= 0 or not np.allclose(steps, dt_ps, rtol=1e-4, atol=1e-3):
        raise ValueError("Expected a uniform time grid")
    mask_len = max(1, round(args.mask_ns * 1000.0 / dt_ps))
    starts = make_starts(length, mask_len, args.mask_stride_samples or mask_len)
    selected = np.random.default_rng(args.seed).permutation(num_events)[:min(args.max_events, num_events)]
    if len(selected) < 10:
        raise ValueError("At least 10 selected events are required")
    val_size = max(1, round(len(selected) * args.val_fraction))
    if val_size >= len(selected):
        raise ValueError("Validation split leaves no training events")
    val_events, train_events = selected[:val_size], selected[val_size:]

    destination = (
        root_relative(args.output) if args.output else
        ROOT / "waveform_analysis" / "imputation_experiments"
        / results.name / args.channel / "csdi_joint"
    )
    destination.mkdir(parents=True, exist_ok=True)
    print(f"Device: {device} | source events={num_events:,} | samples={length:,}", flush=True)
    print(
        f"Selected={len(selected):,} | train={len(train_events):,} | "
        f"validation={len(val_events):,}", flush=True
    )
    print(
        f"Joint paired training | mask width={mask_len} samples ({args.mask_ns:g} ns) | "
        f"positions={len(starts)} | masks/event/epoch={args.masks_per_event_per_epoch}",
        flush=True,
    )
    print(
        f"Training examples/epoch={len(train_events) * args.masks_per_event_per_epoch:,} | "
        f"validation examples={len(val_events):,}", flush=True
    )
    print(f"Output: {destination}", flush=True)

    if args.evaluate_only:
        checkpoint_path = root_relative(args.checkpoint) if args.checkpoint else destination / "best.pt"
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        if checkpoint.get("architecture") != "csdi_conditional_ddpm_dilated_depthwise_conv":
            raise ValueError("Unsupported checkpoint architecture")
        if checkpoint.get("dataset_identity") != manifest.get("analysis_protocol_identity"):
            raise ValueError("Prepared dataset identity differs from checkpoint")
        if int(checkpoint["length"]) != length or int(checkpoint["mask_samples"]) != mask_len:
            raise ValueError("Waveform length or mask width differs from checkpoint")
        if int(checkpoint["mask_stride_samples"]) != (args.mask_stride_samples or mask_len):
            raise ValueError("Mask stride differs from checkpoint")
        if int(checkpoint["mask_positions"]) != len(starts):
            raise ValueError("Mask positions differ from checkpoint")
        if "seed" in checkpoint and int(checkpoint["seed"]) != args.seed:
            raise ValueError("Split seed differs from checkpoint; pass the training --seed")
        # A different event cap would change the original event-level split.
        existing_summary_path = destination / "summary.json"
        if existing_summary_path.is_file():
            existing_summary = json.loads(existing_summary_path.read_text(encoding="utf-8"))
            if (int(existing_summary.get("selected_events", -1)) != len(selected)
                    or int(existing_summary.get("validation_events", -1)) != len(val_events)):
                raise ValueError("Event selection differs from the original training split")
        model = CSDIImputer(
            length=length,
            hidden_channels=int(checkpoint["hidden_channels"]),
            residual_layers=int(checkpoint["residual_layers"]),
            diffusion_steps=int(checkpoint["diffusion_steps"]),
            beta_start=float(checkpoint["beta_start"]),
            beta_end=float(checkpoint["beta_end"]),
            beta_schedule=checkpoint["beta_schedule"],
            diffusion_dim=int(checkpoint["diffusion_embedding_dim"]),
            time_dim=int(checkpoint["time_embedding_dim"]),
        ).to(device)
        model.load_state_dict(checkpoint["state_dict"])
        model.eval()
        default_mlp_root = (
            ROOT / "waveform_analysis" / "imputation_experiments"
            / results.name / args.channel / "detector_specific_mlp"
        )
        mlp_root = root_relative(args.mlp_root) if args.mlp_root else default_mlp_root
        mlp_models, statuses = load_mlp_baselines(
            mlp_root, length, mask_len, starts,
            manifest.get("analysis_protocol_identity"), device
        )
        print(f"MLP status: {statuses}", flush=True)
        # Separate output keeps existing training files and examples untouched.
        evaluation_dir = root_relative(args.eval_output) if args.eval_output else destination / "evaluation"
        evaluation_dir.mkdir(parents=True, exist_ok=True)
        eval_events = val_events[:min(args.eval_events, len(val_events))]
        evaluate_position_scan(
            model, waves, eval_events, time_ps, starts, mask_len, device,
            args.eval_samples, args.eval_batch_size, mlp_models, evaluation_dir
        )
        print(f"Evaluation finished: {evaluation_dir}", flush=True)
        return

    train_ds = EventMaskDataset(
        waves, train_events, starts, random_masks=True,
        masks_per_event=args.masks_per_event_per_epoch,
    )
    val_ds = EventMaskDataset(waves, val_events, starts, random_masks=False)
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0,
        pin_memory=device.type == "cuda",
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0,
        pin_memory=device.type == "cuda",
    )
    model = CSDIImputer(
        length=length,
        hidden_channels=args.hidden_channels,
        residual_layers=args.residual_layers,
        diffusion_steps=args.diffusion_steps,
        beta_start=args.beta_start,
        beta_end=args.beta_end,
        beta_schedule=args.beta_schedule,
        diffusion_dim=args.diffusion_embedding_dim,
        time_dim=args.time_embedding_dim,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    print(
        f"Model parameters={sum(p.numel() for p in model.parameters() if p.requires_grad):,} | "
        f"diffusion steps={args.diffusion_steps} | residual layers={args.residual_layers}",
        flush=True,
    )

    best_loss = float("inf")
    best_epoch = 0
    remaining = args.patience
    history = []
    for epoch in range(1, args.epochs + 1):
        epoch_start = time.perf_counter()
        model.train()
        running, points = 0.0, 0
        print(f"Epoch {epoch}/{args.epochs} started", flush=True)
        for batch_idx, (data, mask_starts) in enumerate(train_loader, start=1):
            data = data.to(device, non_blocking=True)
            target_mask = target_mask_from_starts(mask_starts, length, mask_len, device)
            cond_mask = 1.0 - target_mask
            optimizer.zero_grad(set_to_none=True)
            loss = model.noise_prediction_loss(data, cond_mask)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            n_points = int(target_mask.sum().item())
            running += float(loss.item()) * n_points
            points += n_points
            if (
                batch_idx == 1 or batch_idx % args.log_every == 0
                or batch_idx == len(train_loader)
            ):
                print(
                    f"epoch {epoch:03d} | batch {batch_idx}/{len(train_loader)} | "
                    f"train noise MSE={running / max(points, 1):.6f} | "
                    f"elapsed={time.perf_counter() - epoch_start:.1f}s",
                    flush=True,
                )

        train_loss = running / max(points, 1)
        val_loss = validation_noise_loss(
            model, val_loader, mask_len, device, seed=args.seed + 12345
        )
        history.append((epoch, train_loss, val_loss, time.perf_counter() - epoch_start))
        print(
            f"epoch {epoch:03d} | train noise MSE={train_loss:.6f} | "
            f"val noise MSE={val_loss:.6f} | elapsed={time.perf_counter() - epoch_start:.1f}s",
            flush=True,
        )
        if val_loss < best_loss - args.min_delta:
            best_loss = val_loss
            best_epoch = epoch
            remaining = args.patience
            torch.save(
                {
                    "state_dict": model.state_dict(),
                    "architecture": "csdi_conditional_ddpm_dilated_depthwise_conv",
                    "channel": args.channel,
                    "length": length,
                    "mask_samples": mask_len,
                    "mask_positions": len(starts),
                    "mask_stride_samples": args.mask_stride_samples or mask_len,
                    "diffusion_steps": args.diffusion_steps,
                    "beta_start": args.beta_start,
                    "beta_end": args.beta_end,
                    "beta_schedule": args.beta_schedule,
                    "hidden_channels": args.hidden_channels,
                    "residual_layers": args.residual_layers,
                    "time_embedding_dim": args.time_embedding_dim,
                    "diffusion_embedding_dim": args.diffusion_embedding_dim,
                    "weight_decay": args.weight_decay,
                    "dt_ps": dt_ps,
                    "dataset_identity": manifest.get("analysis_protocol_identity"),
                    "seed": args.seed,
                    "best_epoch": epoch,
                    "best_validation_noise_mse": val_loss,
                },
                destination / "best.pt",
            )
        else:
            remaining -= 1
            if remaining <= 0:
                print("Early stopping: validation noise-prediction loss did not improve.", flush=True)
                break

    with (destination / "history.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(("epoch", "train_noise_mse", "val_noise_mse", "elapsed_seconds"))
        writer.writerows(history)
    if not (destination / "best.pt").is_file():
        raise RuntimeError("Training did not produce a checkpoint")
    checkpoint = torch.load(destination / "best.pt", map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    default_mlp_root = (
        ROOT / "waveform_analysis" / "imputation_experiments"
        / results.name / args.channel / "detector_specific_mlp"
    )
    mlp_root = root_relative(args.mlp_root) if args.mlp_root else default_mlp_root
    mlp_models, mlp_status = load_mlp_baselines(
        mlp_root, length, mask_len, starts,
        manifest.get("analysis_protocol_identity"), device,
    )
    for detector, status in mlp_status.items():
        print(f"MLP baseline {detector}: {status}", flush=True)
    eval_events = val_events[:min(args.eval_events, len(val_events))]
    print(
        f"Sampling conditional reconstructions for {len(eval_events)} held-out events, "
        f"{args.eval_samples} draws and three mask regions...", flush=True
    )
    metrics = evaluate_reconstructions(
        model, waves, eval_events, time_ps, starts, mask_len, device,
        args.eval_samples, args.eval_batch_size, mlp_models, destination,
    )
    summary = {
        "source_run": str(source_run),
        "channel": args.channel,
        "dataset_identity": manifest.get("analysis_protocol_identity"),
        "architecture": "CSDI-style conditional DDPM with dilated depthwise-separable temporal convolutions",
        "architecture_note": (
            "The conditional diffusion noise-prediction objective and DDPM reverse sampler follow CSDI. "
            "The reference full-sequence Transformer denoiser is replaced with dilated temporal convolutions "
            "to avoid quadratic attention cost on waveforms with thousands of samples."
        ),
        "tshap_integration": False,
        "conditioning": "both detector waveforms outside the shared target interval",
        "target_mask": "one contiguous interval, same temporal position for both detectors",
        "training_masking": "uniform random mask position per event per epoch",
        "diffusion_steps": args.diffusion_steps,
        "beta_schedule": args.beta_schedule,
        "beta_start": args.beta_start,
        "beta_end": args.beta_end,
        "hidden_channels": args.hidden_channels,
        "residual_layers": args.residual_layers,
        "receptive_field_samples": 1 + 2 * sum(2 ** min(i, 10) for i in range(args.residual_layers)),
        "weight_decay": args.weight_decay,
        "learning_rate": args.lr,
        "seed": args.seed,
        "development_events": int(num_events),
        "selected_events": int(len(selected)),
        "train_events": int(len(train_events)),
        "validation_events": int(len(val_events)),
        "evaluation_events": int(len(eval_events)),
        "mask_ns": args.mask_ns,
        "mask_samples": mask_len,
        "mask_stride_samples": args.mask_stride_samples or mask_len,
        "mask_positions": len(starts),
        "masks_per_event_per_epoch": args.masks_per_event_per_epoch,
        "epochs_completed": len(history),
        "best_epoch": best_epoch,
        "best_validation_noise_mse": float(best_loss),
        "mlp_baseline_root": str(mlp_root),
        "mlp_baseline_status": mlp_status,
        "comparison_metrics": metrics,
        "interpretation_note": (
            "CSDI conditional-mean RMSE averages generated samples before scoring. "
            "Mean-draw RMSE scores individual stochastic draws. These are different quantities; "
            "the validation noise MSE is not directly comparable to waveform RMSE."
        ),
        "arguments": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
    }
    (destination / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Saved checkpoint, history, plots, comparison metrics and summary to {destination}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--channel", choices=("energy", "timing"), required=True)
    parser.add_argument("--cache-root", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--mlp-root", type=Path)
    parser.add_argument("--max-events", type=int, default=5000)
    parser.add_argument("--mask-ns", type=float, default=1.0)
    parser.add_argument("--mask-stride-samples", type=int)
    parser.add_argument("--masks-per-event-per-epoch", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=6)
    parser.add_argument("--min-delta", type=float, default=1e-5)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--val-fraction", type=float, default=0.15)
    parser.add_argument("--hidden-channels", type=int, default=32)
    parser.add_argument("--residual-layers", type=int, default=11)
    parser.add_argument("--time-embedding-dim", type=int, default=32)
    parser.add_argument("--diffusion-embedding-dim", type=int, default=128)
    parser.add_argument("--diffusion-steps", type=int, default=50)
    parser.add_argument("--beta-start", type=float, default=1e-4)
    parser.add_argument("--beta-end", type=float, default=0.5)
    parser.add_argument("--beta-schedule", choices=("quad", "linear"), default="quad")
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--evaluate-only", action="store_true")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--eval-output", type=Path)
    parser.add_argument("--eval-events", type=int, default=100)
    parser.add_argument("--eval-samples", type=int, default=4)
    parser.add_argument("--eval-batch-size", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()
    if args.max_events < 10:
        parser.error("--max-events must be at least 10")
    if args.mask_stride_samples is not None and args.mask_stride_samples < 1:
        parser.error("--mask-stride-samples must be positive")
    if args.mask_ns <= 0 or args.lr <= 0 or args.weight_decay < 0:
        parser.error("mask-ns and lr must be positive; weight-decay must be non-negative")
    if not 0 < args.val_fraction < 1:
        parser.error("--val-fraction must be between 0 and 1")
    if min(args.epochs, args.patience, args.batch_size, args.log_every,
           args.masks_per_event_per_epoch, args.hidden_channels, args.residual_layers,
           args.time_embedding_dim, args.diffusion_embedding_dim, args.diffusion_steps,
           args.eval_events, args.eval_samples, args.eval_batch_size) < 1:
        parser.error("Training, architecture and evaluation integer parameters must be positive")
    if args.time_embedding_dim < 2 or args.diffusion_embedding_dim < 2:
        parser.error("Embedding dimensions must be at least 2")
    if not 0 < args.beta_start < args.beta_end < 1:
        parser.error("Require 0 < beta-start < beta-end < 1")
    if args.grad_clip <= 0 or args.min_delta < 0:
        parser.error("grad-clip must be positive and min-delta non-negative")
    train(args)


if __name__ == "__main__":
    main()
