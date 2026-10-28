#!/usr/bin/env python3
"""Train detector-specific fully-connected ReLU MLPs with systematic wide-window masking (128, 128 hidden units).

From repository root:
    python -m waveform_analysis.scripts.train_wide_unet_imputer_detector_specific \
        --results waveform_analysis/results/UC/FINAL --channel energy --max-events 1000

Direct execution is supported too:
    python waveform_analysis/scripts/train_wide_unet_imputer_detector_specific.py \
        --results waveform_analysis/results/UC/FINAL --channel energy --max-events 1000

Only development prepared events are used. Each event is split *before* masks are
generated, preventing correlated event/mask copies crossing the validation split.
"""
from __future__ import annotations

import argparse
import csv
import json
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

from waveform_analysis.data.results import discover_runs, resolve_run_config, read_run_manifest
from waveform_analysis.engine.xai_replay import PreparedCacheLookup, _cache_root


def root_relative(path):
    if path is None:
        return None
    path = Path(path).expanduser()
    return path if path.is_absolute() else ROOT / path


def load_wide_development(results, channel, cache_root):
    matches = []
    for run in discover_runs(results):
        directory = Path(run["directory"])
        cfg = resolve_run_config(directory)
        if "wide" in str(cfg.get("window_name", "")).lower() and cfg.get("mode") == f"{channel}_to_{channel}":
            matches.append((directory, cfg))
    if not matches:
        raise ValueError(f"No wide {channel} runs found under {results}")
    directory, cfg = matches[0]
    manifest = read_run_manifest(directory)
    cache = _cache_root(cfg, directory, cache_root, roles=("development",))
    dataset = PreparedCacheLookup().find(cache, "development", cfg, manifest)
    waves = dataset.energy_windows if channel == "energy" else dataset.timing_windows
    times = dataset.energy_time_ps if channel == "energy" else dataset.timing_time_ps
    if waves is None or times is None or waves.ndim != 3 or waves.shape[1] != 2:
        raise ValueError("Expected development waveforms [events, 2 detectors, samples]")
    return waves, np.asarray(times), directory, dataset.manifest


class MaskedWaveformDataset(Dataset):
    """Lazy exhaustive Cartesian product of events and temporal mask positions."""
    def __init__(self, waves, events, detector, mask_len, stride):
        self.waves = waves
        self.events = np.asarray(events, dtype=np.int64)
        self.detector = int(detector)
        self.mask_len = int(mask_len)
        total = int(waves.shape[-1])
        if not (0 < mask_len < total and stride > 0):
            raise ValueError("Invalid mask length/stride")
        self.starts = list(range(0, total - mask_len + 1, stride))
        if self.starts[-1] != total - mask_len:
            self.starts.append(total - mask_len)

    def __len__(self):
        return len(self.events) * len(self.starts)

    def __getitem__(self, idx):
        event = int(self.events[idx // len(self.starts)])
        start = self.starts[idx % len(self.starts)]
        signal = np.asarray(self.waves[event, self.detector], dtype=np.float32).copy()
        if not np.isfinite(signal).all():
            raise ValueError(f"Non-finite samples in event {event}")
        observed = np.ones(signal.size, dtype=np.float32)
        observed[start:start + self.mask_len] = 0
        return torch.from_numpy(signal), torch.from_numpy(observed)


class MaskedMLP(nn.Module):
    """Fully connected 2*T -> 128 -> 128 -> T masked reconstruction."""
    def __init__(self, length):
        super().__init__()
        self.length = int(length)
        self.net = nn.Sequential(
            nn.Linear(2 * self.length, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, self.length),
        )

    def forward(self, signal, observed):
        if signal.shape != observed.shape or signal.shape[-1] != self.length:
            raise ValueError("Unexpected waveform/mask shape")
        z = torch.cat((signal * observed, observed), dim=-1)
        guess = self.net(z)
        return signal * observed + guess * (1 - observed)


def masked_mse(pred, truth, observed):
    absent = 1 - observed
    return ((pred - truth).square() * absent).sum() / absent.sum().clamp_min(1)


@torch.no_grad()
def validation_rmse(model, loader, device):
    model.eval()
    error, count = 0.0, 0.0
    for signal, observed in loader:
        signal, observed = signal.to(device), observed.to(device)
        prediction = model(signal, observed)
        absent = 1 - observed
        error += ((prediction - signal).square() * absent).sum().item()
        count += absent.sum().item()
    return (error / count) ** 0.5


@torch.no_grad()
def save_examples(model, dataset, time_ps, device, destination, mask_len):
    """Fixed leading-edge, 20 ns and final-region examples, with local zoom."""
    fig, axes = plt.subplots(3, 1, figsize=(10, 8))
    regions = [("leading_edge", 0.), ("late_20ns", 20000.), ("final_window", float(time_ps[-1]))]
    metrics = []
    length = len(time_ps)
    for idx, ((name, center), ax) in enumerate(zip(regions, axes)):
        signal, _ = dataset[idx % len(dataset)]
        original = signal.numpy()
        start = int(np.clip(np.argmin(abs(time_ps - center)) - mask_len // 2, 0, length - mask_len))
        obs = np.ones(length, dtype=np.float32)
        obs[start:start + mask_len] = 0
        model_signal = model(signal[None].to(device), torch.from_numpy(obs[None]).to(device))[0].cpu().numpy()
        missing = obs == 0
        interpolated = np.interp(np.arange(length), np.flatnonzero(~missing), original[~missing])
        err = float(np.sqrt(np.mean((model_signal[missing] - original[missing]) ** 2)))
        err_interp = float(np.sqrt(np.mean((interpolated[missing] - original[missing]) ** 2)))
        metrics.append(dict(region=name, unet_rmse=err, linear_interp_rmse=err_interp))
        t = time_ps / 1000
        ax.plot(t, original, label="True", lw=1)
        ax.plot(t, model_signal, label="U-Net", lw=1)
        ax.plot(t, interpolated, "--", label="Linear interp.", lw=1)
        ax.axvspan(t[start], t[start + mask_len - 1], color="gray", alpha=.2)
        ax.set_xlim(t[max(0, start - 2 * mask_len)], t[min(length - 1, start + 3 * mask_len)])
        ax.set_ylabel("Normalized signal")
        ax.set_title(f"{name}: U-Net RMSE={err:.4g}; interpolation RMSE={err_interp:.4g}")
        ax.legend(fontsize=8)
    axes[-1].set_xlabel("Time (ns)")
    fig.tight_layout()
    fig.savefig(destination / "reconstruction_examples.png", dpi=180)
    plt.close(fig)
    return metrics


def train(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")
    results = root_relative(args.results)
    waves, time_ps, source_run, manifest = load_wide_development(
        results, args.channel, root_relative(args.cache_root))
    num_events, _, samples = waves.shape
    if num_events < 10:
        raise ValueError("At least 10 development events are required")
    steps = np.diff(time_ps)
    dt_ps = float(np.median(steps))
    if dt_ps <= 0 or not np.allclose(steps, dt_ps, rtol=1e-4, atol=1e-3):
        raise ValueError("Expected a uniform time grid")
    mask_len = max(1, round(args.mask_ns * 1000 / dt_ps))
    if mask_len >= samples:
        raise ValueError("Mask exceeds the waveform length")
    stride = args.mask_stride_samples or mask_len
    selected = np.random.default_rng(args.seed).permutation(num_events)[:args.max_events]
    if len(selected) < 10:
        raise ValueError("Select at least 10 events")
    val_size = max(1, round(len(selected) * args.val_fraction))
    val_events, train_events = selected[:val_size], selected[val_size:]
    destination = (root_relative(args.output) if args.output else
                   ROOT / "waveform_analysis" / "imputation_experiments" /
                   results.name / args.channel / "detector_specific_mlp")
    destination.mkdir(parents=True, exist_ok=True)
    positions = len(MaskedWaveformDataset(waves, [train_events[0]], 0, mask_len, stride).starts)
    print(f"Device: {device} | {num_events:,} development events | samples={samples:,}", flush=True)
    print(f"Selected={len(selected):,} | train={len(train_events):,} | val={len(val_events):,}", flush=True)
    print(f"Systematic masks: {positions} positions/event | width={mask_len} samples | stride={stride}", flush=True)
    print(f"Train cases per detector/epoch={len(train_events)*positions:,}; "
          f"validation cases={len(val_events)*positions:,}", flush=True)
    print(f"Output: {destination}", flush=True)
    summary = dict(source_run=str(source_run), channel=args.channel,
                   dataset_identity=manifest.get("analysis_protocol_identity"),
                   development_events=num_events, selected_events=len(selected),
                   train_events=len(train_events), validation_events=len(val_events),
                   mask_ns=args.mask_ns, mask_samples=mask_len,
                   mask_stride_samples=stride, mask_positions=positions,
                   training_masking="uniform_exhaustive",
                   limitation="Deterministic reconstruction; not conditional sampling",
                   detectors={})
    for detector in range(2):
        torch.manual_seed(args.seed + detector)
        label = f"detector_{detector + 1}"
        folder = destination / label
        folder.mkdir(parents=True, exist_ok=True)
        ds_train = MaskedWaveformDataset(waves, train_events, detector, mask_len, stride)
        ds_val = MaskedWaveformDataset(waves, val_events, detector, mask_len, stride)
        train_loader = DataLoader(ds_train, batch_size=args.batch_size, shuffle=True,
                                  num_workers=0, pin_memory=device.type == "cuda")
        val_loader = DataLoader(ds_val, batch_size=args.batch_size, shuffle=False,
                                num_workers=0, pin_memory=device.type == "cuda")
        model = MaskedMLP(samples).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
        best, remaining, history = float("inf"), args.patience, []
        print(f"\n=== {label} | {len(train_loader)} train / {len(val_loader)} validation batches ===", flush=True)
        for epoch in range(1, args.epochs + 1):
            start = time.perf_counter()
            running, points = 0.0, 0.0
            model.train()
            print(f"{label}: epoch {epoch}/{args.epochs} started", flush=True)
            for batch_idx, (signal, observed) in enumerate(train_loader, start=1):
                signal, observed = signal.to(device), observed.to(device)
                optimizer.zero_grad(set_to_none=True)
                pred = model(signal, observed)
                loss = masked_mse(pred, signal, observed)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.)
                optimizer.step()
                missing = (1 - observed).sum().item()
                running += loss.item() * missing
                points += missing
                if batch_idx == 1 or batch_idx % args.log_every == 0 or batch_idx == len(train_loader):
                    print(f"{label} epoch {epoch:03d} | {batch_idx}/{len(train_loader)} "
                          f"| train RMSE={(running/points)**.5:.6f} | elapsed={time.perf_counter()-start:.1f}s",
                          flush=True)
            train_error = (running / points) ** .5
            print(f"{label} epoch {epoch:03d} | validating", flush=True)
            val_error = validation_rmse(model, val_loader, device)
            history.append((epoch, train_error, val_error))
            print(f"{label} epoch {epoch:03d} | train={train_error:.6f} "
                  f"| val={val_error:.6f} | elapsed={time.perf_counter()-start:.1f}s", flush=True)
            if val_error < best - 1e-6:
                best, remaining = val_error, args.patience
                torch.save(dict(state_dict=model.state_dict(), architecture='mlp_relu_128_128',
                                channel=args.channel, detector=detector + 1,
                                length=samples, mask_samples=mask_len, dt_ps=dt_ps,
                                dataset_identity=manifest.get("analysis_protocol_identity")),
                           folder / "best.pt")
            else:
                remaining -= 1
                if remaining <= 0:
                    print(f"{label}: early stopping", flush=True)
                    break
        with (folder / "history.csv").open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(("epoch", "train_rmse", "val_rmse"))
            writer.writerows(history)
        checkpoint = torch.load(folder / "best.pt", map_location=device, weights_only=True)
        model.load_state_dict(checkpoint["state_dict"])
        model.eval()
        metrics = save_examples(model, ds_val, time_ps, device, folder, mask_len)
        summary["detectors"][label] = dict(best_val_rmse_normalized=best,
                                           epochs_completed=len(history),
                                           checkpoint=str(folder / "best.pt"),
                                           example_metrics=metrics)
        print(f"{label}: saved checkpoint, history and examples to {folder}", flush=True)
        del model, optimizer
        if device.type == "cuda":
            torch.cuda.empty_cache()
    (destination / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Completed both detectors: {destination}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--channel", choices=("energy", "timing"), required=True)
    parser.add_argument("--cache-root", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--max-events", type=int, default=None)
    parser.add_argument("--mask-ns", type=float, default=1.)
    parser.add_argument("--mask-stride-samples", type=int)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--val-fraction", type=float, default=.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()
    if args.max_events is not None and args.max_events < 10:
        parser.error("--max-events must be at least 10")
    if args.mask_stride_samples is not None and args.mask_stride_samples < 1:
        parser.error("--mask-stride-samples must be positive")
    if args.mask_ns <= 0 or args.lr <= 0 or not 0 < args.val_fraction < 1:
        parser.error("Invalid mask-ns, lr or val-fraction")
    if min(args.epochs, args.patience, args.batch_size, args.log_every) < 1:
        parser.error("Epochs, patience, batch-size and log-every must be positive")
    train(args)


if __name__ == "__main__":
    main()
