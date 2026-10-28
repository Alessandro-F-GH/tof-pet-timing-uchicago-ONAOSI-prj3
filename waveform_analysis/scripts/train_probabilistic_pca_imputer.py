#!/usr/bin/env python3
"""Probabilistic PCA (isotropic factor-analysis) imputation of full wide waveforms.

Fits independently per detector on prepared DEVELOPMENT waveforms:
  x = mean + W z + eps, z ~ N(0,I), eps ~ N(0,sigma^2 I).
Posterior z|observed is exact under this model. Missing samples are sampled
conditionally and all observed samples are copied unchanged.

Requires no additional dependencies beyond the waveform_analysis requirements.
Compares conditional-mean RMSE, linear interpolation and compatible compact-MLP
checkpoints on identical held-out events and 1-ns mask positions.
No TSHAP integration.
"""
from __future__ import annotations
import argparse
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if __package__ in (None, "") and str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
import torch
from waveform_analysis.scripts.train_wide_unet_imputer_detector_specific import (
    MaskedMLP, load_wide_development, root_relative,
)


def positions(length, mask_len, stride):
    if not 0 < mask_len < length or stride < 1:
        raise ValueError("Invalid mask width or stride")
    starts = list(range(0, length - mask_len + 1, stride))
    if starts[-1] != length - mask_len:
        starts.append(length - mask_len)
    return starts


class ProbabilisticPCA:
    """Full-waveform PPCA: low-rank Gaussian covariance plus isotropic noise."""
    def __init__(self, components, noise_floor=1e-10):
        self.components = int(components)
        self.noise_floor = float(noise_floor)

    def fit(self, data):
        data = np.asarray(data, dtype=np.float64)
        n, length = data.shape
        if not 1 <= self.components < min(n, length):
            raise ValueError("PPCA components must be less than samples and events")
        pca = PCA(n_components=self.components, svd_solver="randomized", random_state=42)
        pca.fit(data)
        self.mean = np.asarray(pca.mean_, dtype=np.float64)
        eigenvalues = pca.explained_variance_
        self.sigma2 = max(float(pca.noise_variance_), self.noise_floor)
        # PPCA maximum-likelihood loading matrix (rotation fixed to PCA axes).
        scales = np.sqrt(np.maximum(eigenvalues - self.sigma2, 0.))
        self.W = np.ascontiguousarray(pca.components_.T * scales)
        self.explained_variance_ratio = float(pca.explained_variance_ratio_.sum())
        return self

    def condition(self, waveform, start, width, samples, rng):
        """Return posterior mean and draws for the missing interval."""
        if samples < 1:
            raise ValueError("samples must be >= 1")
        stop = start + width
        if start < 0 or stop > len(self.mean):
            raise ValueError("Invalid interval")
        obs = np.ones(len(self.mean), dtype=bool)
        obs[start:stop] = False
        W_o, W_m = self.W[obs], self.W[start:stop]
        observed = np.asarray(waveform, dtype=np.float64)[obs] - self.mean[obs]
        # Posterior precision: I + W_O^T W_O/sigma^2
        precision = np.eye(self.components) + W_o.T @ W_o / self.sigma2
        factor = np.linalg.cholesky(precision)
        rhs = W_o.T @ observed / self.sigma2
        posterior_mean = np.linalg.solve(factor.T, np.linalg.solve(factor, rhs))
        missing_mean = self.mean[start:stop] + W_m @ posterior_mean
        # Sigma_z = L^{-T} L^{-1}; draw z using L^{-T} standard_normal.
        latent_noise = np.linalg.solve(
            factor.T, rng.standard_normal((self.components, samples))
        ).T
        predicted = (
            missing_mean[None, :] + latent_noise @ W_m.T
            + rng.normal(0., np.sqrt(self.sigma2), size=(samples, width))
        )
        return missing_mean, predicted


def load_mlp_models(results, channel, length, width, starts, identity):
    root = (ROOT / "waveform_analysis/imputation_experiments" /
            results.name / channel / "detector_specific_mlp")
    found, status = {}, {}
    for d in range(2):
        label = f"detector_{d+1}"
        checkpoint = root / label / "best.pt"
        if not checkpoint.is_file():
            status[label] = "not found"
            continue
        saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
        if (saved.get("architecture") != "mlp_relu_128_128_masked_output"
            or saved.get("length") != length
            or saved.get("mask_samples") != width
            or saved.get("mask_positions") != len(starts)
            or (identity and saved.get("dataset_identity") != identity)):
            status[label] = "incompatible checkpoint"
            continue
        model = MaskedMLP(length - width, width, len(starts))
        model.load_state_dict(saved["state_dict"])
        model.eval()
        found[d] = model
        status[label] = "loaded"
    return found, status


@torch.no_grad()
def evaluate(models, waves, validation, starts, width, time_ps, draws, mlp_models,
             destination, seed):
    rows = []
    rng = np.random.default_rng(seed + 567)
    length = len(time_ps)
    for position_idx, start in enumerate(starts):
        stop = start + width
        observed = np.ones(length, dtype=bool)
        observed[start:stop] = False
        totals = np.zeros((2, 4), dtype=np.float64)
        coverage = np.zeros((2, 2), dtype=np.float64)
        for event_id in validation:
            for det in range(2):
                signal = np.asarray(waves[int(event_id), det], dtype=np.float64)
                truth = signal[start:stop]
                mu, generated = models[det].condition(signal, start, width, draws, rng)
                totals[det, 0] += np.square(mu - truth).sum()
                totals[det, 1] += np.square(generated - truth[None]).sum() / draws
                interpolated = np.interp(
                    np.arange(start, stop), np.flatnonzero(observed), signal[observed]
                )
                totals[det, 2] += np.square(interpolated - truth).sum()
                if det in mlp_models:
                    context = np.concatenate((signal[:start], signal[stop:]))
                    code = np.zeros(len(starts), dtype=np.float32)
                    code[position_idx] = 1.
                    features = torch.from_numpy(
                        np.concatenate((context, code)).astype(np.float32)[None]
                    )
                    predicted = mlp_models[det](features)[0].numpy()
                    totals[det, 3] += np.square(predicted - truth).sum()
                for j, level in enumerate((0.5, .9)):
                    lower = np.quantile(generated, (1. - level) / 2, axis=0)
                    upper = np.quantile(generated, (1. + level) / 2, axis=0)
                    coverage[det, j] += np.count_nonzero((truth >= lower) & (truth <= upper))
        n = len(validation) * width
        for det in range(2):
            rows.append(dict(
                detector=det + 1, mask_position=position_idx,
                start_sample=start, start_ns=float(time_ps[start] / 1000.),
                evaluation_events=len(validation), conditional_samples=draws,
                ppca_mean_rmse=float(np.sqrt(totals[det, 0] / n)),
                ppca_draw_rmse=float(np.sqrt(totals[det, 1] / n)),
                interpolation_rmse=float(np.sqrt(totals[det, 2] / n)),
                mlp_rmse=float(np.sqrt(totals[det, 3] / n)) if det in mlp_models else None,
                coverage_50=float(coverage[det, 0] / n),
                coverage_90=float(coverage[det, 1] / n),
            ))
        print(f"Position {position_idx+1}/{len(starts)} evaluated", flush=True)
    with (destination / "position_metrics.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    for d in range(2):
        matching = [r for r in rows if r["detector"] == d + 1]
        times = [r["start_ns"] for r in matching]
        for key, label in (
            ("ppca_mean_rmse", "PPCA conditional mean"),
            ("interpolation_rmse", "Interpolation"),
            ("mlp_rmse", "MLP"),
        ):
            if all(r[key] is not None for r in matching):
                axes[d].plot(times, [r[key] for r in matching], label=label)
        axes[d].set_ylabel(f"Detector {d+1} RMSE")
        axes[d].grid(alpha=.25)
        axes[d].legend()
    axes[-1].set_xlabel("Masked region start (ns)")
    fig.tight_layout()
    fig.savefig(destination / "rmse_by_position.png", dpi=180)
    fig.savefig(destination / "rmse_by_position.pdf")
    plt.close(fig)
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--channel", choices=("energy", "timing"), required=True)
    parser.add_argument("--cache-root", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--max-events", type=int, default=5000)
    parser.add_argument("--components", type=int, default=32)
    parser.add_argument("--mask-ns", type=float, default=1.)
    parser.add_argument("--mask-stride-samples", type=int)
    parser.add_argument("--eval-events", type=int, default=100)
    parser.add_argument("--draws", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val-fraction", type=float, default=.15)
    args = parser.parse_args()
    if (args.max_events < 10 or args.components < 1 or args.eval_events < 1
        or args.draws < 1 or args.mask_ns <= 0 or not 0 < args.val_fraction < 1
        or args.mask_stride_samples is not None and args.mask_stride_samples < 1):
        parser.error("Invalid parameter")
    results = root_relative(args.results)
    waves, time_ps, source, manifest = load_wide_development(
        results, args.channel, root_relative(args.cache_root))
    length = waves.shape[-1]
    dt_ps = float(np.median(np.diff(time_ps)))
    if dt_ps <= 0 or not np.allclose(np.diff(time_ps), dt_ps, rtol=1e-4, atol=1e-3):
        raise ValueError("Expected uniform time grid")
    width = max(1, round(args.mask_ns * 1000. / dt_ps))
    starts = positions(length, width, args.mask_stride_samples or width)
    selected = np.random.default_rng(args.seed).permutation(len(waves))[:args.max_events]
    n_val = max(1, round(len(selected) * args.val_fraction))
    validation, training = selected[:n_val], selected[n_val:]
    if args.components >= min(len(training), length):
        parser.error("--components must be below the number of training events and samples")
    destination = (root_relative(args.output) if args.output else
                   ROOT / "waveform_analysis/imputation_experiments" /
                   results.name / args.channel / "ppca_detector_specific")
    destination.mkdir(parents=True, exist_ok=True)
    print(f"Fitting PPCA on {len(training)} events per detector; "
          f"components={args.components}; full length={length}", flush=True)
    models = []
    for d in range(2):
        training_waves = np.asarray(waves[training, d, :], dtype=np.float64)
        model = ProbabilisticPCA(args.components).fit(training_waves)
        models.append(model)
        print(f"Detector {d+1}: residual variance={model.sigma2:.6g}, "
              f"PCA explained variance={model.explained_variance_ratio:.4f}", flush=True)
    mlp, status = load_mlp_models(
        results, args.channel, length, width, starts,
        manifest.get("analysis_protocol_identity")
    )
    evaluation = validation[:min(len(validation), args.eval_events)]
    rows = evaluate(models, waves, evaluation, starts, width, np.asarray(time_ps),
                    args.draws, mlp, destination, args.seed)
    np.savez_compressed(
        destination / "model.npz",
        mean=np.stack([m.mean for m in models]),
        loading=np.stack([m.W for m in models]),
        noise_variance=np.asarray([m.sigma2 for m in models]),
    )
    summary = dict(
        method="probabilistic_pca_isotropic_factor_model",
        dataset_identity=manifest.get("analysis_protocol_identity"),
        source_run=str(source), channel=args.channel,
        selected_events=len(selected), training_events=len(training),
        validation_events=len(validation), evaluation_events=len(evaluation),
        mask_samples=width, mask_positions=len(starts),
        components=args.components, draws=args.draws, seed=args.seed,
        detector_noise_variances=[float(m.sigma2) for m in models],
        detector_explained_variance_ratios=[m.explained_variance_ratio for m in models],
        mlp_status=status,
        note="Marginal predictive coverage from finite draws. Isotropic PPCA residual noise.",
    )
    (destination / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Finished: {destination}", flush=True)


if __name__ == "__main__":
    main()
