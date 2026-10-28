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

    def prepare_mask(self, start, width):
        """Cache the mask-specific posterior and predictive covariance factors."""
        stop = start + width
        if start < 0 or stop > len(self.mean):
            raise ValueError("Invalid mask")
        observed = np.ones(len(self.mean), dtype=bool)
        observed[start:stop] = False
        W_o, W_m = self.W[observed], self.W[start:stop]
        precision = np.eye(self.components) + W_o.T @ W_o / self.sigma2
        factor = np.linalg.cholesky(precision)
        # Missing mean = mu_M + (x_O - mu_O) @ coefficient_map
        # Solve with multiple RHS once, not once per event.
        coefficient_map = np.linalg.solve(
            factor.T, np.linalg.solve(factor, W_o.T / self.sigma2)
        ).T @ W_m.T
        # Integrate out z: x_M|x_O ~ N(mean_M, W_M Cov(z|x_O) W_M^T + sigma²I)
        latent_to_missing = np.linalg.solve(factor, W_m.T)
        covariance = latent_to_missing.T @ latent_to_missing
        covariance.flat[::width + 1] += self.sigma2
        sampling_factor = np.linalg.cholesky(covariance)
        return observed, coefficient_map, sampling_factor

    def condition_batch(self, waveforms, start, width, samples, rng, prepared=None):
        """Batch means and exact Gaussian conditional draws, shape [events, draws, width]."""
        if samples < 1:
            raise ValueError("samples must be >= 1")
        observed, coefficient_map, sampling_factor = (
            prepared if prepared is not None else self.prepare_mask(start, width)
        )
        batch = np.asarray(waveforms, dtype=np.float64)
        mean = self.mean[start:start + width] + (
            batch[:, observed] - self.mean[observed]
        ) @ coefficient_map
        random = rng.standard_normal((len(batch), samples, width))
        draws = mean[:, None, :] + random @ sampling_factor.T
        return mean, draws

    def condition(self, waveform, start, width, samples, rng):
        mean, draws = self.condition_batch(
            np.asarray(waveform)[None], start, width, samples, rng
        )
        return mean[0], draws[0]

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
             destination, seed, batch_size=128):
    rows = []
    rng = np.random.default_rng(seed + 567)
    validation = np.asarray(validation, dtype=np.int64)
    length = len(time_ps)
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    for position_idx, start in enumerate(starts):
        stop = start + width
        totals = np.zeros((2, 4), dtype=np.float64)
        coverage = np.zeros((2, 2), dtype=np.float64)
        for det in range(2):
            model = models[det]
            prepared = model.prepare_mask(start, width)
            observed = prepared[0]
            code = np.zeros(len(starts), dtype=np.float32)
            code[position_idx] = 1.
            for offset in range(0, len(validation), batch_size):
                ids = validation[offset:offset + batch_size]
                signals = np.asarray(waves[ids, det, :], dtype=np.float64)
                truth = signals[:, start:stop]
                mu, generated = model.condition_batch(
                    signals, start, width, draws, rng, prepared=prepared
                )
                totals[det, 0] += np.square(mu - truth).sum()
                totals[det, 1] += np.square(generated - truth[:, None, :]).sum() / draws
                # The gap is contiguous. np.interp uses constant extrapolation
                # for intervals at either boundary.
                sample_positions = np.arange(start, stop)[None, :]
                if start == 0:
                    interpolated = np.broadcast_to(signals[:, stop, None], truth.shape)
                elif stop == length:
                    interpolated = np.broadcast_to(signals[:, start - 1, None], truth.shape)
                else:
                    alpha = (sample_positions - (start - 1)) / (width + 1.)
                    interpolated = (
                        signals[:, start - 1, None] * (1. - alpha)
                        + signals[:, stop, None] * alpha
                    )
                totals[det, 2] += np.square(interpolated - truth).sum()
                if det in mlp_models:
                    context = np.concatenate(
                        (signals[:, :start], signals[:, stop:]), axis=1
                    ).astype(np.float32)
                    features = np.concatenate(
                        (context, np.broadcast_to(code[None], (len(ids), len(starts)))),
                        axis=1
                    )
                    prediction = mlp_models[det](
                        torch.from_numpy(features)
                    ).numpy()
                    totals[det, 3] += np.square(prediction - truth).sum()
                # Same empirical quantile definition as the previous evaluator.
                for j, level in enumerate((.5, .9)):
                    lower = np.quantile(generated, (1. - level) / 2., axis=1)
                    upper = np.quantile(generated, (1. + level) / 2., axis=1)
                    coverage[det, j] += np.count_nonzero(
                        (truth >= lower) & (truth <= upper)
                    )
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


def plot_imputation_examples(models, waves, validation, starts, width, time_ps,
                             destination, seed, n_events=3, n_draws=5,
                             boundary_strength=1.0, boundary_slope_points=5):
    """Show true, masked and PPCA-imputed waveforms at informative mask positions.

    Choose masks nearest 0 ns, 20 ns and a smoother late region. A reproducible
    subset of held-out events is used; only missing regions are sampled.
    """
    if n_events <= 0:
        return
    rng = np.random.default_rng(seed + 2027)
    chosen = np.asarray(validation[:min(n_events, len(validation))], dtype=np.int64)
    times = np.asarray(time_ps, dtype=np.float64) / 1000.0
    targets = (0.0, 20.0, 40.0)
    mask_indices = list(dict.fromkeys(
        min(range(len(starts)), key=lambda i: abs(times[starts[i]] - t))
        for t in targets
    ))
    out = destination / "imputation_examples"
    out.mkdir(parents=True, exist_ok=True)
    for det in range(2):
        model = models[det]
        for index in mask_indices:
            start = starts[index]
            stop = start + width
            prepared = model.prepare_mask(start, width)
            batch = np.asarray(waves[chosen, det, :], dtype=np.float64)
            mean, samples = model.condition_batch(
                batch, start, width, n_draws, rng, prepared=prepared
            )
            corrected_mean = boundary_correct_batch(
                mean, batch, start, width, boundary_strength, boundary_slope_points
            )
            repeated_batch = np.repeat(batch, n_draws, axis=0)
            corrected_samples = boundary_correct_batch(
                samples.reshape(-1, width), repeated_batch, start, width,
                boundary_strength, boundary_slope_points
            ).reshape(samples.shape)
            for j, event_id in enumerate(chosen):
                fig, axes = plt.subplots(2, 1, figsize=(11, 7), height_ratios=(1, 1.2))
                original = batch[j]
                masked = original.copy()
                masked[start:stop] = np.nan
                reconstructed = original.copy()
                reconstructed[start:stop] = mean[j]
                boundary_reconstructed = original.copy()
                boundary_reconstructed[start:stop] = corrected_mean[j]
                for ax, zoom in zip(axes, (False, True)):
                    if zoom:
                        left = max(0, start - 2 * width)
                        right = min(len(original), stop + 2 * width)
                    else:
                        left, right = 0, len(original)
                    segment = slice(left, right)
                    ax.plot(times[segment], original[segment], color="black",
                            lw=1.3, label="Ground truth")
                    ax.plot(times[segment], masked[segment], color="tab:gray",
                            lw=1, ls="--", label="Observed (masked)")
                    ax.plot(times[segment], reconstructed[segment], color="tab:blue",
                            lw=1.4, label="PPCA conditional mean")
                    ax.plot(times[segment], boundary_reconstructed[segment], color="tab:green",
                            lw=1.5, label="Boundary-corrected mean")
                    for k in range(n_draws):
                        imputed = original.copy()
                        imputed[start:stop] = samples[j, k]
                        ax.plot(times[segment], imputed[segment], color="tab:orange",
                                lw=.7, alpha=.22,
                                label="Raw conditional samples" if k == 0 else None)
                        imputed_corrected = original.copy()
                        imputed_corrected[start:stop] = corrected_samples[j, k]
                        ax.plot(times[segment], imputed_corrected[segment], color="tab:purple",
                                lw=.7, alpha=.22,
                                label="Corrected samples" if k == 0 else None)
                    ax.axvspan(times[start], times[stop - 1],
                               color="tab:red", alpha=.08, label="1 ns mask")
                    ax.set_ylabel("Normalized signal")
                    ax.grid(alpha=.2)
                axes[0].set_title(
                    f"Detector {det + 1}, held-out event {event_id}, "
                    f"mask at {times[start]:.2f} ns"
                )
                axes[0].legend(ncol=3, fontsize=8, loc="best")
                axes[1].set_xlabel("Time (ns)")
                fig.tight_layout()
                fig.savefig(
                    out / f"detector_{det+1}_event_{event_id}_position_{index:02d}.png",
                    dpi=160
                )
                plt.close(fig)
    print(f"Imputation examples: {out}", flush=True)

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
    parser.add_argument("--eval-batch-size", type=int, default=128)
    parser.add_argument("--plot-examples", type=int, default=3,
                        help="Held-out events to plot for each detector/region (0 disables)")
    parser.add_argument("--example-draws", type=int, default=5)
    parser.add_argument("--boundary-correction-strength", type=float, default=1.0,
                        help="Cubic boundary correction strength in [0, 1]; 0 disables")
    parser.add_argument("--boundary-slope-points", type=int, default=5,
                        help="Observed samples per side for local linear slope estimates")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val-fraction", type=float, default=.15)
    args = parser.parse_args()
    if (args.max_events < 10 or args.components < 1 or args.eval_events < 1
        or args.draws < 1 or args.eval_batch_size < 1 or args.plot_examples < 0
        or args.example_draws < 1 or args.mask_ns <= 0 or not 0 < args.val_fraction < 1
        or not 0 <= args.boundary_correction_strength <= 1
        or args.boundary_slope_points < 2
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
                    args.draws, mlp, destination, args.seed, args.eval_batch_size,
                    args.boundary_correction_strength, args.boundary_slope_points)
    if args.plot_examples:
        plot_imputation_examples(models, waves, evaluation, starts, width,
                                 time_ps, destination, args.seed,
                                 args.plot_examples, args.example_draws,
                                 args.boundary_correction_strength,
                                 args.boundary_slope_points)
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
        evaluation_batch_size=args.eval_batch_size,
        plot_examples=args.plot_examples, example_draws=args.example_draws,
        boundary_correction_strength=args.boundary_correction_strength,
        boundary_slope_points=args.boundary_slope_points,
        detector_noise_variances=[float(m.sigma2) for m in models],
        detector_explained_variance_ratios=[m.explained_variance_ratio for m in models],
        mlp_status=status,
        note="Marginal predictive coverage from finite draws. Isotropic PPCA residual noise.",
    )
    (destination / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Finished: {destination}", flush=True)


if __name__ == "__main__":
    main()
