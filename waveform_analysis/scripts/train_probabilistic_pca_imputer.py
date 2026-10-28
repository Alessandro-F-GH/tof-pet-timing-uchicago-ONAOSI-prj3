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

    def conditional_mean_batch(self, waveforms, start, width, prepared=None):
        """Compute conditional means without generating stochastic draws."""
        observed, coefficient_map, _ = (
            prepared if prepared is not None else self.prepare_mask(start, width)
        )
        batch = np.asarray(waveforms, dtype=np.float64)
        return self.mean[start:start + width] + (
            batch[:, observed] - self.mean[observed]
        ) @ coefficient_map

    def condition_batch(self, waveforms, start, width, samples, rng, prepared=None):
        """Batch means and exact Gaussian conditional draws, shape [events, draws, width]."""
        if samples < 1:
            raise ValueError("samples must be >= 1")
        prepared = prepared if prepared is not None else self.prepare_mask(start, width)
        _, _, sampling_factor = prepared
        batch = np.asarray(waveforms, dtype=np.float64)
        mean = self.conditional_mean_batch(batch, start, width, prepared=prepared)
        random = rng.standard_normal((len(batch), samples, width))
        draws = mean[:, None, :] + random @ sampling_factor.T
        return mean, draws

    def condition(self, waveform, start, width, samples, rng):
        mean, draws = self.condition_batch(
            np.asarray(waveform)[None], start, width, samples, rng
        )
        return mean[0], draws[0]


def estimate_boundary_slopes(signals, start, stop, points):
    """Estimate boundary slopes by least-squares fits to nearby observed samples."""
    signals = np.asarray(signals, dtype=np.float64)
    length = signals.shape[1]
    left_slope = right_slope = None
    if start >= 2:
        idx = np.arange(max(0, start - points), start)
        if len(idx) >= 2:
            centered = idx.astype(np.float64) - idx.mean()
            left_slope = (signals[:, idx] @ centered) / np.dot(centered, centered)
    if stop + 1 < length:
        idx = np.arange(stop, min(length, stop + points))
        if len(idx) >= 2:
            centered = idx.astype(np.float64) - idx.mean()
            right_slope = (signals[:, idx] @ centered) / np.dot(centered, centered)
    return left_slope, right_slope


def boundary_correct_batch(predictions, signals, start, width,
                           strength=1.0, slope_points=5):
    """Apply cubic Hermite correction matching extrapolated boundary values/slopes."""
    pred = np.asarray(predictions, dtype=np.float64)
    signals = np.asarray(signals, dtype=np.float64)
    if strength <= 0 or width < 3:
        return pred.copy()
    stop = start + width
    left_slope, right_slope = estimate_boundary_slopes(
        signals, start, stop, slope_points
    )
    d0 = np.zeros(len(pred), dtype=np.float64)
    d1 = np.zeros(len(pred), dtype=np.float64)
    m0 = np.zeros(len(pred), dtype=np.float64)
    m1 = np.zeros(len(pred), dtype=np.float64)
    if left_slope is not None:
        d0 = signals[:, start - 1] + left_slope - pred[:, 0]
        m0 = (width - 1) * (left_slope - (pred[:, 1] - pred[:, 0]))
    if right_slope is not None:
        d1 = signals[:, stop] - right_slope - pred[:, -1]
        m1 = (width - 1) * (right_slope - (pred[:, -1] - pred[:, -2]))

    u = np.linspace(0.0, 1.0, width, dtype=np.float64)[None, :]
    h00 = 2 * u**3 - 3 * u**2 + 1
    h10 = u**3 - 2 * u**2 + u
    h01 = -2 * u**3 + 3 * u**2
    h11 = u**3 - u**2
    correction = (
        d0[:, None] * h00 + m0[:, None] * h10
        + d1[:, None] * h01 + m1[:, None] * h11
    )
    return pred + float(strength) * correction


def boundary_slope_error(predictions, signals, start, width, slope_points=5):
    """Squared mismatch of imputed boundary-crossing slopes, and count."""
    pred = np.asarray(predictions, dtype=np.float64)
    signals = np.asarray(signals, dtype=np.float64)
    stop = start + width
    left_slope, right_slope = estimate_boundary_slopes(
        signals, start, stop, slope_points
    )
    sse = 0.0
    count = 0
    if left_slope is not None:
        error = (pred[:, 0] - signals[:, start - 1]) - left_slope
        sse += float(np.square(error).sum())
        count += len(pred)
    if right_slope is not None:
        error = (signals[:, stop] - pred[:, -1]) - right_slope
        sse += float(np.square(error).sum())
        count += len(pred)
    return sse, count


def optimize_boundary_strengths(models, waves, tuning, starts, width,
                                continuity_weight, slope_points, batch_size):
    """Analytically optimize continuous correction strength per detector/mask.

    The objective is masked MSE plus a weighted mean squared boundary-slope
    mismatch. Both terms are quadratic in strength, so the constrained optimum
    on [0, 1] is available in closed form without grid search.
    """
    tuning = np.asarray(tuning, dtype=np.int64)
    strengths = np.zeros((2, len(starts)), dtype=np.float64)
    rows = []
    for det, model in enumerate(models):
        print(f"Optimizing boundary strengths: detector {det + 1}", flush=True)
        for position_idx, start in enumerate(starts):
            stop = start + width
            prepared = model.prepare_mask(start, width)
            sse_base = correction_sq = correction_cross = 0.0
            slope_base_sq = slope_correction_sq = slope_correction_cross = 0.0
            slope_count = 0
            for offset in range(0, len(tuning), batch_size):
                ids = tuning[offset:offset + batch_size]
                signals = np.asarray(waves[ids, det, :], dtype=np.float64)
                truth = signals[:, start:stop]
                mean = model.conditional_mean_batch(signals, start, width, prepared)
                corrected_at_one = boundary_correct_batch(
                    mean, signals, start, width, 1.0, slope_points
                )
                correction = corrected_at_one - mean
                error = mean - truth
                sse_base += float(np.square(error).sum())
                correction_sq += float(np.square(correction).sum())
                correction_cross += float((error * correction).sum())
                left_slope, right_slope = estimate_boundary_slopes(
                    signals, start, stop, slope_points
                )
                if left_slope is not None:
                    base_error = (mean[:, 0] - signals[:, start - 1]) - left_slope
                    delta = correction[:, 0]
                    slope_base_sq += float(np.square(base_error).sum())
                    slope_correction_sq += float(np.square(delta).sum())
                    slope_correction_cross += float((base_error * delta).sum())
                    slope_count += len(signals)
                if right_slope is not None:
                    base_error = (signals[:, stop] - mean[:, -1]) - right_slope
                    delta = -correction[:, -1]
                    slope_base_sq += float(np.square(base_error).sum())
                    slope_correction_sq += float(np.square(delta).sum())
                    slope_correction_cross += float((base_error * delta).sum())
                    slope_count += len(signals)

            n_values = max(len(tuning) * width, 1)
            base_mse = sse_base / n_values
            a_mse = correction_sq / n_values
            b_mse = correction_cross / n_values
            n_slopes = max(slope_count, 1)
            base_slope_mse = slope_base_sq / n_slopes
            a_slope = slope_correction_sq / n_slopes
            b_slope = slope_correction_cross / n_slopes
            denominator = a_mse + continuity_weight * a_slope
            numerator = b_mse + continuity_weight * b_slope
            strength = 0.0 if denominator <= 1e-18 else float(
                np.clip(-numerator / denominator, 0.0, 1.0)
            )
            strengths[det, position_idx] = strength
            corrected_mse = max(base_mse + 2 * strength * b_mse + strength**2 * a_mse, 0.0)
            corrected_slope_mse = max(
                base_slope_mse + 2 * strength * b_slope + strength**2 * a_slope, 0.0
            )
            rows.append(dict(
                detector=det + 1, mask_position=position_idx, start_sample=start,
                tuning_events=len(tuning), correction_strength=strength,
                tuning_ppca_rmse=float(np.sqrt(max(base_mse, 0.0))),
                tuning_corrected_rmse=float(np.sqrt(corrected_mse)),
                tuning_ppca_slope_mismatch_rmse=float(np.sqrt(max(base_slope_mse, 0.0))),
                tuning_corrected_slope_mismatch_rmse=float(np.sqrt(corrected_slope_mse)),
                continuity_weight=continuity_weight,
                objective_before=float(base_mse + continuity_weight * base_slope_mse),
                objective_after=float(corrected_mse + continuity_weight * corrected_slope_mse),
            ))
            if position_idx % 10 == 0 or position_idx == len(starts) - 1:
                print(f"Detector {det + 1}: section {position_idx + 1}/{len(starts)} "
                      f"lambda={strength:.3f}", flush=True)
    return strengths, rows

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
             destination, seed, batch_size=128, boundary_strengths=None,
             boundary_slope_points=5):
    rows = []
    rng = np.random.default_rng(seed + 567)
    validation = np.asarray(validation, dtype=np.int64)
    length = len(time_ps)
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    if boundary_strengths is None:
        boundary_strengths = np.ones((2, len(starts)), dtype=np.float64)
    if np.shape(boundary_strengths) != (2, len(starts)):
        raise ValueError("boundary_strengths must have shape [2, mask_positions]")
    for position_idx, start in enumerate(starts):
        stop = start + width
        # raw mean, raw draws, interpolation, MLP, corrected mean, corrected draws
        totals = np.zeros((2, 6), dtype=np.float64)
        coverage = np.zeros((2, 2, 2), dtype=np.float64)
        slope_sse = np.zeros((2, 2), dtype=np.float64)
        slope_count = np.zeros((2, 2), dtype=np.float64)
        for det in range(2):
            model = models[det]
            prepared = model.prepare_mask(start, width)
            strength = float(boundary_strengths[det, position_idx])
            code = np.zeros(len(starts), dtype=np.float32)
            code[position_idx] = 1.
            for offset in range(0, len(validation), batch_size):
                ids = validation[offset:offset + batch_size]
                signals = np.asarray(waves[ids, det, :], dtype=np.float64)
                truth = signals[:, start:stop]
                mu, generated = model.condition_batch(
                    signals, start, width, draws, rng, prepared=prepared
                )
                corrected_mu = boundary_correct_batch(
                    mu, signals, start, width, strength, boundary_slope_points
                )
                repeated_signals = np.repeat(signals, draws, axis=0)
                corrected_generated = boundary_correct_batch(
                    generated.reshape(-1, width), repeated_signals, start, width,
                    strength, boundary_slope_points
                ).reshape(generated.shape)

                totals[det, 0] += np.square(mu - truth).sum()
                totals[det, 1] += np.square(generated - truth[:, None, :]).sum() / draws
                totals[det, 4] += np.square(corrected_mu - truth).sum()
                totals[det, 5] += np.square(corrected_generated - truth[:, None, :]).sum() / draws

                if start == 0:
                    interpolated = np.broadcast_to(signals[:, stop, None], truth.shape)
                elif stop == length:
                    interpolated = np.broadcast_to(signals[:, start - 1, None], truth.shape)
                else:
                    alpha = (np.arange(start, stop) - (start - 1)) / (width + 1.)
                    interpolated = (
                        signals[:, start - 1, None] * (1. - alpha[None, :])
                        + signals[:, stop, None] * alpha[None, :]
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
                    prediction = mlp_models[det](torch.from_numpy(features)).numpy()
                    totals[det, 3] += np.square(prediction - truth).sum()

                for variant, samples_array in enumerate((generated, corrected_generated)):
                    for j, level in enumerate((.5, .9)):
                        lower = np.quantile(samples_array, (1. - level) / 2., axis=1)
                        upper = np.quantile(samples_array, (1. + level) / 2., axis=1)
                        coverage[det, variant, j] += np.count_nonzero(
                            (truth >= lower) & (truth <= upper)
                        )
                for variant, prediction in enumerate((mu, corrected_mu)):
                    sse, count = boundary_slope_error(
                        prediction, signals, start, width, boundary_slope_points
                    )
                    slope_sse[det, variant] += sse
                    slope_count[det, variant] += count

        n = len(validation) * width
        for det in range(2):
            rows.append(dict(
                detector=det + 1, mask_position=position_idx,
                start_sample=start, start_ns=float(time_ps[start] / 1000.),
                evaluation_events=len(validation), conditional_samples=draws,
                boundary_correction_strength=float(boundary_strengths[det, position_idx]),
                ppca_mean_rmse=float(np.sqrt(totals[det, 0] / n)),
                ppca_draw_rmse=float(np.sqrt(totals[det, 1] / n)),
                boundary_corrected_mean_rmse=float(np.sqrt(totals[det, 4] / n)),
                boundary_corrected_draw_rmse=float(np.sqrt(totals[det, 5] / n)),
                interpolation_rmse=float(np.sqrt(totals[det, 2] / n)),
                mlp_rmse=float(np.sqrt(totals[det, 3] / n)) if det in mlp_models else None,
                ppca_slope_mismatch_rmse=float(np.sqrt(slope_sse[det, 0] / max(slope_count[det, 0], 1))),
                boundary_corrected_slope_mismatch_rmse=float(np.sqrt(slope_sse[det, 1] / max(slope_count[det, 1], 1))),
                coverage_50=float(coverage[det, 0, 0] / n),
                coverage_90=float(coverage[det, 0, 1] / n),
                boundary_coverage_50=float(coverage[det, 1, 0] / n),
                boundary_coverage_90=float(coverage[det, 1, 1] / n),
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
            ("boundary_corrected_mean_rmse", "Boundary-corrected PPCA"),
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
                             boundary_strengths=None, boundary_slope_points=5):
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
    if boundary_strengths is None:
        boundary_strengths = np.ones((2, len(starts)), dtype=np.float64)
    out = destination / "imputation_examples"
    out.mkdir(parents=True, exist_ok=True)
    for det in range(2):
        model = models[det]
        for index in mask_indices:
            start = starts[index]
            stop = start + width
            prepared = model.prepare_mask(start, width)
            strength = float(boundary_strengths[det, index])
            batch = np.asarray(waves[chosen, det, :], dtype=np.float64)
            mean, samples = model.condition_batch(
                batch, start, width, n_draws, rng, prepared=prepared
            )
            corrected_mean = boundary_correct_batch(
                mean, batch, start, width, strength, boundary_slope_points
            )
            repeated_batch = np.repeat(batch, n_draws, axis=0)
            corrected_samples = boundary_correct_batch(
                samples.reshape(-1, width), repeated_batch, start, width,
                strength, boundary_slope_points
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
                    f"mask at {times[start]:.2f} ns, lambda={strength:.2f}"
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
    parser.add_argument("--tuning-events", type=int, default=500,
                        help="Calibration events used to optimize correction strengths")
    parser.add_argument("--continuity-weight", type=float, default=1.0,
                        help="Weight of boundary-slope mismatch in the analytic objective")
    parser.add_argument("--boundary-slope-points", type=int, default=5,
                        help="Observed samples per side for local linear slope estimates")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val-fraction", type=float, default=.15)
    args = parser.parse_args()
    if (args.max_events < 10 or args.components < 1 or args.eval_events < 1
        or args.draws < 1 or args.eval_batch_size < 1 or args.plot_examples < 0
        or args.example_draws < 1 or args.mask_ns <= 0 or not 0 < args.val_fraction < 1
        or args.tuning_events < 1 or args.continuity_weight < 0
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
                   results.name / args.channel / "ppca_adaptive_boundary")
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
    if len(validation) < 2:
        parser.error("Need at least two validation events for separate tuning and evaluation")
    n_tuning = min(args.tuning_events, max(1, len(validation) // 2))
    tuning = validation[:n_tuning]
    evaluation = validation[n_tuning:n_tuning + args.eval_events]
    if len(evaluation) == 0:
        parser.error("No held-out evaluation events remain after strength tuning")
    print(f"Correction tuning: {len(tuning)} events; independent evaluation: "
          f"{len(evaluation)} events; continuity weight={args.continuity_weight}", flush=True)
    strengths, tuning_rows = optimize_boundary_strengths(
        models, waves, tuning, starts, width, args.continuity_weight,
        args.boundary_slope_points, args.eval_batch_size
    )
    for row in tuning_rows:
        row["start_ns"] = float(time_ps[row["start_sample"]] / 1000.0)
    with (destination / "optimized_boundary_strengths.csv").open(
        "w", newline="", encoding="utf-8"
    ) as f:
        writer = csv.DictWriter(f, fieldnames=list(tuning_rows[0]))
        writer.writeheader()
        writer.writerows(tuning_rows)
    strength_fig, strength_ax = plt.subplots(figsize=(10, 4))
    for det in range(2):
        strength_ax.plot(np.asarray(time_ps)[starts] / 1000.0, strengths[det],
                         marker=".", ms=4, label=f"Detector {det + 1}")
    strength_ax.set_xlabel("Masked region start (ns)")
    strength_ax.set_ylabel("Optimized correction strength")
    strength_ax.set_ylim(-0.03, 1.03)
    strength_ax.grid(alpha=.25)
    strength_ax.legend()
    strength_fig.tight_layout()
    strength_fig.savefig(destination / "optimized_boundary_strengths.png", dpi=180)
    plt.close(strength_fig)

    rows = evaluate(models, waves, evaluation, starts, width, np.asarray(time_ps),
                    args.draws, mlp, destination, args.seed, args.eval_batch_size,
                    strengths, args.boundary_slope_points)
    if args.plot_examples:
        plot_imputation_examples(models, waves, evaluation, starts, width,
                                 time_ps, destination, args.seed,
                                 args.plot_examples, args.example_draws,
                                 strengths, args.boundary_slope_points)
    np.savez_compressed(
        destination / "model.npz",
        mean=np.stack([m.mean for m in models]),
        loading=np.stack([m.W for m in models]),
        noise_variance=np.asarray([m.sigma2 for m in models]),
    )
    summary = dict(
        method="ppca_analytic_position_specific_boundary_correction",
        dataset_identity=manifest.get("analysis_protocol_identity"),
        source_run=str(source), channel=args.channel,
        selected_events=len(selected), training_events=len(training),
        validation_events=len(validation), tuning_events=len(tuning),
        evaluation_events=len(evaluation),
        mask_samples=width, mask_positions=len(starts),
        components=args.components, draws=args.draws, seed=args.seed,
        evaluation_batch_size=args.eval_batch_size,
        plot_examples=args.plot_examples, example_draws=args.example_draws,
        continuity_weight=args.continuity_weight,
        boundary_slope_points=args.boundary_slope_points,
        optimized_boundary_strengths=strengths.tolist(),
        detector_noise_variances=[float(m.sigma2) for m in models],
        detector_explained_variance_ratios=[m.explained_variance_ratio for m in models],
        mlp_status=status,
        note="Correction strengths optimized continuously and independently per detector/mask on calibration events using masked MSE plus weighted boundary-slope mismatch; final metrics use disjoint evaluation events. Isotropic PPCA residual noise.",
    )
    (destination / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Finished: {destination}", flush=True)


if __name__ == "__main__":
    main()
