#!/usr/bin/env python3
"""Train the standard PyPOTS GP-VAE for waveform imputation.

The full temporal span is retained, but waveforms are block-averaged by a
configurable factor before GP-VAE training. This is important because the
standard GP-VAE uses dense time-covariance operations. For a 1 ns mask and
factor 10 on 12.5 ps data, the model sees 8 steps per mask.

The script compares GP-VAE with linear interpolation on the same held-out
events. It reports metrics both at the model's coarse resolution and after
linear upsampling the GP-VAE reconstruction to the original sample grid.
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

from waveform_analysis.scripts.train_probabilistic_pca_imputer import positions
from waveform_analysis.scripts.train_ppca_interpolation_blend import (
    linear_interpolation_batch,
)
from waveform_analysis.scripts.train_wide_unet_imputer_detector_specific import (
    load_wide_development,
    root_relative,
)


def block_average(signals, factor):
    """Average consecutive blocks without crossing aligned mask boundaries."""
    signals = np.asarray(signals, dtype=np.float64)
    n_samples, length = signals.shape
    n_blocks = (length + factor - 1) // factor
    output = np.empty((n_samples, n_blocks), dtype=np.float64)
    for j, left in enumerate(range(0, length, factor)):
        output[:, j] = signals[:, left:min(left + factor, length)].mean(axis=1)
    return output


def block_time_axis(time_ps, factor):
    time_ps = np.asarray(time_ps, dtype=np.float64)
    return np.asarray([
        time_ps[left:min(left + factor, len(time_ps))].mean()
        for left in range(0, len(time_ps), factor)
    ])


def mask_random_blocks(data, valid_starts, width, rng):
    """Create one randomly positioned contiguous missing block per event."""
    masked = np.asarray(data, dtype=np.float32).copy()
    chosen = rng.choice(np.asarray(valid_starts, dtype=np.int64), size=len(masked))
    for i, start in enumerate(chosen):
        masked[i, start:start + width] = np.nan
    return masked, chosen


def extract_imputation(model, masked):
    result = model.predict({"X": np.asarray(masked, dtype=np.float32)[..., None]})
    prediction = np.asarray(result["imputation"])
    if prediction.ndim == 4:
        prediction = prediction.mean(axis=1)
    if prediction.ndim != 3 or prediction.shape[-1] != 1:
        raise RuntimeError(f"Unexpected GP-VAE output shape: {prediction.shape}")
    return prediction[..., 0]


def read_comparison_metrics(path):
    if path is None:
        return {}
    mapping = {}
    with Path(path).open("r", newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            value = row.get("blend_rmse")
            if value not in (None, ""):
                mapping[(int(row["detector"]), int(row["mask_position"]))] = float(value)
    return mapping


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--channel", choices=("energy", "timing"), required=True)
    parser.add_argument("--cache-root", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--max-events", type=int, default=1000)
    parser.add_argument("--eval-events", type=int, default=100)
    parser.add_argument("--mask-ns", type=float, default=1.0)
    parser.add_argument("--mask-stride-samples", type=int)
    parser.add_argument("--downsample-factor", type=int, default=10,
                        help="Block-average this many original samples into one GP-VAE step")
    parser.add_argument("--eval-position-stride", type=int, default=4,
                        help="Evaluate every Nth mask position; use 1 for all positions")
    parser.add_argument("--latent-size", type=int, default=8)
    parser.add_argument("--hidden-size", type=int, default=32)
    parser.add_argument("--kernel", choices=("cauchy", "diffusion", "rbf", "matern"),
                        default="cauchy")
    parser.add_argument("--length-scale", type=float, default=7.0)
    parser.add_argument("--window-size", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--patience", type=int, default=4)
    parser.add_argument("--plot-examples", type=int, default=3)
    parser.add_argument("--comparison-metrics", type=Path,
                        help="Optional position_metrics.csv from the PPCA/interpolation blend experiment")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--normalization", choices=("scale_only", "zscore"), default="scale_only",
                        help="scale_only preserves the positive signal baseline so zero-filled missing blocks are distinguishable")
    parser.add_argument("--val-fraction", type=float, default=0.15)
    args = parser.parse_args()

    if (
        args.max_events < 30 or args.eval_events < 1 or args.mask_ns <= 0
        or args.downsample_factor < 1 or args.eval_position_stride < 1
        or args.latent_size < 1 or args.hidden_size < 1 or args.batch_size < 1
        or args.epochs < 1 or args.patience < 1 or args.plot_examples < 0
        or args.window_size < 1 or args.length_scale <= 0
        or not 0 < args.val_fraction < 0.5
        or (args.mask_stride_samples is not None and args.mask_stride_samples < 1)
    ):
        parser.error("Invalid parameter")

    try:
        from pypots.imputation import GPVAE
    except ImportError as exc:
        raise SystemExit(
            "PyPOTS GP-VAE is required. Install it in the active environment with: pip install pypots"
        ) from exc

    import torch

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
        device = "cuda"
    else:
        device = "cpu"
    torch.set_num_threads(min(4, torch.get_num_threads()))

    results = root_relative(args.results)
    waves, time_ps, source, manifest = load_wide_development(
        results, args.channel, root_relative(args.cache_root)
    )
    waves = np.asarray(waves, dtype=np.float32)
    time_ps = np.asarray(time_ps, dtype=np.float64)
    length = waves.shape[-1]
    dt_ps = float(np.median(np.diff(time_ps)))
    if dt_ps <= 0 or not np.allclose(np.diff(time_ps), dt_ps, rtol=1e-4, atol=1e-3):
        raise ValueError("Expected a uniformly sampled time grid")

    mask_width = max(1, round(args.mask_ns * 1000.0 / dt_ps))
    original_stride = args.mask_stride_samples or mask_width
    if mask_width % args.downsample_factor != 0:
        parser.error(
            f"Mask width ({mask_width} samples) must be divisible by downsample factor "
            f"({args.downsample_factor}); use factor 10 for the standard 80-sample mask"
        )
    if original_stride % args.downsample_factor != 0:
        parser.error("Mask stride must be divisible by --downsample-factor")

    all_starts = positions(length, mask_width, original_stride)
    aligned_positions = [
        (idx, start) for idx, start in enumerate(all_starts)
        if start % args.downsample_factor == 0
    ]
    skipped = len(all_starts) - len(aligned_positions)
    if not aligned_positions:
        parser.error("No mask positions align with the block-averaging grid")
    coarse_width = mask_width // args.downsample_factor
    coarse_starts = [start // args.downsample_factor for _, start in aligned_positions]

    rng = np.random.default_rng(args.seed)
    selected = rng.permutation(len(waves))[:args.max_events]
    if len(selected) < 30:
        parser.error("At least 30 events must be available")
    n_val = max(2, round(len(selected) * args.val_fraction))
    n_eval_pool = max(2, round(len(selected) * args.val_fraction))
    if n_val + n_eval_pool >= len(selected):
        parser.error("Not enough events for train/validation/evaluation splits")
    training_ids = selected[n_val + n_eval_pool:]
    validation_ids = selected[:n_val]
    evaluation_pool = selected[n_val:n_val + n_eval_pool]
    evaluation_ids = evaluation_pool[:min(args.eval_events, len(evaluation_pool))]

    destination = (
        root_relative(args.output) if args.output else
        ROOT / "waveform_analysis" / "imputation_experiments"
        / results.name / args.channel / "gpvae"
    )
    destination.mkdir(parents=True, exist_ok=True)
    comparison = read_comparison_metrics(root_relative(args.comparison_metrics))

    coarse_all = block_average(waves.reshape(-1, length), args.downsample_factor)
    coarse_all = coarse_all.reshape(len(waves), 2, -1)
    coarse_time_ps = block_time_axis(time_ps, args.downsample_factor)
    coarse_length = coarse_all.shape[-1]
    eval_positions = [
        item for i, item in enumerate(aligned_positions)
        if i % args.eval_position_stride == 0
    ]
    if not eval_positions:
        eval_positions = [aligned_positions[0]]

    print(
        f"GP-VAE via PyPOTS; device={device}; channel={args.channel}; "
        f"selected={len(selected)} train={len(training_ids)} "
        f"validation={len(validation_ids)} evaluation={len(evaluation_ids)}; "
        f"original length={length}; coarse length={coarse_length}; "
        f"normalization={args.normalization}; downsample factor={args.downsample_factor}; mask={mask_width} samples "
        f"({coarse_width} GP-VAE steps); aligned positions={len(aligned_positions)}; "
        f"evaluated positions={len(eval_positions)}; skipped unaligned positions={skipped}",
        flush=True,
    )

    all_rows = []
    model_configs = []
    for det in range(2):
        print(f"\nDetector {det + 1}: preparing data", flush=True)
        coarse = coarse_all[:, det, :]
        # PyPOTS GP-VAE encodes X but does not pass missing_mask into the encoder.
        # NaNs are replaced internally with zero, so z-scoring makes the missing
        # marker coincide with the signal mean. By default preserve the positive
        # waveform baseline and scale only, leaving zero as an out-of-range marker.
        scale = float(coarse[training_ids].std())
        if not np.isfinite(scale) or scale < 1e-8:
            raise ValueError(f"Invalid normalization scale for detector {det + 1}")
        center = float(coarse[training_ids].mean()) if args.normalization == "zscore" else 0.0
        normalized = ((coarse - center) / scale).astype(np.float32)

        train_rng = np.random.default_rng(args.seed + 100 + det)
        val_rng = np.random.default_rng(args.seed + 200 + det)
        train_masked, train_mask_starts = mask_random_blocks(
            normalized[training_ids], coarse_starts, coarse_width, train_rng
        )
        val_masked, val_mask_starts = mask_random_blocks(
            normalized[validation_ids], coarse_starts, coarse_width, val_rng
        )
        val_full = normalized[validation_ids].astype(np.float32)[..., None]

        model = GPVAE(
            n_steps=coarse_length,
            n_features=1,
            latent_size=args.latent_size,
            encoder_sizes=(args.hidden_size, args.hidden_size),
            decoder_sizes=(args.hidden_size, args.hidden_size),
            kernel=args.kernel,
            beta=0.2,
            M=1,
            K=1,
            sigma=1.0,
            length_scale=args.length_scale,
            kernel_scales=1,
            window_size=args.window_size,
            batch_size=args.batch_size,
            epochs=args.epochs,
            patience=args.patience,
            num_workers=0,
            device=device,
            saving_path=None,
            model_saving_strategy=None,
            verbose=True,
        )
        print(
            f"Detector {det + 1}: training GP-VAE on {len(training_ids)} events; "
            f"latent={args.latent_size}, hidden={args.hidden_size}, "
            f"batch={args.batch_size}, epochs<={args.epochs}, patience={args.patience}",
            flush=True,
        )
        model.fit(
            {"X": train_masked[..., None]},
            {
                "X": val_masked[..., None],
                "X_ori": val_full,
            },
        )
        torch.save(
            model.model.state_dict(),
            destination / f"detector_{det + 1}_state_dict.pt",
        )
        model_configs.append({
            "detector": det + 1,
            "normalization_mean": center,
            "normalization_std": scale,
            "latent_size": args.latent_size,
            "hidden_size": args.hidden_size,
            "kernel": args.kernel,
            "length_scale": args.length_scale,
            "window_size": args.window_size,
            "coarse_length": coarse_length,
        })

        original_eval = waves[evaluation_ids, det, :].astype(np.float64)
        coarse_eval = coarse[evaluation_ids]
        for local_pos_idx, (position_idx, start) in enumerate(eval_positions):
            coarse_start = start // args.downsample_factor
            coarse_stop = coarse_start + coarse_width
            masked = normalized[evaluation_ids].copy()
            masked[:, coarse_start:coarse_stop] = np.nan
            imputed_normalized = extract_imputation(model, masked)
            imputed_coarse = imputed_normalized * scale + center

            truth_coarse = coarse_eval[:, coarse_start:coarse_stop]
            gpvae_coarse = imputed_coarse[:, coarse_start:coarse_stop]
            interp_coarse = linear_interpolation_batch(
                coarse_eval, coarse_start, coarse_width
            )
            gpvae_coarse_rmse = float(np.sqrt(np.mean((gpvae_coarse - truth_coarse) ** 2)))
            interp_coarse_rmse = float(np.sqrt(np.mean((interp_coarse - truth_coarse) ** 2)))

            gpvae_full = np.stack([
                np.interp(time_ps, coarse_time_ps, row) for row in imputed_coarse
            ])
            interp_full = linear_interpolation_batch(original_eval, start, mask_width)
            truth_full = original_eval[:, start:start + mask_width]
            gpvae_full_segment = gpvae_full[:, start:start + mask_width]
            gpvae_full_rmse = float(np.sqrt(np.mean((gpvae_full_segment - truth_full) ** 2)))
            interp_full_rmse = float(np.sqrt(np.mean((interp_full - truth_full) ** 2)))
            blend_rmse = comparison.get((det + 1, position_idx))

            row = dict(
                detector=det + 1,
                mask_position=position_idx,
                start_sample=start,
                start_ns=float(time_ps[start] / 1000.0),
                coarse_start=coarse_start,
                evaluation_events=len(evaluation_ids),
                downsample_factor=args.downsample_factor,
                gpvae_coarse_rmse=gpvae_coarse_rmse,
                interpolation_coarse_rmse=interp_coarse_rmse,
                gpvae_original_grid_rmse=gpvae_full_rmse,
                interpolation_original_grid_rmse=interp_full_rmse,
                blend_original_grid_rmse=blend_rmse,
                gpvae_minus_interpolation_original_grid=gpvae_full_rmse - interp_full_rmse,
                gpvae_minus_blend_original_grid=(
                    gpvae_full_rmse - blend_rmse if blend_rmse is not None else None
                ),
            )
            all_rows.append(row)
            print(
                f"Detector {det + 1}: mask {local_pos_idx + 1}/{len(eval_positions)} "
                f"at {row['start_ns']:.2f} ns; "
                f"coarse RMSE GP-VAE={gpvae_coarse_rmse:.5g}, "
                f"interp={interp_coarse_rmse:.5g}; "
                f"original-grid RMSE GP-VAE={gpvae_full_rmse:.5g}, "
                f"interp={interp_full_rmse:.5g}",
                flush=True,
            )

            if args.plot_examples and local_pos_idx in {
                0, len(eval_positions) // 2, len(eval_positions) - 1
            }:
                plot_dir = destination / "imputation_examples"
                plot_dir.mkdir(parents=True, exist_ok=True)
                count = min(args.plot_examples, len(evaluation_ids))
                event_rng = np.random.default_rng(args.seed + 3000 + det * 100 + position_idx)
                selected_local = event_rng.choice(len(evaluation_ids), size=count, replace=False)
                for local_event in selected_local:
                    original = original_eval[local_event]
                    interp_segment = interp_full[local_event]
                    gpvae_segment = gpvae_full_segment[local_event]
                    observed = original.copy()
                    observed[start:start + mask_width] = np.nan
                    interp_reconstruction = original.copy()
                    interp_reconstruction[start:start + mask_width] = interp_segment
                    gpvae_reconstruction = original.copy()
                    gpvae_reconstruction[start:start + mask_width] = gpvae_segment
                    fig, axes = plt.subplots(2, 1, figsize=(11, 7), height_ratios=(1, 1.2))
                    for ax, zoom in zip(axes, (False, True)):
                        ax.plot(time_ps / 1000.0, original, color="black", lw=1.0, label="Ground truth")
                        ax.plot(time_ps / 1000.0, observed, color="gray", lw=1.0, ls="--",
                                label="Observed (masked)")
                        ax.plot(time_ps / 1000.0, interp_reconstruction, lw=1.2,
                                label="Linear interpolation")
                        ax.plot(time_ps / 1000.0, gpvae_reconstruction, lw=1.3,
                                label="GP-VAE (coarse, upsampled)")
                        ax.axvspan(time_ps[start] / 1000.0,
                                   time_ps[start + mask_width - 1] / 1000.0,
                                   color="tab:red", alpha=0.08)
                        ax.grid(alpha=0.25)
                        ax.set_ylabel("Signal")
                        ax.legend(fontsize="small", ncol=2)
                        if zoom:
                            margin_ps = 3000.0
                            left = max(time_ps[0], time_ps[start] - margin_ps)
                            right = min(time_ps[-1], time_ps[start + mask_width - 1] + margin_ps)
                            ax.set_xlim(left / 1000.0, right / 1000.0)
                            ax.set_xlabel("Time (ns)")
                    fig.suptitle(
                        f"Detector {det + 1}, event {evaluation_ids[local_event]}, "
                        f"mask at {time_ps[start] / 1000.0:.2f} ns"
                    )
                    fig.tight_layout()
                    fig.savefig(
                        plot_dir / f"detector_{det + 1}_position_{position_idx:03d}_event_{evaluation_ids[local_event]}.png",
                        dpi=160,
                    )
                    plt.close(fig)

    csv_path = destination / "position_metrics.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(all_rows[0]))
        writer.writeheader()
        writer.writerows(all_rows)

    fig, axes = plt.subplots(2, 1, figsize=(11, 8), sharex=True)
    for det in range(2):
        subset = [r for r in all_rows if r["detector"] == det + 1]
        x = [r["start_ns"] for r in subset]
        axes[det].plot(x, [r["gpvae_original_grid_rmse"] for r in subset],
                       marker="o", ms=3, label="GP-VAE, upsampled")
        axes[det].plot(x, [r["interpolation_original_grid_rmse"] for r in subset],
                       marker=".", ms=3, label="Linear interpolation")
        if any(r["blend_original_grid_rmse"] is not None for r in subset):
            axes[det].plot(x, [r["blend_original_grid_rmse"] for r in subset],
                           linestyle="--", label="PPCA/interpolation blend")
        axes[det].set_ylabel(f"Detector {det + 1} RMSE")
        axes[det].grid(alpha=0.25)
        axes[det].legend(fontsize="small")
    axes[-1].set_xlabel("Masked region start (ns)")
    fig.tight_layout()
    fig.savefig(destination / "rmse_by_position.png", dpi=180)
    fig.savefig(destination / "rmse_by_position.pdf")
    plt.close(fig)

    fig, axes = plt.subplots(2, 1, figsize=(11, 8), sharex=True)
    for det in range(2):
        subset = [r for r in all_rows if r["detector"] == det + 1]
        x = [r["start_ns"] for r in subset]
        axes[det].plot(x, [r["gpvae_coarse_rmse"] for r in subset],
                       marker="o", ms=3, label="GP-VAE at coarse resolution")
        axes[det].plot(x, [r["interpolation_coarse_rmse"] for r in subset],
                       marker=".", ms=3, label="Interpolation at coarse resolution")
        axes[det].set_ylabel(f"Detector {det + 1} RMSE")
        axes[det].grid(alpha=0.25)
        axes[det].legend(fontsize="small")
    axes[-1].set_xlabel("Masked region start (ns)")
    fig.tight_layout()
    fig.savefig(destination / "rmse_coarse_grid.png", dpi=180)
    plt.close(fig)

    summary = dict(
        method="PyPOTS standard GPVAE",
        implementation="pypots.imputation.GPVAE",
        dataset_identity=manifest.get("analysis_protocol_identity"),
        source_run=str(source),
        channel=args.channel,
        selected_events=len(selected),
        training_events=len(training_ids),
        validation_events=len(validation_ids),
        evaluation_events=len(evaluation_ids),
        original_waveform_length=length,
        coarse_waveform_length=coarse_length,
        downsample_factor=args.downsample_factor,
        mask_width_original_samples=mask_width,
        mask_width_model_steps=coarse_width,
        aligned_mask_positions=len(aligned_positions),
        evaluated_mask_positions=len(eval_positions),
        skipped_unaligned_mask_positions=skipped,
        latent_size=args.latent_size,
        hidden_size=args.hidden_size,
        kernel=args.kernel,
        length_scale=args.length_scale,
        window_size=args.window_size,
        batch_size=args.batch_size,
        epochs=args.epochs,
        patience=args.patience,
        device=device,
        normalization_method=args.normalization,
        normalization_parameters=model_configs,
        comparison_metrics=str(root_relative(args.comparison_metrics)) if args.comparison_metrics else None,
        note=(
            "Standard PyPOTS GP-VAE. Full temporal span retained after block averaging; "
            "fine-scale fluctuations are not represented at model resolution. PyPOTS GP-VAE "
            "encoder consumes X without an explicit missing-mask input; NaNs are internally "
            "zero-filled, so scale_only normalization preserves a positive waveform baseline "
            "and makes zero-filled gaps more distinguishable. Training and validation use "
            "randomly positioned contiguous masks. Native validation criterion is not masked "
            "RMSE. Original-grid RMSE is computed after linear upsampling and is not a like-for-like "
            "full-resolution architecture comparison."
        ),
    )
    (destination / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Finished: {destination}", flush=True)


if __name__ == "__main__":
    main()
