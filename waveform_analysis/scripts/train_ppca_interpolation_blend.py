#!/usr/bin/env python3
"""Fit a position-specific convex blend of linear interpolation and standard PPCA.

Linear interpolation is the baseline. For each detector and masked interval,
an analytically optimized alpha in [0, 1] blends interpolation and the standard
PPCA conditional mean. Alpha is fitted on calibration events and evaluated on
disjoint held-out events. No boundary correction or stochastic PPCA draws are
used.
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

from waveform_analysis.models.ppca_imputer import ProbabilisticPCA, positions
from waveform_analysis.data.imputation import load_wide_development, root_relative


def linear_interpolation_batch(signals, start, width):
    """Linear interpolation between the samples bracketing the missing interval.

    At a waveform edge, where only one bracketing sample exists, use constant
    extrapolation to match the baseline used in the PPCA comparison script.
    """
    signals = np.asarray(signals, dtype=np.float64)
    stop = start + width
    if start == 0:
        return np.broadcast_to(signals[:, stop, None], (len(signals), width)).copy()
    if stop == signals.shape[1]:
        return np.broadcast_to(signals[:, start - 1, None], (len(signals), width)).copy()
    fraction = (
        np.arange(start, stop, dtype=np.float64) - (start - 1)
    ) / (width + 1.0)
    return (
        signals[:, start - 1, None] * (1.0 - fraction[None, :])
        + signals[:, stop, None] * fraction[None, :]
    )


def optimize_blend_weights(models, waves, tuning, starts, width,
                           prepared_masks, batch_size):
    """Find the exact least-squares blend alpha per detector and mask position.

    x_blend = x_interp + alpha * (x_ppca - x_interp), alpha constrained to [0, 1].
    The least-squares objective is quadratic in alpha, so no grid search is needed.
    """
    tuning = np.asarray(tuning, dtype=np.int64)
    alphas = np.zeros((2, len(starts)), dtype=np.float64)
    rows = []

    for det, model in enumerate(models):
        print(f"Optimizing blend weights: detector {det + 1}", flush=True)
        for pos_idx, start in enumerate(starts):
            stop = start + width
            prepared = prepared_masks[det][pos_idx]
            numerator = 0.0
            denominator = 0.0
            interpolation_sse = 0.0
            ppca_sse = 0.0

            for offset in range(0, len(tuning), batch_size):
                ids = tuning[offset:offset + batch_size]
                signals = np.asarray(waves[ids, det, :], dtype=np.float64)
                truth = signals[:, start:stop]
                ppca = model.conditional_mean_batch(
                    signals, start, width, prepared=prepared
                )
                interp = linear_interpolation_batch(signals, start, width)
                delta = ppca - interp
                target_delta = truth - interp
                numerator += float(np.sum(delta * target_delta))
                denominator += float(np.sum(delta * delta))
                interpolation_sse += float(np.square(interp - truth).sum())
                ppca_sse += float(np.square(ppca - truth).sum())

            alpha = (
                float(np.clip(numerator / denominator, 0.0, 1.0))
                if denominator > 1e-18 else 0.0
            )
            alphas[det, pos_idx] = alpha
            blend_sse = max(
                interpolation_sse - 2.0 * alpha * numerator
                + alpha * alpha * denominator,
                0.0,
            )
            n_values = max(len(tuning) * width, 1)
            rows.append(dict(
                detector=det + 1,
                mask_position=pos_idx,
                start_sample=start,
                tuning_events=len(tuning),
                alpha_ppca=alpha,
                ppca_weight=alpha,
                interpolation_weight=1.0 - alpha,
                tuning_interpolation_rmse=float(np.sqrt(interpolation_sse / n_values)),
                tuning_ppca_rmse=float(np.sqrt(ppca_sse / n_values)),
                tuning_blend_rmse=float(np.sqrt(blend_sse / n_values)),
                tuning_blend_minus_interpolation_rmse=float(
                    np.sqrt(blend_sse / n_values) - np.sqrt(interpolation_sse / n_values)
                ),
            ))
            if pos_idx % 10 == 0 or pos_idx == len(starts) - 1:
                print(
                    f"Detector {det + 1}: section {pos_idx + 1}/{len(starts)} "
                    f"alpha={alpha:.3f}",
                    flush=True,
                )
    return alphas, rows


def evaluate_blend(models, waves, evaluation, starts, width, time_ps,
                   alphas, prepared_masks, destination, batch_size):
    """Compare interpolation, standard PPCA conditional mean and their blend."""
    evaluation = np.asarray(evaluation, dtype=np.int64)
    rows = []

    for pos_idx, start in enumerate(starts):
        stop = start + width
        squared_error = np.zeros((2, 3), dtype=np.float64)
        for det, model in enumerate(models):
            alpha = float(alphas[det, pos_idx])
            prepared = prepared_masks[det][pos_idx]
            for offset in range(0, len(evaluation), batch_size):
                ids = evaluation[offset:offset + batch_size]
                signals = np.asarray(waves[ids, det, :], dtype=np.float64)
                truth = signals[:, start:stop]
                ppca = model.conditional_mean_batch(
                    signals, start, width, prepared=prepared
                )
                interp = linear_interpolation_batch(signals, start, width)
                blend = interp + alpha * (ppca - interp)
                squared_error[det, 0] += np.square(interp - truth).sum()
                squared_error[det, 1] += np.square(ppca - truth).sum()
                squared_error[det, 2] += np.square(blend - truth).sum()

        n_values = max(len(evaluation) * width, 1)
        for det in range(2):
            interp_rmse, ppca_rmse, blend_rmse = np.sqrt(
                np.maximum(squared_error[det] / n_values, 0.0)
            )
            rows.append(dict(
                detector=det + 1,
                mask_position=pos_idx,
                start_sample=start,
                start_ns=float(time_ps[start] / 1000.0),
                evaluation_events=len(evaluation),
                alpha_ppca=float(alphas[det, pos_idx]),
                interpolation_rmse=float(interp_rmse),
                ppca_mean_rmse=float(ppca_rmse),
                blend_rmse=float(blend_rmse),
                blend_minus_interpolation_rmse=float(blend_rmse - interp_rmse),
                blend_minus_ppca_rmse=float(blend_rmse - ppca_rmse),
            ))
        if pos_idx % 10 == 0 or pos_idx == len(starts) - 1:
            print(f"Evaluation: section {pos_idx + 1}/{len(starts)}", flush=True)

    with (destination / "position_metrics.csv").open(
        "w", newline="", encoding="utf-8"
    ) as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    for det in range(2):
        matching = [r for r in rows if r["detector"] == det + 1]
        x = [r["start_ns"] for r in matching]
        for key, label in (
            ("interpolation_rmse", "Linear interpolation"),
            ("ppca_mean_rmse", "PPCA conditional mean"),
            ("blend_rmse", "Optimized blend"),
        ):
            axes[det].plot(x, [r[key] for r in matching], label=label)
        axes[det].set_ylabel(f"Detector {det + 1} RMSE")
        axes[det].grid(alpha=0.25)
        axes[det].legend()
    axes[-1].set_xlabel("Masked region start (ns)")
    fig.tight_layout()
    fig.savefig(destination / "rmse_by_position.png", dpi=180)
    fig.savefig(destination / "rmse_by_position.pdf")
    plt.close(fig)
    return rows


def plot_examples(models, waves, evaluation, starts, width, time_ps, alphas,
                  prepared_masks, destination, seed, n_events=3):
    """Plot held-out examples comparing interpolation, PPCA and the fitted blend."""
    if n_events <= 0:
        return
    out = destination / "imputation_examples"
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed + 2028)
    evaluation = np.asarray(evaluation, dtype=np.int64)
    chosen = rng.choice(evaluation, size=min(n_events, len(evaluation)), replace=False)
    times = np.asarray(time_ps, dtype=np.float64) / 1000.0
    targets_ns = (0.0, 20.0, 40.0)
    mask_indices = list(dict.fromkeys(
        min(range(len(starts)), key=lambda i: abs(times[starts[i]] - target))
        for target in targets_ns
    ))
    dt_ps = float(np.median(np.diff(time_ps)))
    margin = max(1, int(round(3000.0 / dt_ps)))

    for det, model in enumerate(models):
        for pos_idx in mask_indices:
            start = starts[pos_idx]
            stop = start + width
            alpha = float(alphas[det, pos_idx])
            signals = np.asarray(waves[chosen, det, :], dtype=np.float64)
            ppca, _ = model.condition_batch(
                signals, start, width, 1, rng,
                prepared=prepared_masks[det][pos_idx],
            )
            interp = linear_interpolation_batch(signals, start, width)
            blend = interp + alpha * (ppca - interp)

            for event_i, event_id in enumerate(chosen):
                original = signals[event_i]
                observed = original.copy()
                observed[start:stop] = np.nan
                reconstructions = []
                for imputed in (interp[event_i], ppca[event_i], blend[event_i]):
                    reconstructed = original.copy()
                    reconstructed[start:stop] = imputed
                    reconstructions.append(reconstructed)

                fig, axes = plt.subplots(
                    2, 1, figsize=(11, 7), height_ratios=(1.0, 1.2)
                )
                for ax, zoom in zip(axes, (False, True)):
                    ax.plot(times, original, color="black", lw=1.2, label="Ground truth")
                    ax.plot(times, observed, color="gray", lw=1.0, ls="--",
                            label="Observed (masked)")
                    ax.plot(times, reconstructions[0], color="tab:green", lw=1.2,
                            label="Linear interpolation")
                    ax.plot(times, reconstructions[1], color="tab:blue", lw=1.2,
                            label="PPCA conditional mean")
                    ax.plot(times, reconstructions[2], color="tab:orange", lw=1.5,
                            label=f"Blend (alpha={alpha:.2f})")
                    ax.axvspan(times[start], times[min(stop, len(times) - 1)],
                               color="tab:red", alpha=0.08)
                    ax.grid(alpha=0.25)
                    if zoom:
                        ax.set_xlim(times[max(0, start - margin)],
                                     times[min(len(times) - 1, stop + margin)])
                        ax.set_xlabel("Time (ns)")
                    ax.set_ylabel("Normalized signal")
                    ax.legend(loc="best", fontsize="small", ncol=2)
                fig.suptitle(
                    f"Detector {det + 1}, held-out event {event_id}, "
                    f"mask at {times[start]:.2f} ns"
                )
                fig.tight_layout()
                fig.savefig(
                    out / f"detector_{det + 1}_position_{pos_idx:03d}_event_{event_id}.png",
                    dpi=160,
                )
                plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--channel", choices=("energy", "timing"), required=True)
    parser.add_argument("--cache-root", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--max-events", type=int, default=5000)
    parser.add_argument("--components", type=int, default=32)
    parser.add_argument("--mask-ns", type=float, default=0.25)
    parser.add_argument("--mask-stride-samples", type=int)
    parser.add_argument("--eval-events", type=int, default=100)
    parser.add_argument("--tuning-events", type=int, default=500)
    parser.add_argument("--eval-batch-size", type=int, default=128)
    parser.add_argument("--plot-examples", type=int, default=3,
                        help="Held-out events to plot per detector/selected mask (0 disables)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val-fraction", type=float, default=0.15)
    args = parser.parse_args()

    if (
        args.max_events < 10
        or args.components < 1
        or args.eval_events < 1
        or args.tuning_events < 1
        or args.eval_batch_size < 1
        or args.plot_examples < 0
        or args.mask_ns <= 0
        or not 0 < args.val_fraction < 1
        or (args.mask_stride_samples is not None and args.mask_stride_samples < 1)
    ):
        parser.error("Invalid parameter")

    results = root_relative(args.results)
    waves, time_ps, source, manifest = load_wide_development(
        results, args.channel, root_relative(args.cache_root)
    )
    length = waves.shape[-1]
    dt_ps = float(np.median(np.diff(time_ps)))
    if dt_ps <= 0 or not np.allclose(np.diff(time_ps), dt_ps, rtol=1e-4, atol=1e-3):
        raise ValueError("Expected uniform time grid")
    width = max(1, round(args.mask_ns * 1000.0 / dt_ps))
    starts = positions(length, width, args.mask_stride_samples or width)

    selected = np.random.default_rng(args.seed).permutation(len(waves))[:args.max_events]
    if len(selected) < 10:
        parser.error("Not enough events were loaded")
    n_val = max(2, round(len(selected) * args.val_fraction))
    validation = selected[:n_val]
    training = selected[n_val:]
    if args.components >= min(len(training), length):
        parser.error("--components must be below the number of training events and samples")

    n_tuning = min(args.tuning_events, max(1, len(validation) // 2))
    tuning = validation[:n_tuning]
    evaluation = validation[n_tuning:n_tuning + args.eval_events]
    if len(evaluation) == 0:
        parser.error("No held-out evaluation events remain; reduce --tuning-events or --eval-events")

    destination = (
        root_relative(args.output) if args.output else
        ROOT / "waveform_analysis" / "imputation_experiments"
        / results.name / args.channel / "ppca_interpolation_blend"
    )
    destination.mkdir(parents=True, exist_ok=True)

    print(
        f"Fitting standard PPCA on {len(training)} events per detector; "
        f"components={args.components}; waveform length={length}",
        flush=True,
    )
    models = []
    for det in range(2):
        model = ProbabilisticPCA(args.components).fit(
            np.asarray(waves[training, det, :], dtype=np.float64)
        )
        models.append(model)
        print(
            f"Detector {det + 1}: residual variance={model.sigma2:.6g}; "
            f"explained variance={model.explained_variance_ratio:.4f}",
            flush=True,
        )

    print(f"Preparing {len(starts)} mask positions per detector", flush=True)
    prepared_masks = [
        [model.prepare_mask(start, width) for start in starts]
        for model in models
    ]

    print(
        f"Blend calibration: {len(tuning)} events; independent evaluation: "
        f"{len(evaluation)} events",
        flush=True,
    )
    alphas, tuning_rows = optimize_blend_weights(
        models, waves, tuning, starts, width, prepared_masks, args.eval_batch_size
    )
    for row in tuning_rows:
        row["start_ns"] = float(time_ps[row["start_sample"]] / 1000.0)
    with (destination / "blend_weights.csv").open(
        "w", newline="", encoding="utf-8"
    ) as f:
        writer = csv.DictWriter(f, fieldnames=list(tuning_rows[0]))
        writer.writeheader()
        writer.writerows(tuning_rows)

    fig, ax = plt.subplots(figsize=(10, 4))
    x = np.asarray(time_ps)[starts] / 1000.0
    for det in range(2):
        ax.plot(x, alphas[det], marker=".", ms=4, label=f"Detector {det + 1}")
    ax.set_xlabel("Masked region start (ns)")
    ax.set_ylabel("Optimized PPCA weight (alpha)")
    ax.set_ylim(-0.03, 1.03)
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(destination / "blend_weights.png", dpi=180)
    plt.close(fig)

    rows = evaluate_blend(
        models, waves, evaluation, starts, width, np.asarray(time_ps),
        alphas, prepared_masks, destination, args.eval_batch_size,
    )
    if args.plot_examples:
        plot_examples(
            models, waves, evaluation, starts, width, time_ps, alphas,
            prepared_masks, destination, args.seed, args.plot_examples,
        )

    np.savez_compressed(
        destination / "model.npz",
        mean=np.stack([m.mean for m in models]),
        loading=np.stack([m.W for m in models]),
        noise_variance=np.asarray([m.sigma2 for m in models]),
        alphas=alphas,
        starts=np.asarray(starts, dtype=np.int64),
        mask_width=np.asarray(width),
    )
    summary = dict(
        method="standard_ppca_linear_interpolation_convex_blend",
        dataset_identity=manifest.get("analysis_protocol_identity"),
        source_run=str(source),
        channel=args.channel,
        selected_events=len(selected),
        training_events=len(training),
        validation_events=len(validation),
        tuning_events=len(tuning),
        evaluation_events=len(evaluation),
        components=args.components,
        mask_samples=width,
        mask_positions=len(starts),
        seed=args.seed,
        ppca_weight_formula="clip(sum((ppca-interp)*(truth-interp))/sum((ppca-interp)^2), 0, 1)",
        optimized_ppca_weights=alphas.tolist(),
        detector_noise_variances=[float(m.sigma2) for m in models],
        detector_explained_variance_ratios=[m.explained_variance_ratio for m in models],
        note=(
            "Standard PPCA conditional mean only; no boundary correction. "
            "A separate alpha in [0,1] is fitted per detector and mask position "
            "on calibration events by least squares, then evaluated on disjoint events."
        ),
    )
    (destination / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(f"Finished: {destination}", flush=True)


if __name__ == "__main__":
    main()
