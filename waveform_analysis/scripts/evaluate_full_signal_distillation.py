"""Full-signal temporal CNN distillation diagnostic (no TSHAP windows).

Fit the same TemporalConditionalCNN architecture used by masked NN-Olsen-style
surrogate to reproduce the SAVED original predictions using all waveform
samples. The full mask is supplied as explicit ones; the original model is
never loaded/called, nor are its transformations run.

Use the same split seed and target input provenance as evaluate_nn_olsen.py.
Standardization is fit in streaming train-only passes, applied to minibatches.
"""
from __future__ import annotations

import argparse
import json
import logging
import random
import time
from pathlib import Path

import numpy as np
import torch

from waveform_analysis.engine.nn_olsen import augment
from waveform_analysis.engine.sample_normalization import fit_time_sample_stats, tensor_stats
from waveform_analysis.engine.temporal_conditional_cnn import TemporalConditionalCNN
from waveform_analysis.engine.inception_time_surrogate import InceptionTimeSurrogate
from waveform_analysis.scripts.evaluate_conditional_knn import _load_inputs, _logger


def full_features(waves, mean, scale):
    """Use every coordinate; no missing windows and no hidden values."""
    mask = torch.ones((waves.shape[0], waves.shape[-1]),
                      dtype=waves.dtype, device=waves.device)
    return augment(waves, mask, sample_mean=mean, sample_scale=scale)


def full_metrics(y, prediction):
    y = np.asarray(y, dtype=np.float64)
    prediction = np.asarray(prediction, dtype=np.float64)
    if y.shape != prediction.shape or y.ndim != 1:
        raise ValueError("Invalid prediction arrays")
    error = prediction - y
    return {
        "n_events": int(len(y)),
        "rmse_ps": float(np.sqrt(np.mean(error ** 2))),
        "mae_ps": float(np.mean(np.abs(error))),
        "bias_ps": float(np.mean(error)),
        "median_abs_error_ps": float(np.median(np.abs(error))),
        "p90_abs_error_ps": float(np.percentile(np.abs(error), 90)),
        "p95_abs_error_ps": float(np.percentile(np.abs(error), 95)),
        "p99_abs_error_ps": float(np.percentile(np.abs(error), 99)),
        "original_mean_ps": float(np.mean(y)),
        "original_std_ps": float(np.std(y)),
        "reconstructed_mean_ps": float(np.mean(prediction)),
        "reconstructed_std_ps": float(np.std(prediction)),
        "correlation": (float(np.corrcoef(y, prediction)[0, 1])
                        if np.std(y) > 0 and np.std(prediction) > 0 else None),
    }


@torch.inference_mode()
def evaluate(model, waves, labels, indices, device, batch_size, ymean, ystd,
             sample_mean, sample_scale):
    model.eval()
    result = np.empty(len(indices), dtype=np.float64)
    for start in range(0, len(indices), batch_size):
        ids = indices[start:start + batch_size]
        wave = torch.as_tensor(np.asarray(waves[ids], dtype=np.float32),
                               device=device)
        out = model(full_features(wave, sample_mean, sample_scale))
        result[start:start + len(ids)] = out.cpu().numpy() * ystd + ymean
    return result, full_metrics(labels[indices], result)


def _plots(output, target, prediction, title, model_label="CNN"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    residual = prediction - target
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(target, prediction, s=8, alpha=0.4, rasterized=True)
    lo = float(min(target.min(), prediction.min()))
    hi = float(max(target.max(), prediction.max()))
    ax.plot([lo, hi], [lo, hi], "k--", linewidth=1)
    ax.set(xlabel="Original prediction (ps)", ylabel=f"{model_label} reconstruction (ps)",
           title=title)
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(output / "holdout_original_vs_reconstructed.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.scatter(target, residual, s=8, alpha=0.4, rasterized=True)
    ax.axhline(0, color="k", linestyle="--", linewidth=1)
    ax.set(xlabel="Original prediction (ps)", ylabel=f"{model_label} - original (ps)",
           title=f"Full-signal {model_label} | holdout residual")
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(output / "holdout_residual_vs_original.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(residual, bins=70)
    ax.set(xlabel="CNN - original (ps)", ylabel="Events",
           title=f"Full-signal {model_label} | holdout residual distribution")
    fig.tight_layout()
    fig.savefig(output / "holdout_residual_hist.png", dpi=180)
    plt.close(fig)


def run(args):
    root = args.output_root.expanduser().resolve()
    log = _logger(root)
    log.info("FULL SIGNAL DISTILLATION | no windows, no original-model inference")
    log.info("Arguments=%s", vars(args))
    if (min(args.batch_size, args.eval_batch, args.epochs, args.patience,
            args.width, args.depth, args.temporal_bins) < 1
            or args.learning_rate <= 0):
        raise ValueError("Invalid training parameters")
    waves, predictions, retained, provenance = _load_inputs(args, log)
    if len(retained) != waves.shape[-1] or not np.asarray(retained).all():
        log.warning("Some samples are excluded by prepared-data retention mask; "
                    "using all prepared samples by design for full-signal control")
    n = len(predictions)
    if args.tuning_queries < 1 or args.holdout_queries < 1:
        raise ValueError("Nonempty tuning and holdout required")
    if args.tuning_queries + args.holdout_queries >= n:
        raise ValueError("No training data remain")
    rng = np.random.default_rng(args.seed)
    selected = rng.choice(n, args.tuning_queries + args.holdout_queries,
                          replace=False)
    tuning = selected[:args.tuning_queries]
    holdout = selected[args.tuning_queries:]
    training_flag = np.ones(n, dtype=bool)
    training_flag[selected] = False
    training = np.flatnonzero(training_flag)
    log.info("EVENT SPLIT | train=%d tuning=%d holdout=%d seed=%d",
             len(training), len(tuning), len(holdout), args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    device = ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    ymean = float(np.mean(predictions[training]))
    ystd = float(np.std(predictions[training]))
    if ystd < 1e-12:
        raise ValueError("Original prediction variance is zero")
    mean_array = scale_array = normalization_details = None
    sample_mean = sample_scale = None
    if args.input_normalization == "per_sample":
        mean_array, scale_array, normalization_details = fit_time_sample_stats(
            waves, training, chunk_size=args.normalization_chunk_size,
            scale_floor_ratio=args.normalization_scale_floor_ratio)
        sample_mean, sample_scale = tensor_stats(mean_array, scale_array, device=device)
        log.info("INPUT NORMALIZATION | per_sample | %s", normalization_details)
    else:
        log.info("INPUT NORMALIZATION | none")
    architecture = getattr(args, "architecture", "temporal_cnn")
    if architecture == "temporal_cnn":
        model = TemporalConditionalCNN(waves.shape[-1], width=args.width,
                                        depth=args.depth, bins=args.temporal_bins).to(device)
    elif architecture == "inception_time":
        model = InceptionTimeSurrogate(
            waves.shape[-1], filters=args.inception_filters,
            bottleneck=args.inception_bottleneck,
            depth=args.inception_depth,
            kernel_size=args.inception_kernel_size,
        ).to(device)
    else:
        raise ValueError(f"Unsupported full-signal architecture: {architecture}")
    log.info("MODEL | %s parameters=%d | device=%s",
             type(model).__name__, sum(p.numel() for p in model.parameters()), device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate,
                                 weight_decay=args.weight_decay)
    best_loss, best_epoch, best_state = float("inf"), -1, None
    history, patience_used = [], 0
    clock = time.monotonic()
    for epoch in range(1, args.epochs + 1):
        model.train()
        permutation = rng.permutation(training)
        sum_squared = 0.
        count = 0
        for start in range(0, len(permutation), args.batch_size):
            ids = permutation[start:start + args.batch_size]
            if len(ids) < 2:
                continue
            wave = torch.as_tensor(np.asarray(waves[ids], dtype=np.float32),
                                   device=device)
            target = torch.as_tensor(
                ((predictions[ids] - ymean) / ystd).astype(np.float32), device=device)
            optimizer.zero_grad(set_to_none=True)
            estimated = model(full_features(wave, sample_mean, sample_scale))
            loss = (estimated - target).square().mean()
            loss.backward()
            optimizer.step()
            sum_squared += float(loss.detach().cpu()) * len(ids)
            count += len(ids)
        tuning_prediction, tuning_metrics = evaluate(
            model, waves, predictions, tuning, device, args.eval_batch,
            ymean, ystd, sample_mean, sample_scale)
        val_mse = tuning_metrics["rmse_ps"] ** 2
        improved = val_mse < best_loss - args.min_delta
        if improved:
            best_loss = val_mse
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone()
                          for k, v in model.state_dict().items()}
            patience_used = 0
        else:
            patience_used += 1
        history.append({
            "epoch": epoch, "train_normalized_mse": sum_squared / max(count, 1),
            "tuning": tuning_metrics,
        })
        log.info("EPOCH %03d/%03d | train normalized MSE=%.6f | "
                 "tuning RMSE=%.4f ps MAE=%.4f ps std=%.4f ps "
                 "| best_epoch=%d patience=%d/%d elapsed=%.1fs",
                 epoch, args.epochs, sum_squared / max(count, 1),
                 tuning_metrics["rmse_ps"], tuning_metrics["mae_ps"],
                 tuning_metrics["reconstructed_std_ps"],
                 best_epoch, patience_used, args.patience, time.monotonic()-clock)
        if patience_used >= args.patience:
            break
    if best_state is None:
        raise RuntimeError("No valid best checkpoint")
    model.load_state_dict(best_state)
    tuning_prediction, tuning_metrics = evaluate(
        model, waves, predictions, tuning, device, args.eval_batch,
        ymean, ystd, sample_mean, sample_scale)
    holdout_prediction, holdout_metrics = evaluate(
        model, waves, predictions, holdout, device, args.eval_batch,
        ymean, ystd, sample_mean, sample_scale)
    log.info("FINAL | best_epoch=%d tuning RMSE=%.4f ps | "
             "holdout RMSE=%.4f ps MAE=%.4f ps | original std=%.4f ps",
             best_epoch, tuning_metrics["rmse_ps"], holdout_metrics["rmse_ps"],
             holdout_metrics["mae_ps"], holdout_metrics["original_std_ps"])
    checkpoint_name = ("temporal_cnn_full_signal.pt" if architecture == "temporal_cnn"
                       else "inception_time_full_signal.pt")
    torch.save({
        "model_state": best_state,
        "architecture_name": architecture + "_full_signal",
        "architecture": {
            "n_samples": int(waves.shape[-1]), "width": args.width,
            "depth": args.depth, "temporal_bins": args.temporal_bins,
            "inception_filters": args.inception_filters,
            "inception_bottleneck": args.inception_bottleneck,
            "inception_depth": args.inception_depth,
            "inception_kernel_size": args.inception_kernel_size,
        },
        "input_normalization": args.input_normalization,
        "input_sample_mean": mean_array,
        "input_sample_scale": scale_array,
        "input_normalization_metadata": normalization_details,
        "target_mean_ps": ymean, "target_std_ps": ystd,
        "best_epoch": best_epoch, "provenance": provenance,
    }, root / checkpoint_name)
    np.savez_compressed(
        root / "diagnostics.npz",
        tuning_query_rows=tuning, holdout_query_rows=holdout,
        tuning_original_prediction_ps=predictions[tuning],
        tuning_reconstructed_prediction_ps=tuning_prediction,
        holdout_original_prediction_ps=predictions[holdout],
        holdout_reconstructed_prediction_ps=holdout_prediction,
    )
    summary = {
        "method": f"full-signal {architecture} distillation control (no masks/windows)",
        "source_model_loaded": False, "source_model_predict_calls": 0,
        "provenance": provenance, "best_epoch": best_epoch,
        "settings": {
            "seed": args.seed, "training_events": len(training),
            "tuning_events": len(tuning), "holdout_events": len(holdout),
            "architecture": architecture,
            "width": args.width, "depth": args.depth,
            "temporal_bins": args.temporal_bins,
            "inception_filters": args.inception_filters,
            "inception_bottleneck": args.inception_bottleneck,
            "inception_depth": args.inception_depth,
            "inception_kernel_size": args.inception_kernel_size,
            "input_normalization": args.input_normalization,
            "normalization_scale_floor_ratio": args.normalization_scale_floor_ratio,
            "batch_size": args.batch_size, "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay, "epochs": args.epochs,
            "patience": args.patience, "device": device,
        },
        "input_normalization_metadata": normalization_details,
        "tuning": tuning_metrics, "holdout": holdout_metrics, "history": history,
    }
    (root / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    _plots(root, predictions[holdout], holdout_prediction,
           f"Full-signal {architecture}: original vs reconstructed (holdout)",
           model_label=architecture)
    log.info("COMPLETE | %s | no original model inference", root)
    return summary


def _parser():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--model-dir", required=True, type=Path)
    p.add_argument("--cache-root", type=Path)
    p.add_argument("--artifact-root", type=Path)
    p.add_argument("--output-root", type=Path, required=True)
    p.add_argument("--tuning-queries", type=int, default=500)
    p.add_argument("--holdout-queries", type=int, default=500)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--architecture", choices=("temporal_cnn", "inception_time"),
                   default="temporal_cnn")
    p.add_argument("--inception-filters", type=int, default=16)
    p.add_argument("--inception-bottleneck", type=int, default=32)
    p.add_argument("--inception-depth", type=int, default=6)
    p.add_argument("--inception-kernel-size", type=int, default=39)
    p.add_argument("--width", type=int, default=64)
    p.add_argument("--depth", type=int, default=3)
    p.add_argument("--temporal-bins", type=int, default=8)
    p.add_argument("--input-normalization", choices=("none", "per_sample"),
                   default="per_sample")
    p.add_argument("--normalization-chunk-size", type=int, default=256)
    p.add_argument("--normalization-scale-floor-ratio", type=float, default=0.05)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--eval-batch", type=int, default=256)
    p.add_argument("--learning-rate", type=float, default=0.001)
    p.add_argument("--weight-decay", type=float, default=0.)
    p.add_argument("--patience", type=int, default=8)
    p.add_argument("--min-delta", type=float, default=0.01)
    p.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    return p


if __name__ == "__main__":
    run(_parser().parse_args())
