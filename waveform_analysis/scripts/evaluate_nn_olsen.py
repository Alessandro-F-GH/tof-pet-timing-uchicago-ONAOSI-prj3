"""NN-Olsen-style surrogate benchmark on saved development predictions.

Uses the SAME prepared waveform pairs, two-player TSHAP masks, random query
cohorts, and conditional target as evaluate_conditional_knn.py. Never loads or
predicts with the underlying ML model or runs feature transformations.

Example:
python -m waveform_analysis.scripts.evaluate_nn_olsen \\
  --model-dir /path/to/saved/run --cache-root ~/UC_cache \\
  --output-root ~/conditional_nn/UC --window-samples 20 --stride-samples 20
"""
from __future__ import annotations

import argparse
import copy
import json
import logging
import random
import time
from pathlib import Path

import numpy as np
import torch

from waveform_analysis.engine.conditional_knn import tshap_windows
from waveform_analysis.engine.temporal_conditional_cnn import TemporalConditionalCNN
from waveform_analysis.engine.nn_olsen import (
    NNOlsenSurrogate, coalitions, paired_batch, predict_coalitions,
)
from waveform_analysis.scripts.evaluate_conditional_knn import _load_inputs, _logger


def _metrics(estimated, true):
    """Coalition-specific RMSE, bias, mean, variance and standard deviation."""
    true = np.asarray(true, dtype=np.float64)
    estimated = np.asarray(estimated, dtype=np.float64)
    if estimated.ndim != 3 or estimated.shape[0] != true.size or estimated.shape[2] != 2:
        raise ValueError("Invalid conditional prediction dimensions")
    diff = estimated - true[:, None, None]
    return {
        "rmse_ps": np.sqrt(np.mean(diff * diff, axis=(0, 1))).tolist(),
        "bias_ps": np.mean(diff, axis=(0, 1)).tolist(),
        "estimated_mean_ps": np.mean(estimated, axis=(0, 1)).tolist(),
        "estimated_variance_ps2": np.var(estimated, axis=(0, 1)).tolist(),
        "estimated_std_ps": np.std(estimated, axis=(0, 1)).tolist(),
        "original_mean_ps": float(np.mean(true)),
        "original_variance_ps2": float(np.var(true)),
        "original_std_ps": float(np.std(true)),
        "original_vs_estimated_std_ratio": (
            np.std(estimated, axis=(0, 1)) / max(float(np.std(true)), 1e-12)
        ).tolist(),
        "rmse_by_window_ps": np.sqrt(np.mean(diff * diff, axis=0)).tolist(),
        "estimated_std_by_window_ps": np.std(estimated, axis=0).tolist(),
        "original_baseline_rmse_ps": float(np.sqrt(
            np.mean((true - np.mean(true)) ** 2)
        )),
    }


def _evaluate(model, waves, labels, indices, starts, keep, args, device, log, name,
              mean, std):
    start = time.monotonic()
    log.info("%s | evaluating %d held-out events x %d windows x 2 coalitions",
             name, len(indices), len(starts))
    # Passing in one batch at a time prevents GPU memory growing with all queries.
    out = np.empty((len(indices), len(starts), 2), dtype=np.float64)
    for i in range(0, len(indices), args.eval_batch):
        ids = indices[i:i + args.eval_batch]
        prediction = predict_coalitions(model, waves[ids], starts, args.window_samples,
                                        keep, batch_size=args.eval_batch, device=device)
        out[i:i+len(ids)] = prediction * std + mean
        if i == 0 or i + len(ids) == len(indices):
            log.info("%s | processed %d/%d queries | elapsed=%.1fs",
                     name, i+len(ids), len(indices), time.monotonic()-start)
    results = _metrics(out, labels[indices])
    log.info("%s | W RMSE=%.4f ps R RMSE=%.4f ps | W std=%.4f ps R std=%.4f ps | original std=%.4f ps",
             name, results["rmse_ps"][0], results["rmse_ps"][1],
             results["estimated_std_ps"][0], results["estimated_std_ps"][1],
             results["original_std_ps"])
    return out, results


def run(args):
    root = args.output_root.expanduser().resolve()
    log = _logger(root)
    architecture = getattr(args, "architecture", "mlp")
    if architecture not in ("mlp", "temporal_cnn"):
        raise ValueError(f"Unsupported surrogate architecture: {architecture}")
    log.info("CONDITIONAL SURROGATE | architecture=%s | no original model inference | W/R complementary masks", architecture)
    log.info("Arguments=%s", vars(args))
    waves, predictions, keep, provenance = _load_inputs(args, log)
    starts, keep = tshap_windows(waves.shape[-1], args.window_samples,
                                args.stride_samples, keep)
    n = len(predictions)
    if args.tuning_queries < 1 or args.holdout_queries < 1:
        raise ValueError("Validation and holdout query counts must be positive")
    if args.tuning_queries + args.holdout_queries >= n:
        raise ValueError("Must reserve at least one independent training event")
    if min(args.batch_size, args.eval_batch, args.epochs, args.patience,
           args.width, args.depth) < 1:
        raise ValueError("Batch sizes, epochs, patience, width and depth must be positive")
    if args.batch_size < 2:
        raise ValueError("Batch normalization needs training batch size >= 2")
    rng = np.random.default_rng(args.seed)
    selected = rng.choice(n, args.tuning_queries + args.holdout_queries, replace=False)
    tune_idx = selected[:args.tuning_queries]
    hold_idx = selected[args.tuning_queries:]
    training_flag = np.ones(n, dtype=bool)
    training_flag[selected] = False
    train_idx = np.flatnonzero(training_flag)
    # Match KNN's reproducible query sampling; unlike KNN, NN training excludes
    # *all* validation and holdout queries to avoid memorization.
    log.info("EVENT SPLIT | train=%d | tuning=%d | holdout=%d | total=%d; seed=%d",
             len(train_idx), len(tune_idx), len(hold_idx), n, args.seed)
    log.info("TARGET | cached original model predictions only, shape=%s", predictions.shape)
    log.info("WINDOWS | count=%d width=%d stride=%d starts=%s",
             len(starts), args.window_samples, args.stride_samples, starts.tolist())
    log.info("METHOD | architecture=%s, shared across all TSHAP windows", architecture)
    log.info("METHOD | masks=(W, complement), paired for each sampled event/window, loss=MSE")
    log.info("METHOD | train on n=%d reference events (no reference cap), never blind", len(train_idx))
    # The original waveforms are passed to the model unchanged; only a binary
    # missingness mask and zero-fill produce surrogate inputs at runtime.
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available")
    log.info("DEVICE | %s | torch=%s", device, torch.__version__)
    ymean = float(np.mean(predictions[train_idx]))
    ystd = float(np.std(predictions[train_idx]))
    if ystd < 1e-12:
        raise ValueError("Original model predictions are effectively constant")
    log.info("ORIGINAL TRAIN TARGET | mean=%.6f ps | variance=%.6f ps2 | std=%.6f ps",
             ymean, ystd**2, ystd)
    if architecture == "mlp":
        model = NNOlsenSurrogate(waves.shape[-1], width=args.width, depth=args.depth).to(device)
    else:
        model = TemporalConditionalCNN(waves.shape[-1], width=args.width, depth=args.depth,
                                       bins=args.temporal_bins).to(device)
    log.info("MODEL | %s | parameters=%d", type(model).__name__,
             sum(p.numel() for p in model.parameters()))
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate,
                                 weight_decay=args.weight_decay)
    masks = coalitions(starts, args.window_samples, keep, device=device)
    best_loss = float("inf")
    best_state = None
    best_epoch = -1
    history = []
    patience_used = 0
    start = time.monotonic()
    log.info("TRAINING START | epochs=%d | batch_size=%d | learning_rate=%.5g | weight_decay=%.5g | patience=%d",
             args.epochs, args.batch_size, args.learning_rate,
             args.weight_decay, args.patience)
    for epoch in range(1, args.epochs+1):
        model.train()
        perm = rng.permutation(train_idx)
        weighted_sum = 0.
        n_outputs = 0
        for left in range(0, len(perm), args.batch_size):
            ids = perm[left:left + args.batch_size]
            if len(ids) < 2:
                continue  # Avoid training BatchNorm on a singleton remainder.
            window_ix = torch.as_tensor(rng.integers(len(starts), size=len(ids)),
                                        dtype=torch.long, device=device)
            wave = torch.as_tensor(np.asarray(waves[ids], dtype=np.float32),
                                   dtype=torch.float32, device=device)
            target = torch.as_tensor(((predictions[ids] - ymean) / ystd).astype(np.float32),
                                     dtype=torch.float32, device=device)
            features, pair_targets = paired_batch(wave, target, window_ix, masks)
            optimizer.zero_grad(set_to_none=True)
            predicted = model(features)
            loss = torch.mean((predicted - pair_targets) ** 2)
            loss.backward()
            optimizer.step()
            weighted_sum += float(loss.detach().cpu()) * len(pair_targets)
            n_outputs += len(pair_targets)
        train_mse = weighted_sum / max(n_outputs, 1)
        tune_out, tune_stats = _evaluate(model, waves, predictions, tune_idx,
                                         starts, keep, args, device, log,
                                         f"TUNING EPOCH {epoch}", ymean, ystd)
        tuning_mse = (tune_stats["rmse_ps"][0]**2 + tune_stats["rmse_ps"][1]**2) / 2
        history.append({"epoch": epoch, "train_normalized_mse": train_mse,
                        "tuning_mse_ps2": tuning_mse,
                        "tuning_rmse_ps": tune_stats["rmse_ps"]})
        improved = tuning_mse < best_loss - args.min_delta
        if improved:
            best_loss = float(tuning_mse)
            best_state = {key: value.detach().cpu().clone()
                          for key, value in model.state_dict().items()}
            best_epoch = epoch
            patience_used = 0
        else:
            patience_used += 1
        log.info("EPOCH %03d/%03d | train normalized MSE=%.6f | tuning W/R RMSE=%s ps | total tuning MSE=%.5f ps2 | best_epoch=%d | patience=%d/%d | total_elapsed=%.1fs",
                 epoch, args.epochs, train_mse,
                 [round(z, 4) for z in tune_stats["rmse_ps"]],
                 tuning_mse, best_epoch, patience_used, args.patience,
                 time.monotonic()-start)
        if patience_used >= args.patience:
            log.info("EARLY STOP | no significant validation improvement in %d epochs", args.patience)
            break
    if best_state is None:
        raise RuntimeError("Training produced no valid model checkpoint")
    model.load_state_dict(best_state)
    model.eval()
    log.info("BEST CHECKPOINT | epoch=%d | tuning average MSE=%.5f ps2",
             best_epoch, best_loss)
    tune_est, tune_stats = _evaluate(model, waves, predictions, tune_idx,
                                     starts, keep, args, device, log,
                                     "TUNING FINAL", ymean, ystd)
    hold_est, hold_stats = _evaluate(model, waves, predictions, hold_idx,
                                     starts, keep, args, device, log,
                                     "HOLDOUT FINAL", ymean, ystd)
    log.info("UNCONDITIONAL REFERENCE | full training predictions mean=%.6f ps, std=%.6f ps",
             ymean, ystd)
    log.info("DIAGNOSTIC NOTE | conditional estimates have smaller expected population variance than original predictions at exact conditional optimum; finite-sample errors may violate this inequality.")
    checkpoint_name = "nn_olsen.pt" if architecture == "mlp" else "temporal_conditional_cnn.pt"
    torch.save({
        "model_state": best_state,
        "architecture_name": architecture,
        "architecture": {"n_samples": int(waves.shape[-1]), "width": args.width,
                         "depth": args.depth,
                         "temporal_bins": args.temporal_bins if architecture == "temporal_cnn" else None},
        "target_mean_ps": ymean, "target_std_ps": ystd,
        "window_starts": starts.tolist(), "window_samples": args.window_samples,
        "retained_sample_mask": keep.tolist(), "provenance": provenance,
        "best_epoch": best_epoch,
    }, root / checkpoint_name)
    np.savez_compressed(root / "diagnostics.npz",
        tuning_query_rows=tune_idx, holdout_query_rows=hold_idx,
        training_rows=train_idx, window_starts=starts,
        tuning_original_prediction_ps=predictions[tune_idx],
        holdout_original_prediction_ps=predictions[hold_idx],
        tuning_conditional_prediction_ps=tune_est,
        holdout_conditional_prediction_ps=hold_est,
    )
    summary = {
        "method": ("NN-Olsen MLP" if architecture == "mlp" else "Temporal residual CNN")
                  + " masked surrogate regression with paired TSHAP W/R masks",
        "source_model_predict_calls": 0, "source_model_loaded": False,
        "source_feature_transform_calls": 0,
        "provenance": provenance,
        "settings": {
            "seed": args.seed, "window_samples": args.window_samples,
            "stride_samples": args.stride_samples,
            "n_windows": len(starts), "training_events": len(train_idx),
            "tuning_events": len(tune_idx), "holdout_events": len(hold_idx),
            "architecture": architecture, "width": args.width, "depth": args.depth,
            "temporal_bins": args.temporal_bins if architecture == "temporal_cnn" else None,
            "model_parameters": sum(p.numel() for p in model.parameters()),
            "epochs_requested": args.epochs,
            "epochs_run": len(history), "batch_size": args.batch_size,
            "learning_rate": args.learning_rate, "weight_decay": args.weight_decay,
            "patience": args.patience, "device": device,
        },
        "best_epoch": best_epoch,
        "training_prediction_variance_ps2": ystd**2,
        "training_prediction_std_ps": ystd,
        "tuning": tune_stats, "holdout": hold_stats, "history": history,
    }
    (root / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    log.info("COMPLETE | output=%s | original model inference calls=0", root)
    return summary


def _parser():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--model-dir", required=True, type=Path)
    p.add_argument("--cache-root", type=Path)
    p.add_argument("--artifact-root", type=Path)
    p.add_argument("--output-root", required=True, type=Path)
    p.add_argument("--window-samples", type=int, default=20)
    p.add_argument("--stride-samples", type=int, default=20)
    p.add_argument("--tuning-queries", type=int, default=500)
    p.add_argument("--holdout-queries", type=int, default=500)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--architecture", choices=["mlp", "temporal_cnn"], default="mlp")
    p.add_argument("--temporal-bins", type=int, default=8)
    p.add_argument("--width", type=int, default=64)
    p.add_argument("--depth", type=int, default=3)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--eval-batch", type=int, default=256)
    p.add_argument("--learning-rate", type=float, default=0.001)
    p.add_argument("--weight-decay", type=float, default=0.)
    p.add_argument("--patience", type=int, default=8)
    p.add_argument("--min-delta", type=float, default=0.01)
    p.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    return p


if __name__ == "__main__":
    run(_parser().parse_args())
