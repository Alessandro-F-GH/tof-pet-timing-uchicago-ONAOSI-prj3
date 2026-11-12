"""Deliberate small-set memorization test for the Temporal CNN (full waveform).

Train on a FIXED subset of training events to test whether the architecture
and optimizer can interpolate the SAVED deterministic teacher outputs.
No W/R masks, original-model inference, holdout selection, early stopping,
dropout, weight decay or feature transforms. Evaluates all training examples
in model.eval() after every epoch to obtain a comparable memorization RMSE.

Preprocessing statistics are fitted on the original 74,714 training cohort,
then applied identically to the fixed small set. Nothing is trained on the
remaining training events. This isolates capacity from changed input scaling.
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import time
from pathlib import Path

import numpy as np
import torch

from waveform_analysis.engine.sample_normalization import (
    fit_time_sample_stats, fit_first_difference_stats, tensor_stats,
)
from waveform_analysis.engine.temporal_conditional_cnn import TemporalConditionalCNN
from waveform_analysis.scripts.evaluate_conditional_knn import _load_inputs, _logger
from waveform_analysis.scripts.evaluate_full_signal_distillation import (
    evaluate, full_features,
)


def training_rows(n, *, seed, tuning_queries, holdout_queries, fit_events):
    """Match the full-distillation split before sampling the memorization set."""
    if fit_events < 2 or tuning_queries < 1 or holdout_queries < 1:
        raise ValueError("Need >=2 fit events and nonempty reserved cohorts")
    if tuning_queries + holdout_queries + fit_events > n:
        raise ValueError("Insufficient disjoint events")
    rng = np.random.default_rng(seed)
    withheld = rng.choice(n, tuning_queries + holdout_queries, replace=False)
    eligible = np.ones(n, dtype=bool)
    eligible[withheld] = False
    reference = np.flatnonzero(eligible)
    # Distinct deterministic selection; do not use reserved tuning/holdout events.
    selected = np.sort(rng.choice(reference, fit_events, replace=False))
    return reference, selected


def run(args):
    output = args.output_root.expanduser().resolve()
    log = _logger(output)
    if min(args.fit_events, args.batch_size, args.eval_batch, args.epochs,
           args.width, args.depth, args.temporal_bins, args.report_every) < 1:
        raise ValueError("All sizes and counts must be positive")
    if args.fit_events < 2 or args.batch_size < 2 or args.learning_rate <= 0:
        raise ValueError("Need >=2 fit events, batch >=2, and positive learning rate")
    if not (0 < args.lr_factor < 1 and args.lr_patience >= 1
            and args.lr_min > 0 and args.lr_threshold_ps >= 0):
        raise ValueError("Invalid plateau scheduler settings")
    if args.weight_decay != 0:
        raise ValueError("Memorization diagnostic requires --weight-decay 0")
    log.info("CNN CAPACITY / MEMORIZATION TEST | target = saved full teacher predictions")
    log.info("No early stopping, no holdout evaluation, no source-model inference")
    log.info("Arguments=%s", vars(args))
    waveforms, labels, retained, provenance = _load_inputs(args, log)
    if not retained.all():
        raise ValueError("Expected full retained waveform for this diagnostic")
    train_reference, selected = training_rows(
        len(labels), seed=args.seed, tuning_queries=args.tuning_queries,
        holdout_queries=args.holdout_queries, fit_events=args.fit_events)
    log.info("FIXED EVENTS | %d selected for training from %d eligible training rows",
             len(selected), len(train_reference))
    log.info("NORMALIZATION | reference stats from entire training cohort, not small subset")
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = ("cuda" if torch.cuda.is_available() else "cpu"
              ) if args.device == "auto" else args.device
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("Requested CUDA is not available")
    sample_mean_array, sample_scale_array, sample_info = fit_time_sample_stats(
        waveforms, train_reference, chunk_size=args.normalization_chunk_size,
        scale_floor_ratio=args.normalization_scale_floor_ratio)
    sample_mean, sample_scale = tensor_stats(
        sample_mean_array, sample_scale_array, device=device)
    deriv_mean_array = deriv_scale_array = deriv_info = None
    deriv_mean = deriv_scale = None
    if args.use_derivative:
        deriv_mean_array, deriv_scale_array, deriv_info = fit_first_difference_stats(
            waveforms, train_reference, chunk_size=args.normalization_chunk_size,
            scale_floor_ratio=args.normalization_scale_floor_ratio)
        deriv_mean, deriv_scale = tensor_stats(deriv_mean_array, deriv_scale_array,
                                               device=device)
    target_mean = float(labels[train_reference].mean())
    target_scale = float(labels[train_reference].std())
    if target_scale <= 1e-12:
        raise ValueError("Constant reference teacher outputs")
    model = TemporalConditionalCNN(
        waveforms.shape[-1], width=args.width, depth=args.depth,
        bins=args.temporal_bins, use_derivative=args.use_derivative).to(device)
    log.info("NETWORK | %s parameters=%d derivative=%s device=%s",
             type(model).__name__, sum(p.numel() for p in model.parameters()),
             args.use_derivative, device)
    log.info("TARGET | mean=%.5f ps std=%.5f ps | selected std=%.5f ps",
             target_mean, target_scale, float(labels[selected].std()))
    opt = torch.optim.Adam(model.parameters(), lr=args.learning_rate,
                           weight_decay=0.)
    scheduler = None if args.no_lr_decay else torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, mode="min", factor=args.lr_factor, patience=args.lr_patience,
        threshold=args.lr_threshold_ps, threshold_mode="abs",
        cooldown=args.lr_cooldown, min_lr=args.lr_min)
    log.info("LR SCHEDULER | %s", "disabled" if scheduler is None else
             f"ReduceLROnPlateau on SAME-EVENT eval RMSE each epoch | "
             f"patience={args.lr_patience} factor={args.lr_factor} "
             f"threshold_ps={args.lr_threshold_ps} min_lr={args.lr_min}")
    rng = np.random.default_rng(args.seed + 1)
    best_rmse = float("inf")
    best_epoch = 0
    best_state = None
    rows = []
    clock = time.monotonic()
    for epoch in range(1, args.epochs + 1):
        model.train()
        shuffled = rng.permutation(selected)
        epoch_loss = 0.
        trained = 0
        for left in range(0, len(shuffled), args.batch_size):
            indices = shuffled[left:left + args.batch_size]
            if len(indices) == 1:
                continue
            x = torch.as_tensor(np.asarray(waveforms[indices], dtype=np.float32),
                                device=device)
            y = torch.as_tensor(
                ((labels[indices] - target_mean) / target_scale).astype(np.float32),
                device=device)
            opt.zero_grad(set_to_none=True)
            out = model(full_features(
                x, sample_mean, sample_scale,
                derivative_mean=deriv_mean, derivative_scale=deriv_scale))
            loss = (out - y).square().mean()
            loss.backward()
            opt.step()
            epoch_loss += float(loss.detach().cpu()) * len(indices)
            trained += len(indices)
        # Evaluate the SAME fixed events EVERY epoch, even when logging less
        # often. This keeps LR plateau patience in actual epochs rather than
        # report intervals, and the best checkpoint is never missed.
        prediction, metrics = evaluate(
            model, waveforms, labels, selected, device, args.eval_batch,
            target_mean, target_scale, sample_mean, sample_scale,
            deriv_mean, deriv_scale)
        metric = float(metrics["rmse_ps"])
        if not np.isfinite(metric):
            raise RuntimeError(f"Non-finite same-event evaluation RMSE at epoch {epoch}")
        if metric < best_rmse:
            best_rmse = metric
            best_epoch = epoch
            best_state = {key: value.detach().cpu().clone()
                          for key, value in model.state_dict().items()}
        old_lr = float(opt.param_groups[0]["lr"])
        if scheduler is not None:
            scheduler.step(metric)
        current_lr = float(opt.param_groups[0]["lr"])
        reduced = current_lr < old_lr
        if reduced:
            log.info("LR REDUCED | epoch=%d | %.8g -> %.8g | eval RMSE=%.5f ps "
                     "| best=%.5f ps", epoch, old_lr, current_lr, metric, best_rmse)
        row = {
            "epoch": epoch,
            "online_train_normalized_mse": epoch_loss / max(trained, 1),
            "eval_train_rmse_ps": metric,
            "eval_train_mae_ps": metrics["mae_ps"],
            "eval_train_max_abs_error_ps": float(
                np.max(np.abs(prediction - labels[selected]))),
            "learning_rate": current_lr,
            "lr_reduced": reduced,
            "elapsed_s": time.monotonic() - clock,
        }
        rows.append(row)
        if epoch % args.report_every == 0 or epoch == 1 or epoch == args.epochs or reduced:
            log.info("EPOCH %04d/%04d | online normalized MSE=%.7f | "
                     "SAME-EVENT eval RMSE=%.5f ps MAE=%.5f ps maxAE=%.5f ps "
                     "| LR=%.8g | best=%.5f ps (epoch %d) | elapsed=%.1fs",
                     epoch, args.epochs, row["online_train_normalized_mse"],
                     metric, metrics["mae_ps"],
                     row["eval_train_max_abs_error_ps"], current_lr,
                     best_rmse, best_epoch, row["elapsed_s"])
    if best_state is None:
        raise RuntimeError("No evaluation of trained network")
    model.load_state_dict(best_state)
    prediction, final_metrics = evaluate(
        model, waveforms, labels, selected, device, args.eval_batch,
        target_mean, target_scale, sample_mean, sample_scale,
        deriv_mean, deriv_scale)
    torch.save({
        "architecture_name": "temporal_cnn",
        "architecture": {
            "n_samples": int(waveforms.shape[-1]), "width": args.width,
            "depth": args.depth, "temporal_bins": args.temporal_bins,
            "use_derivative": args.use_derivative,
        },
        "model_state": best_state, "best_epoch": best_epoch,
        "target_mean_ps": target_mean, "target_std_ps": target_scale,
        "input_normalization": "per_sample",
        "input_sample_mean": sample_mean_array,
        "input_sample_scale": sample_scale_array,
        "input_derivative_mean": deriv_mean_array,
        "input_derivative_scale": deriv_scale_array,
        "selected_training_rows": selected,
        "provenance": provenance,
    }, output / "best_capacity_model.pt")
    np.savez_compressed(
        output / "diagnostics.npz",
        training_rows=selected, teacher_prediction_ps=labels[selected],
        surrogate_prediction_ps=prediction)
    with (output / "learning_curve.csv").open("w", encoding="utf-8",
                                               newline="") as file:
        writer = csv.DictWriter(file, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    from matplotlib import pyplot as plt
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot([r["epoch"] for r in rows],
            [r["eval_train_rmse_ps"] for r in rows], marker=".", markersize=2)
    ax.set(xlabel="Epoch", ylabel="RMSE on fixed training events (ps)",
           title="Full-signal CNN memorization diagnostic")
    ax.grid(alpha=.25)
    fig.tight_layout()
    fig.savefig(output / "memorization_rmse.png", dpi=180)
    plt.close(fig)
    summary = {
        "purpose": "deliberate overfit on frozen full-signal subset",
        "source_model_predict_calls": 0,
        "architecture": "TemporalConditionalCNN",
        "selected_fit_events": int(len(selected)),
        "reference_training_events": int(len(train_reference)),
        "normalization_population": "all eligible training events",
        "tuning_or_holdout_used_for_training": False,
        "early_stopping": False, "weight_decay": 0.,
        "lr_schedule": "disabled" if scheduler is None else "ReduceLROnPlateau",
        "final_learning_rate": float(opt.param_groups[0]["lr"]),
        "lr_reductions": int(sum(row["lr_reduced"] for row in rows)),
        "settings": vars(args) | {
            "model_dir": str(args.model_dir),
            "cache_root": str(args.cache_root) if args.cache_root else None,
            "artifact_root": str(args.artifact_root) if args.artifact_root else None,
            "output_root": str(output),
        },
        "best_epoch": best_epoch,
        "best_train": final_metrics,
        "normalization": sample_info,
        "derivative_normalization": deriv_info,
        "history": rows,
        "provenance": provenance,
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2),
                                          encoding="utf-8")
    log.info("COMPLETE | best epoch %d training RMSE %.5f ps | %s",
             best_epoch, final_metrics["rmse_ps"], output)
    return summary


def _parser():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--model-dir", required=True, type=Path)
    p.add_argument("--cache-root", type=Path)
    p.add_argument("--artifact-root", type=Path)
    p.add_argument("--output-root", required=True, type=Path)
    p.add_argument("--fit-events", type=int, default=256)
    p.add_argument("--tuning-queries", type=int, default=500)
    p.add_argument("--holdout-queries", type=int, default=500)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--use-derivative", action="store_true")
    p.add_argument("--width", type=int, default=64)
    p.add_argument("--depth", type=int, default=3)
    p.add_argument("--temporal-bins", type=int, default=8)
    p.add_argument("--epochs", type=int, default=400)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--eval-batch", type=int, default=64)
    p.add_argument("--learning-rate", type=float, default=0.0005)
    p.add_argument("--weight-decay", type=float, default=0.)
    p.add_argument("--no-lr-decay", action="store_true",
                   help="Disable adaptive LR decay to reproduce old fixed-LR behavior")
    p.add_argument("--lr-factor", type=float, default=0.5)
    p.add_argument("--lr-patience", type=int, default=20,
                   help="Plateau patience in epochs (eval every epoch)")
    p.add_argument("--lr-threshold-ps", type=float, default=0.05,
                   help="Minimum RMSE reduction treated as an improvement")
    p.add_argument("--lr-min", type=float, default=1e-6)
    p.add_argument("--lr-cooldown", type=int, default=5)
    p.add_argument("--normalization-chunk-size", type=int, default=256)
    p.add_argument("--normalization-scale-floor-ratio", type=float, default=0.05)
    p.add_argument("--report-every", type=int, default=5)
    p.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    return p


if __name__ == "__main__":
    run(_parser().parse_args())
