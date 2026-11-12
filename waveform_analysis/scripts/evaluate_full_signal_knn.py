"""Exact full-waveform kNN regression of saved teacher predictions.

Training reference is the SAME 74,714-event training cohort as full-signal
CNN distillation; tuning and holdout are completely excluded from the
reference pool. Mean/scale are fitted solely on training waveforms using the
same per-time-sample standardization as the neural surrogates.

Uses batched BLAS distance calculations and streaming reference blocks.
No teacher inference, internal teacher features, windows, or GPU required.
"""
from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np

from waveform_analysis.engine.sample_normalization import fit_time_sample_stats
from waveform_analysis.scripts.evaluate_conditional_knn import _load_inputs, _logger
from waveform_analysis.scripts.evaluate_full_signal_distillation import full_metrics


def _standardized_flat(waves, indices, mean, scale):
    data = np.asarray(waves[indices], dtype=np.float32)
    data = (data - mean[None, :, :]) / scale[None, :, :]
    return np.ascontiguousarray(data.reshape(len(indices), -1))


def _predict_knn(dist, label, k_values, weighted):
    """Aggregate sorted nearest distances/labels for each query and k."""
    order = np.argsort(dist, axis=1, kind="stable")
    distances = np.take_along_axis(dist, order, axis=1)
    values = np.take_along_axis(label, order, axis=1)
    sum_value = np.cumsum(values, axis=1)
    uniform = sum_value[:, k_values - 1] / k_values[None, :]
    if not weighted:
        return uniform[:, :, None]
    # Exact matches receive all weight; other neighbours are ignored.
    exact = distances <= 1e-12
    exact_count = np.cumsum(exact, axis=1)
    exact_sum = np.cumsum(np.where(exact, values, 0), axis=1)
    weights = 1.0 / np.maximum(np.sqrt(np.maximum(distances, 0)), 1e-8)
    weighted_value = np.cumsum(values * weights, axis=1)
    weight_sum = np.cumsum(weights, axis=1)
    weighted_prediction = weighted_value[:, k_values - 1] / weight_sum[:, k_values - 1]
    has_exact = exact_count[:, k_values - 1] > 0
    weighted_prediction = np.where(
        has_exact, exact_sum[:, k_values - 1] /
        np.maximum(exact_count[:, k_values - 1], 1), weighted_prediction)
    return np.stack([uniform, weighted_prediction], axis=-1)


def _reference_search(waves, labels, training, queries, mean, scale,
                      k_values, weighted, reference_batch, query_batch, log,
                      log_every):
    """Return [queries, k, weights] estimates, with bounded reference RAM."""
    topk = int(k_values[-1])
    results = np.empty((len(queries), len(k_values), 2 if weighted else 1),
                       dtype=np.float64)
    nearest = np.empty(len(queries), dtype=np.float64)
    start = time.monotonic()
    log.info("SEARCH START | queries=%d reference=%d dimensions=%d kmax=%d "
             "reference_batch=%d query_batch=%d", len(queries), len(training),
             int(np.prod(waves.shape[1:])), topk, reference_batch, query_batch)
    for query_start in range(0, len(queries), query_batch):
        query_rows = queries[query_start:query_start + query_batch]
        q = _standardized_flat(waves, query_rows, mean, scale)
        q_norm = np.einsum("ij,ij->i", q, q, dtype=np.float64)
        best_distance = np.full((len(q), topk), np.inf, dtype=np.float64)
        best_labels = np.zeros((len(q), topk), dtype=np.float64)
        for ref_start in range(0, len(training), reference_batch):
            ids = training[ref_start:ref_start + reference_batch]
            reference = _standardized_flat(waves, ids, mean, scale)
            r_norm = np.einsum("ij,ij->i", reference, reference, dtype=np.float64)
            # float64 accumulation for squared norms; SGEMM for cross products.
            inner = q @ reference.T
            d2 = q_norm[:, None] + r_norm[None, :] - 2.0 * inner.astype(np.float64)
            np.maximum(d2, 0.0, out=d2)
            distances = np.concatenate((best_distance, d2), axis=1)
            candidate_labels = np.concatenate(
                (best_labels, np.broadcast_to(labels[ids], (len(q), len(ids)))), axis=1)
            take = np.argpartition(distances, topk - 1, axis=1)[:, :topk]
            best_distance = np.take_along_axis(distances, take, axis=1)
            best_labels = np.take_along_axis(candidate_labels, take, axis=1)
        results[query_start:query_start + len(q)] = _predict_knn(
            best_distance, best_labels, k_values, weighted)
        nearest[query_start:query_start + len(q)] = np.sqrt(best_distance.min(axis=1))
        done = query_start + len(q)
        if done == len(queries) or done == len(q) or (
                done // query_batch) % log_every == 0:
            elapsed = time.monotonic() - start
            log.info("SEARCH PROGRESS | %d/%d queries | elapsed=%.1fs | ETA=%.1fs",
                     done, len(queries), elapsed,
                     elapsed / done * (len(queries) - done))
    return results, nearest


def _score_grid(target, estimates):
    return {
        "rmse_ps": np.sqrt(np.mean((estimates - target[:, None, None]) ** 2,
                                    axis=0)).tolist(),
        "mae_ps": np.mean(np.abs(estimates - target[:, None, None]), axis=0).tolist(),
    }


def run(args):
    output = args.output_root.expanduser().resolve()
    log = _logger(output)
    log.info("FULL-WAVEFORM KNN | train-only per-time normalization | no windows")
    log.info("Configuration=%s", vars(args))
    k_values = np.unique(np.asarray(args.k, dtype=np.int64))
    if k_values.size == 0 or k_values[0] < 1 or min(
            args.reference_batch, args.query_batch, args.tuning_queries,
            args.holdout_queries, args.log_every) < 1:
        raise ValueError("Invalid k, block size, query count or log interval")
    waves, labels, retained, provenance = _load_inputs(args, log)
    if not np.all(retained):
        log.warning("Using all prepared coordinates; retained mask differs")
    rng = np.random.default_rng(args.seed)
    n = len(labels)
    if args.tuning_queries + args.holdout_queries >= n:
        raise ValueError("Train cohort must not be empty")
    chosen = rng.choice(n, args.tuning_queries + args.holdout_queries,
                        replace=False)
    tuning = chosen[:args.tuning_queries]
    holdout = chosen[args.tuning_queries:]
    training_flag = np.ones(n, dtype=bool)
    training_flag[chosen] = False
    training = np.flatnonzero(training_flag)
    if int(k_values[-1]) > len(training):
        raise ValueError("k must not exceed training reference size")
    log.info("SPLIT | train=%d tuning=%d holdout=%d seed=%d",
             len(training), len(tuning), len(holdout), args.seed)
    mean, scale, normalization_details = fit_time_sample_stats(
        waves, training, chunk_size=args.normalization_chunk_size,
        scale_floor_ratio=args.normalization_scale_floor_ratio)
    log.info("NORMALIZATION | per_sample | %s", normalization_details)
    log.info("KNN uses entire paired standardized waveform, no teacher features")
    tuning_predictions, tuning_nearest = _reference_search(
        waves, labels, training, tuning, mean, scale, k_values, args.weighted,
        args.reference_batch, args.query_batch, log, args.log_every)
    tuning_grid = _score_grid(labels[tuning], tuning_predictions)
    tuning_mse = np.mean((tuning_predictions - labels[tuning, None, None]) ** 2,
                         axis=0)
    best_k_index, best_weight_index = np.unravel_index(
        np.argmin(tuning_mse), tuning_mse.shape)
    selected = {
        "k": int(k_values[best_k_index]),
        "weight": "inverse_distance" if best_weight_index else "uniform",
        "tuning_rmse_ps": float(np.sqrt(tuning_mse[best_k_index, best_weight_index]))
    }
    log.info("TUNING CHOICE | %s", selected)
    holdout_predictions, holdout_nearest = _reference_search(
        waves, labels, training, holdout, mean, scale, k_values, args.weighted,
        args.reference_batch, args.query_batch, log, args.log_every)
    holdout_grid = _score_grid(labels[holdout], holdout_predictions)
    final_tuning = tuning_predictions[:, best_k_index, best_weight_index]
    final_holdout = holdout_predictions[:, best_k_index, best_weight_index]
    tuning_metrics = full_metrics(labels[tuning], final_tuning)
    holdout_metrics = full_metrics(labels[holdout], final_holdout)
    log.info("FINAL | k=%d weight=%s tuning RMSE=%.4f ps "
             "holdout RMSE=%.4f ps MAE=%.4f ps | target std=%.4f ps",
             selected["k"], selected["weight"], tuning_metrics["rmse_ps"],
             holdout_metrics["rmse_ps"], holdout_metrics["mae_ps"],
             holdout_metrics["original_std_ps"])
    np.savez_compressed(output / "diagnostics.npz",
        tuning_query_rows=tuning, holdout_query_rows=holdout,
        tuning_original_prediction_ps=labels[tuning],
        tuning_reconstructed_prediction_ps=final_tuning,
        holdout_original_prediction_ps=labels[holdout],
        holdout_reconstructed_prediction_ps=final_holdout,
        tuning_nearest_distance=tuning_nearest,
        holdout_nearest_distance=holdout_nearest,
        k_values=k_values, tuning_estimates=tuning_predictions,
        holdout_estimates=holdout_predictions)
    with (output / "metrics.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["k", "weight", "tuning_rmse_ps", "holdout_rmse_ps",
                         "tuning_mae_ps", "holdout_mae_ps"])
        for ki, value in enumerate(k_values):
            for wi, name in enumerate(("uniform", "inverse_distance")
                                      if args.weighted else ("uniform",)):
                writer.writerow([int(value), name,
                    tuning_grid["rmse_ps"][ki][wi], holdout_grid["rmse_ps"][ki][wi],
                    tuning_grid["mae_ps"][ki][wi], holdout_grid["mae_ps"][ki][wi]])
    summary = {
        "method": "full-waveform normalized Euclidean kNN regression",
        "teacher_model_calls": 0,
        "teacher_internal_features_used": False,
        "provenance": provenance,
        "settings": {
            "seed": args.seed,
            "training_events": len(training),
            "tuning_events": len(tuning),
            "holdout_events": len(holdout),
            "normalization": "per_sample",
            "normalization_details": normalization_details,
            "k_values": k_values.tolist(),
            "weighted": args.weighted,
            "reference_batch": args.reference_batch,
            "query_batch": args.query_batch,
        },
        "selection": selected,
        "tuning": tuning_metrics,
        "holdout": holdout_metrics,
        "tuning_grid": tuning_grid,
        "holdout_grid": holdout_grid,
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2),
                                         encoding="utf-8")
    log.info("COMPLETE | %s", output)
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
    p.add_argument("--k", type=int, nargs="+", default=[1, 3, 5, 10, 20, 50, 100])
    p.add_argument("--weighted", action="store_true")
    p.add_argument("--normalization-chunk-size", type=int, default=256)
    p.add_argument("--normalization-scale-floor-ratio", type=float, default=0.05)
    p.add_argument("--reference-batch", type=int, default=1024)
    p.add_argument("--query-batch", type=int, default=16)
    p.add_argument("--log-every", type=int, default=5,
                   help="Log every N query batches")
    return p


if __name__ == "__main__":
    run(_parser().parse_args())
