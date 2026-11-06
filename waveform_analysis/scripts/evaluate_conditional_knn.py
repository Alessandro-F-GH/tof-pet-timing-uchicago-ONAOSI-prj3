"""Validate conditional KNN on saved development predictions, without model inference.

Run from the repository root:
  python -m waveform_analysis.scripts.evaluate_conditional_knn \\
    --model-dir waveform_analysis/results/UC/.../direct_multirocket_rdst \\
    --cache-root "$HOME/UC_cache" --output-root "$HOME/conditional_knn" \\
    --weighted --k 5 10 20 50 100 200 500

This command NEVER trains, loads or invokes an ML model/feature transformer.
Missing or incompatible saved development predictions are a hard error.
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
from pathlib import Path
import time

import numpy as np

from waveform_analysis.data.results import read_run, resolve_run_config
from waveform_analysis.data.prepared_cache import PreparedCacheLookup
from waveform_analysis.data.storage import RunStore
from waveform_analysis.data.view import mode_family
from waveform_analysis.engine.conditional_knn import ConditionalKNN, tshap_windows


def _logger(output):
    output.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("conditional_knn")
    logger.handlers.clear()
    logger.setLevel(logging.INFO)
    logger.propagate = False
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    for handler in (logging.StreamHandler(), logging.FileHandler(output / "evaluation.log", mode="w")):
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger


def _prediction_payload(path, dataset, manifest):
    """Strictly align saved predictions with prepared development event IDs."""
    if not path.is_file():
        raise FileNotFoundError(
            f"Saved development predictions not found: {path}. "
            "Run the existing diagnostic separately if needed; conditional KNN never predicts."
        )
    with np.load(path, allow_pickle=False) as archive:
        required = ("event_id", "prediction_ps", "final_fit_fingerprint",
                    "analysis_protocol_identity", "dataset_role")
        absent = [name for name in required if name not in archive]
        if absent:
            raise ValueError(f"Development prediction artifact lacks {absent}: {path}")
        ids = np.asarray(archive["event_id"], dtype=np.int64)
        predictions = np.asarray(archive["prediction_ps"], dtype=np.float64)
        metadata = {key: str(np.asarray(archive[key]).item())
                    for key in required[2:]}
    expected = np.asarray(dataset.event_index, dtype=np.int64)
    if ids.ndim != 1 or predictions.shape != ids.shape or ids.size != expected.size:
        raise ValueError("Saved prediction/event count does not match prepared development population")
    if np.unique(ids).size != ids.size or np.unique(expected).size != expected.size:
        raise ValueError("Duplicate event IDs: cannot establish unambiguous alignment")
    if metadata["dataset_role"] != "development":
        raise ValueError("Saved predictions do not belong to the development population")
    fingerprint = manifest.get("stage_fingerprints", {}).get("final_fit")
    if not fingerprint or metadata["final_fit_fingerprint"] != str(fingerprint):
        raise ValueError("Saved predictions do not match the final fitted model fingerprint")
    # Protocol fingerprints can change after the artifact-store refactor even
    # when the selected, prepared event population is unchanged. Enforce the
    # saved final-fit fingerprint and exact event IDs instead.
    order = np.argsort(ids, kind="stable")
    if not np.array_equal(ids[order], np.sort(expected)):
        raise ValueError("Saved prediction IDs differ from the prepared development event IDs")
    lookup = np.searchsorted(ids[order], expected)
    aligned = predictions[order[lookup]]
    if not np.all(np.isfinite(aligned)):
        raise ValueError("Saved model predictions include NaN or infinity")
    return aligned


def _load_inputs(args, log):
    run_path = args.model_dir.expanduser().resolve()
    run = read_run(run_path)
    config = resolve_run_config(run_path)
    if args.cache_root is not None and args.artifact_root is not None:
        raise ValueError("Choose --cache-root OR --artifact-root")
    if args.cache_root is not None:
        root = args.cache_root.expanduser().resolve()
        if not (root / "artifact_store").is_dir():
            raise FileNotFoundError(f"Prepared cache artifact store missing: {root / 'artifact_store'}")
        config["preprocessing"]["cache_dir"] = str(root / "preprocessing")
        config["preprocessing"]["artifact_root"] = str(root / "artifact_store")
        config["preprocessing"]["named_cache_root"] = str(root)
    elif args.artifact_root is not None:
        config["preprocessing"]["artifact_root"] = str(args.artifact_root.expanduser().resolve())
        config["preprocessing"].pop("named_cache_root", None)
    log.info("Selected existing run=%s, model=%s, mode=%s", run_path, run["manifest"]["model"], config["mode"])
    log.info("Data source: PREPARED development waveforms; prediction source: SAVED development predictions")
    log.info("Inference, feature transformation, model loading and refitting: DISABLED")
    dataset = PreparedCacheLookup().find("development", config, run["manifest"])
    family = mode_family(config["mode"])
    waveforms = getattr(dataset, f"{family}_windows")
    if waveforms is None:
        raise ValueError(f"No prepared {family} waveform array")
    if waveforms.ndim != 3 or waveforms.shape[1] != 2 or waveforms.shape[0] != dataset.n_events:
        raise ValueError("Prepared waveforms are not [events, two detectors, time]")
    # Historical run population metadata may differ after cache refactoring.
    # Do not compare it here: rely on exact event-ID alignment and saved
    # prediction provenance instead.
    store = RunStore(run_path, artifact_root=config["preprocessing"].get("artifact_root"))
    path = store.development_predictions_path
    # Legacy run-local artifacts are accepted only with full embedded provenance.
    if not path.is_file() and (run_path / "artifacts" / "development_pred.npz").is_file():
        path = run_path / "artifacts" / "development_pred.npz"
    predictions = _prediction_payload(path, dataset, run["manifest"])
    keep = (np.ones(waveforms.shape[-1], dtype=bool)
            if config["mode"] == "energy_to_energy"
            else dataset.sample_mask)
    if keep is None:
        raise ValueError("Timing sample mask is unavailable")
    log.info("Prepared artifact=%s | shape=%s | dtype=%s", dataset.directory, waveforms.shape, waveforms.dtype)
    log.info("Saved predictions=%s | n=%d | range=[%.6g, %.6g] ps",
             path, predictions.size, predictions.min(), predictions.max())
    log.info("Retained time samples=%d/%d, reference population=%d (no train cap)",
             int(np.sum(keep)), waveforms.shape[-1], dataset.n_events)
    return waveforms, predictions, np.asarray(keep, bool), {
        "run": str(run_path), "model": run["manifest"]["model"], "mode": config["mode"],
        "dataset": str(dataset.directory), "prediction_artifact": str(path),
        "final_fit_fingerprint": run["manifest"]["stage_fingerprints"]["final_fit"],
        "analysis_protocol_identity": dataset.manifest.get("analysis_protocol_identity"),
        "reference_events": int(dataset.n_events),
    }


def _scores(errors):
    """MSE and signed bias per coalitional class, k, weighting."""
    return {"rmse": np.sqrt(np.mean(errors * errors, axis=(0, 1))),
            "bias": np.mean(errors, axis=(0, 1)),
            "by_window_rmse": np.sqrt(np.mean(errors * errors, axis=0))}


def _choose(errors, k_values, weighted):
    # Balance W and complement equally. Each event and window contributes once.
    mse = np.mean(errors * errors, axis=(0, 1))
    # shape: [W,R,k,weight].
    combined = np.mean(mse, axis=0)
    ix = np.unravel_index(np.argmin(combined), combined.shape)
    one = {"k": int(k_values[ix[0]]),
           "weight": "inverse_distance" if ix[1] else "uniform",
           "validation_mse": float(combined[ix])}
    two = {}
    for side, name in enumerate(("window", "complement")):
        sub = np.unravel_index(np.argmin(mse[side]), mse[side].shape)
        two[name] = {"k": int(k_values[sub[0]]),
                     "weight": "inverse_distance" if sub[1] else "uniform",
                     "validation_mse": float(mse[side][sub])}
    return {"single": one, "separate": two}


def _selected_mse(errors, choice, k_values):
    squared = errors * errors
    def at(side, c):
        k = int(np.flatnonzero(k_values == c["k"])[0])
        w = 1 if c["weight"] == "inverse_distance" else 0
        return float(np.mean(squared[:, :, side, k, w]))
    single = float(np.mean([at(0, choice["single"]), at(1, choice["single"])]))
    separate = float(np.mean([at(0, choice["separate"]["window"]),
                              at(1, choice["separate"]["complement"])]))
    return {"single_rmse_ps": float(np.sqrt(single)),
            "separate_rmse_ps": float(np.sqrt(separate)),
            "single_mse_ps2": single, "separate_mse_ps2": separate}


def _evaluate(engine, indices, labels, log, title, log_every):
    nwin = len(engine.starts)
    values = np.empty((len(indices), nwin, 2, len(engine.k_values),
                       2 if engine.weighted else 1), dtype=np.float64)
    nearest = np.empty((len(indices), nwin, 2), dtype=np.float64)
    radius = np.empty_like(nearest)
    start = time.monotonic()
    log.info("%s START | queries=%d | windows=%d | coalitions=%d | reference=%d | k=%s | weighted=%s",
             title, len(indices), nwin, 2, len(engine.labels), engine.k_values.tolist(), engine.weighted)
    for step, idx in enumerate(indices, 1):
        predictions, nearest[step-1], radius[step-1] = engine.estimate(int(idx))
        values[step-1] = predictions - labels[int(idx)]
        if step == 1 or step % log_every == 0 or step == len(indices):
            elapsed = time.monotonic() - start
            rate = elapsed / step
            log.info("%s PROGRESS | %d/%d (%.1f%%) | elapsed=%.1fs | mean/query=%.3fs | ETA=%.1fs | last_event_row=%d",
                     title, step, len(indices), 100 * step / len(indices), elapsed,
                     rate, rate * (len(indices)-step), int(idx))
    log.info("%s DONE | total=%.2fs | error_shape=%s", title, time.monotonic()-start, values.shape)
    return values, nearest, radius


def _write_csv(path, k_values, weighted, tuning, holdout):
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["coalition", "k", "weights", "tuning_rmse_ps", "tuning_bias_ps",
                         "holdout_rmse_ps", "holdout_bias_ps"])
        for side, name in enumerate(("window", "complement")):
            for ki, k in enumerate(k_values):
                for wi, weight in enumerate(("uniform", "inverse_distance") if weighted else ("uniform",)):
                    writer.writerow([name, int(k), weight,
                                     float(tuning["rmse"][side, ki, wi]),
                                     float(tuning["bias"][side, ki, wi]),
                                     float(holdout["rmse"][side, ki, wi]),
                                     float(holdout["bias"][side, ki, wi])])


def run(args):
    log = _logger(args.output_root.expanduser().resolve())
    log.info("Conditional KNN benchmark | saved predictions only | method=exact normalized Euclidean")
    log.info("Configuration: %s", vars(args))
    waves, predictions, keep, provenance = _load_inputs(args, log)
    starts, keep = tshap_windows(waves.shape[-1], args.window_samples,
                                args.stride_samples, keep)
    n = len(predictions)
    k = np.unique(np.asarray(args.k, dtype=np.int64))
    if k.size == 0 or k[0] < 1 or k[-1] >= n:
        raise ValueError("All k must satisfy 1 <= k < total development events")
    if args.tuning_queries < 1 or args.holdout_queries < 1:
        raise ValueError("Both tuning and holdout need at least one query")
    if args.tuning_queries + args.holdout_queries > n:
        raise ValueError("Requested more distinct queries than development events")
    rng = np.random.default_rng(args.seed)
    selected = rng.choice(n, size=args.tuning_queries + args.holdout_queries, replace=False)
    tuning_ids = selected[:args.tuning_queries]
    holdout_ids = selected[args.tuning_queries:]
    log.info("Validation scheme=leave-one-out, separate query cohorts | tuning=%d | holdout=%d | seed=%d",
             len(tuning_ids), len(holdout_ids), args.seed)
    log.info("Mask geometry: width=%d stride=%d | eligible interior windows=%d | starts=%s",
             args.window_samples, args.stride_samples, len(starts), starts.tolist())
    log.info("Search strategy: one batched reference-distance pass per query; shared window/complement prefix sums")
    log.info("Memory strategy: retain only per-query window distances, never full query x reference x window tensor")
    engine = ConditionalKNN(waves, predictions, starts=starts, width=args.window_samples,
                            keep=keep, k_values=k, weighted=args.weighted,
                            reference_batch=args.reference_batch)
    tune, near_tune, rad_tune = _evaluate(engine, tuning_ids, predictions, log,
                                          "TUNING", args.log_every)
    tuning_scores = _scores(tune)
    selection = _choose(tune, k, args.weighted)
    log.info("Selection on tuning only: %s", json.dumps(selection))
    test, near_test, rad_test = _evaluate(engine, holdout_ids, predictions, log,
                                          "HOLDOUT", args.log_every)
    holdout_scores = _scores(test)
    test_choice = _selected_mse(test, selection, k)
    log.info("Independent LOO query holdout | selected single/separate k metrics: %s", test_choice)
    output = args.output_root.expanduser().resolve()
    np.savez_compressed(output / "diagnostics.npz",
        window_starts=starts, k_values=k,
        tuning_query_rows=tuning_ids, holdout_query_rows=holdout_ids,
        tuning_rmse_by_window=tuning_scores["by_window_rmse"],
        holdout_rmse_by_window=holdout_scores["by_window_rmse"],
        tuning_nearest_distance=near_tune,
        holdout_nearest_distance=near_test,
        tuning_kmax_distance=rad_tune,
        holdout_kmax_distance=rad_test,
    )
    _write_csv(output / "metrics.csv", k, args.weighted, tuning_scores, holdout_scores)
    summary = {
        "purpose": "conditional expectation regression, not waveform imputation",
        "inference_calls": 0, "model_retrained": False, "feature_transforms_applied": False,
        "provenance": provenance,
        "settings": {"k": k.tolist(), "weighted": bool(args.weighted),
                     "window_samples": args.window_samples,
                     "stride_samples": args.stride_samples, "seed": args.seed,
                     "tuning_queries": len(tuning_ids), "holdout_queries": len(holdout_ids),
                     "reference_batch": args.reference_batch},
        "selection": selection, "holdout_selected": test_choice,
        "tuning_rmse_by_coalition": tuning_scores["rmse"].tolist(),
        "holdout_rmse_by_coalition": holdout_scores["rmse"].tolist(),
        "output_files": ["evaluation.log", "metrics.csv", "diagnostics.npz", "summary.json"],
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    log.info("COMPLETE | output=%s | no predictions were recalculated", output)
    return summary


def _parser():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--model-dir", required=True, type=Path,
                        help="One completed saved run directory")
    parser.add_argument("--cache-root", type=Path,
                        help="Portable prepared-cache bundle, e.g. ~/UC_cache")
    parser.add_argument("--artifact-root", type=Path,
                        help="Optional existing artifact store (instead of --cache-root)")
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--window-samples", type=int, default=20)
    parser.add_argument("--stride-samples", type=int, default=20)
    parser.add_argument("--k", type=int, nargs="+", default=[5, 10, 20, 50, 100, 200, 500])
    parser.add_argument("--weighted", action="store_true",
                        help="Also evaluate inverse-distance weighted means")
    parser.add_argument("--tuning-queries", type=int, default=500)
    parser.add_argument("--holdout-queries", type=int, default=500)
    parser.add_argument("--reference-batch", type=int, default=2048)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    return parser


if __name__ == "__main__":
    options = _parser().parse_args()
    if options.log_every < 1:
        raise ValueError("--log-every must be >= 1")
    run(options)
