"""Full-signal distillation from independently control-fitted windowed Rocket/RDST.

Feature extraction is entirely independent of the saved teacher targets.
Each window's fitted transforms and development features are cached on disk.
Ridge is fit on development-training predictions, alpha selected by tuning,
and holdout evaluated once. No original-model inference or perturbation.
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import os
import pickle
from pathlib import Path

import numpy as np
from scipy.sparse.linalg import LinearOperator, lsqr

from waveform_analysis.core.io import canonical_hash
from waveform_analysis.data.prepared_cache import PreparedCacheLookup
from waveform_analysis.data.results import read_run, resolve_run_config
from waveform_analysis.models.kernel import multirocket_common as mr
from waveform_analysis.models.kernel import rdst_common as rd
from waveform_analysis.scripts.evaluate_conditional_knn import _load_inputs, _logger
from waveform_analysis.scripts.evaluate_full_signal_distillation import full_metrics

VERSION = 1


def disjoint_windows(samples: int, length: int):
    """Cover the signal exactly; merge too-short final segment into prior one."""
    if length < 11 or samples < 11:
        raise ValueError("Window length and signal length must both be >= 11 samples")
    blocks = [(s, min(s + length, samples)) for s in range(0, samples, length)]
    if len(blocks) > 1 and blocks[-1][1] - blocks[-1][0] < 11:
        blocks[-2] = (blocks[-2][0], blocks[-1][1])
        blocks.pop()
    return blocks


def split_budgets(total: int, family: str):
    if total < 6:
        raise ValueError("--features-per-window must be >= 6")
    if family == "multirocket":
        return total, 0
    if family == "rdst":
        return 0, total
    return total // 2, total - total // 2


def select_columns(actual: int, budget: int):
    if actual < 1 or budget < 1:
        return np.empty(0, dtype=np.int64)
    return np.linspace(0, actual - 1, min(actual, budget), dtype=np.int64)


def control_and_development(args, log):
    """Resolve prepared control from same canonical run/config as saved labels."""
    waves, labels, retained, provenance = _load_inputs(args, log)
    path = args.model_dir.expanduser().resolve()
    run = read_run(path)
    config = resolve_run_config(path)
    if args.cache_root is not None:
        base = args.cache_root.expanduser().resolve()
        config["preprocessing"]["cache_dir"] = str(base / "preprocessing")
        config["preprocessing"]["artifact_root"] = str(base / "artifact_store")
        config["preprocessing"]["named_cache_root"] = str(base)
    else:
        base = args.artifact_root.expanduser().resolve()
        config["preprocessing"]["artifact_root"] = str(base)
        config["preprocessing"]["cache_dir"] = str(base.parent / "preprocessing")
        config["preprocessing"].pop("named_cache_root", None)
    lookup = PreparedCacheLookup()
    control = lookup.find("control", config, run["manifest"])
    development = lookup.find("development", config, run["manifest"])
    family = "energy" if config["mode"] == "energy_to_energy" else "timing"
    control_waves = getattr(control, f"{family}_windows")
    control_time = getattr(control, f"{family}_time_ps")
    dev_time = getattr(development, f"{family}_time_ps")
    if control_waves is None or control_waves.ndim != 3:
        raise ValueError("No prepared control waveforms available")
    if control_waves.shape[1:] != waves.shape[1:]:
        raise ValueError("Control/development waveform grids differ")
    if control_time is None or dev_time is None:
        raise ValueError("Missing control/development time axes")
    if not np.allclose(control_time, dev_time, rtol=0, atol=1e-5):
        raise ValueError("Control/development time axes are not aligned")
    if not np.all(retained):
        control_waves = control_waves[:, :, retained]
        waves = waves[:, :, retained]
    if control_waves.shape[-1] != waves.shape[-1]:
        raise ValueError("Control and development retained lengths differ")
    if control.n_events < 2 or development.n_events != len(labels):
        raise ValueError("Invalid control/development population")
    log.info("FITTING POPULATION | control only | %d events | development=%d events",
             control.n_events, development.n_events)
    return control_waves, waves, labels, control, development, provenance


def _family_parameters(args, width: int, mr_count: int, rd_count: int):
    cfg = {"transform": {"n_jobs": args.n_jobs,
                         "batch_size": args.transform_batch_size}}
    mr_params = None
    rd_params = None
    if mr_count:
        mr_params = mr.transform_parameters({
            "n_kernels": max(84, math.ceil(mr_count / 4)),
            "max_dilations_per_kernel": min(32, max(1, width // 9)),
            "n_features_per_kernel": 4,
            "normalise": False,
        }, cfg)
    if rd_count:
        rd_params = rd.transform_parameters({
            "max_shapelets": max(1, math.ceil(rd_count / 3)),
            "shapelet_lengths": [11],
            "proba_normalization": 0.8,
            "threshold_percentiles": [5, 10],
            "alpha_similarity": 0.5,
            "use_prime_dilations": False,
        }, cfg)
    return cfg, mr_params, rd_params


def _save_pickle(path: Path, obj):
    tmp = path.with_name(path.name + ".partial")
    with tmp.open("wb") as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(tmp, path)


def _save_json(path: Path, payload):
    tmp = path.with_name(path.name + ".partial")
    tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    os.replace(tmp, path)


def _transform_features(artifacts, pair):
    pieces = []
    for name, module in (("multirocket", mr), ("rdst", rd)):
        fitted = artifacts.get(name)
        if fitted is None:
            continue
        result = np.asarray(module.transform(fitted["artifact"], pair),
                            dtype=np.float32)
        pieces.append(result[:, fitted["columns"]])
    if not pieces:
        raise RuntimeError("Window has no fitted features")
    return np.concatenate(pieces, axis=1)


def cached_window_features(args, control, development, block, number,
                           transform_base, development_identity, log):
    start, stop = block
    width = stop - start
    mr_count, rd_count = split_budgets(args.features_per_window, args.feature_family)
    cfg, mr_params, rd_params = _family_parameters(args, width, mr_count, rd_count)
    identity_payload = {
        "version": VERSION, "control_identity": transform_base,
        "start": start, "stop": stop, "seed": args.seed + number,
        "feature_family": args.feature_family,
        "features_per_window": args.features_per_window,
        "mr_params": mr_params, "rd_params": rd_params,
    }
    identity = canonical_hash(identity_payload)
    directory = args.transform_cache_root / identity
    directory.mkdir(parents=True, exist_ok=True)
    transform_path = directory / "transforms.pkl"
    transform_manifest = directory / "transform_manifest.json"
    log.info("TRANSFORM WINDOW %d | samples=[%d,%d) | %d samples | cache=%s",
             number + 1, start, stop, width, directory)
    if transform_path.is_file() and transform_manifest.is_file():
        state = json.loads(transform_manifest.read_text(encoding="utf-8"))
        if state.get("identity") != identity:
            raise ValueError("Cached transform identity mismatch")
        with transform_path.open("rb") as file:
            artifacts = pickle.load(file)
        log.info("TRANSFORM WINDOW %d | CACHE HIT | %s", number + 1,
                 state.get("selected_feature_counts"))
    else:
        control_pair = np.asarray(control[:, :, start:stop], dtype=np.float32)
        artifacts = {}
        for kind, module, params, count in (
            ("multirocket", mr, mr_params, mr_count),
            ("rdst", rd, rd_params, rd_count),
        ):
            if not count:
                continue
            log.info("TRANSFORM WINDOW %d | fitting %s on CONTROL | n=%d | requested=%d",
                     number + 1, kind, len(control_pair), count)
            artifact, _ = module.fit_transform(
                params, control_pair, seed=args.seed + 1009 * number + (0 if kind == "multirocket" else 1),
                config={**cfg, "_frozen_transform_only": True}, shared=False)
            available = int(artifact.metadata["feature_count"])
            columns = select_columns(available, count)
            if not len(columns):
                raise RuntimeError(f"{kind} returned no features in window {number}")
            artifacts[kind] = {"artifact": artifact, "columns": columns}
            log.info("TRANSFORM WINDOW %d | %s FIT COMPLETE | available=%d selected=%d",
                     number + 1, kind, available, len(columns))
        del control_pair
        _save_pickle(transform_path, artifacts)
        _save_json(transform_manifest, {
            **identity_payload, "identity": identity,
            "fit_role": "control",
            "selected_feature_counts": {
                name: len(value["columns"]) for name, value in artifacts.items()
            },
        })
        log.info("TRANSFORM WINDOW %d/%d | FIT COMPLETE and saved",
                 number + 1, args.total_windows)

    count_features = sum(len(v["columns"]) for v in artifacts.values())
    if count_features > args.features_per_window:
        raise ValueError("Cache exceeded feature budget")
    output = directory / f"development_{development_identity}.npy"
    done = directory / f"development_{development_identity}.json"
    if output.is_file() and done.is_file():
        meta = json.loads(done.read_text(encoding="utf-8"))
        if meta.get("identity") != identity or meta.get("n_events") != len(development):
            raise ValueError("Cached feature matrix identity/row mismatch")
        matrix = np.load(output, mmap_mode="r")
        if matrix.shape != (len(development), count_features):
            raise ValueError("Cached feature matrix shape mismatch")
        log.info("FEATURE WINDOW %d/%d | CACHE HIT | events=%d columns=%d",
                 number + 1, args.total_windows, len(development), count_features)
        return matrix, directory
    partial = output.with_name(output.name + ".partial")
    if partial.exists():
        partial.unlink()
    matrix = np.lib.format.open_memmap(
        partial, mode="w+", dtype=np.float32,
        shape=(len(development), count_features))
    try:
        for begin in range(0, len(development), args.transform_batch_size):
            end = min(begin + args.transform_batch_size, len(development))
            pair = np.asarray(development[begin:end, :, start:stop], dtype=np.float32)
            values = _transform_features(artifacts, pair)
            if values.shape != (end - begin, count_features) or not np.isfinite(values).all():
                raise RuntimeError("Nonfinite features or unexpected transform shape")
            matrix[begin:end] = values
            if (end == len(development) or begin == 0 or
                    end % args.log_every_events < args.transform_batch_size):
                log.info("FEATURE WINDOW %d/%d | %d/%d events (%.1f%%)",
                         number + 1, args.total_windows, end, len(development),
                         100. * end / len(development))
        matrix.flush()
    finally:
        mm = getattr(matrix, "_mmap", None)
        if mm is not None:
            mm.close()
        del matrix
    os.replace(partial, output)
    _save_json(done, {
        "identity": identity, "development_identity": development_identity,
        "n_events": len(development), "n_features": count_features,
        "fit_role": "control",
    })
    log.info("FEATURE WINDOW %d/%d | COMPLETE | events=%d features=%d",
             number + 1, args.total_windows, len(development), count_features)
    return np.load(output, mmap_mode="r"), directory


class BlockRidgeOperator:
    """Centered training feature operator over disk-resident block matrices."""

    def __init__(self, blocks, training_indices, chunk_size=1024):
        self.blocks = blocks
        self.rows = np.asarray(training_indices, dtype=np.int64)
        self.chunk = int(chunk_size)
        self.sizes = [int(a.shape[1]) for a in blocks]
        self.offsets = np.cumsum([0, *self.sizes]).astype(np.int64)
        self.means = []
        for block in blocks:
            total = np.zeros(block.shape[1], dtype=np.float64)
            for begin in range(0, len(self.rows), self.chunk):
                ids = self.rows[begin:begin + self.chunk]
                total += np.asarray(block[ids], dtype=np.float64).sum(axis=0)
            self.means.append(total / len(self.rows))
        self.col_mean = np.concatenate(self.means)

    @property
    def feature_count(self):
        return int(self.offsets[-1])

    def matvec(self, vector):
        vector = np.asarray(vector, dtype=np.float64)
        out = np.zeros(len(self.rows), dtype=np.float64)
        for j, block in enumerate(self.blocks):
            coeff = vector[self.offsets[j]:self.offsets[j + 1]]
            for begin in range(0, len(self.rows), self.chunk):
                end = min(begin + self.chunk, len(self.rows))
                out[begin:end] += np.asarray(
                    block[self.rows[begin:end]], dtype=np.float64) @ coeff
        out -= float(self.col_mean @ vector)
        return out

    def rmatvec(self, vector):
        vector = np.asarray(vector, dtype=np.float64)
        out = np.zeros(self.feature_count, dtype=np.float64)
        for j, block in enumerate(self.blocks):
            result = np.zeros(block.shape[1], dtype=np.float64)
            for begin in range(0, len(self.rows), self.chunk):
                end = min(begin + self.chunk, len(self.rows))
                result += np.asarray(block[self.rows[begin:end]], dtype=np.float64).T @ vector[begin:end]
            out[self.offsets[j]:self.offsets[j + 1]] = result
        out -= self.col_mean * float(vector.sum())
        return out

    def scipy_operator(self):
        return LinearOperator(
            shape=(len(self.rows), self.feature_count),
            matvec=self.matvec, rmatvec=self.rmatvec, dtype=np.float64)

    def predict(self, coefficients, indices):
        indices = np.asarray(indices, dtype=np.int64)
        result = np.zeros(len(indices), dtype=np.float64)
        for j, block in enumerate(self.blocks):
            weight = coefficients[self.offsets[j]:self.offsets[j + 1]]
            for begin in range(0, len(indices), self.chunk):
                end = min(begin + self.chunk, len(indices))
                result[begin:end] += np.asarray(
                    block[indices[begin:end]], dtype=np.float64) @ weight
        return result


def run(args):
    root = args.output_root.expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    log = _logger(root)
    if args.cache_root is None and args.artifact_root is None:
        raise ValueError("Supply --cache-root or --artifact-root for prepared data")
    if args.cache_root is not None and args.artifact_root is not None:
        raise ValueError("Supply only one of --cache-root and --artifact-root")
    if args.transform_cache_root is None:
        args.transform_cache_root = (
            args.cache_root if args.cache_root else args.artifact_root.parent
        ) / "windowed_rocket_surrogate"
    args.transform_cache_root = args.transform_cache_root.expanduser().resolve()
    args.transform_cache_root.mkdir(parents=True, exist_ok=True)
    if (args.lambda_min <= 0 or args.lambda_max < args.lambda_min or
            args.n_lambda < 1 or args.transform_batch_size < 1 or
            args.log_every_events < 1 or args.ridge_chunk_size < 1 or
            args.ridge_iter_limit < 1 or args.ridge_tol <= 0):
        raise ValueError("Invalid lambda, transform, or Ridge settings")
    log.info("FULL-SIGNAL WINDOWED MR/RDST SURROGATE | teacher black-box; saved predictions only")
    log.info("PARAMETERS | %s", vars(args))
    control, waves, teacher, control_data, development_data, provenance = (
        control_and_development(args, log))
    n = len(teacher)
    rng = np.random.default_rng(args.seed)
    if args.tuning_queries < 1 or args.holdout_queries < 1:
        raise ValueError("Tuning and holdout must be nonempty")
    if args.tuning_queries + args.holdout_queries >= n:
        raise ValueError("No training events remain")
    chosen = rng.choice(n, args.tuning_queries + args.holdout_queries, replace=False)
    tuning, holdout = chosen[:args.tuning_queries], chosen[args.tuning_queries:]
    training_mask = np.ones(n, dtype=bool)
    training_mask[chosen] = False
    training = np.flatnonzero(training_mask)
    log.info("SPLIT | training=%d tuning=%d holdout=%d seed=%d",
             len(training), len(tuning), len(holdout), args.seed)
    windows = disjoint_windows(waves.shape[-1], args.window_length)
    args.total_windows = len(windows)
    log.info("WINDOWS | count=%d requested_length=%d sample_lengths=%s",
             len(windows), args.window_length, [b-a for a,b in windows])
    transform_identity = str(control_data.manifest["analysis_protocol_identity"])
    dev_identity = canonical_hash({
        "development_identity": development_data.manifest["analysis_protocol_identity"],
        "events": n, "version": VERSION,
    })[:24]
    matrices, locations = [], []
    for i, block in enumerate(windows):
        matrix, cache_dir = cached_window_features(
            args, control, waves, block, i, transform_identity, dev_identity, log)
        matrices.append(matrix)
        locations.append(cache_dir)
        log.info("TOTAL TRANSFORM PROGRESS | %d/%d windows complete | %.1f%%",
                 i + 1, len(windows), 100 * (i + 1) / len(windows))
    operator = BlockRidgeOperator(matrices, training, args.ridge_chunk_size)
    log.info("FEATURE CACHE COMPLETE | total_features=%d | disk_estimate=%.2f GiB | %s",
             operator.feature_count,
             sum(m.nbytes for m in matrices) / (1024. ** 3),
             args.transform_cache_root)
    centered_y = teacher[training] - float(teacher[training].mean())
    source_mean = float(teacher[training].mean())
    linear_operator = operator.scipy_operator()
    candidates = np.geomspace(args.lambda_min, args.lambda_max, args.n_lambda)
    results = []
    best_score = np.inf
    best = None
    for index, alpha in enumerate(candidates, 1):
        log.info("RIDGE LAMBDA %d/%d | lambda=%.9g | fitting %d rows x %d features",
                 index, len(candidates), alpha, len(training), operator.feature_count)
        solution = lsqr(
            linear_operator, centered_y, damp=np.sqrt(float(alpha)),
            atol=args.ridge_tol, btol=args.ridge_tol,
            iter_lim=args.ridge_iter_limit)
        coef = np.asarray(solution[0], dtype=np.float64)
        intercept = source_mean - float(operator.col_mean @ coef)
        tuning_pred = intercept + operator.predict(coef, tuning)
        metrics = full_metrics(teacher[tuning], tuning_pred)
        entry = {
            "lambda": float(alpha), "tuning": metrics,
            "lsqr_stop": int(solution[1]),
            "lsqr_iterations": int(solution[2]),
            "lsqr_residual_norm": float(solution[3]),
        }
        results.append(entry)
        log.info("RIDGE LAMBDA %d/%d | alpha=%.9g | tuning RMSE=%.5f ps "
                 "MAE=%.5f ps bias=%.5f ps | LSQR stop=%d iterations=%d",
                 index, len(candidates), alpha,
                 metrics["rmse_ps"], metrics["mae_ps"], metrics["bias_ps"],
                 entry["lsqr_stop"], entry["lsqr_iterations"])
        if entry["lsqr_stop"] == 7:
            log.warning("RIDGE LAMBDA %.9g | hit LSQR iteration limit; "
                        "increase --ridge-iter-limit to improve numerical convergence", alpha)
        if metrics["rmse_ps"] < best_score:
            best_score = metrics["rmse_ps"]
            best = (float(alpha), coef.copy(), intercept)
        # Preserve partial scan progress even if interrupted.
        _save_json(root / "lambda_scan.json", results)
    if best is None:
        raise RuntimeError("No Ridge candidate was evaluated")
    best_alpha, coefficients, intercept = best
    tuning_pred = intercept + operator.predict(coefficients, tuning)
    holdout_pred = intercept + operator.predict(coefficients, holdout)
    tuning_stats = full_metrics(teacher[tuning], tuning_pred)
    holdout_stats = full_metrics(teacher[holdout], holdout_pred)
    log.info("SELECTED | alpha=%.9g | tuning RMSE=%.5f ps MAE=%.5f ps",
             best_alpha, tuning_stats["rmse_ps"], tuning_stats["mae_ps"])
    log.info("HOLDOUT (one time) | RMSE=%.5f ps MAE=%.5f ps bias=%.5f ps",
             holdout_stats["rmse_ps"], holdout_stats["mae_ps"], holdout_stats["bias_ps"])
    np.savez_compressed(
        root / "surrogate_ridge.npz",
        coefficients=coefficients, intercept=np.array(intercept),
        lambda_selected=np.array(best_alpha),
        window_limits=np.asarray(windows, dtype=np.int64),
        column_offsets=operator.offsets,
        training_column_means=operator.col_mean)
    np.savez_compressed(
        root / "diagnostics.npz",
        tuning_rows=tuning, holdout_rows=holdout,
        tuning_reference_ps=teacher[tuning], tuning_reconstructed_ps=tuning_pred,
        holdout_reference_ps=teacher[holdout], holdout_reconstructed_ps=holdout_pred)
    summary = {
        "method": "full-signal windowed MultiRocket/RDST -> Ridge distillation",
        "teacher_loaded": False, "teacher_inference_calls": 0,
        "control_fit_role": "control",
        "control_events": control_data.n_events,
        "development_events": n, "training_events": len(training),
        "tuning_events": len(tuning), "holdout_events": len(holdout),
        "seed": args.seed, "parameters": {
            "window_length": args.window_length,
            "features_per_window": args.features_per_window,
            "feature_family": args.feature_family,
            "lambda_min": args.lambda_min,
            "lambda_max": args.lambda_max,
            "n_lambda": args.n_lambda,
            "ridge_tol": args.ridge_tol,
            "ridge_iter_limit": args.ridge_iter_limit,
        },
        "windows": windows, "column_offsets": operator.offsets.tolist(),
        "cache_directories": [str(x) for x in locations],
        "total_features": operator.feature_count,
        "selected_lambda": best_alpha,
        "lambda_scan": results,
        "tuning": tuning_stats, "holdout": holdout_stats,
        "source": provenance,
    }
    _save_json(root / "summary.json", summary)
    log.info("COMPLETE | %s | control transforms cached and development blocks reusable",
             root)
    return summary


def parser():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--model-dir", required=True, type=Path)
    p.add_argument("--cache-root", type=Path)
    p.add_argument("--artifact-root", type=Path)
    p.add_argument("--transform-cache-root", type=Path,
                   help="Dedicated cache for control-fitted transforms and development feature blocks")
    p.add_argument("--output-root", required=True, type=Path)
    p.add_argument("--window-length", type=int, default=256)
    p.add_argument("--features-per-window", type=int, default=256)
    p.add_argument("--feature-family", choices=("both", "multirocket", "rdst"),
                   default="both")
    p.add_argument("--lambda-min", type=float, default=1e2)
    p.add_argument("--lambda-max", type=float, default=1e7)
    p.add_argument("--n-lambda", type=int, default=7)
    p.add_argument("--tuning-queries", type=int, default=500)
    p.add_argument("--holdout-queries", type=int, default=500)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--n-jobs", type=int, default=2)
    p.add_argument("--transform-batch-size", type=int, default=128)
    p.add_argument("--log-every-events", type=int, default=8192)
    p.add_argument("--ridge-chunk-size", type=int, default=1024)
    p.add_argument("--ridge-iter-limit", type=int, default=100)
    p.add_argument("--ridge-tol", type=float, default=1e-4)
    return p


if __name__ == "__main__":
    run(parser().parse_args())
