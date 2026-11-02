#!/usr/bin/env python3
"""Exploratory development-only CV as time-block features accumulate.

Each independent temporal block is transformed ONCE using control-only fitted
MultiRocket/RDST, and its development features are cached. At cutoff t the
Ridge sees concatenated blocks whose END is <= t. Never loads blind data.

This is deliberately NOT a standard increasing-window MultiRocket/RDST:
independent block transforms can learn scale-dependent features.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import Ridge

from waveform_analysis.core.io import atomic_json
from waveform_analysis.data.imputation import load_wide_development, root_relative
from waveform_analysis.data.results import resolve_run_config
from waveform_analysis.data.splits import make_cv_split, semantic_seed
from waveform_analysis.engine.search import candidate_manifest, grid_candidates
from waveform_analysis.engine.validation import best_complete_candidate, evaluate_candidate
from waveform_analysis.models.kernel import multirocket_common as mr, rdst_common as rd
from waveform_analysis.reporting.stats import metric_values


def temporal_blocks(time_ps, sample_mask, start_ns, end_ns, step_ns):
    """Return nonoverlapping time intervals, using only retained model samples."""
    times = np.asarray(time_ps, np.float64) / 1000.
    valid = np.asarray(sample_mask, bool)
    boundaries = np.arange(start_ns, end_ns, step_ns)
    boundaries = np.r_[boundaries, end_ns]
    blocks = []
    for a, b in zip(boundaries[:-1], boundaries[1:]):
        indices = np.flatnonzero((times >= a - 1e-8) & (times < b - 1e-8) & valid)
        if indices.size >= 16:
            blocks.append((float(a), float(b), indices))
    if not blocks:
        raise ValueError("No blocks with at least 16 retained samples")
    return blocks


def fingerprint(payload):
    return hashlib.sha256(json.dumps(payload, sort_keys=True, default=str).encode()).hexdigest()[:20]


def block_features(control, development, indices, args, seed):
    """Fit two existing transformer families on control; transform development."""
    x_ctrl = np.ascontiguousarray(control[:, :, indices], dtype=np.float32)
    x_dev = np.ascontiguousarray(development[:, :, indices], dtype=np.float32)
    pieces = []
    counts = {}
    if args.family in ("multirocket", "both"):
        parameters = mr.transform_parameters({
            "n_kernels": args.kernels,
            "max_dilations_per_kernel": args.dilations,
            "n_features_per_kernel": 4, "normalise": False,
        }, {"transform": {"n_jobs": args.jobs, "batch_size": args.batch_size}})
        model, _ = mr.fit_transform(parameters, x_ctrl, seed=seed, config={
            "_frozen_transform_only": True}, shared=False)
        feat = mr.transform(model, x_dev)
        pieces.append(np.asarray(feat, np.float32))
        counts["multirocket"] = int(feat.shape[1])
    if args.family in ("rdst", "both"):
        parameters = rd.transform_parameters({
            "max_shapelets": args.shapelets,
            "shapelet_lengths": [args.shapelet_length],
            "proba_normalization": 0.8,
            "threshold_percentiles": [5, 10],
            "alpha_similarity": 0.5, "use_prime_dilations": False,
        }, {"transform": {"n_jobs": args.jobs}})
        model, _ = rd.fit_transform(parameters, x_ctrl, seed=seed+1, config={},
                                    shared=False)
        feat = rd.transform(model, x_dev)
        pieces.append(np.asarray(feat, np.float32))
        counts["rdst"] = int(feat.shape[1])
    return np.concatenate(pieces, axis=1), counts


def alpha_grid(family):
    """Use the exact candidate spaces configured for the batch-study models."""
    name = {"both": "direct_multirocket_rdst", "multirocket": "direct_multirocket",
            "rdst": "direct_rdst"}[family]
    path = Path(__file__).resolve().parents[1] / "config" / "model_spaces" / (name + ".json")
    space = json.loads(path.read_text(encoding="utf-8"))
    # Kernel and shapelet counts are intentionally overridden for this study.
    # Only the ORIGINAL Ridge penalty grid is reused here.
    candidates = [
        {key: float(value) for key, value in candidate.items()
         if key in ("multirocket_alpha", "rdst_alpha", "ridge_alpha")}
        for candidate in grid_candidates(space)
    ]
    return candidate_manifest(candidates)


def regularized_feature_matrix(features, rdst_ranges, family, parameters):
    """Identical two-alpha scaling to kernel.multirocket_rdst_common.fit.

    min ||y - X_m b_m - X_r b_r||²
        + alpha_m ||b_m||² + alpha_r ||b_r||².
    Scaling X_r by sqrt(alpha_m/alpha_r), then Ridge(alpha_m), is
    algebraically equivalent. Never mutate cached features.
    """
    if family != "both":
        return features, float(parameters["ridge_alpha"])
    mr_alpha = float(parameters["multirocket_alpha"])
    rdst_alpha = float(parameters["rdst_alpha"])
    ratio = np.sqrt(mr_alpha / rdst_alpha)
    if not np.isfinite(ratio) or ratio <= 0:
        raise ValueError("Invalid family penalty ratio")
    x = features.copy()
    for start, stop in rdst_ranges:
        x[:, start:stop] *= ratio
    return x, mr_alpha


def evaluate_prefix(features, rdst_ranges, target, split, family, candidates,
                    pruning, criterion, fit_cfg, seed):
    """Run the SAME grid-CV/pruning implementation used by batch studies.

    The only scores produced here are development CV fold scores. There is
    no final fit and no independent evaluation population.
    """
    led_folds = {
        int(fold.fold_id): metric_values(
            target[np.asarray(fold.validation, np.int64)], fit_cfg,
            seed=semantic_seed(seed, "led", int(fold.fold_id))
        )
        for fold in split.folds
    }
    summaries = []
    fold_rows = []
    by_candidate = {}
    rows_by_candidate = {}
    startup = int(pruning["startup_complete_candidates"])

    for identifier, parameters in candidates.items():
        incumbent = None if sum(
            not row["pruned"] and row["completed_folds"] == row["total_folds"]
            for row in summaries
        ) < startup else best_complete_candidate(summaries, criterion)
        incumbent_id = incumbent["candidate_id"] if incumbent is not None else None
        incumbent_rows = (
            rows_by_candidate.get(incumbent_id) if incumbent_id is not None else None
        )
        x, alpha = regularized_feature_matrix(
            features, rdst_ranges, family, parameters
        )
        def evaluate_fold(parameters, fold):
            tr = np.asarray(fold.train, np.int64)
            va = np.asarray(fold.validation, np.int64)
            model = Ridge(alpha=alpha, fit_intercept=True,
                          solver="lsqr", tol=1e-4)
            model.fit(x[tr], target[tr])
            residual = target[va] - model.predict(x[va])
            result = metric_values(
                residual, fit_cfg,
                seed=semantic_seed(seed, identifier, int(fold.fold_id), "metric")
            )
            led = led_folds[int(fold.fold_id)]
            return {
                "n_train": int(len(tr)),
                "n_validation": int(len(va)),
                "ctr_ps": result["ctr_ps"], "rmse_ps": result["rmse_ps"],
                "led_ctr_ps": led["ctr_ps"], "led_rmse_ps": led["rmse_ps"],
            }
        candidate, rows = evaluate_candidate(
            candidate_id=identifier,
            parameters=parameters,
            folds=split.folds,
            evaluate_fold=evaluate_fold,
            metric=criterion,
            pruning=pruning,
            incumbent_candidate_id=incumbent_id,
            incumbent_rows=incumbent_rows,
        )
        summaries.append(candidate)
        rows_by_candidate[identifier] = rows
        by_candidate[identifier] = (x, alpha)
        # Candidate-level details, including incomplete pruned candidates.
        fold_rows.extend(rows)
        print(
            f"    alpha={parameters} | {criterion.upper()}="
            f"{candidate[criterion + '_mean_ps']:.2f} ps | "
            f"folds={candidate['completed_folds']}/{candidate['total_folds']} | "
            f"{'PRUNED (' + str(candidate['pruning_reason']) + ')' if candidate['pruned'] else 'complete'}",
            flush=True,
        )
        del x

    winner = best_complete_candidate(summaries, criterion)
    if winner is None:
        # Same batch-study fallback: fully evaluate the best partial candidate
        # only so that a comparison score is available at this cutoff.
        finite = [
            row for row in summaries
            if np.isfinite(float(row[criterion + "_mean_ps"]))
        ]
        if not finite:
            raise RuntimeError("Every candidate failed to produce a CV score")
        pick = min(finite, key=lambda row: (
            float(row[criterion + "_mean_ps"]), row["candidate_id"]))
        identifier = pick["candidate_id"]
        params = candidates[identifier]
        x, alpha = regularized_feature_matrix(
            features, rdst_ranges, family, params
        )
        def evaluate_fold(parameters, fold):
            tr = np.asarray(fold.train, np.int64)
            va = np.asarray(fold.validation, np.int64)
            model = Ridge(alpha=alpha, fit_intercept=True,
                          solver="lsqr", tol=1e-4)
            model.fit(x[tr], target[tr])
            residual = target[va] - model.predict(x[va])
            result = metric_values(
                residual, fit_cfg,
                seed=semantic_seed(seed, identifier, int(fold.fold_id), "metric")
            )
            led = led_folds[int(fold.fold_id)]
            return {
                "n_train": int(len(tr)), "n_validation": int(len(va)),
                "ctr_ps": result["ctr_ps"], "rmse_ps": result["rmse_ps"],
                "led_ctr_ps": led["ctr_ps"], "led_rmse_ps": led["rmse_ps"],
            }
        winner, rows = evaluate_candidate(
            candidate_id=identifier, parameters=params, folds=split.folds,
            evaluate_fold=evaluate_fold, metric=criterion, pruning=pruning,
            existing_fold_rows=rows_by_candidate[identifier],
            force_complete=True,
        )
        summaries = [winner if row["candidate_id"] == identifier else row
                     for row in summaries]
        fold_rows = [row for row in fold_rows if row["candidate_id"] != identifier]
        fold_rows.extend(rows)
        del x
    return winner, summaries, fold_rows


def csv_output(path, rows):
    if not rows:
        return
    with open(path, "w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def plot_summary(rows, output, label):
    fig, ax = plt.subplots(figsize=(8, 4.3))
    x = [r["cutoff_ns"] for r in rows]
    ax.errorbar(x, [r["ctr_mean_ps"] for r in rows],
                yerr=[r["ctr_std_ps"] for r in rows],
                marker="o", capsize=3, label="CV CTR")
    ax.set_xlabel("Last included time (ns)")
    ax.set_ylabel("CV CTR (ps)")
    ax.set_title(label + " — cumulative independent time-block features")
    ax.grid(alpha=.3)
    fig.tight_layout()
    fig.savefig(output / "cv_ctr_vs_cutoff.pdf")
    fig.savefig(output / "cv_ctr_vs_cutoff.png", dpi=170)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(8, 4.3))
    ax.errorbar(x, [r["rmse_mean_ps"] for r in rows],
                yerr=[r["rmse_std_ps"] for r in rows],
                marker="o", capsize=3, label="CV RMSE")
    ax.set_xlabel("Last included time (ns)")
    ax.set_ylabel("CV RMSE (ps)")
    ax.grid(alpha=.3)
    fig.tight_layout()
    fig.savefig(output / "cv_rmse_vs_cutoff.pdf")
    fig.savefig(output / "cv_rmse_vs_cutoff.png", dpi=170)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--channel", choices=("energy", "timing"), default="energy")
    parser.add_argument("--output", type=Path, default=Path("progressive_window_cv"))
    parser.add_argument("--family", choices=("both", "multirocket", "rdst"), default="both")
    parser.add_argument("--start-ns", type=float, default=-2.)
    parser.add_argument("--end-ns", type=float, default=50.)
    parser.add_argument("--step-ns", type=float, default=2.)
    parser.add_argument("--kernels", type=int, default=84)
    parser.add_argument("--dilations", type=int, default=4)
    parser.add_argument("--shapelets", type=int, default=16)
    parser.add_argument("--shapelet-length", type=int, default=11)
    parser.add_argument("--ridge-alpha", type=float, default=10000.)
    parser.add_argument("--max-control", type=int, default=1024)
    parser.add_argument("--max-development", type=int, default=3000)
    parser.add_argument("--folds", type=int, default=3)
    parser.add_argument("--jobs", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fit-bin-width-ps", type=float, default=10.)
    args = parser.parse_args()
    if args.step_ns <= 0 or args.end_ns <= args.start_ns or args.kernels < 84:
        parser.error("Invalid time range or kernels (<84)")
    if (args.ridge_alpha <= 0 or args.folds < 2 or args.shapelets < 1
            or args.max_control < 10 or args.max_development < args.folds * 5
            or args.dilations < 1 or args.shapelet_length < 2):
        parser.error("Invalid Ridge alpha, folds or shapelets")
    output = root_relative(args.output)
    output.mkdir(parents=True, exist_ok=True)
    waves, times, run, manifest, _, control, mask, target = load_wide_development(
        root_relative(args.results), args.channel, root_relative(args.cache_root),
        return_transform=True, include_control=True, include_sample_mask=True,
        include_target=True)
    selected_control = np.sort(np.random.default_rng(args.seed+7).choice(
        len(control), min(len(control), args.max_control), replace=False))
    selected_dev = np.sort(np.random.default_rng(args.seed+13).choice(
        len(waves), min(len(waves), args.max_development), replace=False))
    controls = np.asarray(control[selected_control], np.float32)
    dev = np.asarray(waves[selected_dev], np.float32)
    y = np.asarray(target[selected_dev], np.float64)
    if len(y) < args.folds * 5:
        raise ValueError("Not enough development events for CV")
    blocks = temporal_blocks(times, mask, args.start_ns, args.end_ns, args.step_ns)
    if args.family in ("rdst", "both") and any(len(b[2]) < args.shapelet_length for b in blocks):
        raise ValueError("Shapelet length exceeds a selected block")
    fit_cfg = {"histogram_bin_width_ps": args.fit_bin_width_ps}
    settings = {k: v for k, v in vars(args).items()
                if k not in ("output", "results", "cache_root")}
    settings.update(control_identity=manifest.get("analysis_protocol_identity"),
                    source_run=str(run), selected_control=selected_control.tolist(),
                    selected_dev=selected_dev.tolist(),
                    blocks=[(a,b,idx.tolist()) for a,b,idx in blocks])
    sig = fingerprint(settings)
    (output/"settings.json").write_text(json.dumps(
        {"fingerprint": sig, "settings": settings, "blind_accessed": False},
        indent=2, default=str), encoding="utf-8")
    splits = list(KFold(n_splits=args.folds, shuffle=True,
                        random_state=args.seed).split(np.arange(len(y))))
    features = []
    summary, folds_out = [], []
    for block_id, (begin, end, indices) in enumerate(blocks):
        name = f"features_{sig}_block_{block_id:03d}.npy"
        path = output/name
        if path.is_file():
            z = np.load(path, mmap_mode="r")
            if z.shape[0] != len(y) or z.ndim != 2 or not np.all(np.isfinite(z)):
                raise ValueError(f"Invalid cached transform: {path}")
            print(f"[{block_id+1}/{len(blocks)}] reused block {begin:g}..{end:g} ns", flush=True)
        else:
            print(f"[{block_id+1}/{len(blocks)}] fitting transforms for {begin:g}..{end:g} ns", flush=True)
            z, counts = block_features(controls, dev, indices, args, args.seed+block_id*3)
            np.save(path, z)
            print(f"  saved {name}: features={z.shape[1]} families={counts}", flush=True)
        features.append(np.asarray(z, np.float32))
        # Only concatenate EXISTING feature matrices, no transform repeated.
        prefix = np.concatenate(features, axis=1)
        metrics = evaluate_prefix(prefix, y, splits, args, fit_cfg)
        for row in metrics:
            folds_out.append(dict(cutoff_ns=end, n_blocks=block_id+1,
                                  features=prefix.shape[1], **row))
        result = dict(cutoff_ns=end, n_blocks=block_id+1,
                      features=prefix.shape[1], n_development=len(y),
                      ctr_mean_ps=float(np.mean([r["ctr_ps"] for r in metrics])),
                      ctr_std_ps=float(np.std([r["ctr_ps"] for r in metrics], ddof=1)),
                      rmse_mean_ps=float(np.mean([r["rmse_ps"] for r in metrics])),
                      rmse_std_ps=float(np.std([r["rmse_ps"] for r in metrics], ddof=1)))
        summary.append(result)
        csv_output(output/"cv_summary.csv", summary)
        csv_output(output/"cv_folds.csv", folds_out)
        print(f"  cutoff {end:g} ns | features={prefix.shape[1]} | "
              f"CV CTR={result['ctr_mean_ps']:.2f} ± {result['ctr_std_ps']:.2f} ps | "
              f"RMSE={result['rmse_mean_ps']:.2f} ps", flush=True)
    plot_summary(summary, output, args.channel.upper())
    print(f"Finished; outputs: {output}", flush=True)


if __name__ == "__main__":
    main()
