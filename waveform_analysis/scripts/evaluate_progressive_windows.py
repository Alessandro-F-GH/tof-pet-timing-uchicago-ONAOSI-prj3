#!/usr/bin/env python3
"""Development-only CV with separately refitted cumulative-window transforms.

For each cutoff, fit MultiRocket/RDST on the FULL [start, cutoff) waveform
using control events only. Evaluate frozen features on development CV folds.
Select Ridge penalties once on the largest window with the original batch
grid/pruning, then KEEP THEIR EFFECTIVE VALUES FIXED for every cutoff.

Never opens blind events; saves only CV results/plots, no features or models.
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


def temporal_cutoffs(time_ps, sample_mask, start_ns, end_ns, step_ns,
                     min_samples=16):
    """Full prefix [start, cutoff) indices at each requested cutoff.

    Samples masked by the model's frozen preprocessing are excluded. No
    independent temporal blocks are constructed or concatenated.
    """
    times = np.asarray(time_ps, np.float64) / 1000.
    valid = np.asarray(sample_mask, bool)
    if times.ndim != 1 or valid.shape != times.shape:
        raise ValueError("Time grid and model sample mask must match")
    if step_ns <= 0 or end_ns <= start_ns:
        raise ValueError("Invalid progressive time limits")
    ends = np.arange(start_ns + step_ns, end_ns, step_ns, dtype=np.float64)
    ends = np.append(ends, float(end_ns))
    windows = []
    for end in ends:
        indices = np.flatnonzero(
            (times >= start_ns - 1e-8) & (times < end - 1e-8) & valid
        )
        if len(indices) >= min_samples:
            windows.append((float(start_ns), float(end), indices))
    if not windows:
        raise ValueError("No cumulative window with enough retained samples")
    return windows


def fingerprint(payload):
    return hashlib.sha256(json.dumps(payload, sort_keys=True, default=str).encode()).hexdigest()[:20]


def batch_model_space(family):
    """Read transform parameters directly from the checked-in batch model space."""
    name = {"both": "direct_multirocket_rdst",
            "multirocket": "direct_multirocket",
            "rdst": "direct_rdst"}[family]
    path = (Path(__file__).resolve().parents[1] / "config" /
            "model_spaces" / f"{name}.json")
    return json.loads(path.read_text(encoding="utf-8"))


def window_features(control, development, indices, family, space, seed):
    """Fit the full prefix using EXACT batch transform settings."""
    x_ctrl = np.ascontiguousarray(control[:, :, indices], dtype=np.float32)
    x_dev = np.ascontiguousarray(development[:, :, indices], dtype=np.float32)
    configured = {key: spec["value"] for key, spec in space["parameters"].items()
                  if spec["type"] == "fixed"}
    execution = {"transform": space["transform"]}
    pieces, counts = [], {}
    if family in ("multirocket", "both"):
        params = mr.transform_parameters(configured, execution)
        model, _ = mr.fit_transform(
            params, x_ctrl, seed=seed,
            config={"_frozen_transform_only": True}, shared=False)
        feat = mr.transform(model, x_dev)
        pieces.append(np.asarray(feat, np.float32))
        counts["multirocket"] = int(feat.shape[1])
        del feat, model
    if family in ("rdst", "both"):
        params = rd.transform_parameters(configured, execution)
        model, _ = rd.fit_transform(
            params, x_ctrl, seed=seed + 1, config={}, shared=False)
        feat = rd.transform(model, x_dev)
        pieces.append(np.asarray(feat, np.float32))
        counts["rdst"] = int(feat.shape[1])
        del feat, model
    return np.concatenate(pieces, axis=1), counts


def alpha_grid(family):
    """Use the exact candidate spaces configured for the batch-study models."""
    name = {"both": "direct_multirocket_rdst", "multirocket": "direct_multirocket",
            "rdst": "direct_rdst"}[family]
    path = Path(__file__).resolve().parents[1] / "config" / "model_spaces" / (name + ".json")
    space = json.loads(path.read_text(encoding="utf-8"))
    # The transform settings and penalty grids both match this batch model space.
    candidates = [
        {key: float(value) for key, value in candidate.items()
         if key in ("multirocket_alpha", "rdst_alpha", "ridge_alpha")}
        for candidate in grid_candidates(space)
    ]
    return candidate_manifest(candidates)


def original_feature_counts(family):
    """Reference feature counts implied by the saved batch model-space settings.

    These are per-family counts, NOT the sum across time blocks. The benchmark
    MultiRocket transform outputs raw + first-difference pooling features.
    RDST outputs three values per fitted shapelet.
    """
    name = {"both": "direct_multirocket_rdst",
            "multirocket": "direct_multirocket",
            "rdst": "direct_rdst"}[family]
    path = (Path(__file__).resolve().parents[1] / "config" /
            "model_spaces" / (name + ".json"))
    params = json.loads(path.read_text(encoding="utf-8"))["parameters"]
    reference = {}
    if family in ("both", "multirocket"):
        reference["multirocket"] = (
            2 * int(params["n_kernels"]["value"]) *
            int(params["n_features_per_kernel"]["value"])
        )
    if family in ("both", "rdst"):
        reference["rdst"] = 3 * int(params["max_shapelets"]["value"])
    return reference


def feature_counts(features, rdst_ranges, family):
    """Exact counts, including any realized RDST shapelet reduction."""
    n_rdst = sum(int(stop - start) for start, stop in rdst_ranges)
    if family == "rdst":
        return {"rdst": int(features.shape[1])}
    if family == "multirocket":
        return {"multirocket": int(features.shape[1])}
    return {
        "multirocket": int(features.shape[1] - n_rdst),
        "rdst": int(n_rdst),
    }


def effective_alphas(features, rdst_ranges, family, parameters,
                     reference_counts):
    """Normalize batch alpha grid ONCE at the largest window.

    Formula preserves the earlier exploratory feature-count correction.
    These numerical alphas MUST then remain fixed across all cutoffs.
    """
    actual = feature_counts(features, rdst_ranges, family)
    names = (("multirocket", "multirocket_alpha"),
             ("rdst", "rdst_alpha")) if family == "both" else ((family, "ridge_alpha"),)
    scaled = {}
    for group, parameter in names:
        count, reference = int(actual[group]), int(reference_counts[group])
        base = float(parameters[parameter])
        if count < 1 or reference < 1 or base <= 0 or not np.isfinite(base):
            raise ValueError("Invalid feature-normalized alpha")
        scaled[parameter] = float(base * count / reference)
    return scaled


def regularized_feature_matrix(features, rdst_ranges, family, alphas):
    """Apply the same two-family Ridge penalty reparameterization as batch.

    Values passed here are ALREADY the final effective alpha values. Do not
    rescale them at each cutoff: this maintains identical regularization.
    """
    if family != "both":
        alpha = float(alphas["ridge_alpha"])
        if alpha <= 0 or not np.isfinite(alpha):
            raise ValueError("Ridge alpha must be positive and finite")
        return features, alpha
    mr_alpha = float(alphas["multirocket_alpha"])
    rdst_alpha = float(alphas["rdst_alpha"])
    if min(mr_alpha, rdst_alpha) <= 0:
        raise ValueError("Family Ridge penalties must be positive")
    ratio = np.sqrt(mr_alpha / rdst_alpha)
    if not np.isfinite(ratio) or ratio <= 0:
        raise ValueError("Invalid family penalty ratio")
    x = np.asarray(features, dtype=np.float32).copy()
    for start, stop in rdst_ranges:
        x[:, start:stop] *= ratio
    return x, mr_alpha


def evaluate_prefix(features, rdst_ranges, target, split, family, candidates,
                    pruning, criterion, fit_cfg, seed, force_complete=False):
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
            incumbent_rows=incumbent_rows, force_complete=force_complete,
        )
        summaries.append(candidate)
        rows_by_candidate[identifier] = rows
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
    parser.add_argument("--max-control", type=int, default=1024)
    parser.add_argument("--max-development", type=int, default=3000)
    parser.add_argument("--seed", type=int, default=None,
                        help="Override the existing batch seed (default: saved run seed)")
    args = parser.parse_args()
    if args.step_ns <= 0 or args.end_ns <= args.start_ns:
        parser.error("Invalid time range")
    if args.max_control < 10 or args.max_development < 10:
        parser.error("Invalid sample counts")
    output = root_relative(args.output)
    output.mkdir(parents=True, exist_ok=True)
    waves, times, run, manifest, _, control, mask, target = load_wide_development(
        root_relative(args.results), args.channel, root_relative(args.cache_root),
        return_transform=True, include_control=True, include_sample_mask=True,
        include_target=True)
    run_config = resolve_run_config(run)
    args.seed = int(run_config["seed"] if args.seed is None else args.seed)
    cv_settings = run_config["cross_validation"]
    pruning = cv_settings["pruning"]
    criterion = cv_settings["metric"]
    fold_count = int(cv_settings["folds"])
    fit_cfg = run_config["fit"]
    if int(pruning["startup_complete_candidates"]) < 0:
        raise ValueError("Invalid batch-study pruning startup setting")

    selected_control = np.sort(np.random.default_rng(args.seed+7).choice(
        len(control), min(len(control), args.max_control), replace=False))
    selected_dev = np.sort(np.random.default_rng(args.seed+13).choice(
        len(waves), min(len(waves), args.max_development), replace=False))
    controls = np.asarray(control[selected_control], np.float32)
    dev = np.asarray(waves[selected_dev], np.float32)
    y = np.asarray(target[selected_dev], np.float64)
    if len(y) < max(fold_count, fold_count * int(cv_settings["minimum_events_per_fold"])):
        raise ValueError("Not enough development events for the configured CV folds")
    windows = temporal_cutoffs(times, mask, args.start_ns, args.end_ns,
                               args.step_ns)
    space = batch_model_space(args.family)
    shapelet_length = (max(space["parameters"]["shapelet_lengths"]["value"])
                       if args.family in ("rdst", "both") else 0)
    if args.family in ("rdst", "both") and any(
            len(indices) < shapelet_length for _, _, indices in windows):
        raise ValueError("Shapelet length exceeds an included cumulative window")
    split = make_cv_split(
        len(y),
        population_identity=str(manifest["analysis_protocol_identity"]) + ":" +
                            fingerprint(selected_dev.tolist()),
        batch_seed=args.seed,
        n_folds=fold_count, shuffle=bool(cv_settings["shuffle"]))
    grid = alpha_grid(args.family)
    reference_counts = original_feature_counts(args.family)
    settings = {k: v for k, v in vars(args).items()
                if k not in ("output", "results", "cache_root")}
    settings.update(
        control_identity=manifest.get("analysis_protocol_identity"),
        source_run=str(run), selected_control=selected_control.tolist(),
        selected_dev=selected_dev.tolist(),
        windows=[(a,b,idx.tolist()) for a,b,idx in windows])
    atomic_json(output/"settings.json", {
        "settings": settings, "blind_accessed": False,
        "cross_validation": cv_settings,
        "alpha_selection": "full-window development CV and batch pruning",
        "fixed_alpha_across_cutoffs": True,
        "reference_feature_counts": reference_counts,
        "batch_transform_settings": space,
        "feature_cache": False,
        "final_fit": False})

    # The largest window is run FIRST to select two effective penalties using
    # development CV. All remaining cutoffs use those exact numerical values.
    # No final fit is performed and no transform or feature matrix is saved.
    start_full, end_full, full_indices = windows[-1]
    print(f"Selecting fixed alphas with full window {start_full:g}..{end_full:g} ns",
          flush=True)
    full_features, full_counts = window_features(
        controls, dev, full_indices, args.family, space,
        args.seed + 3 * (len(windows)-1))
    full_rdst_ranges = ([(int(full_counts["multirocket"]),
                          int(full_features.shape[1]))]
                        if args.family == "both" else [])
    # Keep batch grid candidates, converting their penalties to effective
    # values only ONCE using the full-window feature counts.
    full_candidates = candidate_manifest([
        effective_alphas(full_features, full_rdst_ranges, args.family, params,
                         reference_counts)
        for params in grid.values()
    ])
    selected, full_candidate_rows, full_fold_rows = evaluate_prefix(
        full_features, full_rdst_ranges, y, split, args.family,
        full_candidates, pruning, criterion, fit_cfg, args.seed)
    fixed_alphas = dict(selected["parameters"])
    atomic_json(output/"selected_alphas.json", {
        "selection_cutoff_ns": end_full, "selected_candidate_id": selected["candidate_id"],
        "fixed_effective_alphas": fixed_alphas,
        "selection_metric": criterion, "cv_score_ps": selected[criterion + "_mean_ps"],
        "selection_cv_only": True,
        "selection_is_not_independent_of_reported_cv": True,
    })
    print(f"Fixed effective Ridge alphas selected by full-window CV: {fixed_alphas}",
          flush=True)

    summary, candidates_out, folds_out = [], [], []
    def store_cutoff(cutoff, counts, features, candidate, candidate_rows, fold_rows):
        row = dict(cutoff_ns=cutoff, features=int(features.shape[1]),
                   n_development=len(y), selection_metric=criterion,
                   alpha_source_cutoff_ns=end_full,
                   **fixed_alphas,
                   ctr_mean_ps=candidate["ctr_mean_ps"],
                   ctr_std_ps=candidate["ctr_std_ps"],
                   rmse_mean_ps=candidate["rmse_mean_ps"],
                   rmse_std_ps=candidate["rmse_std_ps"])
        summary.append(row)
        for c in candidate_rows:
            candidates_out.append(dict(
                cutoff_ns=cutoff, candidate_id=c["candidate_id"],
                parameters=json.dumps(c["parameters"], sort_keys=True),
                pruned=c["pruned"], completed_folds=c["completed_folds"],
                total_folds=c["total_folds"], pruning_reason=c["pruning_reason"],
                ctr_mean_ps=c["ctr_mean_ps"], rmse_mean_ps=c["rmse_mean_ps"]))
        for fold in fold_rows:
            folds_out.append(dict(cutoff_ns=cutoff, **fold))
        csv_output(output/"cv_summary.csv", summary)
        csv_output(output/"cv_candidates.csv", candidates_out)
        csv_output(output/"cv_folds.csv", folds_out)
        print(f"  cutoff={cutoff:g} ns | features={features.shape[1]} | "
              f"CV CTR={candidate['ctr_mean_ps']:.2f} ps | "
              f"CV RMSE={candidate['rmse_mean_ps']:.2f} ps", flush=True)

    for i, (_, cutoff, indices) in enumerate(windows):
        if i == len(windows)-1:
            features, counts = full_features, full_counts
            candidates, fold_rows, winner = full_candidate_rows, full_fold_rows, selected
        else:
            print(f"[{i+1}/{len(windows)}] fitting entire window "
                  f"{args.start_ns:g}..{cutoff:g} ns", flush=True)
            features, counts = window_features(
                controls, dev, indices, args.family, space, args.seed + 3*i)
            rdst_ranges = ([(int(counts["multirocket"]),
                             int(features.shape[1]))]
                           if args.family == "both" else [])
            one_candidate = candidate_manifest([fixed_alphas])
            winner, candidates, fold_rows = evaluate_prefix(
                features, rdst_ranges, y, split, args.family,
                one_candidate, pruning, criterion, fit_cfg, args.seed,
                force_complete=True)
        store_cutoff(cutoff, counts, features, winner, candidates, fold_rows)
        if i != len(windows)-1:
            del features
    del full_features
    summary.sort(key=lambda row: row["cutoff_ns"])
    csv_output(output/"cv_summary.csv", summary)
    plot_summary(summary, output, args.channel.upper())
    print(f"CV complete, no final model and no blind evaluation: {output}", flush=True)


if __name__ == "__main__":
    main()
