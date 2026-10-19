"""Sparse Lasso distillation of saved frozen-feature regression models.

Train on ORIGINAL transformed development features and saved teacher-model outputs;
blind inputs/outputs are evaluation-only. No synthetic samples or group penalties.
Run with: python -m waveform_analysis.scripts.lasso_surrogate --results PATH
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
from pathlib import Path
import tempfile
import warnings

import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import Lasso
from sklearn.preprocessing import StandardScaler

from waveform_analysis.core.io import atomic_json, canonical_hash
from waveform_analysis.data.results import discover_runs, resolve_run_config
from waveform_analysis.data.splits import semantic_seed
from waveform_analysis.data.view import model_target
from waveform_analysis.engine.diagnostics import diagnostic_random_state
from waveform_analysis.engine.regression_explainer import (
    fidelity, frozen_features, predict_features,
)
from waveform_analysis.engine.train import (
    load_fitted_model, release_training_memory, saved_model_complete,
)
from waveform_analysis.engine.xai_replay import PreparedCacheLookup, _cache_root
from waveform_analysis.models import get_model

DEFAULT_ALPHA_RATIOS = (1., .3, .1, .03, .01, .003, .001)


def _positive_unique(values, label):
    values = [float(value) for value in values]
    if not values or not all(np.isfinite(v) and v > 0 for v in values):
        raise ValueError(f"{label} must contain finite positive values")
    if len(set(values)) != len(values):
        raise ValueError(f"{label} must not contain duplicates")
    return sorted(values, reverse=True)


def _validate_inputs(development, development_output, blind, blind_output):
    xdev = np.asarray(development)
    xblind = np.asarray(blind)
    ydev = np.asarray(development_output, dtype=np.float64)
    yblind = np.asarray(blind_output, dtype=np.float64)
    if (xdev.ndim != 2 or xblind.ndim != 2 or xdev.shape[1] != xblind.shape[1]
            or xdev.shape[1] < 1 or xdev.shape[0] < 2 or xblind.shape[0] < 1):
        raise ValueError("Expected nonempty, matching two-dimensional frozen features")
    if ydev.shape != (len(xdev),) or yblind.shape != (len(xblind),):
        raise ValueError("Teacher outputs must align with development and blind rows")
    if (not np.all(np.isfinite(xdev)) or not np.all(np.isfinite(xblind))
            or not np.all(np.isfinite(ydev)) or not np.all(np.isfinite(yblind))):
        raise ValueError("Features and teacher outputs must be finite")
    return ydev, yblind


def fit_lasso_path(development, development_output, blind, blind_output, *,
                   alphas=None, alpha_ratios=DEFAULT_ALPHA_RATIOS,
                   max_iter=10000, tol=1e-4, clip_limit=None):
    """Fit one Lasso per lambda on development ONLY; evaluate each on both roles.

    Returns rows, coefficient path in ORIGINAL frozen-feature units, intercepts,
    standardized feature scales, prediction matrices, and resolved alphas.
    The scaling is fitted only on development (without mean subtraction); Lasso
    internally fits its own intercept. Original input matrices are never altered.
    """
    dev_y, blind_y = _validate_inputs(development, development_output, blind, blind_output)
    if int(max_iter) != max_iter or max_iter < 1 or not np.isfinite(tol) or tol <= 0:
        raise ValueError("max_iter must be positive integer and tol finite positive")
    if clip_limit is not None and (not np.isfinite(clip_limit) or clip_limit <= 0):
        raise ValueError("clip_limit must be finite positive")
    scaler = StandardScaler(with_mean=False, copy=True)
    dev_x = scaler.fit_transform(np.asarray(development, dtype=np.float32))
    blind_x = scaler.transform(np.asarray(blind, dtype=np.float32))
    if not np.all(np.isfinite(dev_x)) or not np.all(np.isfinite(blind_x)):
        raise ValueError("Feature scaling generated non-finite values")
    # This is the smallest alpha that produces an intercept-only Lasso.
    centered_y = dev_y - np.mean(dev_y)
    alpha_max = float(np.max(np.abs(dev_x.T @ centered_y)) / len(dev_y))
    if alphas is None:
        if not np.isfinite(alpha_max) or alpha_max <= 0:
            raise ValueError("Original development predictions have no variable linear signal")
        ratios = _positive_unique(alpha_ratios, "alpha_ratios")
        strengths = [alpha_max * ratio for ratio in ratios]
    else:
        strengths = _positive_unique(alphas, "alphas")
    estimator = Lasso(alpha=strengths[0], fit_intercept=True, max_iter=int(max_iter),
                      tol=float(tol), warm_start=True, selection="cyclic")
    rows, coefficients, intercepts, predictions_dev, predictions_blind = [], [], [], [], []
    for alpha in strengths:
        estimator.set_params(alpha=float(alpha))
        with warnings.catch_warnings(record=True) as found:
            warnings.simplefilter("always", ConvergenceWarning)
            estimator.fit(dev_x, dev_y)
        converged = not any(issubclass(w.category, ConvergenceWarning) for w in found)
        pred_dev = np.asarray(estimator.predict(dev_x), dtype=np.float64)
        pred_blind = np.asarray(estimator.predict(blind_x), dtype=np.float64)
        if clip_limit is not None:
            pred_dev = np.clip(pred_dev, -clip_limit, clip_limit)
            pred_blind = np.clip(pred_blind, -clip_limit, clip_limit)
        # Original frozen-feature units: Lasso is trained on z/scale.
        weights = np.asarray(estimator.coef_, dtype=np.float64) / scaler.scale_
        active = int(np.count_nonzero(estimator.coef_))
        rows.append({
            "alpha": float(alpha), "alpha_over_alpha_max": float(alpha / alpha_max)
            if alpha_max > 0 else None,
            "n_features": int(len(weights)), "n_nonzero": active,
            "n_zero": int(len(weights) - active),
            "sparsity_fraction": float(1 - active / len(weights)),
            "converged": bool(converged),
            "n_iter": int(estimator.n_iter_),
            "development": fidelity(dev_y, pred_dev),
            "blind": fidelity(blind_y, pred_blind),
        })
        coefficients.append(weights.copy())
        intercepts.append(float(estimator.intercept_))
        predictions_dev.append(pred_dev)
        predictions_blind.append(pred_blind)
    return {
        "rows": rows,
        "alphas": np.asarray(strengths),
        "alpha_max": alpha_max,
        "coefficients": np.asarray(coefficients),
        "intercepts": np.asarray(intercepts),
        "feature_scales": np.asarray(scaler.scale_, dtype=np.float64),
        "predictions_development": np.asarray(predictions_dev),
        "predictions_blind": np.asarray(predictions_blind),
    }


def _timing_metrics(path_result, *, target_development, target_blind, teacher_development,
                    teacher_blind, fit_config, seed):
    """Targets are used for reporting ONLY, never for fitting Lasso."""
    from waveform_analysis.reporting.stats import metric_values
    sources = {"development": (np.asarray(target_development), teacher_development),
               "blind": (np.asarray(target_blind), teacher_blind)}
    original = {}
    for role, (target, teacher) in sources.items():
        original[role] = metric_values(target - teacher, fit_config,
                                       seed=semantic_seed(seed, "lasso", role))
        for i, values in enumerate(path_result[f"predictions_{role}"]):
            correction = metric_values(target - values, fit_config,
                                       seed=semantic_seed(seed, "lasso", role))
            path_result["rows"][i][f"{role}_ctr_ps"] = correction["ctr_ps"]
            path_result["rows"][i][f"{role}_timing_rmse_ps"] = correction["rmse_ps"]
    return original


def _plot(path_result, original, destination):
    import matplotlib.pyplot as plt
    rows = path_result["rows"]
    n = np.asarray([row["n_nonzero"] for row in rows])
    order = np.argsort(n, kind="stable")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for role, label in (("development", "Development (fit)"), ("blind", "Blind (test)")):
        axes[0].plot(n[order], [rows[i][role]["rmse_ps"] for i in order], "o-", label=label)
        axes[1].plot(n[order], [rows[i][role]["r2"] if rows[i][role]["r2"] is not None
                                   else np.nan for i in order], "o-", label=label)
    axes[0].set(ylabel="RMSE to original prediction [ps]", title="Lasso fidelity")
    axes[1].set(ylabel="R² to original prediction", title="Lasso fidelity")
    for ax in axes:
        ax.set(xlabel="Nonzero Lasso coefficients")
        ax.grid(alpha=.25)
        ax.legend()
    fig.tight_layout()
    fig.savefig(destination / "fidelity_vs_nonzero.png", dpi=170)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for role, label in (("development", "Development (fit)"), ("blind", "Blind (test)")):
        y = [row[f"{role}_ctr_ps"] for row in rows]
        axes[0].plot(n[order], np.asarray(y)[order], "o-", label=label)
        axes[0].axhline(original[role]["ctr_ps"], linestyle="--", alpha=.5)
    axes[0].set(xlabel="Nonzero Lasso coefficients", ylabel="Timing CTR [ps]",
                title="Original Ridge CTR shown by dashed lines")
    axes[0].grid(alpha=.25)
    axes[0].legend()
    for role, label in (("development", "Development (fit)"), ("blind", "Blind (test)")):
        y = [row[f"{role}_timing_rmse_ps"] for row in rows]
        axes[1].plot(n[order], np.asarray(y)[order], "o-", label=label)
        axes[1].axhline(original[role]["rmse_ps"], linestyle="--", alpha=.5)
    axes[1].set(xlabel="Nonzero Lasso coefficients", ylabel="Timing residual RMSE [ps]",
                title="Original Ridge RMSE shown by dashed lines")
    axes[1].grid(alpha=.25)
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(destination / "timing_vs_nonzero.png", dpi=170)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    dev_rmse = np.asarray([row["development"]["rmse_ps"] for row in rows])
    blind_rmse = np.asarray([row["blind"]["rmse_ps"] for row in rows])
    dev_ctr = np.asarray([row["development_ctr_ps"] for row in rows])
    blind_ctr = np.asarray([row["blind_ctr_ps"] for row in rows])
    for ax, a, b, title, units in [
        (axes[0], dev_rmse, blind_rmse, "Fidelity RMSE", "RMSE to teacher [ps]"),
        (axes[1], dev_ctr, blind_ctr, "Timing CTR", "CTR [ps]"),
    ]:
        ax.scatter(a, b, c=n)
        low, high = float(min(a.min(), b.min())), float(max(a.max(), b.max()))
        margin = max((high - low) * .05, 1e-6)
        ax.plot([low - margin, high + margin], [low - margin, high + margin],
                "--", color="gray", linewidth=1)
        for i, r in enumerate(rows):
            ax.annotate(str(r["n_nonzero"]), (a[i], b[i]), xytext=(4, 3),
                        textcoords="offset points", fontsize=7)
        ax.set(xlabel=f"Development {units}", ylabel=f"Blind {units}",
               title=f"{title} (labels = nonzero coefficients)")
        ax.grid(alpha=.2)
    fig.tight_layout()
    fig.savefig(destination / "development_vs_blind.png", dpi=170)
    plt.close(fig)


def _csv_rows(rows):
    for r in rows:
        flat = {key: value for key, value in r.items()
                if key not in {"development", "blind"}}
        for role in ("development", "blind"):
            for key, value in r[role].items():
                flat[f"{role}_fidelity_{key}"] = value
        yield flat


def analyze_run(run, *, cache_override=None, alphas=None,
                alpha_ratios=DEFAULT_ALPHA_RATIOS, max_development_events=None,
                max_blind_events=None, max_iter=10000, tol=1e-4, logger=None):
    directory, manifest = Path(run["directory"]), run["manifest"]
    spec = get_model(manifest["model"])
    if spec.feature_transform is None:
        raise ValueError(f"{spec.name} has no frozen feature transform")
    model_dir = directory / "model"
    if not saved_model_complete(spec, model_dir):
        raise FileNotFoundError(f"Saved original model is missing: {model_dir}")
    config = resolve_run_config(directory)
    cache_root = _cache_root(config, directory, cache_override,
                             roles=("development", "blind"))
    lookup = PreparedCacheLookup()
    development = lookup.find(cache_root, "development", config, manifest)
    blind = lookup.find(cache_root, "blind", config, manifest)
    seed = semantic_seed(int(config["seed"]), spec.name, "lasso_surrogate")
    rng = np.random.default_rng(seed)

    def indices(dataset, maximum):
        if maximum is None or maximum >= dataset.n_events:
            return np.arange(dataset.n_events, dtype=np.int64)
        if maximum < 2:
            raise ValueError("Event caps must be >= 2")
        return np.sort(rng.choice(dataset.n_events, size=maximum, replace=False))

    dev_idx, blind_idx = indices(development, max_development_events), indices(blind, max_blind_events)
    identity = {
        "method": "lasso_frozen_feature_distillation_v1",
        "model": spec.name,
        "final_fit": manifest["stage_fingerprints"]["final_fit"],
        "development_protocol": development.manifest["analysis_protocol_identity"],
        "blind_protocol": blind.manifest["analysis_protocol_identity"],
        "seed": int(seed),
        "alphas": alphas,
        "alpha_ratios": list(alpha_ratios) if alphas is None else None,
        "max_development_events": max_development_events,
        "max_blind_events": max_blind_events,
        "max_iter": max_iter, "tol": tol,
    }
    destination = directory / "explanations" / "lasso_surrogate" / canonical_hash(identity)[:16]
    if (destination / "summary.json").is_file():
        if logger:
            logger.info("Lasso surrogate | reuse | %s", destination)
        return destination

    destination.parent.mkdir(parents=True, exist_ok=True)
    matrices, sources = {}, {}
    with diagnostic_random_state():
        fitted = load_fitted_model(spec, model_dir, run["best"]["parameters"], config)
        try:
            with tempfile.TemporaryDirectory(prefix=".lasso-features-", dir=destination.parent) as scratch:
                try:
                    for role, dataset, idx in (("development", development, dev_idx),
                                               ("blind", blind, blind_idx)):
                        matrices[role], sources[role] = frozen_features(
                            spec, fitted, dataset, config, idx, cache_root, role,
                            Path(scratch), logger=logger)
                    chunk = int(config["runtime"]["prediction_chunk_size"])
                    teacher_dev = predict_features(spec, fitted, matrices["development"], chunk,
                                                   logger=logger, role="development")
                    teacher_blind = predict_features(spec, fitted, matrices["blind"], chunk,
                                                     logger=logger, role="blind")
                    if logger:
                        logger.info("Lasso surrogate | %s | features=%d | development=%d blind=%d",
                                    spec.name, matrices["development"].shape[1],
                                    len(dev_idx), len(blind_idx))
                    result = fit_lasso_path(
                        matrices["development"], teacher_dev, matrices["blind"], teacher_blind,
                        alphas=alphas, alpha_ratios=alpha_ratios, max_iter=max_iter,
                        tol=tol, clip_limit=fitted.output_max_abs_ps)
                finally:
                    for matrix in matrices.values():
                        if isinstance(matrix, np.memmap):
                            matrix._mmap.close()
                    matrices.clear()
        finally:
            del fitted
            release_training_memory()

    # The targets are used ONLY for timing-performance reporting; they never
    # enter feature scaling, alpha_max, or Lasso fitting.
    original = _timing_metrics(
        result, target_development=model_target(development, config["mode"])[dev_idx],
        target_blind=model_target(blind, config["mode"])[blind_idx],
        teacher_development=teacher_dev, teacher_blind=teacher_blind,
        fit_config=config["fit"], seed=seed,
    )
    destination.mkdir(parents=True, exist_ok=True)
    rows = list(_csv_rows(result["rows"]))
    with (destination / "metrics.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    np.savez_compressed(
        destination / "coefficient_path.npz",
        alphas=result["alphas"], coefficients=result["coefficients"],
        intercepts=result["intercepts"], feature_scales=result["feature_scales"],
        development_event_ids=development.event_index[dev_idx],
        blind_event_ids=blind.event_index[blind_idx],
    )
    _plot(result, original, destination)
    atomic_json(destination / "summary.json", {
        **identity, "transform_identity": sources["development"].get("transform_identity")
        or sources["development"].get("path"),
        "feature_sources": sources,
        "n_development": len(dev_idx), "n_blind": len(blind_idx),
        "n_features": len(result["feature_scales"]),
        "alpha_max": result["alpha_max"],
        "alpha_convention": "sklearn Lasso alpha; default ratios relative to development alpha_max",
        "standardization": "StandardScaler(with_mean=False), fitted on development only; saved coefficients expressed in original transformed feature units",
        "teacher_predictions": "Original saved model on frozen transformed features, including output clipping",
        "surrogate_prediction_policy": "Clip predictions to original output_max_abs_ps after linear Lasso inference",
        "teacher_timing": original,
        "training_targets": "original_model_development_predictions_only",
        "blind_used_for_selection": False,
        "note": "Development fidelity is in-sample; no lambda selected on blind. Each alpha reported independently.",
    })
    if logger:
        logger.info("Lasso surrogate | wrote %s | %d strengths", destination, len(result["rows"]))
    return destination


def run(results, *, models=None, cache_root=None, alphas=None,
        alpha_ratios=DEFAULT_ALPHA_RATIOS, max_development_events=None,
        max_blind_events=None, max_iter=10000, tol=1e-4, logger=None):
    if alphas is not None:
        _positive_unique(alphas, "alphas")
    else:
        _positive_unique(alpha_ratios, "alpha_ratios")
    for value in (max_development_events, max_blind_events):
        if value is not None and value < 2:
            raise ValueError("Event caps must be >= 2")
    runs = discover_runs(Path(results).expanduser().resolve())
    if not runs:
        raise ValueError(f"No completed runs under {results}")
    names = {r["manifest"]["model"] for r in runs}
    if models is not None and set(models) - names:
        raise ValueError(f"Unknown models in results: {sorted(set(models) - names)}")
    outputs, failures = [], []
    for item in runs:
        if models is not None and item["manifest"]["model"] not in models:
            continue
        try:
            outputs.append(analyze_run(
                item, cache_override=cache_root, alphas=alphas, alpha_ratios=alpha_ratios,
                max_development_events=max_development_events,
                max_blind_events=max_blind_events, max_iter=max_iter, tol=tol, logger=logger))
        except Exception as exc:
            failures.append(f"{item['directory']}: {exc}")
            if logger:
                logger.exception("Lasso surrogate failed for %s", item["directory"])
    if failures:
        raise RuntimeError("Some Lasso surrogate analyses failed:\n" + "\n".join(failures))
    return outputs


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, required=True,
                        help="Completed canonical model run or batch root")
    parser.add_argument("--models", nargs="+", metavar="MODEL")
    parser.add_argument("--cache-root", type=Path, help="Root of original prepared/model_inputs caches")
    choices = parser.add_mutually_exclusive_group()
    choices.add_argument("--alphas", type=float, nargs="+",
                         help="Absolute sklearn Lasso penalties")
    choices.add_argument("--alpha-ratios", type=float, nargs="+", default=None,
                         help="Fractions of alpha_max derived from development only")
    parser.add_argument("--max-development-events", type=int)
    parser.add_argument("--max-blind-events", type=int)
    parser.add_argument("--max-iter", type=int, default=10000)
    parser.add_argument("--tol", type=float, default=1e-4)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    for output in run(args.results, models=args.models, cache_root=args.cache_root,
                      alphas=args.alphas,
                      alpha_ratios=args.alpha_ratios if args.alpha_ratios is not None else DEFAULT_ALPHA_RATIOS,
                      max_development_events=args.max_development_events,
                      max_blind_events=args.max_blind_events,
                      max_iter=args.max_iter, tol=args.tol,
                      logger=logging.getLogger(__name__)):
        print(output)


if __name__ == "__main__":
    main()
