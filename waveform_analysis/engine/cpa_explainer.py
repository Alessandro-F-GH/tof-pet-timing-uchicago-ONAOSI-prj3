"""Chronological Prefix Attribution for frozen wide-window TOF-PET models.

No correction-model refitting and no transformed-feature cache. Imputers are
selected on held-out development waveforms and never tuned on blind events.
The statistic is deterministic prefix-imputation attribution, not exact SHAP.
"""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from waveform_analysis.data.results import (batch_root, read_json, read_run, resolve_run_config)
from waveform_analysis.data.named_cache import safe_component
from waveform_analysis.data.splits import semantic_seed
from waveform_analysis.data.view import mode_family, model_target, waveform_view
from waveform_analysis.engine.train import (
    load_fitted_model, predict_indices, release_training_memory, saved_model_complete,
)
from waveform_analysis.engine.tshap_explainer import (
    _predict_bounded, _resolve_tshap_model_directory,
)
from waveform_analysis.data.prepared_cache import PreparedCacheLookup
from waveform_analysis.data.storage import RunStore
from waveform_analysis.models import get_model
from waveform_analysis.scripts.evaluate_mean_ar import (
    fit_moments, fit_ridge_orders, forecast_suffix, valid_intervals,
)
from waveform_analysis.signal.metrics import fit_ctr_ps


DEFAULT_ORDERS = (8, 16, 32, 64, 128)
DEFAULT_MODELS = {
    "energy": ("direct_multirocket_rdst", "onishi_cnn", "shared_linear_ridge"),
    "timing": ("direct_multirocket", "onishi_cnn", "shared_linear_ridge"),
}


def cutoff_schedule_ns(last_ns):
    """Requested positions on the aligned, ML-input-relative time axis."""
    values = [-1., 0., 2., 4., 6., 8., 10.]
    values.extend(float(v) for v in np.arange(15., last_ns + 1e-9, 5.))
    return [v for v in values if v <= last_ns + 1e-9]


def timing_cutoff_schedule_ns(time_ps, mask):
    """Restart a 1,3,5,7,9,14,19,... ns schedule after each masked gap.

    The initial (pre-first-gap) segment retains the ordinary base schedule;
    subsequent segments use their actual first valid sample as the origin.
    All positions are quantized by cutoff_samples and never moved across gaps.
    """
    t = np.asarray(time_ps, np.float64) / 1000.
    keep = np.asarray(mask, bool)
    if t.shape != keep.shape or t.ndim != 1:
        raise ValueError("Timing grid and mask must have matching dimensions")
    from waveform_analysis.scripts.evaluate_mean_ar import valid_intervals
    intervals = valid_intervals(keep)
    if not intervals:
        return []
    # The absolute schedule belongs ONLY to the first retained segment.
    # Mixing it with later mask-relative schedules creates spurious 10/15/20...
    # ns cutoffs between (for example) 8.925, 10.925, 12.925 ns.
    first_start, first_end = intervals[0]
    requested = [
        v for v in cutoff_schedule_ns(float(t[-1]))
        if float(t[first_start]) <= v <= float(t[first_end-1])
    ]
    # Every subsequent retained segment starts a fresh local schedule.
    for start, end in intervals[1:]:
        base = float(t[start])
        offset = 1.
        while base + offset <= float(t[end-1]) + 1e-9:
            requested.append(base + offset)
            offset += 2. if offset < 9. else 5.
    return sorted(set(requested))


def masked_time_intervals(time_ps, mask):
    """Physical-time spans of excluded samples for plot shading."""
    t = np.asarray(time_ps, np.float64) / 1000.
    if len(t) < 2:
        return []
    dt = np.diff(t)
    edges = np.r_[t[0]-dt[0]/2, (t[:-1]+t[1:])/2, t[-1]+dt[-1]/2]
    missing = ~np.asarray(mask, bool)
    transitions = np.diff(np.r_[False, missing, False].astype(np.int8))
    return [(float(edges[a]), float(edges[b]))
            for a,b in zip(np.flatnonzero(transitions == 1),
                           np.flatnonzero(transitions == -1))]


def cutoff_samples(time_ps, mask, requested_ns=None):
    """Do not shift cutoffs into or across masked timing regions."""
    t = np.asarray(time_ps, np.float64) / 1000.
    keep = np.asarray(mask, bool)
    if t.ndim != 1 or len(t) != len(keep) or not np.all(np.diff(t) > 0):
        raise ValueError("Expected a monotonically increasing ML waveform time grid")
    schedule = cutoff_schedule_ns(float(t[-1])) if requested_ns is None else requested_ns
    selected, omitted = [], []
    for requested in schedule:
        if not (t[0] <= requested <= t[-1]):
            omitted.append((float(requested), "outside grid"))
            continue
        j = int(np.argmin(abs(t-float(requested))))
        if not keep[j]:
            omitted.append((float(requested), "masked"))
            continue
        if j < 1:
            omitted.append((float(requested), "empty observed prefix"))
            continue
        if j not in selected:
            selected.append(j)
    return sorted(selected), omitted


def segment_for_cutoff(mask, cutoff):
    """Return observed-history start and next masked boundary."""
    m = np.asarray(mask, bool)
    k = int(cutoff)
    if not 0 <= k < len(m) or not m[k]:
        raise ValueError("AR cutoff must be on a retained sample")
    start = k
    while start > 0 and m[start-1]:
        start -= 1
    stop = k
    while stop < len(m) and m[stop]:
        stop += 1
    return start, stop


def suffix_with_imputer(waves, mean, std, mask, cutoff, candidate):
    """Observed prefix stays exact; AR stops at first future masked sample."""
    x = np.asarray(waves, np.float32)
    if x.ndim != 2 or x.shape[1] != len(mean):
        raise ValueError("Expected [events, samples] for one detector")
    k = int(cutoff)
    result = np.broadcast_to(np.asarray(mean, np.float32), x.shape).copy()
    result[:, :k] = x[:, :k]
    if candidate["method"] == "mean":
        return result
    lo, hi = segment_for_cutoff(mask, k)
    coeff = np.asarray(candidate["coefficients"], np.float64)
    if k-lo < len(coeff):
        raise ValueError("Not enough contiguous observed history")
    result[:, k:hi] = forecast_suffix(
        x[:, :k], mean, std, coeff, k, hi,
        standardized=candidate["method"] == "standardized_ar",
    ).astype(np.float32)
    return result


def _validation_score(truth, mean, forecast, end, cutoff, local_samples):
    """Balanced local and full-valid-region MSE relative to mean baseline."""
    n = min(int(local_samples), int(end-cutoff))
    if n <= 0:
        return float("inf")
    target = truth[:, cutoff:end]
    baseline = mean[None, cutoff:end]
    predicted = forecast[:, cutoff:end]
    def ratio(size):
        baseline_mse = float(np.mean((target[:, :size]-baseline[:, :size])**2))
        candidate_mse = float(np.mean((target[:, :size]-predicted[:, :size])**2))
        return (candidate_mse + 1e-12) / (baseline_mse + 1e-12)
    return .5*(ratio(n) + ratio(end-cutoff))


def _draw_indices(total, count, seed):
    count = int(count)
    if count < 0:
        raise ValueError("Event cap must be >= 0 (0 means all)")
    n = total if count == 0 else min(int(total), count)
    if n < 2:
        raise ValueError("At least two events are required")
    return np.sort(np.random.default_rng(int(seed)).choice(total, size=n, replace=False))


def select_imputers(development, mask, cutoffs, *, seed=42, fit_events=1024,
                    validation_events=128, orders=DEFAULT_ORDERS, local_samples=100,
                    ridge_alpha=1e-3, train_stride=4):
    """Validate independent detector autoregressors, then refit chosen models.

    A common development permutation splits fitting/validation. Both selected
    detectors use the same input event split, but fit separate coefficients.
    """
    if validation_events < 2 or fit_events < 0:
        raise ValueError("Invalid AR fit/validation event limits")
    count = len(development)
    rng = np.random.default_rng(int(seed))
    permutation = rng.permutation(count)
    n_val = min(validation_events, max(2, count//5))
    fit_ids = permutation[n_val:n_val+fit_events] if fit_events else permutation[n_val:]
    val_ids = permutation[:n_val]
    if len(fit_ids) < 2:
        raise ValueError("Insufficient independent development events for AR fitting")
    # Refit on selected development population *including* validation events,
    # but only after order and method selection have been frozen.
    n_refit = min(count, fit_events) if fit_events else count
    refit_ids = permutation[:n_refit]
    x_fit = np.asarray(development[fit_ids], np.float64)
    x_val = np.asarray(development[val_ids], np.float64)
    x_refit = np.asarray(development[refit_ids], np.float64)
    m = np.asarray(mask, bool)
    candidates = {}
    diagnostics = []
    for detector in range(2):
        training = x_fit[:, detector, :]
        testing = x_val[:, detector, :]
        mu, sigma, _ = fit_moments(training)
        intervals = valid_intervals(m)
        # Only train on contiguous portions longer than the maximum requested order.
        long_intervals = [pair for pair in intervals if pair[1]-pair[0] > max(orders)]
        trial = (fit_ridge_orders(training, mu, sigma, long_intervals, orders,
                                 alpha=ridge_alpha, stride=train_stride)
                 if long_intervals else {})
        for cutoff in cutoffs:
            lo, end = segment_for_cutoff(m, cutoff)
            winner = {"method": "mean", "order": 0}
            best_score = 1.
            diagnostics.append(dict(detector=detector+1, cutoff_sample=cutoff,
                                    method="mean", order=0, score=1., eligible=True))
            for method in ("mean_ar", "standardized_ar"):
                for order in orders:
                    fitted = trial.get(method, {}).get(order)
                    eligible = bool(
                        cutoff-lo >= order and fitted is not None and
                        fitted["stable"] and np.isfinite(fitted["root_radius"])
                    )
                    score = float("inf")
                    if eligible:
                        predicted = suffix_with_imputer(
                            testing, mu, sigma, m, cutoff,
                            {"method": method, "coefficients": fitted["coefficients"]},
                        )
                        score = _validation_score(testing, mu, predicted, end,
                                                  cutoff, local_samples)
                    diagnostics.append(dict(
                        detector=detector+1, cutoff_sample=cutoff,
                        method=method, order=order, score=score,
                        eligible=eligible,
                    ))
                    if score < best_score:
                        best_score = score
                        winner = {"method": method, "order": int(order)}
            candidates[(detector, int(cutoff))] = {
                **winner, "validation_score": best_score,
            }
    # Fit means/standard deviations and required AR orders on the refit cohort.
    refitted = []
    for detector in range(2):
        full = x_refit[:, detector, :]
        mu, sigma, std_floor = fit_moments(full)
        required = sorted({row["order"] for (d, _), row in candidates.items()
                           if d == detector and row["order"] > 0})
        coefficients = {}
        if required:
            intervals = [pair for pair in valid_intervals(m)
                         if pair[1]-pair[0] > max(required)]
            models = fit_ridge_orders(full, mu, sigma, intervals, required,
                                      alpha=ridge_alpha, stride=train_stride)
            for method in ("mean_ar", "standardized_ar"):
                for order in required:
                    coefficients[(method, order)] = models[method][order]["coefficients"]
        refitted.append({"mean": mu, "std": sigma, "std_floor": std_floor,
                         "coefficients": coefficients})
    for (detector, cutoff), choice in candidates.items():
        if choice["order"]:
            coeff = refitted[detector]["coefficients"][
                (choice["method"], choice["order"])]
            # Recheck stability after refit: otherwise fall back to mean.
            radius = float(max(abs(np.roots(np.r_[1., -coeff]))))
            choice["refit_root_radius"] = radius
            if not np.isfinite(radius) or radius >= 1.:
                choice.update(method="mean", order=0,
                              fallback="refitted AR was unstable")
            else:
                choice["coefficients"] = coeff
    return {
        "detectors": refitted, "selected": candidates, "scores": diagnostics,
        "fit_ids": fit_ids, "validation_ids": val_ids, "refit_ids": refit_ids,
    }


def reconstruct_pair(waves, imputer, mask, cutoff, *, mean_only=False):
    """Reconstruct paired detector waveforms without future-sample leakage."""
    pair = np.asarray(waves, np.float32)
    result = np.empty_like(pair)
    for d in range(2):
        trained = imputer["detectors"][d]
        choice = ({"method": "mean", "order": 0} if mean_only else
                  imputer["selected"][(d, int(cutoff))])
        result[:, d, :] = suffix_with_imputer(
            pair[:, d, :], trained["mean"], trained["std"], mask, cutoff, choice,
        )
    return result


def _centered_width(residual):
    r = np.asarray(residual, np.float64)
    return float(np.std(r, ddof=0))


def _ctr(residual, seed):
    try:
        return float(fit_ctr_ps(residual, seed=int(seed), bootstrap=False).ctr_ps)
    except (ValueError, FloatingPointError):
        return float("nan")


def summarize_predictions(predictions, targets, *, seed, bootstrap=100):
    """Bootstrap paired events, reusing each resample across all cutoffs."""
    p = np.asarray(predictions, np.float64)
    target = np.asarray(targets, np.float64)
    if p.ndim != 2 or p.shape[1] != len(target):
        raise ValueError("Predictions must be [cutoffs, events]")
    residuals = target[None, :] - p
    rng = np.random.default_rng(int(seed))
    draws = [rng.integers(0, len(target), len(target)) for _ in range(int(bootstrap))]
    summary = []
    for j, residual in enumerate(residuals):
        rmse = _centered_width(residual)
        ctr = _ctr(residual, int(seed)+j)
        rmses = np.array([_centered_width(residual[ids]) for ids in draws])
        ctrs = np.array([_ctr(residual[ids], int(seed)+j*1000+i+1)
                         for i, ids in enumerate(draws)])
        def ci(values):
            valid = np.asarray(values, np.float64)
            valid = valid[np.isfinite(valid)]
            return (float(np.percentile(valid, 2.5)),
                    float(np.percentile(valid, 97.5))) if valid.size else (float("nan"),)*2
        rlo, rhi = ci(rmses)
        clo, chi = ci(ctrs)
        summary.append(dict(centered_rmse_ps=rmse, centered_rmse_ci_low_ps=rlo,
                            centered_rmse_ci_high_ps=rhi, ctr_ps=ctr,
                            ctr_ci_low_ps=clo, ctr_ci_high_ps=chi,
                            ctr_valid_bootstrap=int(np.isfinite(ctrs).sum())))
    return summary


def _csv(path, rows):
    if not rows:
        return
    with Path(path).open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _plot_metrics(path, rows, title, *, masked_intervals=()):
    """Separate physical RMSE and CTR scales, with masked time shaded."""
    ordered = sorted((r for r in rows if r["kind"] == "prefix"),
                     key=lambda r: r["cutoff_ns"])
    full = next(r for r in rows if r["kind"] == "full")
    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    positions = np.asarray([r["cutoff_ns"] for r in ordered])
    for ax, key, low, high, label in (
        (axes[0], "centered_rmse_ps", "centered_rmse_ci_low_ps",
         "centered_rmse_ci_high_ps", "Centered residual RMSE (ps)"),
        (axes[1], "ctr_ps", "ctr_ci_low_ps", "ctr_ci_high_ps",
         "CTR, F1 FWHM (ps)"),
    ):
        values = np.asarray([r[key] for r in ordered], float)
        lower = np.asarray([r[low] for r in ordered], float)
        upper = np.asarray([r[high] for r in ordered], float)
        ax.plot(positions, values, "o-", markersize=3, label="Prefix")
        if np.any(np.isfinite(lower) & np.isfinite(upper)):
            ax.fill_between(positions, lower, upper, alpha=.15)
        ax.axhline(full[key], linestyle="--", color="gray",
                   label="Full waveform")
        for index,(left,right) in enumerate(masked_intervals):
            ax.axvspan(left, right, color="gray", alpha=.14,
                       label="Masked samples" if index == 0 else None)
        ax.set_ylabel(label)
        ax.grid(alpha=.2)
        ax.legend(fontsize=8)
    axes[-1].set_xlabel("Aligned waveform cutoff (ns)")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _json_safe(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    return value


def _identity(run, development, blind, board, mode):
    # Include actual cohort identities, not just dataset display names.
    data = dict(board=board, mode=mode, batch=str(batch_root(run["directory"])),
                development=str(development.directory), blind=str(blind.directory),
                development_population=development.manifest.get("event_population_identity"),
                blind_population=blind.manifest.get("event_population_identity"))
    return hashlib.sha256(json.dumps(data, sort_keys=True).encode()).hexdigest()[:16]


def _cpa_destination(output_root, board, run_directory, mode, model):
    """Human-readable paths; dataset/run fingerprint stays in summary metadata.

    A canonical batch has exactly one trio of control/development/blind
    datasets. The batch directory is its natural stable human-readable label.
    """
    return (Path(output_root) / safe_component(board) /
            safe_component(batch_root(run_directory).name) /
            safe_component(mode) / safe_component(model))


def _guard_dataset_identity(destination, identity):
    """Never confuse two cohorts that happen to share a display label.

    This check applies even with --overwrite: overwriting a different cohort
    would silently replace results from an unrelated experiment.
    """
    summary_path = Path(destination) / "summary.json"
    if not summary_path.is_file():
        if Path(destination).exists() and any(Path(destination).iterdir()):
            raise FileExistsError(
                f"Nonempty CPA output without a summary: {destination}. "
                "Move it aside or choose another --output-root."
            )
        return False
    summary = read_json(summary_path)
    prior = summary.get("dataset_identity")
    if prior != identity:
        raise ValueError(
            f"CPA output naming collision: {destination} refers to "
            f"dataset identity {prior!r}, but this run has identity {identity!r}. "
            "Use separate named batch directories or a different --output-root."
        )
    return True


def _cpa_run_inputs(results, model_dir, *, models=None, mode=None, logger=None):
    """Discover usable CPA runs without letting unrelated partial runs abort scans.

    Unlike the batch-wide result reader, discovery reads the small model label
    before inspecting its manifest. This is deliberately CPA-local: the
    training pipeline and reporting keep their stricter completeness rules.
    An incomplete run is eligible once its required *saved model* exists; its
    final blind metrics and training-stage status are not prerequisites.
    """
    explicit = model_dir is not None
    if explicit:
        selected = Path(model_dir).expanduser().resolve()
        if selected.name == "model" and (selected.parent / "metadata").is_dir():
            selected = selected.parent
        paths = [selected]
    else:
        root = Path(results).expanduser().resolve()
        paths = [p.parent.parent for p in sorted(root.rglob("metadata/config.json"))]

    eligible = []
    for directory in paths:
        config_path = directory / "metadata" / "config.json"
        try:
            local = read_json(config_path)
            model_name = local["model"]
            if models is not None and model_name not in models:
                continue
            config = resolve_run_config(directory)
            if str(config.get("window_name", "")).lower() != "wide":
                continue
            family = mode_family(config["mode"])
            if mode is not None and family != mode:
                continue
            if models is None and model_name not in DEFAULT_MODELS[family]:
                continue

            # Saved models may be available before the run has reached the
            # metadata/manifest stage. For a named base cache the canonical
            # run config is sufficient for locating development/blind inputs.
            manifest_path = directory / "metadata" / "manifest.json"
            if manifest_path.is_file():
                run = read_run(directory)
            else:
                population_path = directory.parent / "population.json"
                population = read_json(population_path, {})
                run = {
                    "directory": directory,
                    "manifest": {"model": model_name, **population},
                    "best": read_json(directory / "metadata" / "best.json", {}),
                    "blind": {},
                    "bootstrap": {},
                }
            # Do not require the pipeline's status == complete: CPA only
            # needs fitted-model artifacts and prepared waveforms.
            eligible.append(run)
        except (OSError, ValueError, KeyError, TypeError) as exc:
            if explicit:
                raise
            if logger:
                logger.warning("CPA skipped unreadable or partial run | %s | %s",
                               directory, exc)
            continue
    return eligible


def explain_cpa(results=None, *, model_dir=None, models=None, mode=None,
                board=None, cache_root=None, artifact_root=None, output_root=None,
                max_events=1024, fit_events=1024, validation_events=128,
                bootstrap=100, seed=42, predict_batch_size=64,
                orders=DEFAULT_ORDERS, local_samples=100,
                ridge_alpha=1e-3, train_stride=4,
                overwrite=False, logger=None):
    """Run one board at a time, using only base prepared waveform cache + fitted models."""
    if (results is None) == (model_dir is None):
        raise ValueError("Pass exactly one of --results or --model-dir")
    orders = tuple(sorted(set(int(p) for p in orders)))
    if not orders or min(orders) < 1 or local_samples < 1 or train_stride < 1 or predict_batch_size < 1:
        raise ValueError("Invalid AR orders, horizon, stride or prediction batch size")
    if (max_events != 0 and max_events < 2) or fit_events < 0 or validation_events < 2 or bootstrap < 0:
        raise ValueError("Invalid event count or bootstrap repetitions")
    runs = _cpa_run_inputs(results, model_dir, models=models, mode=mode, logger=logger)
    if not runs:
        raise ValueError("No matching wide-window model runs with readable configurations found")
    output_root = Path(output_root or "cpa_results").expanduser().resolve()
    source_root = Path(results or model_dir).expanduser().resolve()
    if output_root == source_root or output_root.is_relative_to(source_root):
        raise ValueError("CPA output directory must be outside the model results")
    if mode not in (None, "energy", "timing"):
        raise ValueError("mode must be energy or timing")
    prepared = PreparedCacheLookup()
    imputer_cache = {}
    outputs = []
    for run in runs:
        name = run["manifest"].get("model")
        config = None
        try:
            config = resolve_run_config(run["directory"])
            if str(config.get("window_name", "")).lower() != "wide":
                continue
            family = mode_family(config["mode"])
            if mode and mode != family:
                continue
            if name not in (models if models is not None else DEFAULT_MODELS[family]):
                continue
            # Base waveforms plus frozen model/transform suffice: no feature cache.
            prep = config["preprocessing"]
            root = Path(cache_root).expanduser().resolve()
            prep["named_cache_root"] = str(root)
            prep["cache_dir"] = str(root / "preprocessing")
            prep["artifact_root"] = str(
                Path(artifact_root).expanduser().resolve()
                if artifact_root is not None else root / "artifact_store"
            )
            batch = read_json(batch_root(run["directory"]) / "config.json")
            actual_board = str(batch.get("board") or "unknown_board")
            if board is not None and actual_board != "unknown_board" and (
                str(board).lower() != actual_board.lower()
            ):
                raise ValueError(f"Requested board {board!r} but run belongs to {actual_board!r}")
            board_name = str(board or actual_board)
            spec = get_model(name)
            store = RunStore(run["directory"],
                             artifact_root=config["preprocessing"]["artifact_root"])
            try:
                model_path = _resolve_tshap_model_directory(
                    spec, store, run["directory"],
                    explicit_model_dir=model_dir is not None)
            except (FileNotFoundError, ValueError) as exc:
                if model_dir is not None:
                    raise
                if logger:
                    logger.warning("CPA skipped run without a usable fitted model | %s | %s",
                                   run["directory"], exc)
                continue
            if not saved_model_complete(spec, model_path):
                if model_dir is not None:
                    raise FileNotFoundError(f"Saved model/transform missing: {model_path}")
                if logger:
                    logger.warning("CPA skipped run without a complete fitted model | %s",
                                   run["directory"])
                continue
            development = prepared.find("development", config, run["manifest"])
            blind = prepared.find("blind", config, run["manifest"])
            times = np.asarray(getattr(blind, f"{family}_time_ps"), np.float64)
            devtime = np.asarray(getattr(development, f"{family}_time_ps"))
            if not np.array_equal(times, devtime):
                raise ValueError("Prepared development/blind waveform time grids differ")
            fitted = load_fitted_model(
                spec, model_path, run.get("best", {}).get("parameters", {}), config)
            try:
                mask = (np.ones(len(times), bool) if fitted.sample_mask is None else
                        np.asarray(fitted.sample_mask, bool))
                if mask.shape != times.shape:
                    raise ValueError("Fitted-model mask does not match ML-input grid")
                schedule = (timing_cutoff_schedule_ns(times, mask)
                            if family == "timing" else None)
                cutoffs, omitted = cutoff_samples(times, mask, schedule)
                if not cutoffs:
                    raise ValueError("No valid chronological cutoffs for this run")
                identity = _identity(run, development, blind, board_name, family)
                dest = _cpa_destination(
                    output_root, board_name, run["directory"], family, name
                )
                already_complete = _guard_dataset_identity(dest, identity)
                if already_complete and not overwrite:
                    previous = read_json(dest / "summary.json")
                    expected = {
                        "n_blind_events": (int(blind.n_events) if int(max_events) == 0
                                           else min(int(max_events), int(blind.n_events))),
                        "bootstrap": int(bootstrap),
                        "seed": int(seed),
                        "orders": list(orders),
                        "local_samples": int(local_samples),
                        "fit_events_requested": int(fit_events),
                        "validation_events_requested": int(validation_events),
                        "ridge_alpha": float(ridge_alpha),
                        "train_stride": int(train_stride),
                        "cutoff_samples": [int(k) for k in cutoffs],
                        "performance_only_version": 2,
                    }
                    if all(previous.get(k) == v for k,v in expected.items()) and all(
                        (dest / p).is_file() for p in
                        ("metrics.csv", "timing_resolution.png", "predictions.npz")
                    ):
                        outputs.append(dest)
                        continue
                    if logger:
                        logger.info("CPA settings/results changed; recomputing %s", dest)
                cache_key = (identity, mask.tobytes(), tuple(cutoffs),
                             int(seed), int(fit_events), int(validation_events),
                             tuple(orders), int(local_samples), float(ridge_alpha),
                             int(train_stride))
                if cache_key not in imputer_cache:
                    dev_waves = getattr(development, f"{family}_windows")
                    if dev_waves is None:
                        raise ValueError("Prepared development waveforms missing")
                    fit_seed = semantic_seed(seed, "cpa-imputer", board_name, identity)
                    imputer_cache[cache_key] = select_imputers(
                        dev_waves, mask, cutoffs, seed=fit_seed,
                        fit_events=fit_events, validation_events=validation_events,
                        orders=orders, local_samples=local_samples,
                        ridge_alpha=ridge_alpha, train_stride=train_stride)
                imputer = imputer_cache[cache_key]
                event_seed = semantic_seed(seed, "cpa-blind", board_name,
                                           identity, name)
                indices = _draw_indices(blind.n_events, max_events, event_seed)
                waves = waveform_view(blind, config["mode"], indices).materialize()
                if not np.all(np.isfinite(waves)):
                    raise ValueError("Nonfinite prepared blind waveforms")
                target = np.asarray(model_target(blind, config["mode"]), np.float64)[indices]
                predictions = np.empty((len(cutoffs)+1, len(indices)), np.float64)
                mean_predictions = np.empty((len(cutoffs), len(indices)), np.float64)
                for j, cutoff in enumerate(cutoffs):
                    reconstructed = reconstruct_pair(waves, imputer, mask, cutoff)
                    predictions[j] = _predict_bounded(
                        spec, fitted, reconstructed, predict_batch_size)
                    mean_reconstructed = reconstruct_pair(
                        waves, imputer, mask, cutoff, mean_only=True)
                    mean_predictions[j] = _predict_bounded(
                        spec, fitted, mean_reconstructed, predict_batch_size)
                    if logger:
                        logger.info("CPA | %s/%s | cutoff=%g ns | %d/%d",
                                    family, name, times[cutoff]/1000., j+1, len(cutoffs))
                # Use the established fitted-model inference path as the unmodified
                # reference; never derive the full result by imputing final samples.
                predictions[-1] = predict_indices(
                    spec, fitted, blind, config["mode"], indices,
                    chunk_size=predict_batch_size)
                metrics = summarize_predictions(predictions, target,
                                                seed=event_seed+14,
                                                bootstrap=bootstrap)
                axis = [float(times[k]/1000.) for k in cutoffs] + [float(times[-1]/1000.)]
                metric_rows = [
                    {**row, "cutoff_sample": cutoffs[j] if j < len(cutoffs) else len(times),
                     "cutoff_ns": axis[j],
                     "kind": "prefix" if j < len(cutoffs) else "full",
                     "n_events": len(indices)} for j, row in enumerate(metrics)
                ]
                dest.mkdir(parents=True, exist_ok=True)
                _csv(dest / "metrics.csv", metric_rows)
                _csv(dest / "imputer_validation.csv", imputer["scores"])
                selected = [
                    {"detector": d+1, "cutoff_sample": int(k),
                     "cutoff_ns": float(times[k]/1000),
                     **{field: value for field, value in choice.items()
                        if field != "coefficients"}}
                    for (d,k), choice in sorted(imputer["selected"].items())
                ]
                _csv(dest / "selected_imputers.csv", selected)
                np.savez_compressed(dest / "predictions.npz",
                    event_index=np.asarray(blind.event_index[indices], np.int64),
                    dataset_indices=indices, target_ps=target,
                    predictions_ps=predictions,
                    mean_only_predictions_ps=mean_predictions,
                    cutoff_samples=np.asarray(cutoffs, np.int64),
                    cutoff_ns=np.asarray(axis, np.float64))
                np.savez_compressed(dest / "imputer.npz",
                    mean=np.stack([d["mean"] for d in imputer["detectors"]]),
                    std=np.stack([d["std"] for d in imputer["detectors"]]),
                    sample_mask=mask,
                    fit_indices=imputer["fit_ids"],
                    validation_indices=imputer["validation_ids"],
                    refit_indices=imputer["refit_ids"],
                    **{f'd{d+1}_{method}_{p}': coeff for d, data in
                       enumerate(imputer["detectors"]) for (method,p), coeff
                       in data["coefficients"].items()})
                (dest / "summary.json").write_text(json.dumps(_json_safe({
                    "method": "deterministic_chronological_prefix_attribution",
                    "board": board_name, "dataset_identity": identity,
                    "study_directory": batch_root(run["directory"]).name,
                    "source_model_directory": str(model_path),
                    "run_directory": str(run["directory"]),
                    "seed": seed, "event_seed": event_seed,
                    "mode": family, "model": name, "window": "wide",
                    "n_blind_events": len(indices), "bootstrap": bootstrap,
                    "fit_events": int(len(imputer["fit_ids"])),
                    "validation_events": int(len(imputer["validation_ids"])),
                    "refit_events": int(len(imputer["refit_ids"])),
                    "local_samples": local_samples, "orders": list(orders),
                    "fit_events_requested": int(fit_events),
                    "validation_events_requested": int(validation_events),
                    "ridge_alpha": float(ridge_alpha),
                    "train_stride": int(train_stride),
                    "cutoff_samples": [int(k) for k in cutoffs],
                    "performance_only_version": 2,
                    "cutoffs": axis, "omitted_cutoffs": omitted,
                    "masked_intervals_ns": masked_time_intervals(times, mask),
                    "selected_imputers": selected,
                    "attribution_interpretation": "deterministic imputation, not exact conditional Shapley",
                }), indent=2), encoding="utf-8")
                _plot_metrics(
                    dest/"timing_resolution.png", metric_rows,
                    f"{board_name} {family} {name}",
                    masked_intervals=masked_time_intervals(times, mask),
                )
                # Clean up obsolete performance-unrelated outputs from old runs.
                for obsolete in ("prediction_increments.png", "attribution.csv"):
                    (dest / obsolete).unlink(missing_ok=True)
                outputs.append(dest)
                print(f"CPA complete | {board_name}/{family}/{name} | "
                      f"events={len(indices)} | {dest}", flush=True)
            finally:
                del fitted
                release_training_memory()
        except Exception:
            if logger:
                logger.exception("CPA failed | %s", run["directory"])
            raise
    if not outputs:
        raise ValueError("No matching wide-window runs with usable saved models were found")
    return outputs
