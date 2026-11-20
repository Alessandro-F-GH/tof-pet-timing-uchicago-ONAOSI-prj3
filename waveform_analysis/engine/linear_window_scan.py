"""Temporal information scan for the saved shared_linear_ridge study.

Refit the exact shared Ridge on subsets of the *existing* prepared waveform
grid. Both scans use Development CV for alpha selection and Test only for CTR.
No new prepared datasets or input transformations are persisted.
"""
from __future__ import annotations

import csv
import json
import time
from pathlib import Path

import numpy as np

from waveform_analysis.core.io import atomic_json, canonical_hash
from waveform_analysis.data.prepared_cache import PreparedCacheLookup
from waveform_analysis.data.results import discover_runs, resolve_run_config
from waveform_analysis.data.splits import make_cv_split
from waveform_analysis.data.view import mode_family, model_sample_mask, model_target
from waveform_analysis.engine.validation import (
    best_complete_candidate, evaluate_candidate,
)
from waveform_analysis.models.linear.linear_ridge_common import (
    candidates as ridge_candidates, fit_ridge, predict,
)
from waveform_analysis.reporting.stats import (
    blind_event_bootstrap, metric_values,
)


def temporal_windows(start_ns, end_ns, step_ns=1.0, width_ns=None, *,
                     progressive_start_ns=None, progressive_initial_end_ns=None,
                     progressive_ends_ns=None, scan_kind="both"):
    """Return (kind, left_ns, right_ns) with [left, right) convention.

    Fixed windows have width=step by default and never overlap. Explicit
    progressive upper bounds replace uniform increments and preserve their
    specified numerical order. The initial Onishi bound is included only when
    supplied in the explicit list. The final incomplete fixed window is excluded.
    """
    start_ns, end_ns, step_ns = map(float, (start_ns, end_ns, step_ns))
    width_ns = step_ns if width_ns is None else float(width_ns)
    if not all(np.isfinite(x) for x in (start_ns, end_ns, step_ns, width_ns)):
        raise ValueError("Window bounds, step and width must be finite")
    if end_ns <= start_ns or step_ns <= 0 or width_ns <= 0:
        raise ValueError("Expected end>start and positive step/width")
    if scan_kind not in ("both", "progressive", "fixed"):
        raise ValueError("scan_kind must be both, progressive or fixed")
    if scan_kind != "progressive" and width_ns > end_ns - start_ns + 1e-9:
        raise ValueError("Fixed width cannot exceed the available waveform duration")
    progressive_start = (
        start_ns if progressive_start_ns is None else float(progressive_start_ns)
    )
    first_end = (
        progressive_start + step_ns
        if progressive_initial_end_ns is None
        else float(progressive_initial_end_ns)
    )
    if not (start_ns <= progressive_start < first_end <= end_ns):
        raise ValueError("Progressive starting window must lie inside wide boundaries")
    if progressive_ends_ns is None:
        ends = [first_end]
        for i in range(1, int(np.ceil((end_ns - first_end) / step_ns)) + 1):
            ends.append(min(first_end + i * step_ns, end_ns))
    else:
        ends = [float(value) for value in progressive_ends_ns]
        if not ends or not all(np.isfinite(value) for value in ends):
            raise ValueError("Explicit progressive ends must be a nonempty finite list")
        if (any(value < first_end - 1e-9 or value > end_ns + 1e-9 for value in ends)
                or any(a >= b for a, b in zip(ends, ends[1:]))):
            raise ValueError(
                "Progressive ends must increase strictly within [Onishi end, wide end]"
            )
    prefix = [("progressive", progressive_start, right) for right in ends]
    fixed = []
    for i in range(int(np.floor((end_ns - start_ns - width_ns) / step_ns + 1e-9)) + 1):
        left = start_ns + i * step_ns
        fixed.append(("fixed", left, left + width_ns))
    return (
        (prefix if scan_kind in ("both", "progressive") else [])
        + (fixed if scan_kind in ("both", "fixed") else [])
    )


def _elapsed_label(seconds):
    """Compact elapsed/ETA formatting for console progress messages."""
    seconds = max(0, int(round(float(seconds))))
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:d}h{minutes:02d}m{seconds:02d}s" if hours else f"{minutes:d}m{seconds:02d}s"


def selected_indices(time_ps, frozen_mask, left_ns, right_ns, *, last_edge=False):
    """Intersect a scan window with the untouched control-frozen sample mask."""
    time_ps = np.asarray(time_ps, np.float64)
    frozen_mask = np.asarray(frozen_mask, bool)
    if time_ps.ndim != 1 or frozen_mask.shape != time_ps.shape:
        raise ValueError("Waveform time grid and frozen sample mask must match")
    if time_ps.size < 2 or not np.all(np.diff(time_ps) > 0):
        raise ValueError("Waveform time grid must be strictly increasing")
    threshold = float(right_ns) * 1000.0
    inside = (time_ps >= float(left_ns) * 1000.0 - 1e-6) & (
        time_ps <= threshold + 1e-6 if last_edge
        else time_ps < threshold - 1e-6
    )
    return np.flatnonzero(inside & frozen_mask)


def _shared_difference(dataset, mode, columns):
    """Read only the selected columns of the existing prepared waveform array."""
    family = mode_family(mode)
    waves = getattr(dataset, f"{family}_windows")
    if waves is None:
        raise ValueError(f"Missing prepared {family} waveforms")
    first = np.take(waves[:, 0, :], columns, axis=1)
    second = np.take(waves[:, 1, :], columns, axis=1)
    return np.asarray(first - second, dtype=np.float32)


def _choose_alpha(x, target, split, alpha_grid, cv_cfg, fit_cfg, seed, output_limit):
    """Reuse standard Ridge fitter, CV folds, pruning and CTR scoring."""
    led_scores = {
        fold.fold_id: metric_values(target[fold.validation], fit_cfg, seed=seed)
        for fold in split.folds
    }
    summaries, rows_by_candidate = [], {}
    pruning = cv_cfg["pruning"]
    criterion = cv_cfg["metric"]
    startup = int(pruning["startup_complete_candidates"])

    def score_fold(parameters, fold):
        artifact = fit_ridge(
            parameters, x[fold.train], target[fold.train],
            model_name="shared_linear_ridge", fit_intercept=False,
            config={}, metadata={},
        )
        prediction = np.clip(
            predict(artifact, x[fold.validation]), -output_limit, output_limit,
        )
        scored = metric_values(
            target[fold.validation] - prediction, fit_cfg, seed=seed,
        )
        led = led_scores[fold.fold_id]
        return {
            **scored, "led_ctr_ps": led["ctr_ps"], "led_rmse_ps": led["rmse_ps"],
            "n_train": len(fold.train), "n_validation": len(fold.validation),
        }

    for alpha in alpha_grid:
        parameters = {"ridge_alpha": float(alpha)}
        identifier = canonical_hash(parameters)
        complete = sum(
            not summary["pruned"]
            and summary["completed_folds"] == summary["total_folds"]
            for summary in summaries
        )
        incumbent = (
            best_complete_candidate(summaries, criterion)
            if complete >= startup else None
        )
        summary, fold_rows = evaluate_candidate(
            candidate_id=identifier, parameters=parameters,
            folds=split.folds, evaluate_fold=score_fold,
            metric=criterion, pruning=pruning,
            incumbent_candidate_id=(
                incumbent["candidate_id"] if incumbent is not None else None
            ),
            incumbent_rows=(
                rows_by_candidate[incumbent["candidate_id"]]
                if incumbent is not None else None
            ),
        )
        summaries.append(summary)
        rows_by_candidate[identifier] = fold_rows

    best = best_complete_candidate(summaries, criterion)
    if best is None:
        # The standard study fallback fully evaluates the most promising
        # partially pruned candidate rather than selecting an incomplete CV.
        finite = [row for row in summaries if np.isfinite(row[f"{criterion}_mean_ps"])]
        if not finite:
            raise RuntimeError("No Ridge alpha produced a finite Development CV score")
        partial = min(
            finite,
            key=lambda row: (row[f"{criterion}_mean_ps"], row["candidate_id"]),
        )
        best, _ = evaluate_candidate(
            candidate_id=partial["candidate_id"],
            parameters=partial["parameters"],
            folds=split.folds, evaluate_fold=score_fold,
            metric=criterion, pruning=pruning,
            existing_fold_rows=rows_by_candidate[partial["candidate_id"]],
            force_complete=True,
        )
    return float(best["parameters"]["ridge_alpha"]), float(best[f"{criterion}_mean_ps"])


def _plot_curve(rows, kind, path, *, led_ctr, full_ctr):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    data = [row for row in rows if row["kind"] == kind]
    if not data:
        return
    x = [row["end_ns"] if kind == "progressive" else
         (row["start_ns"] + row["end_ns"]) / 2.0 for row in data]
    y = [row["ctr_ps"] for row in data]
    e = [row["ctr_bootstrap_std_ps"] for row in data]
    fig, ax = plt.subplots(figsize=(10, 5), constrained_layout=True)
    ax.errorbar(x, y, yerr=e, fmt="-o", markersize=3, linewidth=1.25,
                capsize=2, label="Shared Ridge (test CTR)")
    ax.axhline(led_ctr, color="0.4", linestyle="--", label="LED (wide)")
    if full_ctr is not None:
        ax.axhline(full_ctr, color="#009E73", linestyle=":",
                   label="Saved full-wide shared Ridge")
    ax.set_xlabel("Window end (ns)" if kind == "progressive" else "Window center (ns)")
    ax.set_ylabel("Test CTR (ps)")
    ax.set_title("Progressive temporal expansion" if kind == "progressive"
                 else "Fixed-width temporal scan")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def scan_linear_windows(
    results, *, mode=None, cache_root=None, output_root="xai/linear/window_scan",
    step_ns=1.0, width_ns=None, progressive_ends_ns=None,
    scan_kind="both", seed=None, overwrite=False, logger=None,
):
    """Run independent temporal scans for energy/timing saved *wide* Ridge runs."""
    root = Path(results).expanduser().resolve()
    all_runs = discover_runs(root)
    selected = [
        run for run in all_runs
        if run["manifest"]["model"] == "shared_linear_ridge"
        and run["manifest"].get("window_name") == "wide"
        and (mode is None or mode_family(run["manifest"]["mode"]) == mode)
    ]
    if not selected:
        raise ValueError("No complete wide shared_linear_ridge runs found")
    lookup = PreparedCacheLookup()
    output_root = Path(output_root).expanduser().resolve()
    if output_root == root or output_root.is_relative_to(root):
        raise ValueError("Window scan output must be outside result root")
    outputs = []
    for run in selected:
        cfg = resolve_run_config(run["directory"])
        if cache_root is not None:
            cfg["preprocessing"]["cache_dir"] = str(
                Path(cache_root).expanduser().resolve() / "preprocessing"
            )
        study_seed = int(cfg["seed"] if seed is None else seed)
        mode_name = mode_family(cfg["mode"])
        destination = output_root / root.parent.name / mode_name
        destination.mkdir(parents=True, exist_ok=True)
        dev = lookup.find("development", cfg, run["manifest"])
        test = lookup.find("blind", cfg, run["manifest"])
        full_time = np.asarray(getattr(dev, f"{mode_name}_time_ps"), np.float64)
        test_time = np.asarray(getattr(test, f"{mode_name}_time_ps"), np.float64)
        if not np.array_equal(full_time, test_time):
            raise ValueError("Development and test time grids differ")
        frozen = model_sample_mask(dev, cfg["mode"])
        if not np.array_equal(frozen, model_sample_mask(test, cfg["mode"])):
            raise ValueError("Development and test sample masks differ")
        start_ns, end_ns = (
            float(cfg["window_ns"]["start"]), float(cfg["window_ns"]["end"])
        )
        windows = temporal_windows(
            start_ns, end_ns, step_ns, width_ns,
            progressive_start_ns=max(start_ns, -1.5),
            progressive_initial_end_ns=min(end_ns, 2.0),
            progressive_ends_ns=progressive_ends_ns,
            scan_kind=scan_kind,
        )
        y_dev = np.asarray(model_target(dev, cfg["mode"]), np.float64)
        y_test = np.asarray(model_target(test, cfg["mode"]), np.float64)
        cv = cfg["cross_validation"]
        split = make_cv_split(
            len(y_dev),
            population_identity=str(dev.manifest["analysis_protocol_identity"]),
            batch_seed=study_seed, n_folds=int(cv["folds"]),
            shuffle=bool(cv["shuffle"]),
        )
        minimum = int(cv["minimum_events_per_fold"])
        if any(len(fold.validation) < minimum for fold in split.folds):
            raise ValueError("Not enough Development events per CV fold")
        alphas = [candidate["ridge_alpha"] for candidate in
                  ridge_candidates(cfg["model"]["space"])]
        settings = {
            "run": str(run["directory"]), "mode": cfg["mode"],
            "step_ns": step_ns, "width_ns": step_ns if width_ns is None else width_ns,
            "progressive_initial_window_ns": [max(start_ns, -1.5), min(end_ns, 2.0)],
            "progressive_ends_ns": (
                None if progressive_ends_ns is None
                else [float(value) for value in progressive_ends_ns]
            ),
            "scan_kind": scan_kind,
            "seed": study_seed, "alphas": alphas, "cv": cv,
            "bootstrap": cfg["bootstrap"],
            "development_identity": dev.manifest["analysis_protocol_identity"],
            "test_identity": test.manifest["analysis_protocol_identity"],
        }
        identity = canonical_hash(settings)
        manifest_path = destination / "settings.json"
        if manifest_path.exists() and not overwrite:
            if json.loads(manifest_path.read_text(encoding="utf-8")).get("identity") != identity:
                raise ValueError("Existing scan has different settings; use --overwrite")
        atomic_json(manifest_path, {"identity": identity, **settings})

        limit = float(cfg["ml_output"]["max_abs_ps"])
        full_ctr = float(run["blind"]["ctr_ps"])
        led_ctr = float(run["blind"]["led_ctr_ps"])
        rows = []
        start_time = time.monotonic()
        computed_seconds = 0.0
        computed_count = 0
        cache_hits = 0
        empty_count = 0
        total = len(windows)
        if logger:
            logger.info(
                "WINDOW SCAN START | board=%s | mode=%s | %d windows "
                "(progressive=%d, fixed=%d) | step=%.3g ns | fixed width=%.3g ns "
                "| development=%d | test=%d | alpha candidates=%d | CV folds=%d "
                "| bootstrap=%d | output=%s",
                root.parent.name, mode_name, total,
                sum(item[0] == "progressive" for item in windows),
                sum(item[0] == "fixed" for item in windows),
                float(step_ns), float(step_ns if width_ns is None else width_ns),
                len(y_dev), len(y_test), len(alphas), len(split.folds),
                int(cfg["bootstrap"]["n_resamples"]), destination,
            )
        previous_kind = None
        for position, (kind, left, right) in enumerate(windows, 1):
            if kind != previous_kind:
                if logger:
                    logger.info("WINDOW SCAN PHASE | %s | %s", mode_name, kind.upper())
                previous_kind = kind
            label = f"{mode_name} | {position}/{total} | {kind} [{left:g}, {right:g}] ns"
            filename = f"{kind}_{left:.4f}_{right:.4f}.json"
            path = destination / filename
            indices = selected_indices(
                full_time, frozen, left, right,
                last_edge=(abs(right - end_ns) < 1e-9),
            )
            if not indices.size:
                empty_count += 1
                if logger:
                    logger.warning("WINDOW SKIP | %s | no samples after frozen mask", label)
                continue
            if path.exists() and not overwrite:
                row = json.loads(path.read_text(encoding="utf-8"))
                if row.get("identity") == identity:
                    cache_hits += 1
                    rows.append(row)
                    if logger:
                        logger.info(
                            "WINDOW CACHE HIT | %s | CTR=%.2f +/- %.2f ps | alpha=%g",
                            label, row["ctr_ps"], row["ctr_bootstrap_std_ps"],
                            row["ridge_alpha"],
                        )
                    continue

            window_start = time.monotonic()
            if logger:
                logger.info(
                    "WINDOW START | %s | %d samples | CV %d candidates x %d folds",
                    label, indices.size, len(alphas), len(split.folds),
                )
            x_dev = _shared_difference(dev, cfg["mode"], indices)
            alpha, cv_score = _choose_alpha(
                x_dev, y_dev, split, alphas, cv, cfg["fit"], study_seed, limit,
            )
            if logger:
                logger.info(
                    "WINDOW CV DONE | %s | selected alpha=%g | CV %s=%.2f ps"
                    " | elapsed=%s | final fit + test bootstrap next",
                    label, alpha, cv["metric"].upper(), cv_score,
                    _elapsed_label(time.monotonic() - window_start),
                )
            model = fit_ridge(
                {"ridge_alpha": alpha}, x_dev, y_dev,
                model_name="shared_linear_ridge", fit_intercept=False,
                config={}, metadata={},
            )
            del x_dev
            x_test = _shared_difference(test, cfg["mode"], indices)
            prediction = np.clip(predict(model, x_test), -limit, limit)
            del x_test, model
            stats, _ = blind_event_bootstrap(
                y_test - prediction, y_test, cfg["fit"],
                n_resamples=int(cfg["bootstrap"]["n_resamples"]),
                seed=study_seed,
            )
            row = {
                "identity": identity, "kind": kind,
                "start_ns": left, "end_ns": right,
                "n_samples": int(indices.size),
                "ridge_alpha": alpha, "cv_score_ps": cv_score,
                "ctr_ps": float(stats["ctr_ps"]),
                "ctr_bootstrap_std_ps": float(stats["ctr_bootstrap_std_ps"]),
                "led_ctr_ps": float(stats["led_ctr_ps"]),
                "n_events_test": int(y_test.size),
            }
            atomic_json(path, row)
            rows.append(row)
            window_seconds = time.monotonic() - window_start
            computed_seconds += window_seconds
            computed_count += 1
            if logger:
                remaining = total - position
                eta = (
                    _elapsed_label(remaining * computed_seconds / computed_count)
                    if computed_count else "n/a"
                )
                logger.info(
                    "WINDOW DONE | %s | test CTR=%.2f +/- %.2f ps | alpha=%g "
                    "| window=%s | elapsed=%s | remaining=%d | ETA~%s",
                    label, row["ctr_ps"], row["ctr_bootstrap_std_ps"], alpha,
                    _elapsed_label(window_seconds),
                    _elapsed_label(time.monotonic() - start_time), remaining, eta,
                )

        if logger:
            logger.info(
                "WINDOW SCAN FINALIZING | %s | completed=%d/%d | cached=%d "
                "| empty=%d | saving plots and CSV",
                mode_name, computed_count, total, cache_hits, empty_count,
            )
        for kind in ("progressive", "fixed"):
            _plot_curve(
                rows, kind, destination / f"{kind}_ctr.pdf",
                led_ctr=led_ctr, full_ctr=full_ctr,
            )
        with (destination / "window_scan.csv").open(
            "w", newline="", encoding="utf-8",
        ) as stream:
            fields = ["kind", "start_ns", "end_ns", "n_samples",
                      "ridge_alpha", "cv_score_ps", "ctr_ps",
                      "ctr_bootstrap_std_ps", "led_ctr_ps", "n_events_test"]
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows({key: row[key] for key in fields} for row in rows)
        outputs.append(destination)
        if logger:
            logger.info(
                "WINDOW SCAN COMPLETE | %s | processed=%d/%d | fitted=%d "
                "| cached=%d | empty=%d | total=%s | output=%s",
                mode_name, len(rows) + empty_count, total,
                computed_count, cache_hits, empty_count,
                _elapsed_label(time.monotonic() - start_time), destination,
            )
    return outputs
