"""Deletion benchmark for saved direct/shared Ridge models using frozen Linear SHAP.

Reuses the grouped global attribution ranking and Development background stored
in global_attributions.npz. Test-set predictions are rebuilt analytically from
the saved Ridge coefficients, not by repeated inference or model refitting.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from waveform_analysis.core.io import atomic_json
from waveform_analysis.data.prepared_cache import PreparedCacheLookup
from waveform_analysis.data.results import batch_root, resolve_run_config
from waveform_analysis.data.storage import RunStore
from waveform_analysis.data.view import mode_family, waveform_view
from waveform_analysis.engine.linear_shap import (
    RIDGE_NAMES, _aligned_blind_predictions, linear_attributions,
)
from waveform_analysis.engine import tshap_explainer as local
from waveform_analysis.engine.train import (
    load_fitted_model, predict_indices, saved_model_complete,
)
from waveform_analysis.models import get_model
from waveform_analysis.reporting.stats import metric_values
from waveform_analysis.data.splits import bootstrap_draw_indices


def _npz_for_run(run_directory, *, global_npz, global_root):
    if global_npz is not None:
        path = Path(global_npz).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        return path
    if global_root is None:
        raise ValueError("Specify --global-npz or --global-root to reuse saved global SHAP")
    root = Path(global_root).expanduser().resolve()
    run_directory = Path(run_directory)
    rel = run_directory.relative_to(batch_root(run_directory))
    wanted = (
        batch_root(run_directory).name, *rel.parts,
        "linear_shap_global", "global_attributions.npz",
    )
    board = batch_root(run_directory).parent.name
    matches = [
        item for item in root.rglob("global_attributions.npz")
        if item.parts[-len(wanted):] == wanted and board in item.parts
    ]
    if len(matches) != 1:
        raise ValueError(
            f"Expected one global SHAP NPZ for {run_directory} under {root}; "
            f"found {len(matches)}: {matches}"
        )
    return matches[0]


def _load_global_ranking(path, *, model, mode, window, time_ps, n_events):
    path = Path(path)
    summary_path = path.with_name("summary.json")
    if not summary_path.is_file():
        raise FileNotFoundError(f"Missing global SHAP summary: {summary_path}")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if (summary.get("model"), summary.get("mode"), summary.get("window")) != (
        model, mode, window
    ):
        raise ValueError("SHAP NPZ model/mode/window differs from the fitted run")
    if summary.get("background") != "complete_prepared_development_mean":
        raise ValueError("Expected the complete prepared Development mean background")
    if summary.get("aggregation") != "sum_positive_negative_per_event_then_mean_across_events":
        raise ValueError("Unknown global SHAP aggregation")
    if summary.get("explained_prediction") != "unbounded_linear_regressor_before_pipeline_clipping":
        raise ValueError("Global SHAP was not computed for unbounded Ridge predictions")
    with np.load(path, allow_pickle=False) as data:
        required = {
            "time_ps", "background_mean_normalized", "group_starts",
            "mean_positive_contribution_ps", "mean_negative_contribution_ps",
            "event_id",
        }
        if not required.issubset(data.files):
            raise ValueError(f"Global SHAP NPZ missing {sorted(required - set(data.files))}")
        saved_time = np.asarray(data["time_ps"], np.float64)
        background = np.asarray(data["background_mean_normalized"], np.float64)
        starts = np.asarray(data["group_starts"], np.int64)
        positive = np.asarray(data["mean_positive_contribution_ps"], np.float64)
        negative = np.asarray(data["mean_negative_contribution_ps"], np.float64)
        explained_ids = np.asarray(data["event_id"], np.int64)
    if (not np.array_equal(saved_time, time_ps)
            or background.shape != (2, len(time_ps))):
        raise ValueError("SHAP grid/background does not match the fitted waveform grid")
    if (starts.ndim != 1 or not len(starts) or starts[0] != 0
            or np.any(np.diff(starts) <= 0) or starts[-1] >= len(time_ps)):
        raise ValueError("Invalid global SHAP group boundaries")
    width = int(summary.get("settings", {}).get("group_samples", 0))
    if width < 1 or not np.array_equal(starts, np.arange(0, len(time_ps), width)):
        raise ValueError("SHAP group widths differ from the saved settings")
    n_channels = 1 if model == "shared_linear_ridge" else 2
    if positive.shape != (n_channels, len(starts)) or negative.shape != positive.shape:
        raise ValueError("SHAP importances have incompatible detector/group dimensions")
    if (not all(np.all(np.isfinite(a)) for a in (background, positive, negative))
            or np.any(positive < -1e-8) or np.any(negative > 1e-8)):
        raise ValueError("Invalid global SHAP importance/background values")
    if (len(explained_ids) != int(summary["n_explained_events"])
            or not len(explained_ids) or len(np.unique(explained_ids)) != len(explained_ids)):
        raise ValueError("Global SHAP sampling metadata is inconsistent")
    if n_events < 5:
        raise ValueError("Test set is too small for CTR evaluation")
    importance = (positive - negative).reshape(-1)
    descending = np.argsort(-importance, kind="stable")
    return background, starts, importance, descending, explained_ids, summary


def _reconstruct_contributions(
    blind, *, mode, mask, background, starts, regressor, shared,
    saved_prediction, chunk_size, limit, max_events,
):
    n = len(saved_prediction) if max_events is None else min(len(saved_prediction), max_events)
    if n < 5:
        raise ValueError("At least five events are required")
    n_groups = len(starts) * (1 if shared else 2)
    contributions = np.empty((n, n_groups), dtype=np.float64)
    unbounded = np.empty(n, dtype=np.float64)
    baseline = None
    for first in range(0, n, chunk_size):
        last = min(n, first + chunk_size)
        indices = np.arange(first, last, dtype=np.int64)
        pair = np.asarray(
            waveform_view(blind, mode, indices).materialize(), np.float64
        )[:, :, mask]
        phi, mean_prediction, raw = linear_attributions(
            pair, background, regressor.coef_,
            shared=shared, intercept=float(regressor.intercept_),
        )
        if baseline is None:
            baseline = mean_prediction
        elif not np.isclose(mean_prediction, baseline, atol=1e-9, rtol=0):
            raise ValueError("Inconsistent mean prediction across chunks")
        grouped = np.add.reduceat(phi, starts, axis=2)
        contributions[first:last] = (
            grouped.sum(axis=1) if shared else grouped.reshape(last - first, -1)
        )
        unbounded[first:last] = raw
    bounded = _clip(unbounded, limit)
    if not np.allclose(bounded, saved_prediction[:n], atol=0.05, rtol=1e-5):
        raise ValueError("Analytical Ridge predictions differ from saved blind predictions")
    if not np.allclose(
        baseline + contributions.sum(axis=1), unbounded, atol=1e-6, rtol=1e-9
    ):
        raise ValueError("Grouped SHAP contributions do not reconstruct the model")
    return contributions, unbounded, float(baseline)


def _clip(values, limit):
    return np.clip(values, -limit, limit) if limit is not None else values


def deletion_predictions(unbounded_prediction, contributions, order, *, limit=None):
    """Predictions at deletion steps 0..K; analytically subtract signed groups."""
    unbounded_prediction = np.asarray(unbounded_prediction, np.float64).reshape(-1)
    contributions = np.asarray(contributions, np.float64)
    order = np.asarray(order, np.int64)
    if (contributions.ndim != 2
            or contributions.shape[0] != len(unbounded_prediction)
            or sorted(order.tolist()) != list(range(contributions.shape[1]))):
        raise ValueError("Invalid contribution matrix or deletion ordering")
    cumulative = np.cumsum(contributions[:, order], axis=1)
    return _clip(
        unbounded_prediction[:, None] - np.column_stack(
            [np.zeros(len(unbounded_prediction)), cumulative]
        ),
        limit,
    )


def _bootstrap_ctr(residual, fit, *, seed, bootstrap_indices):
    """Conditional event-bootstrap CTR uncertainty; fitted model stays frozen.

    The identical event resamples are reused at every deletion point and for
    every ordering. The central CTR always uses the full unresampled test set.
    """
    if not bootstrap_indices:
        return 0.0
    values = np.array([
        metric_values(residual[idx], fit, seed=seed)["ctr_ps"]
        for idx in bootstrap_indices
    ], dtype=np.float64)
    if not np.all(np.isfinite(values)):
        raise ValueError("Non-finite CTR from deletion bootstrap")
    return float(np.std(values, ddof=1))


def _evaluate_order(unbounded, contributions, order, *, target, fit, seed, limit,
                    ordering, step_groups, original_ctr, bootstrap_indices=()):
    n_groups = contributions.shape[1]
    steps = sorted(set([0, n_groups] + list(range(step_groups, n_groups, step_groups))))
    removed = np.zeros(len(unbounded), dtype=np.float64)
    rows = []
    for step in range(n_groups + 1):
        if step:
            removed += contributions[:, order[step - 1]]
        if step not in steps:
            continue
        residual = target - _clip(unbounded - removed, limit)
        metrics = metric_values(residual, fit, seed=seed)
        uncertainty = _bootstrap_ctr(
            residual, fit, seed=seed, bootstrap_indices=bootstrap_indices
        )
        rows.append({
            "ordering": ordering, "n_removed": step, "n_groups": n_groups,
            "fraction_removed": step / n_groups,
            "ctr_ps": metrics["ctr_ps"], "ctr_bootstrap_std_ps": uncertainty,
            "rmse_ps": metrics["rmse_ps"],
            "delta_ctr_ps": metrics["ctr_ps"] - original_ctr,
        })
    return rows


def _plot_curves(path, rows, *, led_ctr, led_ctr_bootstrap_std, title):
    """Academic grayscale CTR plot with event-bootstrap uncertainty bands.

    SHAP bands: +/- 1 event-bootstrap standard deviation. The random band
    additionally includes variation between deletion order permutations.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8.3, 4.8))
    lines = (
        ("shap_descending", "SHAP descending", "0.12", "-", 2.0, 0.14),
        ("shap_ascending", "SHAP ascending", "0.48", "-.", 1.7, 0.12),
    )
    for key, label, color, style, linewidth, alpha in lines:
        subset = sorted(
            (r for r in rows if r["ordering"] == key),
            key=lambda r: r["n_removed"],
        )
        x = 100 * np.array([r["fraction_removed"] for r in subset])
        y = np.array([r["ctr_ps"] for r in subset])
        error = np.array([r["ctr_bootstrap_std_ps"] for r in subset])
        ax.plot(x, y, color=color, linestyle=style, linewidth=linewidth, label=label)
        ax.fill_between(x, y - error, y + error, color=color,
                        alpha=alpha, linewidth=0)

    random_names = sorted({
        r["ordering"] for r in rows if r["ordering"].startswith("random_")
    })
    if random_names:
        subsets = [
            sorted((r for r in rows if r["ordering"] == name),
                   key=lambda r: r["n_removed"])
            for name in random_names
        ]
        positions = np.array(
            [r["fraction_removed"] for r in subsets[0]], dtype=float
        )
        if any(not np.array_equal(
            positions, [r["fraction_removed"] for r in subset]
        ) for subset in subsets[1:]):
            raise ValueError("Random deletion curves have different x coordinates")
        matrix = np.array([[r["ctr_ps"] for r in subset] for subset in subsets])
        boot_std = np.array([
            [r["ctr_bootstrap_std_ps"] for r in subset] for subset in subsets
        ])
        avg = matrix.mean(axis=0)
        # Total descriptive spread: event-level bootstrap variance within each
        # ordering + variation of mean CTR between random orderings.
        order_var = matrix.var(axis=0, ddof=1) if len(subsets) > 1 else np.zeros_like(avg)
        band = np.sqrt(np.mean(boot_std ** 2, axis=0) + order_var)
        ax.plot(100 * positions, avg, color="0.30", linestyle="--",
                linewidth=1.7, label="Random deletion (mean)")
        ax.fill_between(100 * positions, avg - band, avg + band,
                        color="0.45", alpha=0.11, linewidth=0)

    ax.axhline(led_ctr, color="0.62", linestyle=":", linewidth=1.3,
               label="LED reference")
    if led_ctr_bootstrap_std > 0:
        ax.axhspan(led_ctr - led_ctr_bootstrap_std,
                   led_ctr + led_ctr_bootstrap_std,
                   facecolor="0.7", alpha=0.10, linewidth=0)
    ax.set_xlabel("Removed temporal regions (%)")
    ax.set_ylabel("CTR (ps)")
    ax.set_xlim(0, 100)
    ax.set_title(title, fontsize=11, fontweight="normal")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", color="0.85", linewidth=0.6)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def run_linear_shap_deletion(
    results=None, *, model_dir=None, models=None, ml_window=None, mode=None,
    cache_root=None, global_npz=None, global_root=None, output_root=None,
    chunk_size=256, max_events=None, step_groups=8, random_permutations=5,
    bootstrap_resamples=30, seed=42, overwrite=False, logger=None,
):
    """Evaluate mean-replacement deletion on the saved blind population."""
    if (global_npz is None) == (global_root is None):
        raise ValueError("Provide exactly one of global_npz and global_root")
    if global_npz is not None and model_dir is None:
        raise ValueError("--global-npz requires a single --model-dir run")
    if min(int(chunk_size), int(step_groups), int(random_permutations)) < 1:
        raise ValueError("chunk_size, step_groups and random_permutations must be positive")
    if max_events is not None and max_events < 5:
        raise ValueError("max_events must be >= 5")
    if isinstance(bootstrap_resamples, bool) or int(bootstrap_resamples) != bootstrap_resamples or bootstrap_resamples < 2:
        raise ValueError("bootstrap_resamples must be an integer >= 2")
    allowed = set(models) if models is not None else RIDGE_NAMES
    if not allowed or not allowed.issubset(RIDGE_NAMES):
        raise ValueError("Deletion requires direct_linear_ridge or shared_linear_ridge")
    runs, source = local._tshap_run_inputs(results, model_dir)
    selected_window = local._effective_tshap_window(ml_window, model_dir=model_dir)
    output_root = (
        Path(output_root).expanduser().resolve()
        if output_root is not None else Path.cwd().resolve() / "xai" / "linear_deletion"
    )
    if output_root.is_relative_to(source):
        raise ValueError("Deletion output-root must be outside the results directory")
    lookup = PreparedCacheLookup()
    outputs = []
    for run in runs:
        name = run["manifest"]["model"]
        if name not in allowed:
            continue
        directory = Path(run["directory"])
        cfg = resolve_run_config(directory)
        local._apply_tshap_cache(cfg, cache_root=cache_root)
        if (not local._matches_tshap_window(cfg, selected_window)
                or not local._matches_tshap_mode(cfg, mode)):
            continue
        npz = _npz_for_run(directory, global_npz=global_npz, global_root=global_root)
        destination = (
            output_root / batch_root(directory).parent.name
            / batch_root(directory).name
            / directory.relative_to(batch_root(directory)) / "linear_shap_deletion"
        )
        settings = {
            "global_npz": str(npz), "chunk_size": chunk_size, "max_events": max_events,
            "step_groups": step_groups, "random_permutations": random_permutations,
            "bootstrap_resamples": bootstrap_resamples, "seed": seed,
        }
        if (destination / "summary.json").is_file() and not overwrite:
            prior = json.loads((destination / "summary.json").read_text(encoding="utf-8"))
            if prior.get("settings") == settings and all(
                (destination / item).is_file() for item in (
                    "deletion_metrics.csv", "region_ranking.csv", "deletion_curve.pdf"
                )
            ):
                outputs.append(destination)
                if logger:
                    logger.info("Linear SHAP deletion | existing | %s", destination)
                continue
        spec = get_model(name)
        store = RunStore(directory)
        if not saved_model_complete(spec, store.model_dir):
            raise FileNotFoundError(f"Missing saved Ridge model: {store.model_dir}")
        fitted = load_fitted_model(spec, store.model_dir, run["best"]["parameters"], cfg)
        mask = np.asarray(fitted.sample_mask, bool)
        blind = lookup.find("blind", cfg, run["manifest"])
        family = mode_family(cfg["mode"])
        time_full = np.asarray(getattr(blind, f"{family}_time_ps"), np.float64)
        if mask.shape != time_full.shape:
            raise ValueError("Fitted sample mask differs from prepared test time grid")
        time_ps = time_full[mask]
        saved, target = _aligned_blind_predictions(store, blind)
        background, starts, importance, desc, explained_ids, shap_meta = _load_global_ranking(
            npz, model=name, mode=cfg["mode"], window=cfg["window_name"],
            time_ps=time_ps, n_events=len(saved),
        )
        if not np.isin(explained_ids, blind.event_index).all():
            raise ValueError("SHAP ranking event IDs do not belong to this test population")
        n = len(saved) if max_events is None else min(max_events, len(saved))
        replay_indices = np.arange(min(n, 16), dtype=np.int64)
        replay = predict_indices(spec, fitted, blind, cfg["mode"], replay_indices)
        if not np.allclose(replay, saved[replay_indices], atol=0.05, rtol=1e-5):
            raise ValueError("Saved Ridge model does not reproduce original test predictions")
        contributions, unbounded, mean_prediction = _reconstruct_contributions(
            blind, mode=cfg["mode"], mask=mask, background=background,
            starts=starts, regressor=fitted.artifact.regressor,
            shared=(name == "shared_linear_ridge"), saved_prediction=saved,
            chunk_size=chunk_size, limit=fitted.output_max_abs_ps,
            max_events=max_events,
        )
        seed_used = int(cfg["seed"])
        led_ctr = metric_values(target[:n], cfg["fit"], seed=seed_used)["ctr_ps"]
        original_ctr = metric_values(
            target[:n] - _clip(unbounded, fitted.output_max_abs_ps),
            cfg["fit"], seed=seed_used
        )["ctr_ps"]
        orders = [
            ("shap_descending", desc),
            ("shap_ascending", desc[::-1]),
        ]
        rng = np.random.default_rng(seed)
        orders += [
            (f"random_{i + 1:02d}", rng.permutation(len(desc)))
            for i in range(random_permutations)
        ]
        # Identical paired event resamples across every step and permutation.
        bootstrap_rng = np.random.default_rng(seed)
        bootstrap_indices = [
            bootstrap_draw_indices(n, bootstrap_rng)
            for _ in range(bootstrap_resamples)
        ]
        led_std = _bootstrap_ctr(
            target[:n], cfg["fit"], seed=seed_used,
            bootstrap_indices=bootstrap_indices,
        )
        rows = []
        for label, order in orders:
            rows.extend(_evaluate_order(
                unbounded, contributions, order, target=target[:n],
                fit=cfg["fit"], seed=seed_used, limit=fitted.output_max_abs_ps,
                ordering=label, step_groups=step_groups, original_ctr=original_ctr,
                bootstrap_indices=bootstrap_indices,
            ))
        last = next(r for r in rows if r["ordering"] == "shap_descending"
                    and r["n_removed"] == len(desc))
        if not np.allclose(
            _clip(unbounded - contributions.sum(axis=1), fitted.output_max_abs_ps),
            _clip(np.full(n, mean_prediction), fitted.output_max_abs_ps),
            atol=1e-6, rtol=1e-9,
        ):
            raise ValueError("Complete deletion does not reproduce the mean-input model")
        if logger:
            logger.info(
                "Linear deletion | %s %s | %d events | original CTR %.3f ps "
                "| full deletion %.3f ps | LED %.3f ps | %d regions",
                name, cfg["mode"], n, original_ctr, last["ctr_ps"], led_ctr,
                len(desc),
            )
        destination.mkdir(parents=True, exist_ok=True)
        with (destination / "deletion_metrics.csv").open(
            "w", newline="", encoding="utf-8"
        ) as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        dt = float(np.median(np.diff(time_ps))) if len(time_ps) > 1 else 0.0
        with (destination / "region_ranking.csv").open(
            "w", newline="", encoding="utf-8"
        ) as stream:
            writer = csv.writer(stream)
            writer.writerow([
                "rank", "detector", "start_ns", "end_ns",
                "importance_abs_ps_per_event",
            ])
            for rank, index in enumerate(desc, 1):
                group = int(index % len(starts))
                detector = "pair" if name == "shared_linear_ridge" else (
                    f"detector_{int(index // len(starts)) + 1}"
                )
                start = int(starts[group])
                end = int(starts[group + 1]) if group + 1 < len(starts) else len(time_ps)
                writer.writerow([
                    rank, detector, float(time_ps[start] / 1000.0),
                    float((time_ps[start] + (end - start) * dt) / 1000.0),
                    float(importance[index]),
                ])
        _plot_curves(
            destination / "deletion_curve.pdf", rows, led_ctr=led_ctr,
            led_ctr_bootstrap_std=led_std,
            title=f"{batch_root(directory).parent.name} | {family} | {name}",
        )
        atomic_json(destination / "summary.json", {
            "method": "global_linear_shap_mean_replacement_deletion",
            "model": name, "mode": cfg["mode"], "window": cfg["window_name"],
            "n_test_events": int(n), "n_regions": len(desc),
            "region_definition": (
                "both_detectors_jointly" if name == "shared_linear_ridge"
                else "each_detector_individually"
            ),
            "ranking_definition": "mean_positive_ps + abs(mean_negative_ps)",
            "ranking_population": shap_meta["population"],
            "ranking_n_events": int(shap_meta["n_explained_events"]),
            "evaluation_population": "saved_blind_test_events",
            "overlap_warning": "Saved global rankings were estimated on a subset of the test population.",
            "original_ctr_ps": original_ctr, "led_ctr_ps": led_ctr,
            "led_ctr_bootstrap_std_ps": led_std,
            "uncertainty": {
                "shap_curves": "event_bootstrap_std_1sigma",
                "random_curve": "sqrt(mean_bootstrap_variance_plus_between_permutation_variance)",
                "paired_event_resamples_across_all_steps": True,
                "n_resamples": bootstrap_resamples,
            },
            "full_deletion_ctr_ps": last["ctr_ps"],
            "background": "complete_prepared_development_mean_from_saved_npz",
            "prediction": "unbounded_Ridge_minus_signed_SHAP_then_pipeline_clipping",
            "no_refit": True, "metrics": "original_fit_ctr_ps_with_event_bootstrap_std_per_step",
            "settings": settings,
        })
        outputs.append(destination)
    if not outputs:
        raise ValueError("No matching saved Ridge runs found")
    return outputs
