"""Local paired TSHAP + signed progressive deletion, three frozen imputers.

One run uses one existing trained model and a tiny blind-event case study.
Across models, pass the SAME --event-ids-file (generated on first run).
PPCA and the KNN reference bank are frozen Development-only; every deletion
step reconditions on original still-observed samples, never previous imputations.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from waveform_analysis.core.io import canonical_hash
from waveform_analysis.data.prepared_cache import PreparedCacheLookup
from waveform_analysis.data.results import read_run, resolve_run_config
from waveform_analysis.data.storage import RunStore
from waveform_analysis.data.view import mode_family, model_sample_mask, waveform_view
from waveform_analysis.engine.diagnostics import diagnostic_random_state
from waveform_analysis.engine.local_tshap_deletion import (
    METHODS, DIRECTIONS, FrozenImputationBank, paired_shapley,
    deletion_curve, validate_windows,
)
from waveform_analysis.engine.train import load_fitted_model, release_training_memory
from waveform_analysis.engine import tshap_explainer as base
from waveform_analysis.models import get_model


def _event_rows(blind, predictions, targets, args, output):
    ids = np.asarray(blind.event_index, dtype=np.int64)
    if len(np.unique(ids)) != len(ids):
        raise ValueError("Duplicate blind event IDs")
    if args.event_ids_file is None:
        selected, _, _ = base._select_local_events(
            predictions, targets, events_per_group=args.events_per_group)
        rows = np.array([v["index"] for v in selected], dtype=np.int64)
        groups = [v["group"] for v in selected]
    else:
        path = Path(args.event_ids_file).expanduser().resolve()
        payload = json.loads(path.read_text(encoding="utf-8"))
        chosen = np.asarray(payload["event_ids"], dtype=np.int64)
        if len(chosen) != len(np.unique(chosen)) or len(chosen) < 1:
            raise ValueError("Need distinct nonempty requested blind event IDs")
        lookup = {int(event_id): row for row, event_id in enumerate(ids)}
        if any(int(event_id) not in lookup for event_id in chosen):
            missing = [int(event_id) for event_id in chosen if int(event_id) not in lookup]
            raise ValueError(f"Requested blind event IDs not present: {missing}")
        rows = np.asarray([lookup[int(event_id)] for event_id in chosen], dtype=np.int64)
        groups = payload.get("groups") or ["shared"] * len(rows)
        if len(groups) != len(rows):
            raise ValueError("Event groups must match event IDs")
    event_ids = ids[rows]
    record = {"event_ids": event_ids.tolist(), "groups": list(groups),
              "selection_rule": ("reused explicit event IDs" if args.event_ids_file
                                 else "fixed good/bad/typical blind cases"),
              "dataset": str(blind.directory)}
    (output / "selected_events.json").write_text(
        json.dumps(record, indent=2), encoding="utf-8")
    return rows, groups, event_ids


def _development_cohort(dataset, count, seed):
    if count < 40:
        raise ValueError("At least 40 Development reference events required")
    if count > dataset.n_events:
        raise ValueError("Requested Development references exceed available population")
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(dataset.n_events, size=count, replace=False))


def _cohort_cache(path, reference, identity, args, retained):
    """Reuse a fitted PPCA only when the EXACT reference identity matches."""
    sha = hashlib.sha256(np.ascontiguousarray(reference).view(np.uint8)).hexdigest()
    key = canonical_hash({
        "version": 1, "reference_sha256": sha,
        "shape": list(reference.shape),
        "retained": np.flatnonzero(retained).tolist(),
        "ppca_components": args.ppca_components,
        "knn_k": args.knn_k,
        "knn_weighting": args.knn_weighting,
        "knn_scale_floor": args.knn_scale_floor,
        "dataset_identity": identity,
    })
    cache = path / f"ppca_{key}.npz"
    state = None
    if cache.is_file():
        with np.load(cache, allow_pickle=False) as saved:
            state = (saved["mean"], saved["W"], float(saved["sigma2"]))
        print(f"PPCA CACHE HIT | {cache}", flush=True)
    bank = FrozenImputationBank(
        reference, retained, knn_k=args.knn_k,
        weighting=args.knn_weighting, knn_block=args.knn_reference_block,
        scale_floor=args.knn_scale_floor,
        ppca_components=args.ppca_components, ppca_state=state)
    if state is None:
        path.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(cache, mean=bank.ppca.mean, W=bank.ppca.W,
                            sigma2=bank.ppca.sigma2,
                            reference_sha256=sha, key=key)
        print(f"PPCA CACHE FIT | {cache}", flush=True)
    return bank, cache


def _plot_event(path, event_id, time_ps, starts, width, scores, deletion,
                *, original, target):
    fig, axes = plt.subplots(2, 1, figsize=(11, 7), constrained_layout=True)
    center = (time_ps[starts] + time_ps[np.minimum(starts + width - 1,
                                                    len(time_ps) - 1)]) / 2000.
    for method in METHODS:
        axes[0].plot(center, scores[method], marker=".", markersize=4,
                     label=method)
    axes[0].axhline(0, color="0.4", linewidth=.8)
    axes[0].set_ylabel("Paired-window signed Shapley (ps)")
    axes[0].set_xlabel("Time relative to LED (ns)")
    axes[0].legend()
    # The second panel includes BOTH signs for each imputer.
    for method in METHODS:
        for direction in DIRECTIONS:
            result = deletion[method][direction]
            ax = axes[1]
            ax.plot(result["step"], result["signed_prediction_change_ps"],
                    marker=".", linewidth=1.2,
                    label=f"{method} / {direction}")
    axes[1].axhline(0, color="0.4", linewidth=.8)
    axes[1].set_xlabel("Deleted windows (within each sign)")
    axes[1].set_ylabel("Signed change of model output (ps)")
    axes[1].set_title(f"Original prediction={original:+.2f} ps; target={target:+.2f} ps")
    axes[1].legend(ncol=2, fontsize=8)
    fig.suptitle(f"Event {int(event_id)} | local TSHAP and progressive deletion")
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _plot_error(path, event_id, deletion):
    fig, ax = plt.subplots(figsize=(10, 4.5), constrained_layout=True)
    for method in METHODS:
        for direction in DIRECTIONS:
            result = deletion[method][direction]
            ax.plot(result["step"], result["absolute_error_change_ps"],
                    marker=".", linewidth=1.2, label=f"{method} / {direction}")
    ax.axhline(0, color="0.4", linewidth=.8)
    ax.set(xlabel="Deleted windows (within each sign)",
           ylabel="Change in absolute residual error (ps)",
           title=f"Event {event_id} | deviation from calibrated target")
    ax.legend(ncol=2, fontsize=8)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def run(args):
    if args.cache_root and args.artifact_root:
        raise ValueError("Use --cache-root OR --artifact-root")
    if args.window_samples < 1 or args.stride_samples < args.window_samples:
        raise ValueError("Require positive width and stride >= width: deletion windows must not overlap")
    if args.reference_events <= args.ppca_components + 1:
        raise ValueError("Reference cohort is too small for the specified PPCA rank")
    output = args.output_root.expanduser().resolve()
    source = base._resolve_model_run_directory(args.model_dir)
    if output == source or output.is_relative_to(source):
        raise ValueError("Output must be outside the model run")
    output.mkdir(parents=True, exist_ok=True)
    run_data = read_run(source)
    config = resolve_run_config(source)
    base._apply_tshap_cache(config, cache_root=args.cache_root,
                            artifact_root=args.artifact_root)
    manifest = run_data["manifest"]
    lookup = PreparedCacheLookup()
    development = lookup.find("development", config, manifest)
    blind = lookup.find("blind", config, manifest)
    family = mode_family(config["mode"])
    time_ps = np.asarray(getattr(blind, f"{family}_time_ps"), dtype=float)
    retained = model_sample_mask(blind, config["mode"])
    if not np.array_equal(time_ps, np.asarray(getattr(development, f"{family}_time_ps"))):
        raise ValueError("Development and Blind temporal grids differ")
    starts, _, _ = base._valid_window_starts(
        len(time_ps), args.window_samples, args.stride_samples, retained)
    retained, starts = validate_windows(retained, starts, args.window_samples)
    print(f"MODEL | {manifest['model']} | {config['mode']} | "
          f"windows={len(starts)} | retained_samples={retained.sum()}", flush=True)
    store = RunStore(source, artifact_root=config["preprocessing"]["artifact_root"])
    predictions, targets = base._load_materialized_local_predictions(
        store, blind, manifest, config["mode"])
    rows, groups, event_ids = _event_rows(blind, predictions, targets, args, output)
    selected = waveform_view(blind, config["mode"], rows).materialize()
    if not np.isfinite(selected).all():
        raise ValueError("Selected blind waveforms are nonfinite")
    ref_rows = _development_cohort(development, args.reference_events, args.seed)
    reference = waveform_view(development, config["mode"], ref_rows).materialize()
    cache_dir = args.imputer_cache_root
    if cache_dir is None:
        cache_dir = output / "frozen_imputers"
    cache_dir = Path(cache_dir).expanduser().resolve()
    # Identity includes the prepared Development store and selected event IDs.
    bank, ppca_cache = _cohort_cache(
        cache_dir, reference, str(development.directory), args, retained)
    spec = get_model(manifest["model"])
    model_dir = base._resolve_tshap_model_directory(
        spec, store, source, explicit_model_dir=True)
    settings = {
        "model_dir": str(source), "mode": config["mode"],
        "final_fit": manifest["stage_fingerprints"]["final_fit"],
        "window_samples": args.window_samples,
        "stride_samples": args.stride_samples,
        "reference_events": args.reference_events,
        "knn_k": args.knn_k,
        "knn_weighting": args.knn_weighting,
        "ppca_components": args.ppca_components,
        "seed": args.seed,
        "event_ids": event_ids.tolist(),
        "ppca_cache": str(ppca_cache),
    }
    np.savez_compressed(output / "reference_details.npz",
                        reference_event_ids=development.event_index[ref_rows],
                        starts=starts, retained_mask=retained, time_ps=time_ps)
    all_rows = []
    with diagnostic_random_state():
        fitted = load_fitted_model(spec, model_dir, run_data["best"]["parameters"], config)
        try:
            if fitted.sample_mask is not None and not np.array_equal(
                    fitted.sample_mask, retained):
                raise ValueError("Saved model mask differs from prepared control-derived mask")
            predict = lambda x: base._predict_bounded(
                spec, fitted, x, args.predict_batch_size)
            replay = predict(selected)
            if not np.allclose(replay, predictions[rows],
                               atol=base.LOCAL_PREDICTION_REPLAY_ATOL_PS,
                               rtol=base.LOCAL_PREDICTION_REPLAY_RTOL):
                raise ValueError(
                    "Saved-vs-replayed model prediction discrepancy: "
                    f"maximum={np.max(np.abs(replay - predictions[rows])):.5f} ps")
            for index, (event_id, group, x, full, target) in enumerate(zip(
                    event_ids, groups, selected, replay, targets[rows])):
                print(f"EVENT | {index + 1}/{len(rows)} | ID={event_id} | "
                      f"group={group}", flush=True)
                context = bank.knn_context(x, starts, args.window_samples)
                scores = {}
                curves = {}
                for method in METHODS:
                    shap = paired_shapley(
                        bank, x, starts, args.window_samples, predict, full,
                        method=method, knn_context=context)
                    scores[method] = shap["scores"]
                    curves[method] = {}
                    for direction in DIRECTIONS:
                        curve = deletion_curve(
                            bank, x, starts, args.window_samples, shap["scores"],
                            predict, full, target, method=method,
                            direction=direction, knn_context=context)
                        curves[method][direction] = curve
                        for step, output_ps in enumerate(curve["prediction_ps"]):
                            all_rows.append({
                                "event_id": int(event_id), "group": str(group),
                                "method": method, "direction": direction,
                                "step": step, "n_windows_total": len(starts),
                                "n_deleted_for_sign": len(curve["removed_window_indices"]),
                                "prediction_ps": float(output_ps),
                                "signed_prediction_change_ps": float(curve["signed_prediction_change_ps"][step]),
                                "absolute_error_change_ps": float(curve["absolute_error_change_ps"][step]),
                            })
                    print(f"  METHOD | {method} | signed windows: "
                          f"+{(shap['scores'] > 0).sum()} / "
                          f"-{(shap['scores'] < 0).sum()}", flush=True)
                event_root = output / f"event_{int(event_id)}"
                event_root.mkdir(parents=True, exist_ok=True)
                payload = {
                    "event_id": int(event_id), "group": str(group),
                    "original_prediction_ps": float(full),
                    "target_ps": float(target),
                    "signed_attributions_ps": {k: v.tolist() for k, v in scores.items()},
                    "deletion": {
                        method: {
                            direction: {
                                key: np.asarray(value).tolist()
                                for key, value in curve.items()
                            } for direction, curve in curves[method].items()
                        } for method in METHODS
                    },
                }
                (event_root / "results.json").write_text(
                    json.dumps(payload, indent=2), encoding="utf-8")
                _plot_event(event_root / "tshap_deletion.png",
                            event_id, time_ps, starts, args.window_samples,
                            scores, curves, original=full, target=target)
                _plot_error(event_root / "deletion_error.png", event_id, curves)
        finally:
            del fitted
            release_training_memory()
    import csv
    if all_rows:
        with (output / "deletion_curves.csv").open(
                "w", newline="", encoding="utf-8") as out:
            writer = csv.DictWriter(out, fieldnames=list(all_rows[0]))
            writer.writeheader()
            writer.writerows(all_rows)
    (output / "summary.json").write_text(json.dumps({
        **settings, "events_explained": len(event_ids),
        "methods": list(METHODS),
        "deletion_directions": list(DIRECTIONS),
        "imputation_definition": "point-conditional means; independent refit forbidden during deletion",
        "deletion_semantics": "cumulative missing union, fresh conditioning on original remaining samples",
        "target_metric": "absolute calibrated residual error and signed model output change",
        "interpretation_warning": "Perturbation-dependent local rankings, not physical causality",
    }, indent=2), encoding="utf-8")
    print(f"COMPLETE | events={len(event_ids)} | output={output}", flush=True)
    return output


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model-dir", required=True, type=Path)
    p.add_argument("--cache-root", type=Path)
    p.add_argument("--artifact-root", type=Path)
    p.add_argument("--output-root", required=True, type=Path)
    p.add_argument("--imputer-cache-root", type=Path)
    p.add_argument("--event-ids-file", type=Path,
                   help="Reuse selected_events.json from first run across models")
    p.add_argument("--events-per-group", type=int, default=1)
    p.add_argument("--window-samples", type=int, default=80)
    p.add_argument("--stride-samples", type=int, default=80)
    p.add_argument("--reference-events", type=int, default=1024)
    p.add_argument("--knn-k", type=int, default=20)
    p.add_argument("--knn-weighting", choices=("distance", "uniform"), default="distance")
    p.add_argument("--knn-reference-block", type=int, default=256)
    p.add_argument("--knn-scale-floor", type=float, default=.05)
    p.add_argument("--ppca-components", type=int, default=32)
    p.add_argument("--predict-batch-size", type=int, default=64)
    p.add_argument("--seed", type=int, default=42)
    return p


if __name__ == "__main__":
    run(parser().parse_args())
