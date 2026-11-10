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
    deletion_curve, deletion_sequence_curve, DELETION_SEQUENCES, validate_windows,
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


PLOT_VERSION = 5
PLOT_FILES = ("event_waveforms.png", "tshap_profiles.png", "deletion.png")
LEGACY_PLOT_FILES = ("tshap_deletion.png", "deletion_error.png",
                     "deletion_positive.png", "deletion_negative.png")
METHOD_LABELS = {"knn": "KNN", "ppca": "PPCA", "mean": "Development mean"}


def _window_plot_geometry(time_ps, starts, width):
    """Time intervals in ns from the actual prepared waveform grid."""
    times = np.asarray(time_ps, dtype=np.float64)
    starts = np.asarray(starts, dtype=np.int64)
    if (times.ndim != 1 or len(times) < 2 or
            not np.all(np.isfinite(times)) or
            np.any(np.diff(times) <= 0) or
            starts.ndim != 1 or not len(starts) or
            width < 1 or np.any(starts < 0) or
            np.any(starts + width > len(times))):
        raise ValueError("Invalid time grid or TSHAP windows for plotting")
    left = times[starts] / 1000.
    last = starts + width - 1
    # Use the following sample boundary when possible; extrapolate only for
    # a window ending on the final available sample.
    next_time = np.where(
        last + 1 < len(times),
        times[np.minimum(last + 1, len(times) - 1)],
        times[last] + (times[last] - times[last - 1]))
    right = next_time / 1000.
    return (left + right) / 2., right - left


def _plot_tshap_profiles(path, event_id, time_ps, starts, width, scores):
    """One signed bar profile per imputer; comparable symmetric y scales."""
    center_ns, width_ns = _window_plot_geometry(time_ps, starts, width)
    arrays = {method: np.asarray(scores[method], dtype=np.float64)
              for method in METHODS}
    if any(v.shape != center_ns.shape or not np.isfinite(v).all()
           for v in arrays.values()):
        raise ValueError("Invalid saved local TSHAP profiles")
    maximum = max(float(np.max(np.abs(v))) for v in arrays.values())
    limit = max(1., maximum * 1.08)

    fig, axes = plt.subplots(len(METHODS), 1, figsize=(11, 8.5),
                             sharex=True, sharey=True, constrained_layout=True)
    for ax, method in zip(np.atleast_1d(axes), METHODS):
        values = arrays[method]
        positive = np.where(values > 0, values, 0.)
        negative = np.where(values < 0, values, 0.)
        ax.bar(center_ns, positive, width=width_ns * .94,
               color="tab:blue", alpha=.85, label="Positive")
        ax.bar(center_ns, negative, width=width_ns * .94,
               color="tab:red", alpha=.80, label="Negative")
        ax.axhline(0, color="0.35", lw=.85)
        ax.set_ylim(-limit, limit)
        ax.set_ylabel(f"{METHOD_LABELS[method]}\nShapley (ps)")
        ax.grid(axis="y", alpha=.22)
        ax.set_axisbelow(True)
        ax.set_title(f"{METHOD_LABELS[method]} | +{np.count_nonzero(values > 0)} / "
                     f"-{np.count_nonzero(values < 0)} windows",
                     fontsize=10, loc="left")
    axes[0].legend(loc="upper right", ncol=2, fontsize=9)
    axes[-1].set_xlabel("Time relative to LED (ns)")
    fig.suptitle(f"Event {int(event_id)} | signed local TSHAP profiles")
    fig.savefig(path, dpi=170)
    plt.close(fig)


def _plot_deletion(path, event_id, deletion, *, original):
    """Positive→negative vs negative→positive across the full deletion range."""
    from matplotlib.ticker import PercentFormatter
    fig, axes = plt.subplots(2, 1, figsize=(10.5, 7.5), sharex=True,
                             constrained_layout=True)
    colors = {"knn": "tab:blue", "ppca": "tab:orange",
              "mean": "tab:green"}
    titles = {
        "positive_negative": "Positive → negative attribution deletion",
        "negative_positive": "Negative → positive attribution deletion",
    }
    for ax, sequence in zip(axes, DELETION_SEQUENCES):
        for method in METHODS:
            curve = deletion[method][sequence]
            fraction = np.asarray(curve["fraction_deleted"], dtype=float)
            change = np.asarray(curve["signed_prediction_change_ps"], dtype=float)
            if (fraction.ndim != 1 or fraction.shape != change.shape or
                    not np.isfinite(fraction).all() or
                    not np.isfinite(change).all() or
                    abs(fraction[0]) > 1e-12 or
                    abs(fraction[-1] - 1.) > 1e-12 or
                    np.any(np.diff(fraction) < -1e-12)):
                raise ValueError(f"Invalid complete deletion curve: {method}/{sequence}")
            ax.plot(fraction, change, lw=1.65, marker=".", markersize=4,
                    color=colors[method], label=METHOD_LABELS[method])
            first = int(curve["first_phase_count"])
            # Transition is between ranked positive and negative windows.
            if first > 0:
                transition = fraction[first]
                ax.axvline(transition, color=colors[method], lw=.9,
                           ls="--", alpha=.35)
        ax.axhline(0, color="0.4", lw=.85)
        ax.set_ylabel("Δ model prediction (ps)")
        ax.set_title(titles[sequence], loc="left")
        ax.grid(alpha=.23)
        ax.set_axisbelow(True)
        ax.set_xlim(0, 1)
        ax.legend(loc="best", fontsize=9)
    axes[-1].xaxis.set_major_formatter(PercentFormatter(xmax=1))
    axes[-1].set_xlabel("Fraction of model-retained input samples deleted")
    fig.suptitle(
        f"Event {int(event_id)} | complete progressive deletion\n"
        f"Reference: original model prediction {float(original):+.2f} ps")
    fig.savefig(path, dpi=170)
    plt.close(fig)


def _plot_event_waveforms(path, event_id, time_ps, waveform_mv):
    """Plot both ORIGINAL detector inputs in physical mV (no imputation)."""
    times = np.asarray(time_ps, dtype=np.float64) / 1000.
    waves = np.asarray(waveform_mv, dtype=np.float64)
    if (times.ndim != 1 or waves.shape != (2, len(times)) or
            not np.all(np.isfinite(waves))):
        raise ValueError("Invalid physical event waveform data")
    fig, axes = plt.subplots(2, 1, figsize=(11, 6), sharex=True,
                             constrained_layout=True)
    for detector, ax in enumerate(axes, start=1):
        ax.plot(times, waves[detector - 1], lw=1.4,
                color=("tab:blue" if detector == 1 else "tab:orange"))
        ax.set_ylabel(f"Detector {detector} (mV)")
        ax.grid(alpha=.24)
    axes[-1].set_xlabel("Time relative to LED (ns)")
    fig.suptitle(f"Event {int(event_id)} | original model input waveforms")
    fig.savefig(path, dpi=170)
    plt.close(fig)


def _normalize_saved_deletion(deletion, original):
    """Migrate old target-error metrics to model-reference metrics for plots.

    The original and every deletion prediction are already persisted, so
    this conversion does not require model inference or access to the target.
    """
    for method in METHODS:
        for direction in DELETION_SEQUENCES:
            curve = deletion[method][direction]
            prediction = np.asarray(curve["prediction_ps"], dtype=np.float64)
            signed = prediction - float(original)
            curve["signed_prediction_change_ps"] = signed.tolist()
            curve["absolute_prediction_deviation_ps"] = np.abs(signed).tolist()
            for obsolete in ("absolute_error_ps", "absolute_error_change_ps",
                             "squared_error_change_ps2"):
                curve.pop(obsolete, None)
    return deletion


def _write_event_plots(folder, event_id, time_ps, starts, width, scores,
                       deletion, *, original, waveform_mv=None):
    """Render from saved numerical arrays; never calls the ML model."""
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    if waveform_mv is not None:
        _plot_event_waveforms(folder / "event_waveforms.png", event_id,
                              time_ps, waveform_mv)
    _plot_tshap_profiles(folder / "tshap_profiles.png", event_id,
                         time_ps, starts, width, scores)
    _plot_deletion(folder / "deletion.png", event_id, deletion,
                   original=original)
    # Discard only superseded plotting artifacts, not raw numerical results.
    for name in LEGACY_PLOT_FILES:
        (folder / name).unlink(missing_ok=True)


def replot_saved(output_root):
    """Regenerate the three figures using existing JSON/NPZ without model inference."""
    root = Path(output_root).expanduser().resolve()
    info_file = root / "reference_details.npz"
    summary_file = root / "summary.json"
    if not info_file.is_file() or not summary_file.is_file():
        raise FileNotFoundError(
            "Replot needs existing reference_details.npz and summary.json")
    with np.load(info_file, allow_pickle=False) as info:
        starts = np.asarray(info["starts"], dtype=np.int64)
        time_ps = np.asarray(info["time_ps"], dtype=np.float64)
    summary = json.loads(summary_file.read_text(encoding="utf-8"))
    width = int(summary["window_samples"])
    validate_windows(np.ones(len(time_ps), dtype=bool), starts, width)
    folders = sorted(folder for folder in root.glob("event_*")
                     if folder.is_dir() and (folder / "results.json").is_file())
    if not folders:
        raise FileNotFoundError("No event_*/results.json files to replot")
    for folder in folders:
        payload = json.loads((folder / "results.json").read_text(encoding="utf-8"))
        scores = payload["signed_attributions_ps"]
        original = float(payload["original_prediction_ps"])
        deletion = payload["deletion"]
        if any(sequence not in deletion.get(method, {})
               for method in METHODS for sequence in DELETION_SEQUENCES):
            raise ValueError(
                "Saved curves use the old sign-only protocol. "
                "Run a new full-sequence experiment; --replot-only cannot "
                "calculate missing model predictions.")
        for method in METHODS:
            for sequence in DELETION_SEQUENCES:
                curve = deletion[method][sequence]
                for obsolete in ("absolute_error_ps", "absolute_error_change_ps",
                                 "squared_error_change_ps2",
                                 "absolute_prediction_deviation_ps"):
                    curve.pop(obsolete, None)
        waveform_mv = payload.get("waveform_mv")
        _write_event_plots(
            folder, int(payload["event_id"]), time_ps, starts, width,
            scores, deletion, original=original, waveform_mv=waveform_mv)
        payload["deletion"] = deletion
        (folder / "results.json").write_text(
            json.dumps(payload, indent=2), encoding="utf-8")
        if waveform_mv is None:
            print(f"REPLOT | event={payload['event_id']} | waveforms not saved "
                  "(old result); only TSHAP/deletion figures regenerated", flush=True)
        else:
            print(f"REPLOT | event={payload['event_id']} | {folder}", flush=True)
    csv_file = root / "deletion_curves.csv"
    if csv_file.is_file():
        import csv
        with csv_file.open("r", newline="", encoding="utf-8") as stream:
            records = list(csv.DictReader(stream))
        if records:
            for record in records:
                root_event = root / f"event_{int(record['event_id'])}"
                event_data = json.loads((root_event / "results.json").read_text(encoding="utf-8"))
                original = float(event_data["original_prediction_ps"])
                deviation = float(record["prediction_ps"]) - original
                record["signed_prediction_change_ps"] = deviation
                for obsolete in ("absolute_prediction_deviation_ps",
                                 "absolute_error_ps", "absolute_error_change_ps",
                                 "squared_error_change_ps2"):
                    record.pop(obsolete, None)
            columns = [column for column in records[0]]
            with csv_file.open("w", newline="", encoding="utf-8") as stream:
                writer = csv.DictWriter(stream, fieldnames=columns)
                writer.writeheader()
                writer.writerows(records)
    summary["plot_version"] = PLOT_VERSION
    summary["event_plot_files"] = list(PLOT_FILES)
    summary["deletion_metrics"] = "signed deviation from untouched model output"
    summary["deletion_sequences"] = list(DELETION_SEQUENCES)
    summary_file.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"REPLOT COMPLETE | events={len(folders)} | output={root}", flush=True)
    return root


def hydrate_saved_waveforms(model_dir, output_root, *, cache_root=None,
                            artifact_root=None):
    """Recover original waveform inputs for old runs without recalculating XAI.

    Existing event IDs are authoritative; we never rerun model-dependent
    case selection. The original frozen waveform transform restores mV.
    """
    if cache_root is not None and artifact_root is not None:
        raise ValueError("Specify --cache-root OR --artifact-root")
    root = Path(output_root).expanduser().resolve()
    info_file = root / "reference_details.npz"
    summary_file = root / "summary.json"
    if not info_file.is_file() or not summary_file.is_file():
        raise FileNotFoundError("No saved numerical XAI results at --output-root")
    source = base._resolve_model_run_directory(model_dir)
    run_data = read_run(source)
    config = resolve_run_config(source)
    base._apply_tshap_cache(config, cache_root=cache_root,
                            artifact_root=artifact_root)
    summary = json.loads(summary_file.read_text(encoding="utf-8"))
    if str(summary.get("final_fit")) != str(
            run_data["manifest"]["stage_fingerprints"]["final_fit"]):
        raise ValueError("Saved XAI model fingerprint differs from selected model run")
    lookup = PreparedCacheLookup()
    blind = lookup.find("blind", config, run_data["manifest"])
    family = mode_family(config["mode"])
    time_ps = np.asarray(getattr(blind, f"{family}_time_ps"), dtype=np.float64)
    transform = getattr(blind, f"{family}_transform")
    if transform is None:
        raise ValueError("Frozen physical waveform transform is unavailable")
    with np.load(info_file, allow_pickle=False) as info:
        saved_time = np.asarray(info["time_ps"], dtype=np.float64)
    if not np.array_equal(saved_time, time_ps):
        raise ValueError("Saved attribution time grid differs from prepared Blind grid")
    indices = {int(event_id): i for i, event_id in
               enumerate(np.asarray(blind.event_index, dtype=np.int64))}
    folders = sorted(folder for folder in root.glob("event_*")
                     if folder.is_dir() and (folder / "results.json").is_file())
    if not folders:
        raise FileNotFoundError("No saved event_*/results.json files found")
    for folder in folders:
        payload_path = folder / "results.json"
        payload = json.loads(payload_path.read_text(encoding="utf-8"))
        event_id = int(payload["event_id"])
        if event_id not in indices:
            raise ValueError(f"Saved event {event_id} is missing from Blind data")
        waveform = waveform_view(blind, config["mode"], [indices[event_id]]).materialize()
        physical = np.asarray(transform.inverse(waveform)[0], dtype=np.float64)
        if physical.shape != (2, len(time_ps)) or not np.isfinite(physical).all():
            raise ValueError(f"Invalid recovered physical waveform for {event_id}")
        payload["waveform_mv"] = physical.tolist()
        payload_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        _plot_event_waveforms(folder / "event_waveforms.png", event_id,
                              time_ps, physical)
        print(f"WAVEFORMS | event={event_id} | {folder / 'event_waveforms.png'}",
              flush=True)
    print(f"WAVEFORMS COMPLETE | events={len(folders)} | output={root}", flush=True)
    return root


def run(args):
    if getattr(args, "waveforms_only", False):
        if args.model_dir is None:
            raise ValueError("--waveforms-only requires --model-dir")
        return hydrate_saved_waveforms(args.model_dir, args.output_root,
                                       cache_root=args.cache_root,
                                       artifact_root=args.artifact_root)
    if getattr(args, "replot_only", False):
        return replot_saved(args.output_root)
    if args.model_dir is None:
        raise ValueError("--model-dir is required unless --replot-only is selected")
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
    physical_transform = getattr(blind, f"{family}_transform")
    if physical_transform is None:
        raise ValueError("Missing frozen waveform scaling for physical input plots")
    selected_mv = np.asarray(physical_transform.inverse(selected), dtype=np.float64)
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
                shared_mean = x.copy()
                shared_mean[:, retained] = bank.mean[:, retained]
                shared_endpoint = float(predict(shared_mean[None])[0])
                for method in METHODS:
                    shap = paired_shapley(
                        bank, x, starts, args.window_samples, predict, full,
                        method=method, knn_context=context)
                    scores[method] = shap["scores"]
                    curves[method] = {}
                    for direction in DELETION_SEQUENCES:
                        curve = deletion_sequence_curve(
                            bank, x, starts, args.window_samples, shap["scores"],
                            predict, full, method=method,
                            sequence=direction, knn_context=context,
                            expected_endpoint=shared_endpoint)
                        curves[method][direction] = curve
                        for step, output_ps in enumerate(curve["prediction_ps"]):
                            all_rows.append({
                                "event_id": int(event_id), "group": str(group),
                                "method": method, "direction": direction,
                                "step": step, "n_windows_total": len(starts),
                                "fraction_deleted": float(curve["fraction_deleted"][step]),
                                "phase": str(curve["stage"][step]),
                                "n_first_sign_windows": int(curve["first_phase_count"]),
                                "n_second_sign_windows": int(curve["second_phase_count"]),
                                "prediction_ps": float(output_ps),
                                "signed_prediction_change_ps": float(curve["signed_prediction_change_ps"][step]),
                                "absolute_prediction_deviation_ps": float(curve["absolute_prediction_deviation_ps"][step]),
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
                    "waveform_mv": selected_mv[index].tolist(),
                    "signed_attributions_ps": {k: v.tolist() for k, v in scores.items()},
                    "deletion_orders": {
                        method: {
                            sequence: curves[method][sequence]["removed_window_indices"].tolist()
                            for sequence in DELETION_SEQUENCES
                        } for method in METHODS
                    },
                    "mean_endpoint_prediction_ps": shared_endpoint,
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
                _write_event_plots(
                    event_root, event_id, time_ps, starts, args.window_samples,
                    scores, curves, original=full, waveform_mv=selected_mv[index])
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
        "deletion_sequences": list(DELETION_SEQUENCES),
        "imputation_definition": "point-conditional means; independent refit forbidden during deletion",
        "deletion_semantics": "cumulative missing union, fresh conditioning on original remaining samples",
        "deletion_metrics": "signed deviation from untouched model output",
        "endpoint": "shared Development mean after removing all model-retained coordinates",
        "plot_version": PLOT_VERSION,
        "event_plot_files": list(PLOT_FILES),
        "interpretation_warning": "Perturbation-dependent local rankings, not physical causality",
    }, indent=2), encoding="utf-8")
    print(f"COMPLETE | events={len(event_ids)} | output={output}", flush=True)
    return output


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model-dir", type=Path,
                   help="Saved model directory (not needed with --replot-only)")
    p.add_argument("--replot-only", action="store_true",
                   help="Regenerate figures from existing numerical results; no ML inference or imputer fit")
    p.add_argument("--waveforms-only", action="store_true",
                   help="Recover original event waveforms from prepared Blind cache, plot and save; no model inference or imputer fit")
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
