"""Saved-prediction Ridge surrogate using the existing conditional heatmaps.

Development heatmap-reference, calibration, and validation events are disjoint.
Blind teacher predictions are used for final fidelity evaluation only. Never
load/fit/invoke the original ML model. Blind physical labels are used solely
for the same diagnostic event selection as recent local TSHAP.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from waveform_analysis.data.prepared_cache import PreparedCacheLookup
from waveform_analysis.data.results import read_run, resolve_run_config
from waveform_analysis.data.storage import RunStore
from waveform_analysis.data.view import mode_family, model_sample_mask
from waveform_analysis.engine import tshap_explainer as tshap
from waveform_analysis.engine.conditional_heatmap_surrogate import (
    calibrate_ridge, fidelity_metrics, heatmap_pointwise_features,
    load_conditional_map, split_development, surrogate_components,
)
from waveform_analysis.scripts.evaluate_conditional_knn import _load_inputs, _logger
from waveform_analysis.scripts.plot_conditional_target_heatmaps import baseline_indices


def _select_events(blind, predictions, targets, args):
    if args.event_ids_file:
        data = json.loads(args.event_ids_file.read_text(encoding="utf-8"))
        requested = [int(v) for v in data["event_ids"]]
        if len(requested) != len(set(requested)) or not requested:
            raise ValueError("Invalid explicit event IDs")
        lookup = {int(v): i for i, v in enumerate(blind.event_index)}
        if any(v not in lookup for v in requested):
            raise ValueError("Requested blind event ID unavailable")
        rows = np.array([lookup[v] for v in requested], dtype=int)
        groups = data.get("groups", ["shared"] * len(rows))
        if len(groups) != len(rows):
            raise ValueError("Mismatch between groups and requested event IDs")
    else:
        cases, _, _ = tshap._select_local_events(
            predictions, targets, events_per_group=args.events_per_group)
        rows = np.array([v["index"] for v in cases], dtype=int)
        groups = [v["group"] for v in cases]
    return rows, groups


def _plot_waveforms(path, time, signal, selected_time, values, good, *,
                    event_id, group, teacher, surrogate, center, radius, title):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection
    from matplotlib.colors import TwoSlopeNorm
    from matplotlib.cm import ScalarMappable

    norm = TwoSlopeNorm(vmin=-radius, vcenter=0, vmax=radius)
    cmap = "coolwarm"
    fig, axes = plt.subplots(2, 1, sharex=True, figsize=(12, 7),
                             constrained_layout=True)
    for d, ax in enumerate(axes):
        ax.plot(time, signal[d], color="0.75", lw=1)
        points = np.column_stack(
            [selected_time, np.interp(selected_time, time, signal[d])])
        segments = np.stack([points[:-1], points[1:]], axis=1)
        kept = good[d, :-1] & good[d, 1:]
        line = LineCollection(segments[kept], cmap=cmap, norm=norm,
                              linewidth=2.5, capstyle="round")
        line.set_array(.5 * (values[d, :-1] + values[d, 1:])[kept])
        ax.add_collection(line)
        ax.set_xlim(time[0], time[-1])
        ax.set_ylabel(f"Detector {d + 1} (mV)")
        ax.grid(alpha=.2)
        ax.text(.98, .90, f"support: {good[d].mean():.1%}",
                transform=ax.transAxes, va="top", ha="right")
    axes[-1].set_xlabel("Time relative to LED (ns)")
    fig.colorbar(ScalarMappable(cmap=cmap, norm=norm),
                 ax=axes, label=f"{title} (ps)", fraction=.025, pad=.02)
    fig.suptitle(f"Event {event_id} | {group} | teacher={teacher:+.2f} ps | "
                 f"Ridge={surrogate:+.2f} ps\n{title} | heatmap mean={center:+.2f} ps")
    fig.savefig(path, dpi=170)
    plt.close(fig)


def _plot_fidelity(path, original, reconstructed, label):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), constrained_layout=True)
    axes[0].scatter(original, reconstructed, s=7, alpha=.4)
    lo = min(np.min(original), np.min(reconstructed))
    hi = max(np.max(original), np.max(reconstructed))
    axes[0].plot([lo, hi], [lo, hi], "--", color="0.3")
    axes[0].set(xlabel="Teacher prediction (ps)", ylabel="Ridge prediction (ps)")
    axes[1].scatter(original, reconstructed - original, s=7, alpha=.4)
    axes[1].axhline(0, linestyle="--", color="0.3")
    axes[1].set(xlabel="Teacher prediction (ps)", ylabel="Ridge minus teacher (ps)")
    for ax in axes:
        ax.grid(alpha=.2)
    fig.suptitle(label)
    fig.savefig(path, dpi=170)
    plt.close(fig)


def run(args):
    if args.cache_root is not None and args.artifact_root is not None:
        raise ValueError("Choose --cache-root OR --artifact-root")
    run_dir = tshap._resolve_model_run_directory(args.model_dir)
    root = args.output_root.expanduser().resolve()
    if root == run_dir or run_dir in root.parents:
        raise ValueError("Output must be outside the trained model directory")
    root.mkdir(parents=True, exist_ok=True)
    manifest = read_run(run_dir)["manifest"]
    config = resolve_run_config(run_dir)
    tshap._apply_tshap_cache(config, cache_root=args.cache_root,
                            artifact_root=args.artifact_root)
    development = PreparedCacheLookup().find("development", config, manifest)
    blind = PreparedCacheLookup().find("blind", config, manifest)
    family = mode_family(config["mode"])
    dev_time = np.asarray(getattr(development, f"{family}_time_ps"), np.float64)
    blind_time = np.asarray(getattr(blind, f"{family}_time_ps"), np.float64)
    if not np.array_equal(dev_time, blind_time):
        raise ValueError("Development/Blind prepared time grids disagree")
    dev_waves = getattr(development, f"{family}_windows")
    blind_waves = getattr(blind, f"{family}_windows")
    if dev_waves.shape[1:] != blind_waves.shape[1:]:
        raise ValueError("Development/Blind sample grids disagree")
    retain = model_sample_mask(development, config["mode"])
    if not np.array_equal(retain, model_sample_mask(blind, config["mode"])):
        raise ValueError("Development/Blind model masks disagree")

    dev_args = argparse.Namespace(
        model_dir=run_dir, cache_root=args.cache_root,
        artifact_root=args.artifact_root)
    _, dev_pred, _, _ = _load_inputs(dev_args, _logger(root))
    store = RunStore(run_dir, artifact_root=config["preprocessing"]["artifact_root"])
    blind_pred, blind_targets = tshap._load_materialized_local_predictions(
        store, blind, manifest, config["mode"])
    if len(dev_pred) != development.n_events or len(blind_pred) != blind.n_events:
        raise ValueError("Prediction/data population mismatch")
    reference, calibration, validation = split_development(
        len(dev_pred), seed=args.seed,
        calibration_count=args.calibration_count,
        validation_count=args.validation_count)
    root_meta = json.loads((args.heatmap_root / "summary.json").read_text(encoding="utf-8"))
    provenance = root_meta["provenance"]
    if (root_meta.get("comparison_provenance") is not None or
            provenance["final_fit_fingerprint"] != manifest["stage_fingerprints"]["final_fit"] or
            provenance["mode"] != config["mode"] or
            provenance["model"] != manifest["model"] or
            int(root_meta["seed"]) != args.seed or
            int(root_meta["events"]) != len(reference) or
            Path(provenance["dataset"]).resolve() != Path(development.directory).resolve()):
        raise ValueError("Heatmap model/split provenance mismatch")
    rows, groups = _select_events(blind, blind_pred, blind_targets, args)
    selected = {"event_ids": np.asarray(blind.event_index[rows], int).tolist(),
                "groups": list(groups)}
    (root / "selected_events.json").write_text(json.dumps(selected, indent=2),
                                               encoding="utf-8")
    (root / "development_splits.json").write_text(json.dumps({
        "reference_event_ids": np.asarray(development.event_index[reference], int).tolist(),
        "calibration_event_ids": np.asarray(development.event_index[calibration], int).tolist(),
        "validation_event_ids": np.asarray(development.event_index[validation], int).tolist(),
        "seed": args.seed,
    }), encoding="utf-8")
    transform = getattr(blind, f"{family}_transform")
    if transform is None:
        raise ValueError("Missing saved physical waveform transform")
    # Required views share the same reference event population and native grid.
    # First-difference data lives in original/ (second feature kind).
    feature_specs = (
        ("original", "waveform"),
        ("original", "first_difference"),
        ("amplitude_normalized", "waveform"),
    )
    feature_arrays = {}
    common_maps = {}
    for representation, kind in feature_specs:
        key = ("first_difference" if kind == "first_difference"
               else representation)
        heatmap = load_conditional_map(
            args.heatmap_root / representation / "conditional_target_heatmaps.npz",
            feature_name=kind)
        rep_meta = json.loads(
            (args.heatmap_root / representation / "summary.json").read_text(encoding="utf-8"))
        expected_baseline = [args.baseline_start_ns, args.baseline_end_ns]
        actual_baseline = rep_meta.get("baseline_window_ns")
        expected_prediction_mean = float(dev_pred[reference].mean())
        checks = {
            "representation": (heatmap.representation == representation,
                               heatmap.representation, representation),
            "root membership": (representation in root_meta["representations"],
                                root_meta["representations"], representation),
            "reference event count": (int(rep_meta["events"]) == len(reference),
                                      int(rep_meta["events"]), len(reference)),
            "min_count": (int(rep_meta["min_count"]) == args.min_count,
                          int(rep_meta["min_count"]), args.min_count),
            "baseline window": (representation == "original" or
                                actual_baseline == expected_baseline,
                                actual_baseline, expected_baseline),
            "teacher prediction mean": (np.isclose(
                heatmap.baseline_ps, expected_prediction_mean, atol=1e-5, rtol=0),
                heatmap.baseline_ps, expected_prediction_mean),
        }
        mismatches = [
            f"{field}: stored={actual!r}, expected={expected!r}"
            for field, (ok, actual, expected) in checks.items() if not ok
        ]
        if mismatches:
            raise ValueError(
                f"Heatmap {representation}/{kind} metadata mismatch:\n" +
                "\n".join(mismatches))
        base = (None if representation == "original" else baseline_indices(
            dev_time, args.baseline_start_ns, args.baseline_end_ns))
        common = dict(full_time_ps=dev_time, baseline_positions=base,
                      retained=retain, min_count=args.min_count,
                      chunk_size=args.chunk_size)
        a_cal, valid_cal = heatmap_pointwise_features(
            dev_waves, calibration, heatmap, **common)
        a_val, valid_val = heatmap_pointwise_features(
            dev_waves, validation, heatmap, **common)
        feature_arrays[key] = {
            "calibration": a_cal, "validation": a_val,
            "calibration_valid": valid_cal, "validation_valid": valid_val,
        }
        common_maps[key] = heatmap
        print(f"FEATURE | {key} | calibration={a_cal.shape}", flush=True)

    configurations = {
        "original": ("original",),
        "original_first_difference": ("original", "first_difference"),
        "original_amplitude_normalized": ("original", "amplitude_normalized"),
        "all_three": ("original", "first_difference", "amplitude_normalized"),
    }
    reference_mean = common_maps["original"].baseline_ps
    for name, heatmap in common_maps.items():
        if (not np.array_equal(heatmap.positions, common_maps["original"].positions)
                or not np.array_equal(heatmap.time_ps, common_maps["original"].time_ps)
                or not np.isclose(heatmap.baseline_ps, reference_mean, rtol=0, atol=1e-8)):
            raise ValueError(f"Heatmap coordinates/baseline mismatch: {name}")
    results = {}
    for config_name, feature_names in configurations.items():
        # Shape [event, representation, detector, time]. All names share time.
        def stacked(part):
            return np.stack([feature_arrays[name][part] for name in feature_names], axis=1)
        a_cal, a_val = stacked("calibration"), stacked("validation")
        v_cal, v_val = stacked("calibration_valid"), stacked("validation_valid")
        ridge, cv = calibrate_ridge(
            a_cal.reshape(len(a_cal), -1), dev_pred[calibration],
            alphas=args.alphas, folds=args.cv_folds, seed=args.seed)
        c_cal, y_cal = surrogate_components(ridge, a_cal)
        c_val, y_val = surrogate_components(ridge, a_val)

        # Stream Blind in bounded-size chunks. Avoid materializing the full
        # [events, representations, detectors, time] float64 tensor.
        y_blind = np.empty(len(blind_pred), dtype=np.float64)
        blind_support = np.empty(len(blind_pred), dtype=np.float32)
        valid_sum = np.zeros(len(feature_names), dtype=np.int64)
        total_valid_cells = 0
        for left in range(0, len(blind_pred), args.chunk_size):
            right = min(left + args.chunk_size, len(blind_pred))
            indices = np.arange(left, right, dtype=np.int64)
            parts, masks = [], []
            for feature in feature_names:
                hm = common_maps[feature]
                baseline = (None if hm.representation == "original"
                            else baseline_indices(dev_time, args.baseline_start_ns,
                                                  args.baseline_end_ns))
                a_part, v_part = heatmap_pointwise_features(
                    blind_waves, indices, hm,
                    full_time_ps=dev_time, baseline_positions=baseline,
                    retained=retain, min_count=args.min_count,
                    chunk_size=args.chunk_size)
                parts.append(a_part)
                masks.append(v_part)
            a_chunk = np.stack(parts, axis=1)
            v_chunk = np.stack(masks, axis=1)
            _, predicted = surrogate_components(ridge, a_chunk)
            y_blind[left:right] = predicted
            blind_support[left:right] = v_chunk.mean(axis=(1, 2, 3))
            valid_sum += v_chunk.sum(axis=(0, 2, 3))
            total_valid_cells += v_chunk.size
        coverage_per_rep = {
            feature: float(valid_sum[j] / (len(blind_pred) * 2 * a_cal.shape[-1]))
            for j, feature in enumerate(feature_names)}
        metrics = {
            "features": list(feature_names),
            "regularization": cv,
            "fidelity_to_saved_model": {
                "calibration": fidelity_metrics(y_cal, dev_pred[calibration], reference_mean),
                "validation": fidelity_metrics(y_val, dev_pred[validation], reference_mean),
                "blind": fidelity_metrics(y_blind, blind_pred, reference_mean),
            },
            "coverage": {
                "calibration": float(v_cal.mean()),
                "validation": float(v_val.mean()),
                "blind": float(valid_sum.sum() / total_valid_cells),
                "blind_per_representation": coverage_per_rep,
            },
            "missing_cell_policy": "zero deviation from reference mean, mask retained",
            "interpretation": "association-based additive surrogate, not causal attribution",
        }
        destination = root / config_name
        destination.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            destination / "surrogate.npz",
            feature_names=np.asarray(feature_names),
            coefficients=ridge.coef_.reshape(len(feature_names), 2, -1),
            intercept_ps=np.array(ridge.intercept_),
            time_ps=common_maps["original"].time_ps,
            positions=common_maps["original"].positions,
            blind_event_ids=np.asarray(blind.event_index, dtype=np.int64),
            blind_teacher_prediction_ps=blind_pred,
            blind_surrogate_prediction_ps=y_blind,
            blind_support_fraction=blind_support)
        (destination / "metrics.json").write_text(
            json.dumps(metrics, indent=2), encoding="utf-8")
        _plot_fidelity(destination / "blind_fidelity.png",
                       blind_pred, y_blind, f"Blind reconstruction | {config_name}")
        _plot_fidelity(destination / "validation_fidelity.png",
                       dev_pred[validation], y_val, f"Development validation | {config_name}")
        # Diagnostic: raw heatmap color per representation, and calibrated
        # additive contributions aggregated across representations.
        shared_time = common_maps["original"].time_ps / 1000.
        # Recompute only selected diagnostic events (small bounded subset).
        local_a = []
        local_valid = []
        for feature in feature_names:
            hm = common_maps[feature]
            baseline = (None if hm.representation == "original"
                        else baseline_indices(dev_time, args.baseline_start_ns,
                                              args.baseline_end_ns))
            a_part, v_part = heatmap_pointwise_features(
                blind_waves, rows, hm, full_time_ps=dev_time,
                baseline_positions=baseline, retained=retain,
                min_count=args.min_count, chunk_size=args.chunk_size)
            local_a.append(a_part)
            local_valid.append(v_part)
        local_a = np.stack(local_a, axis=1)
        local_valid = np.stack(local_valid, axis=1)
        local_c, local_pred = surrogate_components(ridge, local_a)
        if not np.allclose(local_pred, y_blind[rows], atol=1e-7, rtol=1e-10):
            raise AssertionError("Local recomputation disagrees with streamed Blind")
        calibrated_radius = float(max(np.nanpercentile(
            np.abs(local_c.sum(axis=1)), 98), 1.))
        for local_index, (row, group) in enumerate(zip(rows, groups)):
            event_id = int(blind.event_index[row])
            folder = destination / group / f"event_{event_id}"
            folder.mkdir(parents=True, exist_ok=True)
            physical = transform.inverse(
                np.asarray(blind_waves[row:row + 1], dtype=np.float32))[0]
            plot_args = dict(
                time=dev_time / 1000., signal=physical,
                selected_time=shared_time,
                event_id=event_id, group=group,
                teacher=float(blind_pred[row]), surrogate=float(y_blind[row]),
                center=reference_mean)
            for j, feature in enumerate(feature_names):
                hm = common_maps[feature]
                radius = float(max(np.nanpercentile(
                    np.abs(hm.target_mean - reference_mean), 98), 1.))
                _plot_waveforms(
                    folder / f"heatmap_{feature}_colored_waveforms.png",
                    values=local_a[local_index, j], good=local_valid[local_index, j],
                    radius=radius, title=f"Heatmap {feature} deviation", **plot_args)
            combined_contributions = local_c[local_index].sum(axis=0)
            combined_valid = np.any(local_valid[local_index], axis=0)
            radius = calibrated_radius
            _plot_waveforms(
                folder / "ridge_combined_colored_waveforms.png",
                values=combined_contributions, good=combined_valid,
                radius=radius, title="Combined calibrated Ridge contributions",
                **plot_args)
            np.savez_compressed(
                folder / "local_profiles.npz",
                feature_names=np.asarray(feature_names),
                time_ps=common_maps["original"].time_ps,
                heatmap_deviation_ps=local_a[local_index],
                ridge_contribution_ps=local_c[local_index],
                combined_ridge_contribution_ps=combined_contributions,
                valid=local_valid[local_index],
                teacher_prediction_ps=blind_pred[row],
                ridge_prediction_ps=y_blind[row])
        results[config_name] = metrics
        print(f"COMPLETE | {config_name} | blind RMSE "
              f"{metrics['fidelity_to_saved_model']['blind']['rmse_ps']:.3f} ps | "
              f"coverage {metrics['coverage']['blind']:.1%}", flush=True)
    report = {
        "model_dir": str(run_dir),
        "heatmap_root": str(args.heatmap_root.resolve()),
        "final_fit_fingerprint": manifest["stage_fingerprints"]["final_fit"],
        "reference_events": len(reference), "calibration_events": len(calibration),
        "validation_events": len(validation), "blind_events": len(blind_pred),
        "selected_events": selected,
        "fidelity_target": "saved teacher prediction, not calibrated physical target",
        "results": results,
    }
    (root / "summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model-dir", type=Path, required=True)
    p.add_argument("--heatmap-root", type=Path, required=True)
    p.add_argument("--output-root", type=Path, required=True)
    p.add_argument("--cache-root", type=Path)
    p.add_argument("--artifact-root", type=Path)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--calibration-count", type=int, default=500)
    p.add_argument("--validation-count", type=int, default=500)
    p.add_argument("--min-count", type=int, default=25)
    p.add_argument("--baseline-start-ns", type=float, default=-2.)
    p.add_argument("--baseline-end-ns", type=float, default=-1.)
    p.add_argument("--chunk-size", type=int, default=128)
    p.add_argument("--cv-folds", type=int, default=5)
    p.add_argument("--alphas", nargs="+", type=float,
                   default=[.1, 1., 10., 100., 1000., 10000.])
    p.add_argument("--events-per-group", type=int, default=2)
    p.add_argument("--event-ids-file", type=Path,
                   help="Reuse selected_events.json from recent local TSHAP")
    return p


if __name__ == "__main__":
    run(parser().parse_args())
