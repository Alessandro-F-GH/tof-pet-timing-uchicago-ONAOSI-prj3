"""Local temporal attribution plots for representative blind events."""
from __future__ import annotations

import csv
import json
import shutil
import tempfile
from pathlib import Path

import numpy as np

from waveform_analysis.core.io import atomic_json
from waveform_analysis.data.results import discover_runs, resolve_run_config
from waveform_analysis.data.view import waveform_view
from waveform_analysis.engine.regression_explainer import _stage, frozen_features, predict_features
from waveform_analysis.engine.stratified_explainer import (
    INTERVALS, _development_mean, select_quantile_groups,
)
from waveform_analysis.engine.train import load_fitted_model, release_training_memory, saved_model_complete
from waveform_analysis.engine.xai_replay import PreparedCacheLookup, _cache_root
from waveform_analysis.models import get_model
from waveform_analysis.signal.sample_mask import apply_sample_mask, apply_sample_mask_to_time


def _confirm_overwrite(destination: Path, yes: bool) -> bool:
    if not destination.exists():
        return True
    if yes:
        return True
    try:
        answer = input(
            f"Representative explanations already exist at {destination}. "
            "Overwrite them? [y/N] "
        ).strip().lower()
    except (EOFError, KeyboardInterrupt):
        return False
    return answer in {"y", "yes"}


def _plot_waveform_map(path, time_ps, waveform, relevance, *, title, label):
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection
    from matplotlib.colors import TwoSlopeNorm

    time_ns = np.asarray(time_ps, dtype=np.float64) / 1000.0
    waveform = np.asarray(waveform, dtype=np.float64)
    relevance = np.asarray(relevance, dtype=np.float64)
    if waveform.ndim != 1 or relevance.shape != waveform.shape:
        raise ValueError("Waveform and local temporal attribution must have matching 1-D shapes")
    fig, ax = plt.subplots(figsize=(10, 3.3))
    ax.plot(time_ns, waveform, color="0.78", linewidth=1.1, zorder=1)
    if len(time_ns) >= 2 and np.any(relevance):
        points = np.column_stack([time_ns, waveform]).reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
        # Attribute each segment using the mean of its endpoint relevance.
        segment_relevance = 0.5 * (relevance[:-1] + relevance[1:])
        extent = float(np.max(np.abs(segment_relevance)))
        norm = TwoSlopeNorm(vmin=-extent, vcenter=0.0, vmax=extent) if extent > 0 else None
        collection = LineCollection(
            segments, cmap="coolwarm", norm=norm,
            linewidths=2.6, zorder=2,
        )
        collection.set_array(segment_relevance)
        ax.add_collection(collection)
        fig.colorbar(collection, ax=ax, pad=0.02, label="Signed local contribution (ps)")
    ax.set_title(title)
    ax.set_xlabel("Time relative to waveform window start (ns)")
    ax.set_ylabel("Prepared waveform amplitude")
    ax.margins(x=0.01)
    ax.grid(alpha=0.18)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _plot_event(folder, stem, event_id, prediction, target, time, waveform, relevance,
                detector_names, method):
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection
    from matplotlib.colors import TwoSlopeNorm

    time_ns = np.asarray(time, dtype=np.float64) / 1000.
    n_channels = waveform.shape[0]
    fig, axes = plt.subplots(n_channels, 1, figsize=(11, 3.1 * n_channels), squeeze=False)
    max_abs = float(np.max(np.abs(relevance))) if relevance.size else 0.
    for ch, ax in enumerate(axes[:, 0]):
        values = np.asarray(waveform[ch], dtype=np.float64)
        rel = np.asarray(relevance[min(ch, relevance.shape[0]-1)], dtype=np.float64)
        ax.plot(time_ns, values, color="0.80", linewidth=1.0, zorder=1)
        if len(time_ns) >= 2 and np.any(rel):
            points = np.column_stack([time_ns, values]).reshape(-1, 1, 2)
            segments = np.concatenate([points[:-1], points[1:]], axis=1)
            segment_rel = .5 * (rel[:-1] + rel[1:])
            local_max = max(float(np.max(np.abs(segment_rel))), 1e-15)
            norm = TwoSlopeNorm(vmin=-local_max, vcenter=0., vmax=local_max)
            line = LineCollection(segments, cmap="coolwarm", norm=norm, linewidths=2.6, zorder=2)
            line.set_array(segment_rel)
            ax.add_collection(line)
            fig.colorbar(line, ax=ax, pad=.015, label="Signed contribution (ps)")
        ax.set_ylabel(f"Detector {ch+1} amplitude")
        ax.set_xlabel("Time relative to window start (ns)")
        ax.set_title(detector_names[ch] if len(detector_names) == n_channels else
                     f"Detector {ch+1} — {detector_names[0]} attribution")
        ax.grid(alpha=.18)
        ax.margins(x=.01)
    fig.suptitle(
        f"{stem} | event {event_id} | prediction={prediction:.3f} ps | "
        f"target={target:.3f} ps\n{method}",
        y=1.01, fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(folder / f"{stem}.png", dpi=180, bbox_inches="tight")
    fig.savefig(folder / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)


def explain_representatives(results, *, models=None, cache_root=None,
                            n_ppv_features=None, yes_overwrite=False, logger=None):
    runs = discover_runs(Path(results).expanduser().resolve())
    if not runs:
        raise ValueError(f"No complete canonical runs found in {results}")
    available = {run["manifest"]["model"] for run in runs}
    if models is not None and set(models) - available:
        raise ValueError(f"Requested models not found: {sorted(set(models) - available)}")
    supported = {
        "direct_rdst", "shared_rdst", "direct_multirocket", "shared_multirocket",
    }
    selected_runs = [
        run for run in runs
        if (models is None or run["manifest"]["model"] in models)
        and run["manifest"]["model"] in supported
    ]
    if not selected_runs:
        raise ValueError("No supported RDST/MultiRocket Ridge runs selected")
    outputs, failures = [], []
    lookup = PreparedCacheLookup()
    for run in selected_runs:
        try:
            output = _explain_one(run, lookup, cache_root, n_ppv_features,
                                  yes_overwrite, logger)
            if output is not None:
                outputs.append(output)
        except Exception as exc:
            failures.append(f"{run['directory']}: {exc}")
            if logger:
                logger.exception("Representative explanation failed | %s", run["directory"])
        finally:
            release_training_memory()
    if failures:
        raise RuntimeError("Representative explanation failures:\n" + "\n".join(failures))
    return outputs


def _explain_one(run, lookup, cache_override, n_ppv_features, yes_overwrite, logger):
    directory, manifest = Path(run["directory"]), run["manifest"]
    spec = get_model(manifest["model"])
    config = resolve_run_config(directory)
    destination = directory / "representatives"
    if not _confirm_overwrite(destination, yes_overwrite):
        if logger:
            logger.info("Representative explanation | skipped overwrite | %s", destination)
        return None
    model_dir = directory / "model"
    if not saved_model_complete(spec, model_dir):
        raise FileNotFoundError(f"Complete saved model with original transform required: {model_dir}")
    root = _cache_root(config, directory, cache_override, roles=("development", "blind"))
    development = lookup.find(root, "development", config, manifest)
    blind = lookup.find(root, "blind", config, manifest)
    mode = config["mode"]
    family = waveform_view(development, mode, [0]).family
    time = np.asarray(getattr(development, f"{family}_time_ps"))
    if not np.array_equal(time, getattr(blind, f"{family}_time_ps")):
        raise ValueError("Development/blind time grids differ")

    fitted = None
    temp_parent = directory
    temp_output = Path(tempfile.mkdtemp(prefix=".representatives-", dir=temp_parent))
    scratch = Path(tempfile.mkdtemp(prefix=".representative-cache-", dir=temp_parent))
    matrices = {}
    try:
        with _stage(logger, f"{spec.name} load frozen model"):
            fitted = load_fitted_model(spec, model_dir, run["best"]["parameters"], config)
        for role, dataset in (("development", development), ("blind", blind)):
            with _stage(logger, f"{spec.name} {role} frozen features"):
                matrices[role], _ = frozen_features(
                    spec, fitted, dataset, config, np.arange(dataset.n_events),
                    root, role, scratch, logger=logger,
                )
        dev_mean = _development_mean(matrices["development"], int(config["runtime"]["prediction_chunk_size"]))
        with _stage(logger, f"{spec.name} blind predictions and representative selection"):
            predictions = predict_features(
                spec, fitted, matrices["blind"],
                int(config["runtime"]["prediction_chunk_size"]), logger=logger, role="blind",
            )
        ids = np.asarray(blind.event_index)
        _, groups = select_quantile_groups(ids, predictions, INTERVALS)
        # Existing rank-band logic selects the median-ranked event in each
        # 2–10%, 46–54%, and 90–98% band. Sort indices for temporal-map APIs.
        reps = {name: int(info["representative_index"]) for name, info in groups.items()}
        event_indices = np.asarray(sorted(set(reps.values())), dtype=np.int64)
        row_for_event = {int(index): row for row, index in enumerate(event_indices)}
        weights = np.asarray(fitted.artifact.regressor.coef_).reshape(-1)
        masked_time = apply_sample_mask_to_time(time, fitted.sample_mask)
        method = ""
        if "rdst" in spec.name:
            from waveform_analysis.engine.rdst_temporal import temporal_match_maps
            with _stage(logger, f"{spec.name} representative RDST temporal maps"):
                maps = temporal_match_maps(
                    fitted=fitted, blind=blind, mode=mode, features=matrices["blind"],
                    reference_mean=dev_mean, coefficients=weights, time_ps=masked_time,
                    chunk_size=max(1, int(config["runtime"]["prediction_chunk_size"])),
                    logger=logger, event_indices=event_indices, position_cache_root=root,
                )
            event_signed = np.asarray(maps["event_sample_signed_ps"])
            event_abs = np.asarray(maps["event_sample_absolute_ps"])
            shared = bool(fitted.feature_transform.artifact.shared)
            detector_names = ["detector_1", "detector_2"] if shared else ["joint_pair"]
            method = "RDST signed Ridge contributions distributed over exact dilated shapelet support"
        else:
            from waveform_analysis.engine.rocket_ppv_lrp import ppv_temporal_map
            with _stage(logger, f"{spec.name} representative MultiRocket temporal maps"):
                maps = ppv_temporal_map(
                    fitted=fitted, blind=blind, mode=mode, features=matrices["blind"],
                    reference_mean=dev_mean, coefficients=weights, time_ps=masked_time,
                    event_indices=event_indices, logger=logger,
                    n_ppv_features=n_ppv_features,
                )
            event_signed = np.asarray(maps["event_input_signed_ps"]).sum(axis=1)
            event_abs = np.asarray(maps["event_input_absolute_ps"]).sum(axis=1)
            shared = bool(getattr(fitted.feature_transform.artifact, "shared", False))
            detector_names = ["detector_1", "detector_2"] if shared else ["joint_pair"]
            method = "MultiRocket PPV pooling attribution redistributed to input samples; raw and difference views combined"
        if event_signed.shape[0] != len(event_indices):
            raise RuntimeError("Temporal attribution output does not align with representative events")
        # Use the same target extraction as the original batch evaluation.
        from waveform_analysis.data.view import model_target
        target = model_target(blind, mode)
        records = []
        masked_blind = apply_sample_mask(
            waveform_view(blind, mode, event_indices).materialize(), fitted.sample_mask
        )
        for name in ("low", "central", "high"):
            idx = reps[name]
            row = row_for_event[idx]
            relevance = event_signed[row]
            waveform = np.asarray(masked_blind[row], dtype=np.float64)
            if relevance.shape[0] == 1 and waveform.shape[0] == 2:
                # A direct paired model yields a joint time map, not a
                # separately identified detector attribution. Show the same
                # joint temporal relevance on both traces and label it honestly.
                plot_relevance = np.repeat(relevance, 2, axis=0)
                plot_names = ["joint_pair attribution"] * 2
            else:
                plot_relevance = relevance
                plot_names = detector_names
            _plot_event(
                temp_output, name, str(ids[idx]), float(predictions[idx]),
                float(target[idx]), masked_time, waveform, plot_relevance,
                plot_names, method,
            )
            records.append({
                "group": name, "quantile_interval": list(INTERVALS[name]),
                "population_index": idx, "event_id": str(ids[idx]),
                "prediction_ps": float(predictions[idx]), "target_ps": float(target[idx]),
                "waveform_plot": f"{name}.png", "method": method,
            })
            np.savez_compressed(
                temp_output / f"{name}_attribution.npz",
                time_ps=masked_time, signed_contribution_ps=relevance,
                absolute_contribution_ps=event_abs[row],
                event_id=np.asarray(str(ids[idx])), prediction_ps=np.asarray(predictions[idx]),
                target_ps=np.asarray(target[idx]),
            )
        with (temp_output / "representatives.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(records[0]))
            writer.writeheader()
            writer.writerows(records)
        atomic_json(temp_output / "summary.json", {
            "model": spec.name, "teacher_run": str(directory), "method": method,
            "selection": "existing stable blind prediction rank bands; median-ranked event per band",
            "quantile_intervals": INTERVALS, "representatives": records,
            "map_semantics": (
                "Signed local contributions indicate direction of model output contribution; "
                "they are model attributions, not causal physical effects."
            ),
            "detector_semantics": (
                "Shared formulation has detector-specific maps. Direct formulation has a joint "
                "temporal map; displaying it on both waveforms does not imply separate detector attribution."
            ),
        })
        if destination.exists():
            shutil.rmtree(destination)
        temp_output.rename(destination)
        if logger:
            logger.info("Representative explanations | complete | %s", destination)
        return destination
    finally:
        for value in matrices.values():
            if isinstance(value, np.memmap):
                value._mmap.close()
        if fitted is not None:
            del fitted
        shutil.rmtree(scratch, ignore_errors=True)
        if temp_output.exists():
            shutil.rmtree(temp_output, ignore_errors=True)
