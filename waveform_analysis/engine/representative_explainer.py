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
    ax.plot(time_ns, waveform, color="0.25", linewidth=1.8, zorder=1)
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


def _importance_scale(relevance):
    """Robustly scale a local map for display only; saved attributions remain in ps."""
    values = np.asarray(relevance, dtype=np.float64)
    finite = np.abs(values[np.isfinite(values)])
    if finite.size == 0 or not np.any(finite):
        return np.zeros_like(values)
    scale = float(np.quantile(finite, 0.98))
    if not np.isfinite(scale) or scale <= np.finfo(float).eps:
        scale = float(np.max(finite))
    return np.clip(values / max(scale, np.finfo(float).eps), -1.0, 1.0)



def _aggregate_signed_importance(time_ns, relevance, *, bin_samples=8):
    """Sum signed sample-level contributions in consecutive fixed-sample bins."""
    time_ns = np.asarray(time_ns, dtype=np.float64)
    values = np.asarray(relevance, dtype=np.float64)
    if values.ndim == 1:
        values = values[None, :]
    if values.ndim != 2 or values.shape[1] != len(time_ns):
        raise ValueError("Attribution must have shape (channels, samples) matching time grid")
    if bin_samples < 1:
        raise ValueError("bin_samples must be positive")
    if len(time_ns) == 0:
        return np.empty(0), np.empty((values.shape[0], 0)), np.empty(0)

    centers, widths, sums = [], [], []
    sample_dt = float(np.median(np.diff(time_ns))) if len(time_ns) > 1 else 0.0
    for start in range(0, len(time_ns), bin_samples):
        stop = min(start + bin_samples, len(time_ns))
        centers.append(float(np.mean(time_ns[start:stop])))
        widths.append(sample_dt * (stop - start))
        sums.append(np.sum(values[:, start:stop], axis=1))
    return np.asarray(centers), np.stack(sums, axis=1), np.asarray(widths)

def _importance_zoom_limits(time_ns, relevance, *, mass_fraction=0.90):
    """Return a padded interval containing most attribution mass, if genuinely local."""
    magnitude = np.sum(np.abs(np.asarray(relevance, dtype=np.float64)), axis=0)
    total = float(magnitude.sum())
    if not np.isfinite(total) or total <= 0 or len(time_ns) < 3:
        return None
    cumulative = np.concatenate([[0.0], np.cumsum(magnitude)])
    target = mass_fraction * total
    best = (0, len(magnitude), len(magnitude))
    right = 1
    for left in range(len(magnitude)):
        right = max(right, left + 1)
        while right <= len(magnitude) and cumulative[right] - cumulative[left] < target:
            right += 1
        if right > len(magnitude):
            break
        width = right - left
        if width < best[2]:
            best = (left, right, width)
    left, right, _ = best
    span = float(time_ns[-1] - time_ns[0])
    lo = float(time_ns[max(0, left)]); hi = float(time_ns[min(len(time_ns)-1, right-1)])
    if hi - lo >= 0.70 * span:
        return None
    pad = max(0.08 * (hi - lo), 0.015 * span)
    lo, hi = max(float(time_ns[0]), lo-pad), min(float(time_ns[-1]), hi+pad)
    min_width = min(0.18 * span, 1.0 if span <= 10 else 4.0)
    if hi-lo < min_width:
        center = 0.5*(lo+hi)
        lo, hi = max(float(time_ns[0]), center-min_width/2), min(float(time_ns[-1]), center+min_width/2)
    return (lo, hi) if hi > lo else None


def _plot_event(folder, stem, event_id, prediction, target, time, waveform, relevance,
                detector_names, method, *, shared, contribution_limit_ps=None):
    import matplotlib.pyplot as plt

    time_ns = np.asarray(time, dtype=np.float64) / 1000.0
    waveform = np.asarray(waveform, dtype=np.float64)
    relevance = np.asarray(relevance, dtype=np.float64)
    n_channels = waveform.shape[0]
    if n_channels != 2:
        raise ValueError(f"Expected two detector waveforms, got {n_channels}")
    if shared and relevance.shape[0] != n_channels:
        raise ValueError("Shared model should provide one attribution map per detector")
    if not shared and relevance.shape[0] != 1:
        raise ValueError("Direct model should provide one joint-pair attribution map")
    # Keep the underlying sample-level arrays untouched. For plotting only,
    # aggregate signed contributions in 8-sample bins (~100 ps at 12.5 ps/sample),
    # then normalize the binned values for readable visualization.
    importance_time_ns, binned_relevance, bin_widths_ns = _aggregate_signed_importance(
        time_ns, relevance, bin_samples=8,
    )
    display_relevance = _importance_scale(binned_relevance)
    zoom = _importance_zoom_limits(time_ns, relevance)

    # Always show the two detector signals together. Waveform colors identify
    # detectors only; attribution colors are reserved for the importance panels.
    rows = [("Detector waveforms", "wave_overlay", None)]
    if shared:
        rows.extend([
            ("Detector 1 signed attribution", "importance", 0),
            ("Detector 2 signed attribution", "importance", 1),
        ])
    else:
        rows.append(("Joint-pair signed attribution", "importance", 0))

    detector_colors = ["tab:blue", "tab:orange"]

    def draw(path, xlim=None):
        fig, axes = plt.subplots(
            len(rows), 1, figsize=(12, 2.4 + 1.8 * (len(rows) - 1)),
            sharex=True, squeeze=False,
            gridspec_kw={
                "height_ratios": [1.15 if row[1] == "wave_overlay" else 0.68 for row in rows],
                "hspace": 0.14,
            },
        )
        axes = axes[:, 0]
        for ax, (label, kind, ch) in zip(axes, rows):
            if kind == "wave_overlay":
                for detector in range(n_channels):
                    ax.plot(
                        time_ns, waveform[detector],
                        color=detector_colors[detector], linewidth=1.8,
                        label=f"Detector {detector + 1}", zorder=2,
                    )
                ax.set_ylabel("Amplitude")
                ax.set_title(label, loc="left", fontsize=9)
                ax.legend(loc="upper right", frameon=True, fontsize=8, ncol=2)
                ax.grid(alpha=0.18)
            else:
                # Bar heights remain the actual net contributions in ps. Color
                # intensity is normalized independently for this event only.
                rel_ps = binned_relevance[ch]
                color_strength = display_relevance[ch]
                cmap = plt.get_cmap("coolwarm")
                colors = cmap((color_strength + 1.0) / 2.0)
                ax.axhline(0.0, color="0.5", linewidth=0.7)
                ax.bar(
                    importance_time_ns, rel_ps,
                    width=np.maximum(bin_widths_ns * 0.92, 1e-6),
                    color=colors, edgecolor="none", alpha=0.95, align="center",
                )
                if contribution_limit_ps is None:
                    local_limit = float(np.max(np.abs(binned_relevance))) if binned_relevance.size else 0.0
                    y_limit = max(local_limit * 1.08, 1.0)
                else:
                    y_limit = max(float(contribution_limit_ps), 1e-9)
                ax.set_ylim(-y_limit, y_limit)
                ax.set_ylabel("Contribution (ps)")
                ax.set_title(label, loc="left", fontsize=9)
                ax.grid(alpha=0.18)
        axes[-1].set_xlabel("Time relative to waveform window start (ns)")
        axes[-1].set_xlim(
            xlim if xlim is not None else (float(time_ns[0]), float(time_ns[-1]))
        )
        fig.suptitle(
            f"{stem} | event {event_id} | prediction={prediction:.3f} ps | target={target:.3f} ps",
            y=0.995, fontsize=10,
        )
        fig.tight_layout(rect=(0, 0, 1, 0.965))
        fig.savefig(path, dpi=180, bbox_inches="tight")
        plt.close(fig)

    draw(folder / f"{stem}.png")
    draw(folder / f"{stem}.pdf")
    if zoom is not None:
        draw(folder / f"{stem}_zoom.png", zoom)
        draw(folder / f"{stem}_zoom.pdf", zoom)

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
        # Share the physical y-axis range across the three representatives from
        # this model run. Colors remain event-normalized for visual contrast.
        physical_limit_ps = 0.0
        for map_row in range(len(event_indices)):
            _, binned_map, _ = _aggregate_signed_importance(
                np.asarray(masked_time, dtype=np.float64) / 1000.0,
                event_signed[map_row], bin_samples=8,
            )
            if binned_map.size:
                physical_limit_ps = max(physical_limit_ps, float(np.max(np.abs(binned_map))))
        physical_limit_ps = max(physical_limit_ps * 1.08, 1.0)
        for name in ("low", "central", "high"):
            idx = reps[name]
            row = row_for_event[idx]
            relevance = event_signed[row]
            waveform = np.asarray(masked_blind[row], dtype=np.float64)
            _plot_event(
                temp_output, name, str(ids[idx]), float(predictions[idx]),
                float(target[idx]), masked_time, waveform, relevance,
                detector_names, method, shared=shared,
                contribution_limit_ps=physical_limit_ps,
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
