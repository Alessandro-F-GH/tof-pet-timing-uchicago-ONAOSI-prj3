"""Global and output-stratified explanations of frozen waveform regressors.

Uses original saved transformations and Ridge coefficients. No predictive refit,
new kernels, surrogate, raw acquisition access, or alteration of benchmark files.
"""
from __future__ import annotations

import csv
from importlib.metadata import version
import json
from pathlib import Path
import shutil
import tempfile

import numpy as np

from waveform_analysis.core.io import atomic_json, canonical_hash
from waveform_analysis.data.results import discover_runs, resolve_run_config
from waveform_analysis.data.view import waveform_view
from waveform_analysis.engine.regression_explainer import (
    _stage, frozen_features, predict_features,
)
from waveform_analysis.engine.train import (
    load_fitted_model, release_training_memory, saved_model_complete,
)
from waveform_analysis.engine.xai_replay import PreparedCacheLookup, _cache_root
from waveform_analysis.models import get_model
from waveform_analysis.models.kernel.rdst_common import export_shapelets
from waveform_analysis.signal.sample_mask import apply_sample_mask, apply_sample_mask_to_time

INTERVALS = {
    "low": (0.02, 0.10),
    "central": (0.46, 0.54),
    "high": (0.90, 0.98),
}
GROUP_NAMES = ("all", "low", "central", "high")


def select_quantile_groups(event_ids, predictions, intervals=INTERVALS):
    """Select non-overlapping equal-mass rank bands without boundary-tie losses.

    Stable ordering breaks ties using the frozen blind population order.
    Percentiles refer to ranks, not numerical thresholds in correction units.
    """
    ids, y = np.asarray(event_ids), np.asarray(predictions, dtype=np.float64)
    if y.ndim != 1 or ids.shape != y.shape or not len(y):
        raise ValueError("Event IDs and predictions must be aligned nonempty vectors")
    if not np.all(np.isfinite(y)) or len(np.unique(ids)) != len(ids):
        raise ValueError("Predictions must be finite and event IDs unique")
    order = np.argsort(y, kind="stable")
    labels = np.full(len(y), -1, dtype=np.int8)
    groups = {}
    for k, (name, (lower, upper)) in enumerate(intervals.items()):
        if not (0 <= lower < upper <= 1):
            raise ValueError(f"Invalid quantile interval: {name}")
        start, stop = int(np.floor(lower * len(y))), int(np.floor(upper * len(y)))
        if stop <= start:
            raise ValueError(f"Quantile interval {name} is empty for n={len(y)}")
        selected = order[start:stop]
        if np.any(labels[selected] >= 0):
            raise ValueError("Quantile groups overlap")
        labels[selected] = k
        representative = int(selected[len(selected) // 2])
        groups[name] = {
            "indices": selected, "representative_index": representative,
            "quantiles": [lower, upper], "n_events": int(len(selected)),
            "min_prediction_ps": float(y[selected].min()),
            "max_prediction_ps": float(y[selected].max()),
            "median_prediction_ps": float(np.median(y[selected])),
        }
    return labels, groups


def aggregate_contributions(features, coefficients, reference_mean, predictions,
                            intervals_labels, *, group_size, intercept, clip, chunk_size):
    """Stream exact pre-clipping additive contributions in O(n_features) memory.

    Net shapelet contributions are added before absolute values. Centering is
    relative to the full *development* reference feature mean; the original
    prediction is still reconstructed from uncentered contributions.
    """
    # Keep the fitted head's original precision for reconstruction tolerances.
    # Casting coefficients to float64 for attribution must NOT imply that
    # sklearn Ridge predicted in float64 (float32 LSQR heads predict in float32).
    coefficient_dtype = np.asarray(coefficients).dtype
    weights = np.asarray(coefficients, dtype=np.float64).reshape(-1)
    mu = np.asarray(reference_mean, dtype=np.float64).reshape(-1)
    preds = np.asarray(predictions, dtype=np.float64)
    labels = np.asarray(intervals_labels, dtype=np.int8)
    n, p = features.shape
    if (weights.shape != (p,) or mu.shape != (p,) or preds.shape != (n,)
            or labels.shape != (n,) or p % group_size or chunk_size <= 0):
        raise ValueError("Mismatched feature, coefficient, prediction or group shapes")
    if not np.all(np.isfinite(weights)) or not np.all(np.isfinite(mu)):
        raise ValueError("Non-finite linear model or reference statistics")
    ng = p // group_size
    counts = np.zeros(4, dtype=np.int64)
    totals = np.zeros((4, ng))
    abs_totals = np.zeros((4, ng))
    squared = np.zeros((4, ng))
    abs_squared = np.zeros((4, ng))
    type_signed = np.zeros((4, group_size))
    type_absolute = np.zeros((4, group_size))
    center = weights * mu
    # Saved float32 features set the conservative dot-product error budget,
    # even when fitted Ridge coefficients are float64. Attributions themselves
    # are still accumulated separately in float64.
    epsilon = np.finfo(features.dtype).eps
    max_error, clipped_count = 0., 0
    for start in range(0, n, chunk_size):
        end = min(start + chunk_size, n)
        original = np.asarray(features[start:end], dtype=np.float64) * weights
        corrected = original - center
        net = corrected.reshape(-1, ng, group_size).sum(axis=2)
        reconstruction = float(intercept) + original.sum(axis=1)
        if clip is not None:
            clipped_count += int(np.count_nonzero(np.abs(reconstruction) > clip))
            reconstruction = np.clip(reconstruction, -clip, clip)
        diff = np.abs(reconstruction - preds[start:end])
        max_error = max(max_error, float(diff.max()))
        # Match the original predictor with a tolerance for BLAS rounding.
        tolerance = 1e-4 + 8 * epsilon * (np.abs(original).sum(axis=1) + abs(intercept))
        if np.any(tolerance > 1.0):
            raise ValueError(
                'Ill-conditioned linear reconstruction: numerical error allowance '
                f'up to {float(tolerance.max()):.5g} ps exceeds the 1 ps safety limit'
            )
        if np.any(diff > tolerance):
            worst = int(np.argmax(diff / np.maximum(tolerance, np.finfo(float).tiny)))
            raise ValueError(
                "Feature contributions fail frozen prediction reconstruction "
                f"(event row={start + worst}, error={diff[worst]:.6g} ps, "
                f"tolerance={tolerance[worst]:.6g} ps, "
                f"input_dtype={features.dtype}, coefficient_dtype={coefficient_dtype})"
            )
        for group_index in range(4):
            take = (np.ones(end-start, dtype=bool) if group_index == 0
                    else labels[start:end] == group_index - 1)
            if not take.any():
                continue
            part = net[take]
            absolute = np.abs(part)
            counts[group_index] += len(part)
            totals[group_index] += part.sum(axis=0)
            abs_totals[group_index] += absolute.sum(axis=0)
            squared[group_index] += np.square(part).sum(axis=0)
            abs_squared[group_index] += np.square(absolute).sum(axis=0)
            # Retain the three RDST feature types, or individual features for others.
            typed = corrected[take].reshape(-1, ng, group_size).sum(axis=1)
            type_signed[group_index] += typed.sum(axis=0)
            type_absolute[group_index] += np.abs(typed).sum(axis=0)
    if np.any(counts == 0):
        raise ValueError("At least one explanation group is empty")
    mean = totals / counts[:, None]
    mean_abs = abs_totals / counts[:, None]
    std = np.sqrt(np.maximum(squared / counts[:, None] - np.square(mean), 0))
    abs_std = np.sqrt(np.maximum(abs_squared / counts[:, None] - np.square(mean_abs), 0))
    return {
        "count": counts,
        "mean_signed_ps": mean, "mean_absolute_ps": mean_abs,
        "sd_signed_ps": std, "sd_absolute_ps": abs_std,
        "se_absolute_ps": abs_std / np.sqrt(counts[:, None]),
        "type_signed_ps": type_signed / counts[:, None],
        "type_absolute_ps": type_absolute / counts[:, None],
        "group_names": np.asarray(GROUP_NAMES),
        "reference_prediction_ps": np.asarray(float(intercept) + center.sum()),
        "max_reconstruction_error_ps": np.asarray(max_error),
        "clipped_events": np.asarray(clipped_count),
    }


def _development_mean(matrix, chunk):
    total = np.zeros(matrix.shape[1], dtype=np.float64)
    for start in range(0, len(matrix), chunk):
        total += np.asarray(matrix[start:start + chunk], dtype=np.float64).sum(axis=0)
    return total / len(matrix)


def _csv_results(path, aggregate, shapelets=None):
    import csv
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["group", "feature_id", "feature_label", "n_events",
                         "mean_signed_contribution_ps", "mean_absolute_contribution_ps",
                         "sd_absolute_ps", "se_absolute_ps"])
        for k, name in enumerate(GROUP_NAMES):
            for j in range(aggregate["mean_signed_ps"].shape[1]):
                writer.writerow([name, j, f"shapelet_{j}" if shapelets is not None else f"original_feature_{j}",
                                 int(aggregate["count"][k]),
                                 float(aggregate["mean_signed_ps"][k, j]),
                                 float(aggregate["mean_absolute_ps"][k, j]),
                                 float(aggregate["sd_absolute_ps"][k, j]),
                                 float(aggregate["se_absolute_ps"][k, j])])


def _plot_results(folder, agg, examples, waveforms, time_ps, shapelets=None, top_k=20):
    import matplotlib.pyplot as plt
    import matplotlib
    order = np.argsort(-agg["mean_absolute_ps"][0], kind="stable")[:top_k]
    labels = [f"S{j}" if shapelets is not None else f"F{j}" for j in order]
    fig, axes = plt.subplots(1, 2, figsize=(13, max(4, .3 * len(order))))
    axes[0].barh(labels, agg["mean_absolute_ps"][0, order])
    axes[0].set(xlabel="Mean absolute centered contribution [ps]", title="Global importance")
    axes[0].invert_yaxis()
    for k, name in enumerate(GROUP_NAMES[1:], start=1):
        axes[1].plot(np.arange(len(order)), agg["mean_absolute_ps"][k, order],
                     marker="o", markersize=3, label=name)
    axes[1].set(xticks=np.arange(len(order)), xticklabels=labels,
                ylabel="Mean absolute centered contribution [ps]",
                title="Stratified contribution magnitudes")
    axes[1].tick_params(axis="x", rotation=90)
    axes[1].legend()
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(folder / f"global_and_stratified.{ext}", dpi=160)
    plt.close(fig)

    if shapelets is not None:
        fig, axes = plt.subplots(min(6, len(order)), 1, figsize=(9, 2.2*min(6, len(order))), squeeze=False)
        for ax, j in zip(axes[:, 0], order[:6]):
            length, dilation = int(shapelets["lengths"][j]), int(shapelets["dilations"][j])
            for c, values in enumerate(shapelets["values"][j]):
                ax.plot(np.arange(length)*dilation, values[:length], label=f"channel {c+1}")
            ax.set(title=f"Original shapelet {j}, dilation={dilation}", xlabel="Retained sample offset")
            if len(shapelets["values"][j]) > 1:
                ax.legend()
        fig.tight_layout()
        for ext in ("png", "pdf"):
            fig.savefig(folder / f"important_shapelets.{ext}", dpi=160)
        plt.close(fig)

    # Display real sample waveforms; do NOT invent a kernel/shapelet-to-time map.
    fig, axes = plt.subplots(3, 2, figsize=(13, 10))
    for i, name in enumerate(GROUP_NAMES[1:]):
        info = examples[name]
        wave = waveforms[name]
        for detector in (0, 1):
            axes[i, 0].plot(time_ps, wave[detector], label=f"detector {detector+1}")
        axes[i, 0].set(title=f"{name}: event {info['event_id']}, correction={info['prediction_ps']:.2f} ps",
                       ylabel="Prepared waveform amplitude", xlabel="Time [ps]")
        axes[i, 0].legend(loc="best")
        relevant = info["top_features"][:8]
        axes[i, 1].barh([f"S{j}" if shapelets is not None else f"F{j}" for j, _ in relevant],
                         [value for _, value in relevant])
        axes[i, 1].axvline(0, color="gray", linewidth=.8)
        axes[i, 1].set(xlabel="Signed contribution relative to development [ps]",
                       title="Top absolute local contributions")
        axes[i, 1].invert_yaxis()
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(folder / f"representative_events.{ext}", dpi=160)
    plt.close(fig)


def explain_stratified(results, *, models=None, cache_root=None, top_k=20, max_events_per_group=None, n_ppv_features=None, logger=None):
    if not isinstance(top_k, int) or isinstance(top_k, bool) or top_k < 1:
        raise ValueError("top_k must be a positive integer")
    if max_events_per_group is not None and (isinstance(max_events_per_group, bool) or max_events_per_group < 1):
        raise ValueError("max_events_per_group must be a positive integer")
    if n_ppv_features is not None and (isinstance(n_ppv_features, bool) or not isinstance(n_ppv_features, int) or n_ppv_features < 1):
        raise ValueError("n_ppv_features must be a positive integer")
    runs = discover_runs(Path(results).expanduser().resolve())
    if not runs:
        raise ValueError(f"No complete canonical runs found in {results}")
    available = {r["manifest"]["model"] for r in runs}
    if models is not None and set(models) - available:
        raise ValueError(f"Requested models not found: {sorted(set(models) - available)}")
    outputs, failures = [], []
    lookup = PreparedCacheLookup()
    for run in runs:
        if models is not None and run["manifest"]["model"] not in models:
            continue
        try:
            outputs.append(_explain_one(run, lookup, cache_root, top_k, logger, max_events_per_group, n_ppv_features))
        except Exception as exc:
            failures.append(f"{run['directory']}: {exc}")
            if logger:
                logger.error("Stratified explanation failed | %s | %s", run["directory"], exc)
        finally:
            release_training_memory()
    if failures:
        raise RuntimeError("Stratified explanation failures (successful outputs retained):\n" + "\n".join(failures))
    return outputs


def _explain_one(run, lookup, cache_override, top_k, logger, max_events_per_group=None, n_ppv_features=None):
    directory, manifest = Path(run["directory"]), run["manifest"]
    config = resolve_run_config(directory)
    spec = get_model(manifest["model"])
    if spec.name not in {"direct_rdst", "shared_rdst", "direct_multirocket", "shared_multirocket",
                          "direct_multirocket_rdst", "shared_multirocket_rdst"}:
        raise ValueError("Stratified explanations currently support RDST and MultiRocket Ridge heads only")
    model_dir = directory / "model"
    if not saved_model_complete(spec, model_dir):
        raise FileNotFoundError("Complete saved model with original transform required")
    root = _cache_root(config, directory, cache_override, roles=("development", "blind"))
    development = lookup.find(root, "development", config, manifest)
    blind = lookup.find(root, "blind", config, manifest)
    if development.n_events < 2 or blind.n_events < 200:
        raise ValueError("Need >=2 development and >=200 blind events for nonempty percentile bands")
    mode = config["mode"]
    family = waveform_view(development, mode, [0]).family
    time = np.asarray(getattr(development, f"{family}_time_ps"))
    if not np.array_equal(time, getattr(blind, f"{family}_time_ps")):
        raise ValueError("Development/blind time grids differ")
    for key in ("minimum", "maximum"):
        if not np.array_equal(getattr(getattr(development, f"{family}_transform"), key),
                              getattr(getattr(blind, f"{family}_transform"), key)):
            raise ValueError("Development/blind normalization differs")
    import sklearn
    identity = {
        "method": "stratified_original_feature_contributions_v1",
        "settings": {"intervals": INTERVALS, "top_k": top_k, "reference": "development_mean",
                     "selection": "model_specific_rank",
                     "temporal_max_events_per_group": max_events_per_group,
                     "temporal_sampling_seed": 1729,
                     "temporal_method_version": 2,
                     "ppv_pooling_method_version": 2,
                     "n_ppv_features": n_ppv_features},
        "final_fit": manifest["stage_fingerprints"]["final_fit"],
        "development": development.manifest["analysis_protocol_identity"],
        "blind": blind.manifest["analysis_protocol_identity"],
        "aeon": version("aeon"), "sklearn": sklearn.__version__,
    }
    destination = directory / "explanations" / "stratified" / canonical_hash(identity)[:16]
    if (destination / "summary.json").is_file():
        if logger:
            logger.info("Stratified explanation | reuse | %s", destination)
        return destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    fitted = load_fitted_model(spec, model_dir, run["best"]["parameters"], config)
    # Preserve the Ridge head's native dtype for numerical reconstruction checks.
    # float32 LSQR coefficients MUST NOT be upcast at this call site: the
    # attribution routine performs its own float64 accumulation separately.
    weights = np.asarray(fitted.artifact.regressor.coef_).reshape(-1)
    intercept = float(fitted.artifact.regressor.intercept_)
    chunk = int(config["runtime"]["prediction_chunk_size"])
    mask = fitted.sample_mask
    shapelets = (export_shapelets(fitted.feature_transform.artifact.transformer)
                 if spec.name in {"direct_rdst", "shared_rdst"} else None)
    group_size = 3 if shapelets is not None else 1
    temp_output = Path(tempfile.mkdtemp(prefix=".stratified-", dir=destination.parent))
    try:
        with tempfile.TemporaryDirectory(prefix=".frozen-stratified-", dir=destination.parent) as scratch:
            matrices = {}
            try:
                for role, data in (("development", development), ("blind", blind)):
                    with _stage(logger, f"{spec.name} {role} frozen features"):
                        matrices[role], _ = frozen_features(
                            spec, fitted, data, config, np.arange(data.n_events),
                            root, role, Path(scratch), logger=logger)
                if weights.shape != (matrices["blind"].shape[1],):
                    raise ValueError("Fitted coefficients do not match frozen feature layout")
                dev_mean = _development_mean(matrices["development"], chunk)
                with _stage(logger, "original blind predictions"):
                    predictions = predict_features(spec, fitted, matrices["blind"], chunk, logger=logger, role="blind")
                ids = np.asarray(blind.event_index)
                labels, groups = select_quantile_groups(ids, predictions)
                with _stage(logger, f"{spec.name} full-blind regression contribution aggregation"):
                    if logger:
                        logger.info("Regression XAI | model=%s | events=%d | features=%d | batch_size=%d",
                                    spec.name, blind.n_events, len(weights), chunk)
                    agg = aggregate_contributions(
                        matrices["blind"], weights, dev_mean, predictions, labels,
                        group_size=group_size, intercept=intercept,
                        clip=fitted.output_max_abs_ps, chunk_size=chunk, logger=logger)
                temporal = None
                if shapelets is not None and config["window_name"] == "wide":
                    from waveform_analysis.engine.rdst_temporal import temporal_match_maps
                    selection = np.arange(blind.n_events, dtype=np.int64)
                    if max_events_per_group is not None:
                        rng = np.random.default_rng(1729)
                        selection = np.sort(rng.choice(
                            selection, size=min(max_events_per_group, len(selection)),
                            replace=False))
                    with _stage(logger, "RDST time-localized contributions"):
                        temporal = temporal_match_maps(
                            fitted=fitted, blind=blind, mode=mode,
                            features=matrices["blind"], reference_mean=dev_mean,
                            coefficients=weights,
                            time_ps=apply_sample_mask_to_time(time, mask),
                            chunk_size=chunk, logger=logger,
                            event_indices=selection, position_cache_root=root)
                ppv_map = None
                if ("multirocket" in spec.name and config["window_name"] == "wide"):
                    from waveform_analysis.engine.rocket_ppv_lrp import ppv_temporal_map
                    ppv_selection = np.arange(blind.n_events, dtype=np.int64)
                    if max_events_per_group is not None:
                        rng = np.random.default_rng(1729)
                        ppv_selection = np.sort(rng.choice(
                            ppv_selection, size=min(max_events_per_group, len(ppv_selection)),
                            replace=False))
                    with _stage(logger, "MultiRocket PPV pooling time attribution"):
                        ppv_map = ppv_temporal_map(
                            fitted=fitted, blind=blind, mode=mode,
                            features=matrices["blind"], reference_mean=dev_mean,
                            coefficients=weights,
                            time_ps=apply_sample_mask_to_time(time, mask),
                            event_indices=ppv_selection, logger=logger,
                            n_ppv_features=n_ppv_features)
                examples, waves = {}, {}
                masked_time = apply_sample_mask_to_time(time, mask)
                for name, group in groups.items():
                    idx = group["representative_index"]
                    c = (np.asarray(matrices["blind"][idx], dtype=np.float64) - dev_mean) * weights
                    c = c.reshape(-1, group_size).sum(axis=1)
                    top = np.argsort(-np.abs(c), kind="stable")[:top_k]
                    examples[name] = {
                        "event_id": str(ids[idx]), "population_index": idx,
                        "prediction_ps": float(predictions[idx]),
                        "top_features": [(int(j), float(c[j])) for j in top],
                    }
                    prepared = waveform_view(blind, mode, [idx]).materialize()
                    waves[name] = np.asarray(apply_sample_mask(prepared, mask)[0], dtype=np.float64)
            finally:
                for value in matrices.values():
                    if isinstance(value, np.memmap):
                        value._mmap.close()
                matrices.clear()
        with (temp_output / "groups.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(["event_id", "prediction_ps", "group"])
            for i, (eid, prediction) in enumerate(zip(ids, predictions)):
                writer.writerow([eid, float(prediction),
                                 GROUP_NAMES[labels[i]+1] if labels[i] >= 0 else "unused"])
        atomic_json(temp_output / "summary.json", {
            **identity, "model": spec.name, "formulation": spec.estimator_formulation,
            "transform_identity": fitted.feature_transform.identity,
            "counts": {name: int(agg["count"][i]) for i, name in enumerate(GROUP_NAMES)},
            "group_prediction_ranges": {name: {
                key: value for key, value in group.items() if key != "indices"
            } for name, group in groups.items()},
            "reference_prediction_ps": float(agg["reference_prediction_ps"]),
            "temporal_explanation": "best_match_start, conserved net shapelet contributions" if temporal is not None else None,
            "clipped_events": int(agg["clipped_events"]),
            "max_reconstruction_error_ps": float(agg["max_reconstruction_error_ps"]),
            "explanation_definition": "coefficient*(original frozen feature - development population mean), before clipping",
            "interpretation_warning": "Contributions describe model behavior, not physical causality. "
                                      "No temporal attribution is inferred from random-kernel feature indices. "
                                      "Groups are selected by model predictions and are descriptive."
        })
        np.savez_compressed(temp_output / "event_groups.npz",
                            event_id=ids, prediction_ps=predictions, group_index=labels,
                            representative_index=np.array([groups[g]["representative_index"] for g in GROUP_NAMES[1:]]))
        np.savez_compressed(temp_output / "aggregate_contributions.npz", **agg)
        _csv_results(temp_output / "group_importance.csv", agg, shapelets=shapelets)
        atomic_json(temp_output / "representative_events.json", examples)
        if temporal is not None:
            from waveform_analysis.reporting.rdst_temporal import plot_temporal_importance
            np.savez_compressed(temp_output / "temporal_importance.npz", **temporal)
            with (temp_output / "temporal_importance.csv").open("w", newline="", encoding="utf-8") as stream:
                writer = csv.writer(stream)
                writer.writerow(["time_ns", "mean_absolute_ps", "mean_signed_ps", "fraction", "matches_per_event"])
                for t in range(len(temporal["time_ps"])):
                    writer.writerow([float(temporal["time_ps"][t] / 1000),
                                     float(temporal["combined_absolute_ps"][t]),
                                     float(temporal["combined_signed_ps"][t]),
                                     float(temporal["combined_fraction"][t]),
                                     float(temporal["combined_match_count_per_event"][t])])
            plot_temporal_importance(temp_output, temporal)
        if ppv_map is not None:
            from waveform_analysis.reporting.rocket_ppv import plot_rocket_ppv
            np.savez_compressed(temp_output / "ppv_temporal_importance.npz", **ppv_map)
            with (temp_output / "ppv_temporal_importance.csv").open("w", newline="", encoding="utf-8") as stream:
                wr = csv.writer(stream)
                wr.writerow(["time_ns", "absolute_ps", "signed_ps", "localized_fraction"])
                for k, t in enumerate(ppv_map["time_ps"]):
                    wr.writerow([float(t / 1000), float(ppv_map["combined_absolute_ps"][k]),
                                 float(ppv_map["combined_signed_ps"][k]),
                                 float(ppv_map["combined_fraction"][k])])
            plot_rocket_ppv(temp_output, ppv_map)
        if shapelets is not None:
            np.savez_compressed(temp_output / "original_shapelets.npz", **shapelets)
            with (temp_output / "feature_families.csv").open("w", newline="", encoding="utf-8") as handle:
                writer = csv.writer(handle)
                writer.writerow(["group", "type", "mean_signed_ps", "mean_absolute_ps"])
                for k, name in enumerate(GROUP_NAMES):
                    for j, feature in enumerate(("minimum_distance", "match_location", "occurrence_count")):
                        writer.writerow([name, feature, float(agg["type_signed_ps"][k, j]),
                                         float(agg["type_absolute_ps"][k, j])])
        _plot_results(temp_output, agg, examples, waves, masked_time, shapelets, top_k)
        temp_output.rename(destination)
        return destination
    finally:
        if temp_output.exists():
            shutil.rmtree(temp_output)
