"""Exact interventional local and global Linear SHAP for saved Ridge models.

Uses the complete Development mean as the fixed background; explanations are
computed on blind waveforms without retraining.
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
from waveform_analysis.engine.train import load_fitted_model, predict_indices, saved_model_complete
from waveform_analysis.engine import tshap_explainer as local
from waveform_analysis.models import get_model


RIDGE_NAMES = frozenset({"direct_linear_ridge", "shared_linear_ridge"})


def linear_attributions(pair, mean_pair, coefficients, *, shared=False, intercept=0.0):
    """Return detector-resolved phi and f(mean), in ps.

    For shared Ridge, phi(1,t)=w(t)*(s1-mu1) and
    phi(2,t)=-w(t)*(s2-mu2), so their sum explains w*(s1-s2).
    This is an exact additive decomposition of the same linear model.
    """
    pair = np.asarray(pair, np.float64)
    mean_pair = np.asarray(mean_pair, np.float64)
    coefficients = np.asarray(coefficients, np.float64).reshape(-1)
    if pair.ndim != 3 or pair.shape[1] != 2 or mean_pair.shape != pair.shape[1:]:
        raise ValueError("Expected waveforms [events,2,samples] and background [2,samples]")
    if not np.all(np.isfinite(pair)) or not np.all(np.isfinite(mean_pair)):
        raise ValueError("Linear SHAP inputs must be finite")
    samples = pair.shape[2]
    if shared:
        if coefficients.size != samples:
            raise ValueError("Shared Ridge coefficient dimension mismatch")
        weights = np.stack([coefficients, -coefficients], axis=0)
    else:
        if coefficients.size != 2 * samples:
            raise ValueError("Direct Ridge coefficient dimension mismatch")
        weights = coefficients.reshape(2, samples)
    phi = (pair - mean_pair[None]) * weights[None]
    baseline_prediction = float(np.sum(mean_pair * weights) + intercept)
    unbounded_prediction = baseline_prediction + phi.sum(axis=(1, 2))
    return phi, baseline_prediction, unbounded_prediction


def development_mean(dataset, mode, mask, *, chunk_size=2048):
    """Exact mean of the prepared Development waveforms on the fitted sample grid."""
    if dataset.n_events < 1:
        raise ValueError("Cannot use an empty Development set as SHAP background")
    total = None
    for start in range(0, dataset.n_events, chunk_size):
        indices = np.arange(start, min(start + chunk_size, dataset.n_events))
        values = np.asarray(waveform_view(dataset, mode, indices).materialize(), np.float64)
        selected = values[:, :, mask]
        if total is None:
            total = np.zeros(selected.shape[1:], dtype=np.float64)
        total += np.sum(selected, axis=0, dtype=np.float64)
    return total / dataset.n_events


def grouped_contributions(phi, width):
    """Sum positive and negative SHAP values separately in each time group.

    Returns (starts, positive, negative) with detector axes preserved.
    Positive and negative contributions never cancel within a group; their
    sum is exactly the signed sum of the original sample-level attribution.
    The final partial group is retained.
    """
    if isinstance(width, bool) or int(width) != width or width < 1:
        raise ValueError("group_samples must be a positive integer")
    phi = np.asarray(phi, dtype=np.float64)
    if phi.ndim != 3 or not np.all(np.isfinite(phi)):
        raise ValueError("Expected finite attributions [events,detectors,samples]")
    n = phi.shape[-1]
    if not n:
        raise ValueError("Cannot group empty attributions")
    starts = np.arange(0, n, int(width), dtype=np.int64)
    positive = np.stack(
        [np.maximum(phi[..., start:min(start + width, n)], 0).sum(axis=-1)
         for start in starts], axis=-1,
    )
    negative = np.stack(
        [np.minimum(phi[..., start:min(start + width, n)], 0).sum(axis=-1)
         for start in starts], axis=-1,
    )
    return starts, positive, negative


def _aligned_blind_predictions(store, blind):
    """Use saved predictions, indexed by event ID; no SHA/path equality checks."""
    with np.load(store.predictions_path, allow_pickle=False) as data:
        required = {"event_id", "prediction_ps", "led_residual_ps", "corrected_ps"}
        if not required.issubset(data.files):
            raise ValueError(f"Missing saved prediction fields: {sorted(required - set(data.files))}")
        event_id = np.asarray(data["event_id"], np.int64).reshape(-1)
        prediction = np.asarray(data["prediction_ps"], np.float64).reshape(-1)
        target = np.asarray(data["led_residual_ps"], np.float64).reshape(-1)
        corrected = np.asarray(data["corrected_ps"], np.float64).reshape(-1)
    expected = np.asarray(blind.event_index, np.int64)
    if (event_id.size != expected.size or np.unique(event_id).size != expected.size
            or np.unique(expected).size != expected.size):
        raise ValueError("Saved blind event IDs are not unique or have different counts")
    locations = {int(key): i for i, key in enumerate(event_id)}
    if set(locations) != set(map(int, expected)):
        raise ValueError("Saved blind predictions do not match prepared event IDs")
    idx = [locations[int(key)] for key in expected]
    prediction, target, corrected = prediction[idx], target[idx], corrected[idx]
    if not all(np.all(np.isfinite(a)) for a in (prediction, target, corrected)):
        raise ValueError("Nonfinite saved blind predictions")
    if not np.allclose(target - prediction, corrected, rtol=1e-9, atol=1e-8):
        raise ValueError("Saved corrected residuals differ from target - prediction")
    return prediction, target


def explain_linear_shap_local(
    results=None, *, model_dir=None, models=None, ml_window=None, mode=None,
    cache_root=None, output_root=None, events_per_group=2, extreme_margin=0.05,
    group_samples=8, overwrite=False, logger=None,
):
    """Explain representative blind events from frozen linear Ridge regressors."""
    if isinstance(group_samples, bool) or group_samples < 1:
        raise ValueError("group_samples must be positive")
    allowed = set(models) if models is not None else RIDGE_NAMES
    if not allowed.issubset(RIDGE_NAMES):
        raise ValueError(f"Linear SHAP requires Ridge models: {sorted(allowed - RIDGE_NAMES)}")
    runs, source = local._tshap_run_inputs(results, model_dir)
    selected_window = local._effective_tshap_window(ml_window, model_dir=model_dir)
    output_root = Path(output_root).expanduser().resolve() if output_root is not None else Path.cwd().resolve() / "xai"
    if output_root.is_relative_to(source):
        raise ValueError("XAI output-root must be outside the results directory")
    lookup = PreparedCacheLookup()
    outputs = []
    matched = 0
    for run in runs:
        name = run["manifest"]["model"]
        if name not in allowed:
            continue
        run_directory = Path(run["directory"])
        config = resolve_run_config(run_directory)
        local._apply_tshap_cache(config, cache_root=cache_root)
        if not local._matches_tshap_window(config, selected_window) or not local._matches_tshap_mode(config, mode):
            continue
        matched += 1
        destination = (
            output_root / batch_root(run_directory).name
            / run_directory.relative_to(batch_root(run_directory)) / "linear_shap_local"
        )
        if (destination / "summary.json").is_file() and not overwrite:
            old = json.loads((destination / "summary.json").read_text(encoding="utf-8"))
            if old.get("plot_version") == 3 and old.get("settings") == {
                "events_per_group": events_per_group,
                "extreme_margin": extreme_margin,
                "group_samples": group_samples,
            } and all((destination / p).is_file() for p in old.get("plots", [])):
                outputs.append(destination)
                if logger:
                    logger.info("Linear SHAP local | existing output | %s", destination)
                continue
        spec = get_model(name)
        store = RunStore(run_directory)
        if not saved_model_complete(spec, store.model_dir):
            raise FileNotFoundError(f"Saved Ridge model missing: {store.model_dir}")
        fitted = load_fitted_model(spec, store.model_dir, run["best"]["parameters"], config)
        mask = np.asarray(fitted.sample_mask, bool)
        development = lookup.find("development", config, run["manifest"])
        blind = lookup.find("blind", config, run["manifest"])
        mean_pair = development_mean(development, config["mode"], mask)
        predictions, targets = _aligned_blind_predictions(store, blind)
        selected, improvements, median_prediction = local._select_local_events(
            predictions, targets, events_per_group=events_per_group,
            extreme_margin=extreme_margin,
        )
        indices = np.asarray([row["index"] for row in selected], np.int64)
        waveforms = np.asarray(waveform_view(blind, config["mode"], indices).materialize(), np.float64)
        full_time = np.asarray(
            getattr(blind, f"{mode_family(config['mode'])}_time_ps"), np.float64
        )
        if mask.shape != (full_time.size,):
            raise ValueError("Stored fitted sample mask does not match prepared time grid")
        time_ps = full_time[mask]
        waveforms = waveforms[:, :, mask]
        regressor = fitted.artifact.regressor
        phi, base_prediction, reconstructed = linear_attributions(
            waveforms, mean_pair, regressor.coef_,
            shared=(name == "shared_linear_ridge"),
            intercept=float(regressor.intercept_),
        )
        replay = predict_indices(spec, fitted, blind, config["mode"], indices)
        limit = fitted.output_max_abs_ps
        bounded = np.clip(reconstructed, -limit, limit) if limit is not None else reconstructed
        if not np.allclose(bounded, predictions[indices], atol=0.05, rtol=1e-5):
            raise ValueError("Reconstructed Ridge predictions disagree with saved blind predictions")
        if not np.allclose(bounded, replay, atol=1e-7, rtol=1e-7):
            raise ValueError("Linear SHAP decomposition disagrees with model inference")
        if np.any(np.abs(reconstructed - bounded) > 1e-7):
            raise ValueError("Clipped predictions are not exactly explained by a linear SHAP decomposition")

        family = mode_family(config["mode"])
        normalization = getattr(blind, f"{family}_transform")
        if normalization is None:
            raise ValueError("Missing fitted waveform normalization")
        waveform_mv = np.asarray(normalization.inverse(waveforms), np.float64)
        mean_mv = np.asarray(normalization.inverse(mean_pair[None])[0], np.float64)
        mean_diff_mv = mean_mv[0] - mean_mv[1]
        led_threshold_mv = float(blind.manifest["fixed_led_threshold_mV"])
        led_baselines = local._native_led_baselines(
            blind, config["mode"], indices,
            config["preprocessing"]["selection"]["baseline_window_ns"],
        )
        # Prepare outputs only after all checks have passed.
        destination.mkdir(parents=True, exist_ok=True)
        starts, positive_scores, negative_scores = grouped_contributions(
            phi, int(group_samples),
        )
        if name == "shared_linear_ridge":
            # Preserve contributions of both detector signals before summation:
            # opposite signs never cancel prematurely.
            positive_scores = positive_scores.sum(axis=1, keepdims=True)
            negative_scores = negative_scores.sum(axis=1, keepdims=True)
        plots, records = [], []
        for i, choice in enumerate(selected):
            event_id = int(blind.event_index[indices[i]])
            filename = f"{choice['group']}_event_{event_id}.pdf"
            local._plot_local_event(
                destination / filename, time_ps=time_ps,
                waveform_mv=waveform_mv[i], dataset_mean_difference_mv=mean_diff_mv,
                led_baseline_mv=led_baselines[i], attribution=phi[i],
                group=choice["group"], event_id=event_id,
                target_ps=targets[indices[i]], prediction_ps=predictions[indices[i]],
                improvement_ps=improvements[indices[i]], led_threshold_mv=led_threshold_mv,
                window_starts=starts,
                positive_window_scores=positive_scores[i],
                negative_window_scores=negative_scores[i],
                window_samples=int(group_samples),
                attribution_method="Linear SHAP",
                separate_detector_importance=(name == "direct_linear_ridge"),
                highlight_time_ranges_ns=((-1.5, 2.0),) if str(config.get("window_name", "")).lower() == "wide" else None,
            )
            plots.append(filename)
            records.append({
                "group": choice["group"], "event_id": event_id,
                "dataset_index": int(indices[i]),
                "target_correction_ps": float(targets[indices[i]]),
                "predicted_correction_ps": float(predictions[indices[i]]),
                "final_residual_ps": float(targets[indices[i]] - predictions[indices[i]]),
                "signed_improvement_ps": float(improvements[indices[i]]),
                "base_value_ps": float(base_prediction),
                "phi_sum_ps": float(phi[i].sum()),
                "reconstructed_prediction_ps": float(reconstructed[i]),
            })
        np.savez_compressed(
            destination / "local_attributions.npz",
            event_id=np.asarray([row["event_id"] for row in records]),
            dataset_index=indices,
            time_ps=time_ps,
            waveform_normalized=waveforms,
            waveform_mv=waveform_mv,
            background_mean_normalized=mean_pair,
            attribution_ps_per_sample=phi,
            group_starts=starts,
            group_positive_scores_ps=positive_scores,
            group_negative_scores_ps=negative_scores,
            group_scores_ps=positive_scores + negative_scores,
            baseline_prediction_ps=np.asarray(base_prediction),
        )
        with (destination / "events.csv").open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(records[0]))
            writer.writeheader()
            writer.writerows(records)
        atomic_json(destination / "summary.json", {
            "method": "exact_interventional_linear_shap",
            "plot_version": 3,
            "grouping": "sum_positive_and_negative_separately",
            "model": name,
            "background": "complete_prepared_development_mean",
            "mode": config["mode"],
            "window": config["window_name"],
            "n_explained_events": len(records),
            "median_predicted_correction_ps": float(median_prediction),
            "settings": {
                "events_per_group": events_per_group,
                "extreme_margin": extreme_margin,
                "group_samples": group_samples,
            },
            "plots": plots,
        })
        outputs.append(destination)
        if logger:
            logger.info("Linear SHAP local | %s | %d events | %s", name, len(records), destination)
    if not matched:
        raise ValueError("No matching saved Ridge runs found")
    return outputs


def _random_blind_indices(n_events, max_events, seed):
    """Uniform, seeded sampling of the prepared blind population without replacement."""
    if isinstance(max_events, bool) or not isinstance(max_events, int) or max_events < 1:
        raise ValueError("max_events must be a positive integer")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("seed must be an integer")
    if n_events < 1:
        raise ValueError("Blind dataset contains no prepared events")
    return np.sort(
        np.random.default_rng(seed).choice(
            n_events, size=min(max_events, n_events), replace=False
        )
    )


def _global_mean_contributions(phi, group_samples, *, shared):
    """Event-wise sign separation, then averaging, in ps per event."""
    starts, positive, negative = grouped_contributions(phi, group_samples)
    if shared:
        # Sum detectors only AFTER splitting per-sample signs, as in local SHAP.
        positive = positive.sum(axis=1, keepdims=True)
        negative = negative.sum(axis=1, keepdims=True)
    return starts, positive.mean(axis=0), negative.mean(axis=0)


def _plot_global_linear_shap(
    path, *, time_ps, mean_waveform_mv, starts, group_samples,
    mean_positive_ps, mean_negative_ps, model_name, n_events, window_name,
):
    """Global counterpart of local Linear SHAP: waveforms + signed importance."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    time_ps = np.asarray(time_ps, dtype=np.float64)
    time_ns = time_ps / 1000.0
    means = np.asarray(mean_waveform_mv, dtype=np.float64)
    positive = np.asarray(mean_positive_ps, dtype=np.float64)
    negative = np.asarray(mean_negative_ps, dtype=np.float64)
    starts = np.asarray(starts, dtype=np.int64)
    shared = model_name == "shared_linear_ridge"
    n_panels = 2 if shared else 3
    expected = (n_panels - 1, starts.size)
    if means.shape != (2, time_ps.size) or positive.shape != expected or negative.shape != expected:
        raise ValueError("Global Linear SHAP waveform or contribution dimensions are inconsistent")
    if time_ps.size < 2 or not np.all(np.diff(time_ps) > 0):
        raise ValueError("Global Linear SHAP requires an increasing waveform time grid")
    if not all(np.all(np.isfinite(a)) for a in (time_ps, means, positive, negative)):
        raise ValueError("Global Linear SHAP plot contains nonfinite data")

    fig, axes = plt.subplots(
        n_panels, 1, figsize=(12.5, 3.1 + 2.4 * (n_panels - 1)),
        sharex=True, constrained_layout=True,
        gridspec_kw={"height_ratios": [1.0] + [1.05] * (n_panels - 1)},
    )
    fig.suptitle(
        f"Global Linear SHAP — {model_name.replace('_', ' ').title()}"
        f" | {n_events} random blind events", fontsize=11,
    )
    axes[0].plot(time_ns, means[0], linewidth=1.15, label="Detector 1")
    axes[0].plot(time_ns, means[1], linewidth=1.15, label="Detector 2")
    axes[0].set_title("Mean detector waveforms")
    axes[0].set_ylabel("Amplitude (mV)")
    axes[0].grid(True, alpha=0.22, linewidth=0.6)
    axes[0].legend(loc="best", fontsize=9)

    dt_ns = float(np.median(np.diff(time_ns)))
    widths = np.minimum(int(group_samples), time_ns.size - starts)
    centers = time_ns[starts] + 0.5 * (widths - 1) * dt_ns
    widths_ns = widths * dt_ns
    for channel in range(n_panels - 1):
        ax = axes[channel + 1]
        ax.bar(centers, positive[channel], width=0.9 * widths_ns,
               color="#4c78a8", label="Mean positive SHAP")
        ax.bar(centers, negative[channel], width=0.9 * widths_ns,
               color="#e45756", label="Mean negative SHAP")
        ax.axhline(0.0, color="0.4", linewidth=0.7)
        ax.set_title(
            "Pair Linear SHAP" if shared
            else f"Detector {channel + 1} Linear SHAP"
        )
        ax.set_ylabel("Mean contribution (ps/event)")
        ax.grid(True, alpha=0.22, linewidth=0.6)
        ax.legend(loc="best", fontsize=9)
    axes[-1].set_xlabel("Waveform time (ns)")
    if str(window_name).lower() == "wide":
        for ax in axes:
            ax.axvspan(-1.5, 2.0, color="0.75", alpha=0.25, linewidth=0, zorder=0)
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def explain_linear_shap_global(
    results=None, *, model_dir=None, models=None, ml_window=None, mode=None,
    cache_root=None, output_root=None, max_events=256, seed=42,
    group_samples=8, overwrite=False, logger=None,
):
    """Average exact Ridge SHAP contributions over random prepared blind events.

    The background is the complete Development mean. Every eligible blind event
    is equally likely to be sampled; no prediction/error-based selection occurs.
    """
    if isinstance(group_samples, bool) or not isinstance(group_samples, int) or group_samples < 1:
        raise ValueError("group_samples must be a positive integer")
    _random_blind_indices(1, max_events, seed)  # Validate settings before I/O.
    allowed = set(models) if models is not None else RIDGE_NAMES
    if not allowed.issubset(RIDGE_NAMES):
        raise ValueError(f"Linear SHAP requires Ridge models: {sorted(allowed - RIDGE_NAMES)}")
    runs, source = local._tshap_run_inputs(results, model_dir)
    selected_window = local._effective_tshap_window(ml_window, model_dir=model_dir)
    output_root = (
        Path(output_root).expanduser().resolve() if output_root is not None
        else Path.cwd().resolve() / "xai"
    )
    if output_root.is_relative_to(source):
        raise ValueError("XAI output-root must be outside the results directory")
    lookup = PreparedCacheLookup()
    outputs = []
    matched = 0
    for run in runs:
        name = run["manifest"]["model"]
        if name not in allowed:
            continue
        directory = Path(run["directory"])
        config = resolve_run_config(directory)
        local._apply_tshap_cache(config, cache_root=cache_root)
        if (not local._matches_tshap_window(config, selected_window)
                or not local._matches_tshap_mode(config, mode)):
            continue
        matched += 1
        destination = (
            output_root / batch_root(directory).name
            / directory.relative_to(batch_root(directory)) / "linear_shap_global"
        )
        settings = {
            "max_events": max_events, "seed": seed, "group_samples": group_samples,
        }
        if (destination / "summary.json").is_file() and not overwrite:
            old = json.loads((destination / "summary.json").read_text(encoding="utf-8"))
            if (old.get("plot_version") == 1
                    and old.get("settings") == settings
                    and all((destination / filename).is_file() for filename in (
                        "global_importance.pdf", "global_attributions.npz",
                        "global_importance.csv",
                    ))):
                outputs.append(destination)
                if logger:
                    logger.info("Linear SHAP global | existing output | %s", destination)
                continue

        spec = get_model(name)
        store = RunStore(directory)
        if not saved_model_complete(spec, store.model_dir):
            raise FileNotFoundError(f"Saved Ridge model missing: {store.model_dir}")
        fitted = load_fitted_model(spec, store.model_dir, run["best"]["parameters"], config)
        development = lookup.find("development", config, run["manifest"])
        blind = lookup.find("blind", config, run["manifest"])
        mask = np.asarray(fitted.sample_mask, dtype=bool)
        mean_pair = development_mean(development, config["mode"], mask)
        indices = _random_blind_indices(blind.n_events, max_events, seed)
        waveforms = np.asarray(
            waveform_view(blind, config["mode"], indices).materialize(), np.float64
        )
        family = mode_family(config["mode"])
        full_time = np.asarray(getattr(blind, f"{family}_time_ps"), np.float64)
        if mask.shape != (full_time.size,):
            raise ValueError("Stored fitted sample mask does not match prepared time grid")
        time_ps = full_time[mask]
        waveforms = waveforms[:, :, mask]
        regressor = fitted.artifact.regressor
        phi, baseline_prediction, reconstructed = linear_attributions(
            waveforms, mean_pair, regressor.coef_,
            shared=(name == "shared_linear_ridge"),
            intercept=float(regressor.intercept_),
        )
        # Clipping is a post-model pipeline operation: explain the unbounded
        # linear estimator; verify its clipped output against saved blind results.
        saved, _ = _aligned_blind_predictions(store, blind)
        replay = predict_indices(spec, fitted, blind, config["mode"], indices)
        limit = fitted.output_max_abs_ps
        bounded = np.clip(reconstructed, -limit, limit) if limit is not None else reconstructed
        if not np.allclose(bounded, saved[indices], atol=0.05, rtol=1e-5):
            raise ValueError("Global Linear SHAP disagrees with saved blind predictions")
        if not np.allclose(bounded, replay, atol=1e-7, rtol=1e-7):
            raise ValueError("Global Linear SHAP disagrees with model inference")

        normalization = getattr(blind, f"{family}_transform")
        if normalization is None:
            raise ValueError("Missing fitted waveform normalization")
        mean_waveform_mv = np.asarray(
            normalization.inverse(waveforms).mean(axis=0), np.float64
        )
        starts, positive, negative = _global_mean_contributions(
            phi, int(group_samples), shared=(name == "shared_linear_ridge"),
        )
        # Write only after the model, waveform grid and prediction checks pass.
        destination.mkdir(parents=True, exist_ok=True)
        _plot_global_linear_shap(
            destination / "global_importance.pdf",
            time_ps=time_ps, mean_waveform_mv=mean_waveform_mv,
            starts=starts, group_samples=group_samples,
            mean_positive_ps=positive, mean_negative_ps=negative,
            model_name=name, n_events=len(indices),
            window_name=config["window_name"],
        )
        np.savez_compressed(
            destination / "global_attributions.npz",
            dataset_index=indices,
            event_id=np.asarray(blind.event_index[indices], dtype=np.int64),
            time_ps=time_ps,
            mean_waveform_mv=mean_waveform_mv,
            background_mean_normalized=mean_pair,
            group_starts=starts,
            mean_positive_contribution_ps=positive,
            mean_negative_contribution_ps=negative,
            mean_signed_contribution_ps=positive + negative,
            baseline_prediction_ps=np.asarray(baseline_prediction),
            mean_unbounded_prediction_ps=np.asarray(reconstructed.mean()),
            n_clipped_events=np.asarray(np.count_nonzero(np.abs(reconstructed - bounded) > 1e-7)),
        )
        dt = float(np.median(np.diff(time_ps)))
        with (destination / "global_importance.csv").open(
            "w", newline="", encoding="utf-8"
        ) as stream:
            writer = csv.writer(stream)
            writer.writerow([
                "detector", "start_ns", "end_ns",
                "mean_positive_ps", "mean_negative_ps", "mean_signed_ps",
            ])
            for channel in range(positive.shape[0]):
                for group, start in enumerate(starts):
                    end = min(int(start) + group_samples, time_ps.size)
                    writer.writerow([
                        "pair" if name == "shared_linear_ridge" else f"detector_{channel + 1}",
                        float(time_ps[start] / 1000),
                        float((time_ps[start] + (end - start) * dt) / 1000),
                        float(positive[channel, group]),
                        float(negative[channel, group]),
                        float(positive[channel, group] + negative[channel, group]),
                    ])
        atomic_json(destination / "summary.json", {
            "method": "exact_interventional_linear_shap_global",
            "plot_version": 1,
            "model": name,
            "mode": config["mode"],
            "window": config["window_name"],
            "background": "complete_prepared_development_mean",
            "population": "prepared_blind_events",
            "selection": "uniform_random_without_replacement_no_quality_selection",
            "aggregation": "sum_positive_negative_per_event_then_mean_across_events",
            "contribution_units": "ps_per_event",
            "explained_prediction": "unbounded_linear_regressor_before_pipeline_clipping",
            "n_clipped_events": int(np.count_nonzero(np.abs(reconstructed - bounded) > 1e-7)),
            "n_explained_events": int(len(indices)),
            "settings": settings,
            "plots": ["global_importance.pdf"],
        })
        outputs.append(destination)
        if logger:
            logger.info(
                "Linear SHAP global | %s | %d random blind events | %s",
                name, len(indices), destination,
            )
    if not matched:
        raise ValueError("No matching saved Ridge runs found")
    return outputs
