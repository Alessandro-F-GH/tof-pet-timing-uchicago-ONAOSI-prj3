"""Exact interventional local Linear SHAP for saved direct/shared Ridge models.

Uses the Development mean as background, the existing TSHAP local-event
selection, and its common waveform/window-attribution plot. Never retrains.
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
    """Nonoverlapping signed sums; final partial window is retained."""
    if isinstance(width, bool) or int(width) != width or width < 1:
        raise ValueError("group_samples must be a positive integer")
    n = phi.shape[-1]
    starts = np.arange(0, n, int(width), dtype=np.int64)
    scores = np.stack(
        [np.sum(phi[..., start:min(start + width, n)], axis=-1)
         for start in starts],
        axis=-1,
    )
    return starts, scores


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
            if old.get("settings") == {
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
        starts, scores = grouped_contributions(
            phi if name == "direct_linear_ridge" else phi.sum(axis=1, keepdims=True),
            int(group_samples),
        )
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
                window_starts=starts, window_scores=scores[i], window_samples=int(group_samples),
                attribution_method="Linear SHAP",
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
            group_scores_ps=scores,
            baseline_prediction_ps=np.asarray(base_prediction),
        )
        with (destination / "events.csv").open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(records[0]))
            writer.writeheader()
            writer.writerows(records)
        atomic_json(destination / "summary.json", {
            "method": "exact_interventional_linear_shap",
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
