"""Conditional-mean KNN TSHAP for frozen TOF-PET regressors.

The value function is hybrid by design: v(empty)=mean f(reference), and
v(S)=f(KNN-conditional-mean-imputed waveform) for proper nonempty coalitions.
A separate KNN weighted mean of reference predictions is diagnostic only.
"""
from __future__ import annotations

import csv
import json
import logging
from pathlib import Path
import shutil
import tempfile

import numpy as np

from waveform_analysis.core.io import atomic_json, canonical_hash
from waveform_analysis.core.artifact_identity import ProductRecipe
from waveform_analysis.data.artifact_store import ArtifactStore
from waveform_analysis.data.prepared_cache import PreparedCacheLookup
from waveform_analysis.data.results import batch_root, discover_runs, read_run, resolve_run_config
from waveform_analysis.data.storage import RunStore
from waveform_analysis.data.view import mode_family, waveform_view
from waveform_analysis.engine.train import load_fitted_model, predict_indices, release_training_memory, saved_model_complete
from waveform_analysis.engine.diagnostics import diagnostic_random_state
from waveform_analysis.engine.conditional_knn_imputer import (
    IMPUTER_VERSION, PairedWindowKNN, KNNSelection, WEIGHT_NAMES,
    digest, reference_validation_split,
)
from waveform_analysis.engine import tshap_explainer as base
from waveform_analysis.models import get_model

METHOD_VERSION = 1
DEFAULT_K = (1, 3, 5, 10, 20, 50, 100)


def _metrics(error):
    error = np.asarray(error, dtype=np.float64)
    return {
        "count": int(error.size),
        "mean_ps": float(np.mean(error)),
        "std_ps": float(np.std(error)),
        "rmse_ps": float(np.sqrt(np.mean(error**2))),
        "mae_ps": float(np.mean(np.abs(error))),
        "p05_ps": float(np.percentile(error, 5)),
        "p50_ps": float(np.median(error)),
        "p95_ps": float(np.percentile(error, 95)),
    }


def _pair_map(scores, starts, width, stride, retained, length):
    """Use the same temporal interpolation and redistribution as upstream TSHAP."""
    scores = np.asarray(scores, dtype=np.float64)
    starts = np.asarray(starts, dtype=np.int64)
    valid_at_start = np.asarray([
        bool(np.all(retained[s:min(s + width, length)])) for s in range(length)
    ])
    line = np.zeros(length, dtype=np.float64)
    for j, (start, score) in enumerate(zip(starts, scores)):
        line[start] = score
        if j + 1 < len(starts):
            next_start = starts[j + 1]
            if next_start - start == stride and valid_at_start[start:next_start + 1].all():
                line[start:next_start + 1] = np.linspace(score, scores[j + 1], next_start - start + 1)
    importance = np.zeros(length, dtype=np.float64)
    for start in range(length):
        if not valid_at_start[start] or line[start] == 0:
            continue
        importance[start:min(length, start + width)] += (
            line[start] / (width * min(start + 1, stride))
        )
    return np.stack((importance, importance))


def _save_csv(path, headings, rows):
    with Path(path).open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(headings)
        writer.writerows(rows)


def _plot_diagnostics(path, time_ps, scores, difference_window, difference_complement):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    t = np.asarray(time_ps) / 1000
    fig, axes = plt.subplots(2, 1, figsize=(11, 6), constrained_layout=True)
    axes[0].plot(t, scores, label="Mean |Shapley| (ps/sample)")
    axes[0].set_ylabel("Attribution (ps/sample)")
    axes[0].legend()
    centers = t[np.asarray(difference_window["starts"], dtype=int)]
    for key, array in (("W observed", difference_window["values"]),
                       ("complement observed", difference_complement["values"])):
        values = np.asarray(array, dtype=np.float64)
        axes[1].plot(centers, np.sqrt(np.mean(values**2, axis=0)), label=f"{key}: prediction gap RMSE")
    axes[1].set_xlabel("Waveform time (ns)")
    axes[1].set_ylabel("Model vs KNN mean (ps)")
    axes[1].legend()
    for ax in axes:
        ax.grid(alpha=0.2)
    fig.savefig(path, dpi=170)
    plt.close(fig)


def _load_tuning(store, recipe, engine, validation, logger):
    def build(directory):
        tuned = engine.tune(validation, logger=logger)
        np.savez_compressed(
            directory / "tuning.npz",
            ks=tuned.ks, selected_k=tuned.selected_k,
            selected_weight=tuned.selected_weight,
            mse=tuned.mse, mae=tuned.mae,
            mse_mv2=tuned.mse_mv2, mae_mv=tuned.mae_mv,
        )
        atomic_json(directory / "tuning_summary.json", {
            "method": "deterministic_knn_conditional_mean",
            "configuration": tuned.summary(engine),
        })
    artifact = store.get_or_build(recipe, build)
    with np.load(artifact.payload_path("tuning.npz"), allow_pickle=False) as a:
        tuned = KNNSelection(*(np.asarray(a[k]) for k in (
            "ks", "selected_k", "selected_weight", "mse", "mae", "mse_mv2", "mae_mv"
        )))
    if logger:
        logger.info("KNN TUNING ARTIFACT | id=%s | path=%s", artifact.product_id, artifact.path)
        for item in tuned.summary(engine):
            logger.info(
                "KNN TUNED | observed=%s | window=%d | k=%d | weight=%s | "
                "MSE=%.8g RMSE=%.6g MAE=%.6g | RMSE_mV=%.6g MAE_mV=%.6g",
                item["observed"], item["window_index"], item["k"], item["weighting"],
                item["mse"], item["rmse"], item["mae"], item["rmse_mv"], item["mae_mv"]
            )
    return tuned, artifact


def _evaluate_events(engine, tuned, waveforms, predict, reference_predictions, logger, event_ids, stride):
    n = len(waveforms)
    nwin = engine.n_windows
    maps = np.zeros((n, 2, engine.reference.shape[-1]), dtype=np.float64)
    scores = np.zeros((n, nwin), dtype=np.float64)
    imputed_prediction = np.zeros((n, 2, nwin), dtype=np.float64)
    neighbor_prediction = np.zeros_like(imputed_prediction)
    original_prediction = np.zeros(n, dtype=np.float64)
    mean_prediction = float(np.mean(reference_predictions))
    recon_mse = np.zeros((n, 2, nwin), dtype=np.float64)
    recon_mae = np.zeros_like(recon_mse)
    recon_rmse_mv = np.zeros_like(recon_mse)
    recon_mae_mv = np.zeros_like(recon_mse)
    for event, waveform in enumerate(waveforms):
        neighbor_ids, distances = engine.search(waveform)
        synthetic = np.empty((engine.n_masks, *waveform.shape), dtype=np.float32)
        knn_predictions = np.empty(engine.n_masks, dtype=np.float64)
        for mask in range(engine.n_masks):
            synthetic[mask], chosen, weights = engine.reconstruct_selected(
                waveform, neighbor_ids, distances, tuned, mask
            )
            knn_predictions[mask] = float(weights @ reference_predictions[chosen])
            hidden = engine.hidden[mask]
            residual = (synthetic[mask][:, hidden] - waveform[:, hidden]).astype(np.float64)
            physical = residual * engine.scale_mv[:, hidden]
            side, window = divmod(mask, nwin)
            recon_mse[event, side, window] = np.mean(residual**2)
            recon_mae[event, side, window] = np.mean(np.abs(residual))
            recon_rmse_mv[event, side, window] = np.sqrt(np.mean(physical**2))
            recon_mae_mv[event, side, window] = np.mean(np.abs(physical))
        estimates = predict(synthetic)
        full = float(predict(waveform[None])[0])
        original_prediction[event] = full
        imputed_prediction[event, 0] = estimates[:nwin]
        imputed_prediction[event, 1] = estimates[nwin:]
        neighbor_prediction[event, 0] = knn_predictions[:nwin]
        neighbor_prediction[event, 1] = knn_predictions[nwin:]
        scores[event] = .5 * (full - estimates[nwin:] + estimates[:nwin] - mean_prediction)
        maps[event] = _pair_map(
            scores[event], engine.starts, engine.ends[0]-engine.starts[0],
            int(stride),
            engine.retained, waveform.shape[-1]
        )
        if logger and (event == 0 or (event+1)%10 == 0 or event+1 == n):
            logger.info(
                "KNN TSHAP | event=%d/%d | event_id=%s | f_original=%.6f ps | "
                "f_reference_mean=%.6f ps | mean_abs_phi=%.6f ps | "
                "prediction_gap_RMSE_W=%.6f ps | prediction_gap_RMSE_complement=%.6f ps",
                event+1, n, event_ids[event], full, mean_prediction,
                float(np.mean(np.abs(scores[event]))),
                float(np.sqrt(np.mean((estimates[:nwin]-knn_predictions[:nwin])**2))),
                float(np.sqrt(np.mean((estimates[nwin:]-knn_predictions[nwin:])**2)))
            )
    return {
        "maps": maps, "scores": scores, "f_original": original_prediction,
        "f_reference_mean": mean_prediction, "f_imputed": imputed_prediction,
        "f_neighbor_mean": neighbor_prediction,
        "gap": imputed_prediction-neighbor_prediction,
        "recon_mse": recon_mse, "recon_mae": recon_mae,
        "recon_rmse_mv": recon_rmse_mv, "recon_mae_mv": recon_mae_mv,
    }


def explain_conditional_knn(
    results=None, *, model_dir=None, models=None, ml_window=None, mode=None,
    cache_root=None, artifact_root=None, output_root=None, local=False,
    max_events=100, events_per_group=2, extreme_margin=.05,
    window_samples=20, stride_samples=10, predict_batch_size=64,
    seed=42, overwrite=False, knn_k=DEFAULT_K, knn_tuning_events=512,
    knn_reference_events=None, knn_reference_block=512, logger=None,
):
    """Compute global or local conditional-imputation TSHAP using saved models."""
    logger = logger or logging.getLogger(__name__)
    if (model_dir is None) == (results is None):
        raise ValueError("Supply exactly one of results and model_dir")
    if not knn_k or any(int(k) < 1 for k in knn_k):
        raise ValueError("KNN candidate k values must be positive")
    if int(knn_tuning_events) < 1 or int(window_samples) < 1 or int(stride_samples) < 1:
        raise ValueError("KNN tuning size, window and stride must be positive")
    if local and not 0 < extreme_margin < .25:
        raise ValueError("Local extreme_margin must be in (0, 0.25)")
    source = Path(model_dir or results).expanduser().resolve()
    output_root = Path(output_root or Path.cwd() / "xai").expanduser().resolve()
    if output_root == source or output_root.is_relative_to(source):
        raise ValueError("XAI output must be outside results/model directory")
    runs, _ = base._tshap_run_inputs(results, model_dir)
    selected_window = base._effective_tshap_window(ml_window, model_dir=model_dir)
    lookup = PreparedCacheLookup()
    completed, failures = [], []
    for run in runs:
        manifest = run["manifest"]
        directory = Path(run["directory"])
        if models is not None and manifest["model"] not in models:
            continue
        try:
            config = resolve_run_config(directory)
            base._apply_tshap_cache(config, cache_root=cache_root, artifact_root=artifact_root)
            if not base._matches_tshap_window(config, selected_window) or not base._matches_tshap_mode(config, mode):
                continue
            spec = get_model(manifest["model"])
            run_store = RunStore(directory, artifact_root=config["preprocessing"]["artifact_root"])
            model_path = base._resolve_tshap_model_directory(
                spec, run_store, directory, explicit_model_dir=model_dir is not None
            )
            if not saved_model_complete(spec, model_path):
                raise FileNotFoundError(f"Missing frozen fitted model: {model_path}")
            development = lookup.find("development", config, manifest)
            blind = lookup.find("blind", config, manifest)
            time_ps = base._check_prepared_grids(development, blind, config["mode"])
            family = mode_family(config["mode"])
            scale = getattr(development, f"{family}_transform")
            if scale is None:
                raise ValueError("Missing frozen physical waveform transform")
            scale_values = np.asarray(scale.maximum-scale.minimum, dtype=np.float64)
            if scale_values.shape == (2,):
                scale_values = scale_values[:, None]
            frozen_scale = np.broadcast_to(scale_values, (2, len(time_ps)))
            with diagnostic_random_state():
                fitted = load_fitted_model(spec, model_path, run["best"]["parameters"], config)
                try:
                    keep = np.ones(len(time_ps), dtype=bool) if fitted.sample_mask is None else np.asarray(fitted.sample_mask, dtype=bool)
                    starts, _, _ = base._valid_window_starts(len(time_ps), window_samples, stride_samples, keep)
                    if not starts:
                        raise ValueError("No retained TSHAP windows for the selected mask")
                    reference_idx, validation_idx = reference_validation_split(
                        development.n_events, knn_tuning_events, knn_reference_events, seed
                    )
                    if max(knn_k) > len(reference_idx):
                        raise ValueError(f"Maximum candidate k={max(knn_k)} exceeds reference count={len(reference_idx)}")
                    reference_waves = waveform_view(development, config["mode"], reference_idx).materialize()
                    validation_waves = waveform_view(development, config["mode"], validation_idx).materialize()
                    engine = PairedWindowKNN(
                        reference_waves, starts, window_samples, keep, knn_k,
                        scale_mv=frozen_scale, reference_block=knn_reference_block,
                    )
                    recipe = ProductRecipe(
                        "knn-conditional-tuning",
                        {
                            "version": IMPUTER_VERSION,
                            "development_protocol": development.manifest.get("analysis_protocol_identity"),
                            "reference_event_hash": digest(np.asarray(development.event_index[reference_idx])),
                            "validation_event_hash": digest(np.asarray(development.event_index[validation_idx])),
                            "reference_waveform_hash": digest(reference_waves),
                            "validation_waveform_hash": digest(validation_waves),
                            "retained_mask_hash": digest(keep),
                            "transform_min_hash": digest(scale.minimum),
                            "transform_max_hash": digest(scale.maximum),
                            "starts": list(map(int, starts)),
                            "width": int(window_samples), "stride": int(stride_samples),
                            "k_candidates": list(map(int, sorted(set(knn_k)))),
                            "metric": "hidden_normalized_MSE",
                            "weights": list(WEIGHT_NAMES),
                        },
                        implementation=f"paired-knn-conditional-v{IMPUTER_VERSION}",
                    )
                    store = ArtifactStore(config["preprocessing"]["artifact_root"])
                    logger.info(
                        "KNN SETUP | run=%s | family=%s | reference=%d | validation=%d | "
                        "windows=%d | k=%s | window_samples=%d | stride_samples=%d | retained_samples=%d",
                        directory, family, len(reference_idx), len(validation_idx),
                        len(starts), sorted(set(knn_k)), window_samples, stride_samples, int(keep.sum())
                    )
                    tuned, tuned_artifact = _load_tuning(
                        store, recipe, engine, validation_waves, logger
                    )
                    suffix = "tshap_conditional_knn_local" if local else "tshap_conditional_knn"
                    destination = base._explanation_destination(output_root, directory).parent / suffix
                    identity = {
                        "method": "paired_conditional_knn_mean_imputation",
                        "version": METHOD_VERSION,
                        "final_fit": manifest["stage_fingerprints"]["final_fit"],
                        "blind": manifest["dataset_populations"]["blind"]["analysis_protocol_identity"],
                        "tuning_artifact": tuned_artifact.product_id,
                        "model": spec.name, "mode": config["mode"], "seed": int(seed),
                        "max_events": None if local else int(max_events),
                        "events_per_group": int(events_per_group) if local else None,
                        "extreme_margin": float(extreme_margin) if local else None,
                        "predict_batch_size": int(predict_batch_size),
                    }
                    fingerprint = canonical_hash(identity)
                    if destination.exists() and not overwrite:
                        summary_path = destination / "summary.json"
                        if summary_path.is_file():
                            cached = json.loads(summary_path.read_text(encoding="utf-8"))
                            if (cached.get("identity_sha256") == fingerprint
                                    and (destination / ("local_attributions.npz" if local else "attributions.npz")).is_file()
                                    and (destination / "knn_diagnostics_per_window.csv").is_file()
                                    and (destination / "knn_tuning_selected.csv").is_file()
                                    and (not local or all((destination / name).is_file()
                                                          for name in cached.get("plots", [])))
                                    and (local or (destination / "global_importance.png").is_file())):
                                logger.info("KNN TSHAP | cache hit | %s", destination)
                                completed.append(destination)
                                continue
                    if local:
                        saved, targets = base._load_materialized_local_predictions(
                            run_store, blind, manifest, config["mode"]
                        )
                        selected, _, _ = base._select_local_events(
                            saved, targets, events_per_group=events_per_group, extreme_margin=extreme_margin
                        )
                        ids = np.asarray([item["index"] for item in selected], dtype=np.int64)
                        groups = [item["group"] for item in selected]
                    else:
                        ids = base._choose_indices(blind.n_events, max_events, base._selection_seed(seed))
                        groups = None
                    waves = waveform_view(blind, config["mode"], ids).materialize()
                    predict = lambda v: base._predict_bounded(spec, fitted, v, predict_batch_size)
                    replay = predict(waves)
                    if local:
                        if not np.allclose(replay, saved[ids],
                                           atol=base.LOCAL_PREDICTION_REPLAY_ATOL_PS,
                                           rtol=base.LOCAL_PREDICTION_REPLAY_RTOL):
                            raise ValueError("Conditional KNN TSHAP replay mismatch with saved blind predictions")
                    else:
                        original = predict_indices(spec, fitted, blind, config["mode"], ids,
                                                   chunk_size=predict_batch_size)
                        if not np.allclose(replay, original, atol=1e-4, rtol=1e-6):
                            raise ValueError("Conditional KNN TSHAP replay mismatch with saved model inference")
                    event_ids = np.asarray(blind.event_index[ids], dtype=np.int64)
                    reference_predictions = predict_indices(
                        spec, fitted, development, config["mode"], reference_idx,
                        chunk_size=predict_batch_size
                    )
                    if not np.isfinite(reference_predictions).all():
                        raise ValueError("Non-finite reference model predictions")
                    values = _evaluate_events(engine, tuned, waves, predict, reference_predictions,
                                              logger, event_ids, stride_samples)
                finally:
                    del fitted
                    release_training_memory()
            destination.parent.mkdir(parents=True, exist_ok=True)
            temporary = Path(tempfile.mkdtemp(prefix=".knn-tshap-", dir=destination.parent))
            try:
                scores = values["scores"]
                gaps = values["gap"]
                metadata = {
                    **identity, "identity_sha256": fingerprint,
                    "n_events": int(len(ids)), "n_windows": int(len(starts)),
                    "reference_events": int(len(reference_idx)),
                    "validation_events": int(len(validation_idx)),
                    "reference_prediction_mean_ps": values["f_reference_mean"],
                    "conditional_value_definition": "v(empty)=mean f(reference); v(S)=f(KNN weighted-mean imputation)",
                    "knn_prediction_mean_diagnostic": "sum neighbor weights * f(reference neighbor)",
                    "tuning_selection": tuned.summary(engine),
                    "prediction_gap_metrics": {
                        "window": _metrics(gaps[:,0,:]),
                        "complement": _metrics(gaps[:,1,:]),
                    },
                    "blind_hidden_reconstruction": {
                        "window_observed": {
                            "mse_normalized": float(np.mean(values["recon_mse"][:,0,:])),
                            "mae_normalized": float(np.mean(values["recon_mae"][:,0,:])),
                            "rmse_mv": float(np.sqrt(np.mean(values["recon_rmse_mv"][:,0,:]**2))),
                            "mae_mv": float(np.mean(values["recon_mae_mv"][:,0,:])),
                        },
                        "complement_observed": {
                            "mse_normalized": float(np.mean(values["recon_mse"][:,1,:])),
                            "mae_normalized": float(np.mean(values["recon_mae"][:,1,:])),
                            "rmse_mv": float(np.sqrt(np.mean(values["recon_rmse_mv"][:,1,:]**2))),
                            "mae_mv": float(np.mean(values["recon_mae_mv"][:,1,:])),
                        },
                    },
                }
                logger.info("KNN TSHAP METRICS | run=%s | prediction_gap_W=%s | prediction_gap_complement=%s",
                            directory, metadata["prediction_gap_metrics"]["window"],
                            metadata["prediction_gap_metrics"]["complement"])
                logger.info("KNN TSHAP IMPUTATION | run=%s | metrics=%s",
                            directory, metadata["blind_hidden_reconstruction"])
                if local:
                    metadata["groups"] = groups
                arrays = {
                    "event_id": event_ids, "dataset_index": ids, "time_ps": time_ps,
                    "window_starts": np.asarray(starts,dtype=np.int64),
                    "window_samples": np.asarray(window_samples,dtype=np.int64),
                    "window_attribution_ps": values["maps"],
                    "raw_window_scores_ps": scores[:,None,:],
                    "original_prediction_ps": values["f_original"],
                    "reference_prediction_mean_ps": np.asarray(values["f_reference_mean"]),
                    "f_imputed_ps": values["f_imputed"],
                    "knn_mean_prediction_ps": values["f_neighbor_mean"],
                    "prediction_gap_ps": values["gap"],
                    "reconstruction_mse_normalized": values["recon_mse"],
                    "reconstruction_mae_normalized": values["recon_mae"],
                    "reconstruction_rmse_mv": values["recon_rmse_mv"],
                    "reconstruction_mae_mv": values["recon_mae_mv"],
                    "selected_k": tuned.selected_k,
                    "selected_weighting": tuned.selected_weight,
                }
                np.savez_compressed(temporary / ("local_attributions.npz" if local else "attributions.npz"),**arrays)
                if local:
                    raw_mv = np.asarray(scale.inverse(waves),dtype=np.float64)
                    difference_mean, _ = base._dataset_mean_difference_mv(
                        blind, config["mode"], scale, int(seed), max_events=1000
                    )
                    led_baselines = base._native_led_baselines(
                        config["preprocessing"]["artifact_root"], blind, config["mode"],
                        ids, config["preprocessing"]["selection"]["baseline_window_ns"]
                    )
                    threshold = float(blind.manifest["fixed_led_threshold_mV"])
                    names, event_rows = [], []
                    for i,(group,event_id) in enumerate(zip(groups,event_ids)):
                        filename = f"{group}__event_{event_id}.png"
                        base._plot_local_event(
                            temporary / filename, time_ps=time_ps,
                            waveform_mv=raw_mv[i],
                            dataset_mean_difference_mv=difference_mean,
                            led_baseline_mv=led_baselines[i],
                            attribution=values["maps"][i], group=group, event_id=int(event_id),
                            target_ps=float(targets[ids[i]]),
                            prediction_ps=float(saved[ids[i]]),
                            improvement_ps=float(abs(targets[ids[i]])-abs(targets[ids[i]]-saved[ids[i]])),
                            led_threshold_mv=threshold,
                            window_starts=np.asarray(starts),
                            window_scores=scores[i][None,:], window_samples=window_samples
                        )
                        names.append(filename)
                        event_rows.append((group,int(event_id),int(ids[i]),
                                           float(targets[ids[i]]),float(saved[ids[i]]),
                                           float(np.mean(np.abs(scores[i]))),
                                           float(np.sqrt(np.mean(gaps[i,0]**2))),
                                           float(np.sqrt(np.mean(gaps[i,1]**2)))))
                    _save_csv(temporary/"events.csv",
                              ("group","event_id","dataset_index","target_ps","prediction_ps",
                               "mean_absolute_shapley_ps","knn_gap_rmse_window_ps",
                               "knn_gap_rmse_complement_ps"),event_rows)
                    metadata["plots"] = names
                else:
                    waveform_mv = np.asarray(scale.inverse(waves),dtype=np.float64)
                    mean_mv = waveform_mv.mean(axis=0)
                    std_mv = waveform_mv.std(axis=0)
                    mean_abs = np.mean(np.abs(values["maps"]),axis=(0,1))
                    positive,negative = base._signed_means(scores[:,None,:])
                    base._write_global_importance_csv(temporary,time_ps,mean_abs)
                    base._write_signed_csv(temporary,time_ps,starts,positive,negative,window_samples)
                    base._plot_global(
                        temporary,time_ps,mean_abs,mean_mv,std_waveform_mv=std_mv,
                        model_name=spec.name,mode=config["mode"],
                        window_name=str(config.get("window_name","wide")),
                        board_name=base._find_board_label(directory,output_root),
                        n_events=len(ids),window_samples=window_samples,
                        window_starts=np.asarray(starts),signed_positive=positive,
                        signed_negative=negative,
                    )
                    _plot_diagnostics(
                        temporary/"knn_prediction_gap.png",time_ps,mean_abs,
                        {"starts":starts,"values":gaps[:,0,:]},
                        {"starts":starts,"values":gaps[:,1,:]},
                    )
                per_window = []
                for i,start in enumerate(starts):
                    for side,name in enumerate(("window","complement")):
                        gap = gaps[:,side,i]
                        per_window.append((
                            int(i),int(start),name,int(tuned.selected_k[side*len(starts)+i]),
                            WEIGHT_NAMES[int(tuned.selected_weight[side*len(starts)+i])],
                            float(np.mean(gap)),float(np.std(gap)),
                            float(np.sqrt(np.mean(gap**2))),float(np.mean(np.abs(gap))),
                            float(np.mean(values["recon_mse"][:,side,i])),
                            float(np.mean(values["recon_mae"][:,side,i])),
                            float(np.sqrt(np.mean(values["recon_rmse_mv"][:,side,i]**2))),
                            float(np.mean(values["recon_mae_mv"][:,side,i])),
                        ))
                for row in per_window:
                    logger.info(
                        "KNN DIAGNOSTIC | window=%d | start=%d | observed=%s | k=%d | weighting=%s | "
                        "gap_mean=%.6f ps | gap_std=%.6f ps | gap_RMSE=%.6f ps | "
                        "gap_MAE=%.6f ps | reconstruction_MSE=%.8g | reconstruction_MAE=%.6g | "
                        "reconstruction_RMSE_mV=%.6g | reconstruction_MAE_mV=%.6g",
                        *row
                    )
                _save_csv(
                    temporary/"knn_diagnostics_per_window.csv",
                    ("window_index","start_sample","observed_region","k","weighting",
                     "prediction_gap_mean_ps","prediction_gap_std_ps","prediction_gap_rmse_ps",
                     "prediction_gap_mae_ps","reconstruction_mse_normalized",
                     "reconstruction_mae_normalized","reconstruction_rmse_mv","reconstruction_mae_mv"),
                    per_window,
                )
                _save_csv(temporary/"knn_tuning_selected.csv",
                          ("observed","window_index","start_sample","end_sample","k","weighting",
                           "mse","rmse","mae","mse_mv2","rmse_mv","mae_mv"),
                          [[row[key] for key in (
                              "observed","window_index","start_sample","end_sample","k",
                              "weighting","mse","rmse","mae","mse_mv2","rmse_mv","mae_mv"
                          )] for row in tuned.summary(engine)])
                atomic_json(temporary/"summary.json",metadata)
                base._publish_directory(temporary,destination,overwrite=True)
                logger.info("KNN TSHAP COMPLETE | run=%s | output=%s | events=%d | windows=%d",
                            directory,destination,len(ids),len(starts))
            finally:
                if temporary.exists():
                    shutil.rmtree(temporary)
            completed.append(destination)
        except Exception as exc:
            failures.append(f"{directory}: {exc}")
            logger.exception("KNN TSHAP FAILED | run=%s",directory)
    if not completed and not failures:
        raise ValueError("No complete saved models matched conditional KNN filters")
    if failures:
        raise RuntimeError("Conditional KNN TSHAP failures:\n"+"\n".join(failures))
    return completed
