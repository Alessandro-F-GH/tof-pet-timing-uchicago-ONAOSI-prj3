"""Two-stage shared-linear then direct wide-window timing correction."""
from __future__ import annotations

import copy
import json
import logging
from dataclasses import replace
from pathlib import Path

import numpy as np

from waveform_analysis.core.io import atomic_json, canonical_hash
from waveform_analysis.core.dependencies import stage_fingerprint
from waveform_analysis.data.results import discover_runs
from waveform_analysis.data.storage import RunStore
from waveform_analysis.data.view import model_target, waveform_view
from waveform_analysis.data.splits import semantic_seed, make_cv_split
from waveform_analysis.engine.preprocessing import fit_control, prepare_role_dataset
from waveform_analysis.data.feature_cache import prepare_frozen_transform, prepare_frozen_features
from waveform_analysis.engine.study import _outer_cv_selection_and_fit, _logger
from waveform_analysis.engine.train import predict_indices, target_fingerprint, release_training_memory
from waveform_analysis.models import get_model
from waveform_analysis.reporting.stats import paired_central_metrics, blind_event_bootstrap, paired_model_bootstrap, metric_values
from waveform_analysis.data.shared_artifacts import SharedCVArtifacts


def make_development_split(n_events, *, seed, stage1_fraction=0.25):
    """Deterministic event-level partition; each paired detector event stays intact."""
    n_events = int(n_events)
    fraction = float(stage1_fraction)
    if n_events < 2:
        raise ValueError("At least two development events are required")
    if not 0.05 <= fraction <= 0.95:
        raise ValueError("stage1_fraction must lie in [0.05, 0.95]")
    order = np.random.default_rng(int(seed)).permutation(n_events)
    n_stage1 = min(max(1, int(round(fraction * n_events))), n_events - 1)
    first, second = np.sort(order[:n_stage1]), np.sort(order[n_stage1:])
    if np.intersect1d(first, second).size or np.union1d(first, second).size != n_events:
        raise RuntimeError("Development split is not disjoint and exhaustive")
    return first, second


def _subset(dataset, indices, *, short_window=None):
    idx = np.asarray(indices, dtype=np.int64)
    changes = {"event_index": np.asarray(dataset.event_index[idx])}
    for name in ("bias_voltage_V", "energy_windows", "timing_windows",
                 "energy_target_ps", "timing_target_ps", "energy_led_time_ps",
                 "timing_led_time_ps"):
        value = getattr(dataset, name)
        if value is not None:
            changes[name] = np.asarray(value[idx])
    manifest = dict(dataset.manifest)
    population = canonical_hash({"parent": dataset.manifest["event_population_identity"],
                                 "indices": idx.tolist(), "window": short_window})
    manifest.update(event_population_identity=population,
                    analysis_protocol_identity=canonical_hash({"parent": dataset.manifest["analysis_protocol_identity"],
                                                               "population": population}),
                    analysis_population_identity=canonical_hash({"parent": dataset.manifest["analysis_population_identity"],
                                                                 "population": population}))
    if short_window is not None:
        time_name = "energy_time_ps" if manifest["mode"] == "energy_to_energy" else "timing_time_ps"
        wave_name = "energy_windows" if manifest["mode"] == "energy_to_energy" else "timing_windows"
        time = np.asarray(getattr(dataset, time_name), dtype=float)
        lo, hi = map(float, (short_window["start"], short_window["end"]))
        keep = np.flatnonzero((time >= lo * 1000.0 - 1e-7) & (time <= hi * 1000.0 + 1e-7))
        if keep.size < 2:
            raise ValueError("The wide-window dataset does not contain the Onishi sample grid")
        expected = np.arange(lo * 1000.0, hi * 1000.0 + 1e-7, np.median(np.diff(time)))
        if expected.size != keep.size or not np.allclose(time[keep], expected, rtol=0, atol=1e-6):
            raise ValueError("Onishi samples cannot be recovered exactly from the wide-window grid")
        changes[time_name] = time[keep]
        changes[wave_name] = np.asarray(getattr(dataset, wave_name)[idx][:, :, keep])
        if dataset.sample_mask is not None:
            changes["sample_mask"] = np.asarray(dataset.sample_mask)[keep]
            manifest["sample_mask_identity"] = canonical_hash({
                "parent": dataset.manifest.get("sample_mask_identity"),
                "sample_indices": keep.tolist(),
                "mask": changes["sample_mask"].astype(bool).tolist(),
            })
            manifest["input_samples_before_mask"] = int(keep.size)
            manifest["input_samples_after_mask"] = int(changes["sample_mask"].sum())
    return replace(dataset, **changes, manifest=manifest)


def _source_transform(config, model_name, development, control_dataset, stage2_space, logger):
    """Resolve the exact completed wide-window run; never fit a replacement transform."""
    runs = discover_runs(Path(config["batch_output_dir"]), complete=True)
    matches = [r for r in runs if r["manifest"].get("model") == model_name
               and r["manifest"].get("mode") == config["mode"]
               and r["manifest"].get("window_name") == "wide"]
    if len(matches) != 1:
        raise FileNotFoundError(
            f"Expected exactly one completed wide/{model_name} source run for {config['mode']}; found {len(matches)}"
        )
    run = matches[0]
    source_config = run["manifest"]
    populations = source_config.get("dataset_populations") or {}
    for role, dataset in (("development", development), ("control", control_dataset)):
        saved_identity = (populations.get(role) or {}).get("analysis_protocol_identity")
        current_identity = dataset.manifest.get("analysis_protocol_identity")
        if not saved_identity or str(saved_identity) != str(current_identity):
            raise ValueError(
                f"Source transform {role} population identity does not match the current prepared dataset"
            )
    from waveform_analysis.data.results import resolve_run_config
    from waveform_analysis.data.feature_cache import _transform_parameters
    saved_config = resolve_run_config(run["directory"])
    saved_spec = get_model(model_name)
    expected_parameters = _transform_parameters(saved_spec, stage2_space)
    saved_parameters = _transform_parameters(saved_spec, saved_config["model"]["space"])
    if canonical_hash(expected_parameters) != canonical_hash(saved_parameters):
        raise ValueError("Source fitted transformation parameters differ from the requested Stage-2 model space")
    source_store = RunStore(run["directory"], artifact_root=config["preprocessing"]["artifact_root"])
    model_dir = source_store.model_dir
    path = model_dir / "feature_transform" / "transform.pkl"
    if not path.is_file():
        raise FileNotFoundError(f"Source model does not contain its fitted transform: {path}")
    import pickle
    from waveform_analysis.data.splits import semantic_seed
    with path.open("rb") as stream:
        artifact = pickle.load(stream)
    metadata = getattr(artifact, "metadata", {})
    expected_seed = semantic_seed(
        int(saved_config["seed"]), saved_config["mode"], saved_config.get("window_name"),
        saved_spec.feature_transform.name, "frozen_transform",
    )
    actual_seed = metadata.get("training_seed")
    if actual_seed is None or int(actual_seed) != int(expected_seed):
        raise ValueError(
            f"Source fitted transformation seed mismatch: expected {expected_seed}, found {actual_seed}"
        )
    return path, run


def _experiment_config(config, stage, model_name, window_name, window, output_root):
    out = copy.deepcopy(config)
    out["model"] = {"name": model_name, "space": copy.deepcopy(config["_two_stage_spaces"][model_name])}
    out["window_name"] = window_name
    out["window_ns"] = dict(window)
    out["output_dir"] = str((Path(output_root) / stage).resolve())
    out["batch_output_dir"] = str(Path(output_root).resolve())
    out["name"] = f"{config['name']}__two_stage__{stage}"
    out["run_id"] = f"{stage}__{config['mode']}"
    return out


def run_two_stage(config, *, split_fraction=0.25, logger=None):
    if not 0.05 <= float(split_fraction) <= 0.95:
        raise ValueError("split_fraction must lie in [0.05, 0.95]")
    mode = config["mode"]
    stage2_model = "direct_multirocket_rdst" if mode == "energy_to_energy" else "direct_multirocket"
    short_window = {"start": -1.5, "end": 2.0}
    wide_window = dict(config["window_ns"])
    if config.get("window_name") != "wide":
        raise ValueError("Two-stage experiment must be configured from the wide-window protocol")
    output_root = Path(config["batch_output_dir"]).resolve() / "two_stage" / str(config.get("_board") or "board") / mode
    output_root.mkdir(parents=True, exist_ok=True)
    logger = logger or _logger(output_root)
    control, _ = fit_control(config, rebuild=False, logger=logger)
    _, _, wide = prepare_role_dataset(config, "development", control, rebuild=False, logger=logger)
    target = np.asarray(model_target(wide, mode), dtype=np.float64)
    n = int(wide.n_events)
    if n < max(20, int(config["cross_validation"]["folds"]) * 5):
        raise ValueError(f"Development population too small for two-stage CV: n={n}")
    split_seed = semantic_seed(config["seed"], config.get("_board"), mode, "two-stage-25-75")
    stage1_indices, stage2_indices = make_development_split(
        n, seed=split_seed, stage1_fraction=split_fraction
    )
    split_fp = canonical_hash({"population": wide.manifest["analysis_protocol_identity"],
                               "seed": split_seed, "fraction": float(split_fraction),
                               "stage1": stage1_indices.tolist(), "stage2": stage2_indices.tolist()})
    split_path = output_root / "split.npz"
    if split_path.exists():
        with np.load(split_path) as old:
            if str(old["fingerprint"].item()) != split_fp:
                raise RuntimeError("Persisted two-stage split differs from current protocol; use a new output folder")
    else:
        np.savez_compressed(split_path, stage1=stage1_indices, stage2=stage2_indices,
                            seed=np.asarray(split_seed), fraction=np.asarray(split_fraction),
                            fingerprint=np.asarray(split_fp))
    if np.intersect1d(stage1_indices, stage2_indices).size or np.union1d(stage1_indices, stage2_indices).size != n:
        raise RuntimeError("Two-stage split is not disjoint and exhaustive")

    short_data = _subset(wide, np.arange(n), short_window=short_window)
    stage1_data = _subset(short_data, stage1_indices)
    stage2_data = _subset(wide, stage2_indices)
    stage1_cfg = _experiment_config(config, "stage1", "shared_linear_ridge", "onishi",
                                     short_window, output_root)
    stage2_cfg = _experiment_config(config, "stage2", stage2_model, "wide",
                                     wide_window, output_root)
    stage1_cfg["seed"] = semantic_seed(config["seed"], split_seed, "stage1")
    stage2_cfg["seed"] = semantic_seed(config["seed"], split_seed, "stage2")

    # Stage 1 uses the standard original target and the existing shared CV/fit pipeline.
    stage1_store = RunStore(stage1_cfg["output_dir"], artifact_root=config["preprocessing"]["artifact_root"])
    stage1_spec = get_model("shared_linear_ridge")
    s1_cv, s1_sel, s1_fp, s1_best, s1_model = _outer_cv_selection_and_fit(
        stage1_store, stage1_spec, stage1_cfg["model"]["space"], stage1_cfg,
        stage1_data, logger, None, None,
    )
    if s1_model is None:
        from waveform_analysis.engine.train import load_fitted_model
        s1_model = load_fitted_model(stage1_spec, stage1_store.model_dir, s1_best["parameters"], stage1_cfg)
    p1_s2 = predict_indices(stage1_spec, s1_model, stage2_data, mode,
                            np.arange(stage2_data.n_events), chunk_size=config["runtime"]["prediction_chunk_size"])
    residual_s2 = target[stage2_indices] - p1_s2
    if residual_s2.shape != (stage2_indices.size,) or not np.all(np.isfinite(residual_s2)):
        raise RuntimeError("Stage-1 residual target is invalid or misaligned")
    residual_fp = target_fingerprint(residual_s2)
    np.savez_compressed(output_root / "stage2_targets.npz", event_id=wide.event_index[stage2_indices],
                        stage1_prediction_ps=p1_s2, original_target_ps=target[stage2_indices],
                        residual_target_ps=residual_s2, target_fingerprint=np.asarray(residual_fp))

    # Reuse the exact fitted transform from the existing original-target wide run.
    control_prepared = prepare_role_dataset(config, "control", control, rebuild=False, logger=None)[2]
    source_path, source_run = _source_transform(config, stage2_model, wide, control_prepared, stage2_cfg["model"]["space"], logger)
    stage2_spec = get_model(stage2_model)
    frozen_transform = prepare_frozen_transform(
        stage2_spec, stage2_cfg["model"]["space"], stage2_cfg,
        control_dataset=control_prepared,
        development_dataset=wide, cache_root=config["preprocessing"]["cache_dir"],
        logger=logger, legacy_transform_path=source_path,
    )
    frozen_full = prepare_frozen_features(stage2_spec, frozen_transform, stage2_cfg, wide,
                                          role="development", cache_root=config["preprocessing"]["cache_dir"], logger=logger)
    from waveform_analysis.data.feature_cache import FrozenFeatureSet
    stage2_matrix = frozen_full.rows(stage2_indices)
    frozen_stage2 = FrozenFeatureSet(
        stage2_matrix, frozen_full.sample_mask, frozen_full.transform,
        canonical_hash({"source_feature_matrix": frozen_full.identity,
                        "stage2_indices": stage2_indices.tolist()}),
        "two_stage_stage2",
    )
    # Feature rows use the subset event order; target override has exactly the same order.
    stage2_store = RunStore(stage2_cfg["output_dir"], artifact_root=config["preprocessing"]["artifact_root"])
    s2_split = make_cv_split(
        stage2_data.n_events,
        population_identity=stage2_data.manifest["event_population_identity"],
        batch_seed=stage2_cfg["seed"],
        n_folds=int(stage2_cfg["cross_validation"]["folds"]),
        shuffle=bool(stage2_cfg["cross_validation"]["shuffle"]),
    )
    s2_led_metrics = {}
    for fold in s2_split.folds:
        vals = residual_s2[np.asarray(fold.validation, dtype=np.int64)]
        score = metric_values(
            vals, stage2_cfg["fit"],
            seed=semantic_seed(s2_split.seed, "stage2-residual-reference", fold.fold_id),
        )
        s2_led_metrics[int(fold.fold_id)] = {
            "fold_id": int(fold.fold_id), "n_validation": int(vals.size),
            "led_ctr_ps": score["ctr_ps"], "led_rmse_ps": score["rmse_ps"],
        }
    s2_shared = SharedCVArtifacts(
        Path(stage2_cfg["batch_output_dir"]) / "artifacts",
        s2_split, s2_led_metrics,
        canonical_hash({"split_seed": s2_split.seed, "residual_target": residual_fp}),
    )
    s2_cv, s2_sel, s2_fp, s2_best, s2_model = _outer_cv_selection_and_fit(
        stage2_store, stage2_spec, stage2_cfg["model"]["space"], stage2_cfg,
        stage2_data, logger, frozen_transform, frozen_stage2, residual_s2,
        shared=s2_shared,
    )
    if s2_model is None:
        from waveform_analysis.engine.train import load_fitted_model
        s2_model = load_fitted_model(stage2_spec, stage2_store.model_dir, s2_best["parameters"], stage2_cfg)

    # Blind evaluation uses original targets only after both fitted stages are frozen.
    _, _, blind = prepare_role_dataset(config, "blind", control, rebuild=False, logger=logger)
    blind_short = _subset(blind, np.arange(blind.n_events), short_window=short_window)
    blind_s1 = predict_indices(stage1_spec, s1_model, blind_short, mode,
                               np.arange(blind.n_events), chunk_size=config["runtime"]["prediction_chunk_size"])
    frozen_blind = prepare_frozen_features(stage2_spec, frozen_transform, stage2_cfg, blind,
                                           role="blind", cache_root=config["preprocessing"]["cache_dir"], logger=logger)
    blind_s2 = predict_indices(stage2_spec, s2_model, blind, mode, np.arange(blind.n_events),
                               chunk_size=config["runtime"]["prediction_chunk_size"], frozen_features=frozen_blind)
    blind_target = np.asarray(model_target(blind, mode), dtype=np.float64)
    combined = blind_s1 + blind_s2
    residual1 = blind_target - blind_s1
    corrected = blind_target - combined
    metrics1 = paired_central_metrics(residual1, blind_target, config["fit"],
                                      seed=semantic_seed(split_seed, "blind", "stage1"))
    metrics2 = paired_central_metrics(corrected, blind_target, config["fit"],
                                      seed=semantic_seed(split_seed, "blind", "combined"))
    bootstrap, draws = blind_event_bootstrap(corrected, blind_target, config["fit"],
                                              n_resamples=config["bootstrap"]["n_resamples"],
                                              seed=semantic_seed(split_seed, "bootstrap"))
    paired = paired_model_bootstrap(blind.event_index, residual1, blind.event_index, corrected,
                                     config["fit"], metric="ctr",
                                     n_resamples=config["bootstrap"]["n_resamples"],
                                     seed=semantic_seed(split_seed, "paired-ctr"))
    np.savez_compressed(output_root / "blind_predictions.npz",
                        event_id=blind.event_index, original_target_ps=blind_target,
                        stage1_prediction_ps=blind_s1, residual_after_stage1_ps=residual1,
                        stage2_prediction_ps=blind_s2, combined_prediction_ps=combined,
                        corrected_residual_ps=corrected)
    atomic_json(output_root / "results.json", {
        "status": "complete", "board": config.get("_board"), "mode": mode,
        "split_fraction_stage1": float(split_fraction), "split_seed": int(split_seed),
        "split_fingerprint": split_fp, "n_development": n,
        "n_stage1": int(stage1_indices.size), "n_stage2": int(stage2_indices.size),
        "stage1": {"model": "shared_linear_ridge", "best": s1_best, **metrics1,
                   "cv_fingerprint": s1_cv, "selection_fingerprint": s1_sel, "final_fit_fingerprint": s1_fp},
        "stage2": {"model": stage2_model, "best": s2_best, **metrics2,
                   "cv_fingerprint": s2_cv, "selection_fingerprint": s2_sel, "final_fit_fingerprint": s2_fp,
                   "residual_target_fingerprint": residual_fp},
        "blind_bootstrap": bootstrap, "paired_stage1_minus_combined_ctr": paired,
        "source_transform_run": str(source_run["directory"]),
        "source_transform_path": str(source_path),
        "source_transform_identity": frozen_transform.identity,
        "single_stage_full_development_benchmark": source_run.get("blind", {}),
        "blind_mean_residual_before_ps": float(np.mean(blind_target)),
        "blind_mean_residual_after_stage1_ps": float(np.mean(residual1)),
        "blind_mean_residual_after_stage2_ps": float(np.mean(corrected)),
    })
    release_training_memory()
    logger.info("Two-stage complete | Stage1 CTR %.2f ps | combined CTR %.2f ps | paired Δ %.2f ± %.2f ps",
                metrics1["ctr_ps"], metrics2["ctr_ps"], paired["difference_ps"], paired["bootstrap_std_ps"])
    return output_root
