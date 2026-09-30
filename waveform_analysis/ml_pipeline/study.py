from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np

from .common import canonical_hash
from .control_preprocessing import fit_control_artifact
from .data import preprocess_selected
from .event_selection import apply_selection_rules
from .hyperparameter_plot import plot_hyperparameter_validation
from .models import get_model
from .prepared_data import prepare_ml_dataset
from .progress import ProgressTracker
from .result_plots import blind_rmse_ctr_correlation, make_study_result_plots
from .search import CandidateScore, candidate_manifest, choose_best
from .shared_artifacts import ExperimentArtifactStore
from .splits import semantic_seed
from .stats import paired_ctr_improvement, rmse_ps
from .storage import RunStore
from .train import (
    FeatureTransformCache,
    FitInputCache,
    detector_swap_rmse,
    fit_on_indices,
    predict_indices,
    save_model,
)
from .view import model_target


def _logger(run_dir):
    logger = logging.getLogger(f"waveform-study:{run_dir}")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    logger.propagate = False
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    for handler in (
        logging.StreamHandler(),
        logging.FileHandler(Path(run_dir) / "study.log", encoding="utf-8"),
    ):
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger


def _should_save_model(policy, replica_index):
    if policy == "all":
        return True
    if policy == "first":
        return int(replica_index) == 1
    if policy == "none":
        return False
    raise ValueError(f"Unknown model save policy: {policy}")


def _validation_row(
    *,
    seed,
    candidate_id,
    selected,
    corrected,
    raw_validation_rmse,
    spec,
    config,
    event_identity,
    protocol_identity,
    sampling_identity,
    train_n,
):
    corrected = np.asarray(corrected, float)
    if not corrected.size or not np.all(np.isfinite(corrected)):
        raise RuntimeError("Fixed-validation RMSE requires one finite residual per validation event")
    corrected_rmse = float(rmse_ps(corrected))
    raw_rmse = float(raw_validation_rmse)
    improvement = raw_rmse - corrected_rmse
    improvement_percent = 100.0 * improvement / raw_rmse if raw_rmse != 0 else float("nan")
    return {
        "phase": "hyperparameter_validation",
        "replica_index": "",
        "seed": int(seed),
        "model": spec.name,
        "estimator_formulation": spec.estimator_formulation,
        "mode": config["mode"],
        "window_start_ns": float(config["window_ns"]["start"]),
        "window_end_ns": float(config["window_ns"]["end"]),
        "population_identity": protocol_identity,
        "event_population_identity": event_identity,
        "analysis_protocol_identity": protocol_identity,
        "sampling_identity": sampling_identity,
        "candidate_id": candidate_id,
        "selected": bool(selected),
        "ctr_ps": float("nan"),
        "uncorrected_ctr_ps": float("nan"),
        "improvement_ps": float("nan"),
        "improvement_percent": float("nan"),
        "rmse_ps": corrected_rmse,
        "uncorrected_rmse_ps": raw_rmse,
        "rmse_improvement_ps": improvement,
        "rmse_improvement_percent": improvement_percent,
        "n": int(corrected.size),
        "train_n": int(train_n),
        "swap_rmse_ps": float("nan"),
    }


def _replica_row(
    *,
    replica_index,
    seed,
    candidate_id,
    corrected,
    led,
    spec,
    config,
    event_identity,
    protocol_identity,
    sampling_identity,
    train_n,
    swap_rmse_ps,
    paired,
):
    return {
        "phase": "replica",
        "replica_index": int(replica_index),
        "seed": int(seed),
        "model": spec.name,
        "estimator_formulation": spec.estimator_formulation,
        "mode": config["mode"],
        "window_start_ns": float(config["window_ns"]["start"]),
        "window_end_ns": float(config["window_ns"]["end"]),
        "population_identity": protocol_identity,
        "event_population_identity": event_identity,
        "analysis_protocol_identity": protocol_identity,
        "sampling_identity": sampling_identity,
        "candidate_id": candidate_id,
        "selected": True,
        "ctr_ps": float(paired.corrected_ctr_ps),
        "uncorrected_ctr_ps": float(paired.led_ctr_ps),
        "improvement_ps": float(paired.improvement_ps),
        "improvement_percent": float(paired.improvement_percent),
        "rmse_ps": float(paired.corrected_rmse_ps),
        "uncorrected_rmse_ps": float(paired.led_rmse_ps),
        "rmse_improvement_ps": float(paired.rmse_improvement_ps),
        "rmse_improvement_percent": float(paired.rmse_improvement_percent),
        "n": int(np.isfinite(corrected).sum()),
        "train_n": int(train_n),
        "swap_rmse_ps": float(swap_rmse_ps),
    }


def run_study(config, *, overwrite=False, resume=False, rebuild_preprocessing=False):
    run_dir = Path(config["output_dir"]).resolve()
    store = RunStore(run_dir, overwrite=overwrite, resume=resume)
    logger = _logger(run_dir)
    public_config = {key: value for key, value in config.items() if not str(key).startswith("_")}
    store.write_resolved_config(public_config)

    spec = get_model(config["model"]["name"])
    candidates = candidate_manifest(list(spec.candidates(config["model"]["space"])))
    n_replicas = int(config["evaluation"]["n_replicas"])

    manifest = {
        "schema_version": 40,
        "status": "running",
        "name": config["name"],
        "study_name": config.get("study_name", config["name"]),
        "run_id": config.get("run_id", config["name"]),
        "window_name": config.get("window_name"),
        "config_fingerprint": config["_config_fingerprint"],
        "reference": config["reference"],
        "analysis": config["analysis"],
        "mode": config["mode"],
        "model": config["model"]["name"],
        "estimator_formulation": spec.estimator_formulation,
        "window_ns": config["window_ns"],
        "feature_transform": None if spec.feature_transform is None else spec.feature_transform.name,
        "batch_seed": int(config["seed"]),
        "model_selection": config["model_selection"],
        "evaluation": config["evaluation"],
        "save_models": config["save_models"],
        "preprocessing": config["preprocessing"],
        "preprocessing_fingerprint": canonical_hash(config["preprocessing"]),
        "statistical_unit": "replica",
        "fit_bootstrap": False,
        "event_level_bootstrap": False,
        "hyperparameter_selection_repeated": False,
        "hyperparameter_selection_metric": "fixed_validation_rmse_ps",
        "fixed_validation_used_in_replicas": False,
    }
    if resume and (run_dir / "manifest.json").is_file():
        old = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
        if int(old.get("schema_version", 0)) != 40:
            raise RuntimeError("Cannot resume results from a different pipeline schema")
        if old.get("hyperparameter_selection_metric") != "fixed_validation_rmse_ps" or old.get("fixed_validation_used_in_replicas") is not False:
            raise RuntimeError("Cannot resume results from the previous fixed-validation replica protocol")
        if old.get("config_fingerprint") != config["_config_fingerprint"]:
            raise RuntimeError("Cannot resume with a different resolved configuration")

    store.write_manifest(manifest)
    store.write_candidates(candidates)
    logger.info(
        "Study | model=%s | mode=%s | window=%s | candidates=%d | replicas=%d | batch_seed=%d | fixed_validation=%.3f | blind=%.3f | selection=RMSE",
        spec.name,
        config["mode"],
        config["window_ns"],
        len(candidates),
        n_replicas,
        int(config["seed"]),
        float(config["model_selection"]["validation_fraction"]),
        float(config["evaluation"]["blind_fraction"]),
    )

    rebuild_control = bool(config.get("_rebuild_control", rebuild_preprocessing))
    rebuild_analysis = bool(config.get("_rebuild_analysis", rebuild_preprocessing))
    rebuild_prepared = bool(config.get("_rebuild_prepared", rebuild_preprocessing))
    control, control_dir = fit_control_artifact(
        config["reference"]["root_file"],
        config["reference"],
        config["preprocessing"],
        config["fit"],
        fit_by_mode=config.get("_control_fit_by_mode"),
        modes=config.get("_control_modes") or [config["mode"]],
        cache_root=config["preprocessing"]["cache_dir"],
        rebuild=rebuild_control,
        logger=logger,
    )
    selection = apply_selection_rules(
        config["analysis"]["root_file"],
        config["analysis"],
        config["preprocessing"],
        control["selection_rules"],
        config["mode"],
        cache_dir=Path(config["preprocessing"]["cache_dir"]) / "analysis_selection",
        rebuild=rebuild_analysis,
        logger=logger,
    )
    native = preprocess_selected(
        config["analysis"]["root_file"],
        selection,
        config["analysis"],
        config["preprocessing"],
        config["mode"],
        cache_dir=Path(config["preprocessing"]["cache_dir"]) / "analysis_native",
        rebuild=rebuild_analysis,
        logger=logger,
    )
    dataset = prepare_ml_dataset(
        native,
        control,
        config,
        cache_dir=config["preprocessing"]["cache_dir"],
        rebuild=rebuild_prepared,
        logger=logger,
    )

    event_identity = str(dataset.manifest["event_population_identity"])
    protocol_identity = str(dataset.manifest["analysis_protocol_identity"])
    artifact_root = config.get("_batch_artifact_root") or str(run_dir / "artifacts")
    shared_store = ExperimentArtifactStore(artifact_root)
    fixed = shared_store.prepare_fixed_validation(dataset, config)
    target = model_target(dataset, config["mode"])

    manifest.update({
        "control_artifact": str(control_dir),
        "control_fingerprint": control["fingerprint"],
        "control_mode_fingerprint": (control.get("mode_fingerprints") or {}).get(
            config["mode"], control["fingerprint"]
        ),
        "fixed_led_threshold_mV": control["selected_led_threshold_mV"][config["mode"]],
        "event_population_identity": event_identity,
        "analysis_protocol_identity": protocol_identity,
        "analysis_population_identity": protocol_identity,
        "prepared_dataset": str(dataset.directory),
        "shared_artifact_root": str(shared_store.root),
        "sampling_identity": fixed.sampling_identity,
        "fixed_validation_artifact": str(fixed.directory),
        "fixed_validation_seed": int(fixed.split.seed),
        "shared_replicas": {},
    })
    store.write_manifest(manifest)

    validation_target = np.asarray(target[fixed.split.validation], float)
    raw_validation_rmse = float(rmse_ps(validation_target))
    logger.info(
        "Hyperparameter tuning | fixed split seed=%d | tuning_train=%d | validation=%d | metric=RMSE | validation excluded from every replica",
        int(fixed.split.seed),
        len(fixed.split.tuning_train),
        len(fixed.split.validation),
    )

    transform_cache = FeatureTransformCache()
    fit_input_cache = FitInputCache()
    transform_seed_base = semantic_seed(
        int(config["seed"]), "hyperparameter_tuning", spec.name, protocol_identity
    )
    scores = []
    for index, (candidate_id, params) in enumerate(candidates.items(), 1):
        existing = next(
            (
                row
                for row in store.read_results()
                if row.get("phase") == "hyperparameter_validation"
                and row.get("candidate_id") == candidate_id
            ),
            None,
        )
        if existing is not None:
            scores.append(CandidateScore(candidate_id, float(existing["rmse_ps"])))
            continue

        logger.info(
            "Hyperparameter candidate %d/%d | %s",
            index,
            len(candidates),
            candidate_id,
        )
        fitted = fit_on_indices(
            spec,
            config["model"]["space"],
            config,
            dataset,
            fixed.split.tuning_train,
            params,
            seed=semantic_seed(fixed.split.seed, spec.name, candidate_id, "tuning_fit"),
            transform_seed_base=transform_seed_base,
            feature_transform_cache=transform_cache,
            fit_input_cache=fit_input_cache,
            logger=logger,
        )
        prediction = predict_indices(
            spec,
            fitted,
            dataset,
            config["mode"],
            fixed.split.validation,
        )
        corrected = validation_target - prediction
        row = _validation_row(
            seed=fixed.split.seed,
            candidate_id=candidate_id,
            selected=False,
            corrected=corrected,
            raw_validation_rmse=raw_validation_rmse,
            spec=spec,
            config=config,
            event_identity=event_identity,
            protocol_identity=protocol_identity,
            sampling_identity=fixed.sampling_identity,
            train_n=len(fixed.split.tuning_train),
        )
        store.upsert_result(row)
        scores.append(CandidateScore(candidate_id, float(row["rmse_ps"])))
        logger.info(
            "Hyperparameter validation | candidate=%s | RMSE=%.3f ps",
            candidate_id,
            row["rmse_ps"],
        )

    best = choose_best(scores)
    selected_candidate = best.candidate_id
    selected_params = candidates[selected_candidate]
    rows = store.read_results()
    for row in rows:
        if row.get("phase") == "hyperparameter_validation":
            row["selected"] = row.get("candidate_id") == selected_candidate
    store._atomic_rows(rows)
    store.write_selected_hyperparameters({
        "candidate_id": selected_candidate,
        "parameters": selected_params,
        "selection_metric": "fixed_validation_rmse_ps",
        "selected_validation_rmse_ps": float(best.score),
        "validation_seed": int(fixed.split.seed),
        "validation_fraction": float(config["model_selection"]["validation_fraction"]),
        "validation_reused_in_replicas": False,
    })
    logger.info(
        "Hyperparameter selection complete | candidate=%s | validation RMSE=%.3f ps | configuration frozen for all replicas",
        selected_candidate,
        float(best.score),
    )

    progress = ProgressTracker(logger, {"replica": n_replicas})
    for replica_index in range(1, n_replicas + 1):
        replica = shared_store.prepare_replica(
            dataset,
            config,
            replica_index,
            target,
            fixed=fixed,
        )
        split = replica.split
        manifest["shared_replicas"][str(replica_index)] = str(replica.directory)
        store.write_manifest(manifest)

        if store.has_result("replica", selected_candidate, replica_index):
            progress.complete("replica", f"replica {replica_index}", announce=False)
            continue

        logger.info(
            "Replica %d/%d | seed=%d | fit once | train=%d | blind=%d | fixed validation excluded",
            replica_index,
            n_replicas,
            int(split.seed),
            len(split.train),
            len(split.test),
        )
        replica_transform_cache = FeatureTransformCache()
        replica_fit_input_cache = FitInputCache()
        fitted = fit_on_indices(
            spec,
            config["model"]["space"],
            config,
            dataset,
            split.train,
            selected_params,
            seed=semantic_seed(split.seed, spec.name, selected_candidate, "replica_fit"),
            transform_seed_base=semantic_seed(split.seed, spec.name, "transform"),
            feature_transform_cache=replica_transform_cache,
            fit_input_cache=replica_fit_input_cache,
            logger=logger,
        )

        prediction = predict_indices(spec, fitted, dataset, config["mode"], split.test)
        corrected = np.asarray(target[split.test], float) - prediction
        swap = detector_swap_rmse(spec, fitted, dataset, config["mode"], split.test)
        paired = paired_ctr_improvement(
            corrected,
            replica.led_ps,
            config["fit"],
            seed=semantic_seed(split.seed, "paired_led_ml", selected_candidate),
            led_ctr_ps=replica.led_ctr_ps,
            led_rmse_ps=replica.led_rmse_ps,
        )
        store.save_blind_residuals(split.seed, selected_candidate, corrected)
        row = _replica_row(
            replica_index=replica_index,
            seed=split.seed,
            candidate_id=selected_candidate,
            corrected=corrected,
            led=replica.led_ps,
            spec=spec,
            config=config,
            event_identity=event_identity,
            protocol_identity=protocol_identity,
            sampling_identity=fixed.sampling_identity,
            train_n=len(split.train),
            swap_rmse_ps=swap,
            paired=paired,
        )
        store.upsert_result(row)

        if _should_save_model(config["save_models"], replica_index):
            save_model(
                spec,
                fitted,
                store.model_dir(replica_index, split.seed, selected_candidate),
                selected_params,
            )
            logger.info(
                "Replica model saved | replica=%d | policy=%s",
                replica_index,
                config["save_models"],
            )

        logger.info(
            "Replica result | replica=%d | CTR=%.3f ps | LED CTR=%.3f ps | improvement=%.3f ps (%.2f%%) | RMSE=%.3f ps",
            replica_index,
            row["ctr_ps"],
            row["uncorrected_ctr_ps"],
            row["improvement_ps"],
            row["improvement_percent"],
            row["rmse_ps"],
        )
        progress.complete("replica", f"replica {replica_index}", announce=False)

    rows = store.read_results()
    plot_hyperparameter_validation(
        rows,
        candidates,
        run_dir / "hyperparameter_validation_rmse.png",
        logger,
    )
    plots = make_study_result_plots(
        rows,
        run_dir,
        model=spec.name,
        mode=config["mode"],
        window_ns=config["window_ns"],
    )

    replica_rows = [row for row in rows if row.get("phase") == "replica"]
    blind_ctr = [float(row["ctr_ps"]) for row in replica_rows]
    blind_rmse = [float(row["rmse_ps"]) for row in replica_rows]
    ctr_improvements = [float(row["improvement_ps"]) for row in replica_rows]
    rmse_improvements = [float(row["rmse_improvement_ps"]) for row in replica_rows]
    correlation, n_correlation = blind_rmse_ctr_correlation(rows)

    manifest.update({
        "status": "complete",
        "selected_candidate_id": selected_candidate,
        "selected_hyperparameters": selected_params,
        "selected_validation_rmse_ps": float(best.score),
        "replica_count": len(replica_rows),
        "blind_ctr_mean_ps": float(np.mean(blind_ctr)) if blind_ctr else None,
        "blind_ctr_std_ps": float(np.std(blind_ctr, ddof=1)) if len(blind_ctr) > 1 else 0.0,
        "blind_rmse_mean_ps": float(np.mean(blind_rmse)) if blind_rmse else None,
        "blind_rmse_std_ps": float(np.std(blind_rmse, ddof=1)) if len(blind_rmse) > 1 else 0.0,
        "paired_ctr_improvement_mean_ps": float(np.mean(ctr_improvements)) if ctr_improvements else None,
        "paired_ctr_improvement_std_ps": float(np.std(ctr_improvements, ddof=1)) if len(ctr_improvements) > 1 else 0.0,
        "paired_rmse_improvement_mean_ps": float(np.mean(rmse_improvements)) if rmse_improvements else None,
        "paired_rmse_improvement_std_ps": float(np.std(rmse_improvements, ddof=1)) if len(rmse_improvements) > 1 else 0.0,
        "blind_rmse_ctr_pearson_r": correlation if np.isfinite(correlation) else None,
        "blind_rmse_ctr_correlation_n": n_correlation,
        "result_plots": {
            key: str(value) if value is not None else None
            for key, value in plots.items()
        },
    })
    store.write_manifest(manifest)
    logger.info(
        "Study complete | selected=%s | validation RMSE=%.3f ps | replicas=%d | blind CTR mean=%.3f ± %.3f ps",
        selected_candidate,
        float(best.score),
        manifest["replica_count"],
        manifest["blind_ctr_mean_ps"],
        manifest["blind_ctr_std_ps"],
    )
    return run_dir
