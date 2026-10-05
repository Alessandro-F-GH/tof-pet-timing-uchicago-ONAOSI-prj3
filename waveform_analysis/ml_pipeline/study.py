from __future__ import annotations

import copy
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
from .search import (
    CandidateScore,
    candidate_id,
    candidate_manifest,
    choose_best,
    fixed_parameters,
    grid_candidates,
    optimization_config,
    suggest_parameters,
)
from .shared_artifacts import ExperimentArtifactStore
from .splits import semantic_seed
from .stats import paired_ctr_improvement, rmse_ps
from .storage import RunStore
from .train import (
    FeatureTransformCache,
    FitInputCache,
    FittedModel,
    detector_swap_rmse,
    fit_on_indices,
    predict_indices,
    release_training_memory,
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


def _format_log_value(value):
    if isinstance(value, str):
        return value
    if value is None:
        return "None"
    if isinstance(value, (bool, np.bool_)):
        return "true" if bool(value) else "false"
    if isinstance(value, np.generic):
        value = value.item()
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def _parameter_log_schema(candidates):
    parameter_sets = list(candidates.values())
    if not parameter_sets:
        return {}, {}, {}

    keys = list(parameter_sets[0])
    for params in parameter_sets[1:]:
        for key in params:
            if key not in keys:
                keys.append(key)

    fixed = {}
    varying = {}
    for key in keys:
        values = [params.get(key) for params in parameter_sets]
        encoded = [_format_log_value(value) for value in values]
        if len(set(encoded)) == 1:
            fixed[key] = values[0]
            continue
        unique_values = []
        seen = set()
        for value, token in zip(values, encoded):
            if token in seen:
                continue
            seen.add(token)
            unique_values.append(value)
        varying[key] = unique_values

    codes = {key: index for index, key in enumerate(varying, 1)}
    return fixed, varying, codes


def _format_parameter_pairs(params):
    if not params:
        return "none"
    return " | ".join(f"{key}={_format_log_value(value)}" for key, value in params.items())


def _format_candidate_codes(params, codes):
    if not codes:
        return _format_parameter_pairs(params)
    return " | ".join(
        f"{code}={_format_log_value(params.get(key))}" for key, code in codes.items()
    )


def _log_hyperparameter_space(logger, candidates):
    fixed, varying, codes = _parameter_log_schema(candidates)
    logger.info("Hyperparameter fixed | %s", _format_parameter_pairs(fixed))
    if varying:
        logger.info(
            "Hyperparameter search | %s",
            " | ".join(
                f"{codes[key]}={key} values={_format_log_value(values)}"
                for key, values in varying.items()
            ),
        )
    else:
        logger.info("Hyperparameter search | no optimized hyperparameters")
    return codes


def _selection_stage_summary(selection):
    counts = dict(selection.manifest.get("stage_counts") or {})
    labels = {
        "initial_valid": "valid_waveforms",
        "photopeak": "photopeak",
        "main_hit": "main_hit",
        "main_hit_tot": "timing_tot",
        "baseline_noise": "baseline_noise",
        "baseline_clipping": "baseline_clipping",
    }
    parts = [f"raw={int(selection.manifest.get('n_raw', 0))}"]
    for key, value in counts.items():
        parts.append(f"{labels.get(key, key)}={int(value)}")
    return " | ".join(parts)


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


def _replica_train_size(dataset, fixed, config):
    n_test = int(round(int(dataset.n_events) * float(config["evaluation"]["blind_fraction"])))
    n_train = int(len(fixed.split.tuning_train) - n_test)
    if n_train < 1:
        raise RuntimeError("Replica protocol leaves no events for model training")
    return n_train


def _tuning_train_subset(dataset, fixed, config, event_identity):
    pool = np.asarray(fixed.split.tuning_train, dtype=np.int64)
    n_train = _replica_train_size(dataset, fixed, config)
    if n_train > pool.size:
        raise RuntimeError("Requested tuning subset is larger than the fixed tuning pool")
    seed = semantic_seed(int(config["seed"]), "hyperparameter_train_subset", event_identity)
    if n_train == pool.size:
        return np.sort(pool.copy()), seed
    rng = np.random.default_rng(seed)
    selected = np.sort(rng.choice(pool, size=n_train, replace=False).astype(np.int64))
    return selected, seed


def _fit_fixed_feature_transform(
    spec,
    model_space,
    config,
    dataset,
    indices,
    parameters,
    *,
    transform_seed_base,
    logger,
):
    if spec.feature_transform is None:
        return None
    fit_cache = FitInputCache()
    prepared = fit_cache.prepare(spec, dataset, config["mode"], indices)
    cfg = copy.deepcopy(model_space)
    cfg["_input_time_ps"] = np.asarray(prepared.time_ps, np.float64)
    if logger is not None:
        cfg["_logger"] = logger
    transform_cache = FeatureTransformCache()
    transform, _ = transform_cache.prepare(
        spec.feature_transform,
        parameters,
        prepared.x,
        seed_base=int(transform_seed_base),
        scope_key=prepared.scope_key,
        config=cfg,
    )
    transform.cache.clear()
    fit_cache.clear()
    transform_cache.clear()
    release_training_memory()
    return transform


def _fit_with_fixed_feature_transform(
    spec,
    model_space,
    config,
    dataset,
    indices,
    parameters,
    *,
    seed,
    feature_transform,
    logger,
):
    fit_cache = FitInputCache()
    prepared = fit_cache.prepare(spec, dataset, config["mode"], indices)
    transformed = feature_transform.apply(prepared.x)
    cfg = copy.deepcopy(model_space)
    cfg["_early_stopping_seed"] = int(seed)
    cfg["_input_time_ps"] = np.asarray(prepared.time_ps, np.float64)
    cfg["_prediction_max_abs_ps"] = float(config["ml_output"]["max_abs_ps"])
    if logger is not None:
        cfg["_logger"] = logger
    artifact = spec.fit(
        dict(parameters or {}),
        np.asarray(transformed, np.float32),
        np.asarray(prepared.y, np.float64),
        seed=int(seed),
        config=cfg,
    )
    metadata = dict(getattr(artifact, "metadata", {}) or {})
    metadata.update(
        {
            "estimator_formulation": spec.estimator_formulation,
            "training_events": int(prepared.y.size),
            "sample_mask_training_events": int(prepared.y.size),
            "input_samples_before_mask": int(prepared.sample_mask.size),
            "input_samples_after_mask": int(prepared.sample_mask.sum()),
            "prediction_chunk_size": int(config["runtime"]["prediction_chunk_size"]),
            "feature_transform": feature_transform.spec.name,
            "feature_transform_identity": feature_transform.identity,
            "feature_transform_parameters": feature_transform.parameters,
            "feature_transform_seed": int(feature_transform.seed),
            "feature_transform_reused_across_replicas": True,
        }
    )
    fitted = FittedModel(
        artifact=artifact,
        metadata=metadata,
        output_max_abs_ps=float(config["ml_output"]["max_abs_ps"]),
        sample_mask=np.asarray(prepared.sample_mask, bool),
        feature_transform=feature_transform,
    )
    fit_cache.clear()
    del transformed
    release_training_memory()
    return fitted


def run_study(config, *, overwrite=False, resume=False, rebuild_preprocessing=False):
    run_dir = Path(config["output_dir"]).resolve()
    store = RunStore(run_dir, overwrite=overwrite, resume=resume)
    logger = _logger(run_dir)
    public_config = {key: value for key, value in config.items() if not str(key).startswith("_")}
    store.write_resolved_config(public_config)

    spec = get_model(config["model"]["name"])
    model_space = config["model"]["space"]
    optimization = optimization_config(model_space)
    n_replicas = int(config["evaluation"]["n_replicas"])
    prediction_chunk_size = int(config["runtime"]["prediction_chunk_size"])

    if optimization.strategy == "fixed":
        candidates = candidate_manifest([fixed_parameters(model_space)])
    elif optimization.strategy == "grid":
        candidates = candidate_manifest(grid_candidates(model_space))
    else:
        candidates = {}
        existing_candidates = run_dir / "candidates.json"
        if resume and existing_candidates.is_file():
            loaded = json.loads(existing_candidates.read_text(encoding="utf-8"))
            if isinstance(loaded, dict):
                candidates = {str(key): dict(value) for key, value in loaded.items()}

    selection_metric = None if optimization.strategy == "fixed" else "fixed_validation_rmse_ps"
    manifest = {
        "schema_version": 42,
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
        "optimization": model_space["optimization"],
        "optimization_strategy": optimization.strategy,
        "evaluation": config["evaluation"],
        "runtime": config["runtime"],
        "save_models": config["save_models"],
        "preprocessing": config["preprocessing"],
        "preprocessing_fingerprint": canonical_hash(config["preprocessing"]),
        "statistical_unit": "replica",
        "fit_bootstrap": False,
        "event_level_bootstrap": False,
        "hyperparameter_selection_repeated": False,
        "hyperparameter_selection_metric": selection_metric,
        "validation_skipped": optimization.strategy == "fixed",
        "fixed_validation_used_in_replicas": False,
        "hyperparameter_tuning_train_sampling": "fixed_random_subset_matching_replica_train_size",
        "minirocket_transform_reused_across_replicas": spec.name in {"direct_minirocket", "shared_minirocket"},
    }
    if resume and (run_dir / "manifest.json").is_file():
        old = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
        if int(old.get("schema_version", 0)) != 42:
            raise RuntimeError("Cannot resume results from a different pipeline schema")
        if old.get("optimization_strategy") != optimization.strategy:
            raise RuntimeError("Cannot resume with a different optimization strategy")
        if old.get("fixed_validation_used_in_replicas") is not False:
            raise RuntimeError("Cannot resume results from an incompatible validation protocol")
        if old.get("config_fingerprint") != config["_config_fingerprint"]:
            raise RuntimeError("Cannot resume with a different resolved configuration")

    store.write_manifest(manifest)
    if candidates:
        store.write_candidates(candidates)
    logger.info(
        "Study | model=%s | mode=%s | window=%s | strategy=%s | replicas=%d | batch_seed=%d | fixed_validation=%.3f | blind=%.3f | prediction_chunk=%d",
        spec.name,
        config["mode"],
        config["window_ns"],
        optimization.strategy,
        n_replicas,
        int(config["seed"]),
        float(config["model_selection"]["validation_fraction"]),
        float(config["evaluation"]["blind_fraction"]),
        prediction_chunk_size,
    )

    rebuild_control = bool(config.get("_rebuild_control", rebuild_preprocessing))
    rebuild_analysis = bool(config.get("_rebuild_analysis", rebuild_preprocessing))
    rebuild_prepared = bool(config.get("_rebuild_prepared", rebuild_preprocessing))
    control_modes = config.get("_control_modes") or [config["mode"]]
    logger.info(
        "Reference preprocessing | start | source=%s | modes=%s",
        Path(config["reference"]["root_file"]).name,
        ",".join(map(str, control_modes)),
    )
    control, control_dir = fit_control_artifact(
        config["reference"]["root_file"],
        config["reference"],
        config["preprocessing"],
        config["fit"],
        fit_by_mode=config.get("_control_fit_by_mode"),
        modes=control_modes,
        cache_root=config["preprocessing"]["cache_dir"],
        rebuild=rebuild_control,
        logger=None,
    )
    logger.info(
        "Reference preprocessing | complete | %s",
        " | ".join(
            f"{mode} LED={float(control['selected_led_threshold_mV'][mode]):.6g} mV"
            for mode in control_modes
        ),
    )

    logger.info(
        "Analysis preprocessing | event selection | start | source=%s | mode=%s",
        Path(config["analysis"]["root_file"]).name,
        config["mode"],
    )
    selection = apply_selection_rules(
        config["analysis"]["root_file"],
        config["analysis"],
        config["preprocessing"],
        control["selection_rules"],
        config["mode"],
        cache_dir=Path(config["preprocessing"]["cache_dir"]) / "analysis_selection",
        rebuild=rebuild_analysis,
        logger=None,
    )
    logger.info(
        "Analysis preprocessing | event selection | complete | source=%s | mode=%s | %s",
        Path(config["analysis"]["root_file"]).name,
        config["mode"],
        _selection_stage_summary(selection),
    )

    logger.info(
        "Analysis preprocessing | waveform materialization | start | source=%s | mode=%s",
        Path(config["analysis"]["root_file"]).name,
        config["mode"],
    )
    native = preprocess_selected(
        config["analysis"]["root_file"],
        selection,
        config["analysis"],
        config["preprocessing"],
        config["mode"],
        cache_dir=Path(config["preprocessing"]["cache_dir"]) / "analysis_native",
        rebuild=rebuild_analysis,
        logger=None,
    )
    logger.info(
        "Analysis preprocessing | waveform materialization | complete | source=%s | mode=%s | n=%d",
        Path(config["analysis"]["root_file"]).name,
        config["mode"],
        native.n_events,
    )

    logger.info(
        "ML input preparation | start | mode=%s | window=%s",
        config["mode"],
        config["window_ns"],
    )
    dataset = prepare_ml_dataset(
        native,
        control,
        config,
        cache_dir=config["preprocessing"]["cache_dir"],
        rebuild=rebuild_prepared,
        logger=None,
    )
    prepared_manifest = dataset.manifest
    logger.info(
        "ML input preparation | complete | LED=%.6g mV | coincidence=%d/%d | window_dropped=%d | final=%d",
        float(prepared_manifest["fixed_led_threshold_mV"]),
        int(prepared_manifest["n_after_fixed_led"]),
        int(prepared_manifest["n_before_fixed_led"]),
        int(prepared_manifest["n_dropped_window"]),
        int(prepared_manifest["n_final"]),
    )

    event_identity = str(dataset.manifest["event_population_identity"])
    protocol_identity = str(dataset.manifest["analysis_protocol_identity"])
    artifact_root = config.get("_batch_artifact_root") or str(run_dir / "artifacts")
    shared_store = ExperimentArtifactStore(artifact_root)
    fixed = shared_store.prepare_fixed_validation(dataset, config)
    target = model_target(dataset, config["mode"])
    tuning_train, tuning_train_seed = _tuning_train_subset(dataset, fixed, config, event_identity)

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
        "hyperparameter_tuning_train_size": int(len(tuning_train)),
        "hyperparameter_tuning_train_seed": int(tuning_train_seed),
        "replica_train_size": int(_replica_train_size(dataset, fixed, config)),
        "shared_replicas": {},
    })
    store.write_manifest(manifest)

    selected_validation_rmse = None
    parameter_codes = {}
    shared_transform_seed_base = semantic_seed(
        int(config["seed"]), "hyperparameter_tuning", spec.name, protocol_identity
    )

    if optimization.strategy == "fixed":
        selected_candidate, selected_params = next(iter(candidates.items()))
        store.write_selected_hyperparameters({
            "candidate_id": selected_candidate,
            "parameters": selected_params,
            "selection_metric": None,
            "selected_validation_rmse_ps": None,
            "validation_seed": None,
            "validation_fraction": None,
            "validation_skipped": True,
            "validation_reused_in_replicas": False,
            "tuning_train_size": int(len(tuning_train)),
            "tuning_train_seed": int(tuning_train_seed),
        })
        logger.info(
            "Fixed configuration | validation fit/evaluation skipped | %s",
            _format_parameter_pairs(selected_params),
        )
    else:
        validation_target = np.asarray(target[fixed.split.validation], float)
        raw_validation_rmse = float(rmse_ps(validation_target))
        logger.info(
            "Hyperparameter tuning | strategy=%s | fixed split seed=%d | tuning_train=%d/%d sampled to match replica train | validation=%d | metric=RMSE | validation excluded from every replica",
            optimization.strategy,
            int(fixed.split.seed),
            len(tuning_train),
            len(fixed.split.tuning_train),
            len(fixed.split.validation),
        )

        if optimization.strategy == "grid":
            parameter_codes = _log_hyperparameter_space(logger, candidates)
        else:
            logger.info(
                "Optuna TPE | trials=%d | startup_trials=%d",
                int(optimization.n_trials),
                int(optimization.n_startup_trials),
            )

        transform_cache = FeatureTransformCache()
        fit_input_cache = FitInputCache()
        transform_seed_base = shared_transform_seed_base

        def evaluate_candidate(params, label):
            nonlocal candidates
            params = dict(params)
            identifier = candidate_id(params)
            candidates[identifier] = params
            candidates = candidate_manifest(candidates.values())
            store.write_candidates(candidates)

            existing = next(
                (
                    row
                    for row in store.read_results()
                    if row.get("phase") == "hyperparameter_validation"
                    and row.get("candidate_id") == identifier
                ),
                None,
            )
            if existing is not None:
                return float(existing["rmse_ps"])

            logger.info("Hyperparameter candidate %s | %s", label, _format_parameter_pairs(params))
            fitted = prediction = corrected = None
            try:
                fitted = fit_on_indices(
                    spec,
                    model_space,
                    config,
                    dataset,
                    tuning_train,
                    params,
                    seed=semantic_seed(fixed.split.seed, spec.name, identifier, "tuning_fit"),
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
                    chunk_size=prediction_chunk_size,
                )
                corrected = validation_target - prediction
                row = _validation_row(
                    seed=fixed.split.seed,
                    candidate_id=identifier,
                    selected=False,
                    corrected=corrected,
                    raw_validation_rmse=raw_validation_rmse,
                    spec=spec,
                    config=config,
                    event_identity=event_identity,
                    protocol_identity=protocol_identity,
                    sampling_identity=fixed.sampling_identity,
                    train_n=len(tuning_train),
                )
                store.upsert_result(row)
                logger.info("Hyperparameter validation | RMSE=%.3f ps", row["rmse_ps"])
                return float(row["rmse_ps"])
            finally:
                del corrected, prediction, fitted
                release_training_memory()

        if optimization.strategy == "grid":
            for index, (_, params) in enumerate(list(candidates.items()), 1):
                evaluate_candidate(params, f"{index}/{len(candidates)}")
        else:
            try:
                import optuna
            except ImportError as exc:
                raise ImportError(
                    "Optuna optimization requires the 'optuna' package from waveform_analysis/requirements.txt"
                ) from exc

            optuna.logging.set_verbosity(optuna.logging.WARNING)
            sampler_seed = (
                int(optimization.seed)
                if optimization.seed is not None
                else semantic_seed(int(config["seed"]), "optuna_tpe", spec.name, protocol_identity)
            )
            sampler = optuna.samplers.TPESampler(
                seed=sampler_seed,
                n_startup_trials=int(optimization.n_startup_trials),
            )
            storage_path = (run_dir / "optuna_study.db").resolve()
            study = optuna.create_study(
                study_name="hyperparameter_optimization",
                direction="minimize",
                sampler=sampler,
                storage=f"sqlite:///{storage_path.as_posix()}",
                load_if_exists=True,
            )

            def objective(trial):
                params = suggest_parameters(trial, model_space)
                return evaluate_candidate(params, f"trial={trial.number + 1}/{optimization.n_trials}")

            remaining = max(0, int(optimization.n_trials) - len(study.trials))
            if remaining:
                study.optimize(objective, n_trials=remaining, gc_after_trial=True)
            logger.info(
                "Optuna TPE complete | trials=%d/%d | sampler_seed=%d",
                len(study.trials),
                int(optimization.n_trials),
                sampler_seed,
            )

        scores = []
        for row in store.read_results():
            identifier = row.get("candidate_id")
            if (
                row.get("phase") == "hyperparameter_validation"
                and identifier in candidates
                and str(row.get("rmse_ps", "")) not in ("", "nan")
            ):
                scores.append(CandidateScore(identifier, float(row["rmse_ps"])))

        best = choose_best(scores)
        selected_candidate = best.candidate_id
        selected_params = candidates[selected_candidate]
        selected_validation_rmse = float(best.score)
        rows = store.read_results()
        for row in rows:
            if row.get("phase") == "hyperparameter_validation":
                row["selected"] = row.get("candidate_id") == selected_candidate
        store._atomic_rows(rows)
        store.write_selected_hyperparameters({
            "candidate_id": selected_candidate,
            "parameters": selected_params,
            "selection_metric": "fixed_validation_rmse_ps",
            "selected_validation_rmse_ps": selected_validation_rmse,
            "validation_seed": int(fixed.split.seed),
            "validation_fraction": float(config["model_selection"]["validation_fraction"]),
            "validation_skipped": False,
            "validation_reused_in_replicas": False,
            "optimization_strategy": optimization.strategy,
            "tuning_train_size": int(len(tuning_train)),
            "tuning_train_seed": int(tuning_train_seed),
        })
        logger.info(
            "Hyperparameter selection complete | %s | validation RMSE=%.3f ps | selected configuration used for replica evaluation",
            _format_candidate_codes(selected_params, parameter_codes),
            selected_validation_rmse,
        )

        transform_cache.clear()
        fit_input_cache.clear()
        del transform_cache, fit_input_cache, validation_target
        release_training_memory()

    fixed_minirocket_transform = None
    if spec.name in {"direct_minirocket", "shared_minirocket"}:
        fixed_minirocket_transform = _fit_fixed_feature_transform(
            spec,
            model_space,
            config,
            dataset,
            tuning_train,
            selected_params,
            transform_seed_base=shared_transform_seed_base,
            logger=logger,
        )
        manifest["fixed_feature_transform_identity"] = fixed_minirocket_transform.identity
        manifest["fixed_feature_transform_seed"] = int(fixed_minirocket_transform.seed)
        manifest["fixed_feature_transform_fit_events"] = int(len(tuning_train))
        store.write_manifest(manifest)
        logger.info(
            "MiniRocket transform fixed for all replicas | id=%s | fit_events=%d | seed=%d",
            fixed_minirocket_transform.identity[:12],
            len(tuning_train),
            int(fixed_minirocket_transform.seed),
        )

    progress = ProgressTracker(logger, {"replica": n_replicas})
    replica_protocol_logged = False
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
            del split, replica
            continue

        if not replica_protocol_logged:
            logger.info(
                "Replica protocol | fit=once per replica | train=%d | blind=%d | fixed validation excluded | feature_transform=%s | model_save=%s | prediction_chunk=%d",
                len(split.train),
                len(split.test),
                "fixed across replicas" if fixed_minirocket_transform is not None else "fit with model",
                config["save_models"],
                prediction_chunk_size,
            )
            replica_protocol_logged = True
        logger.info(
            "Replica %d/%d | seed=%d",
            replica_index,
            n_replicas,
            int(split.seed),
        )
        replica_transform_cache = FeatureTransformCache()
        replica_fit_input_cache = FitInputCache()
        fitted = prediction = corrected = paired = None
        try:
            if fixed_minirocket_transform is not None:
                fitted = _fit_with_fixed_feature_transform(
                    spec,
                    model_space,
                    config,
                    dataset,
                    split.train,
                    selected_params,
                    seed=semantic_seed(split.seed, spec.name, selected_candidate, "replica_fit"),
                    feature_transform=fixed_minirocket_transform,
                    logger=logger,
                )
            else:
                fitted = fit_on_indices(
                    spec,
                    model_space,
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

            prediction = predict_indices(
                spec,
                fitted,
                dataset,
                config["mode"],
                split.test,
                chunk_size=prediction_chunk_size,
            )
            corrected = np.asarray(target[split.test], float) - prediction
            swap = detector_swap_rmse(
                spec,
                fitted,
                dataset,
                config["mode"],
                split.test,
                forward_prediction=prediction,
                chunk_size=prediction_chunk_size,
            )
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
                logger.info("Replica model saved | replica=%d", replica_index)

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
        finally:
            replica_transform_cache.clear()
            replica_fit_input_cache.clear()
            del paired, corrected, prediction, fitted
            del replica_transform_cache, replica_fit_input_cache, split, replica
            release_training_memory()

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
        "selected_validation_rmse_ps": selected_validation_rmse,
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
    if selected_validation_rmse is None:
        logger.info(
            "Study complete | fixed configuration | replicas=%d | blind CTR mean=%.3f ± %.3f ps",
            manifest["replica_count"],
            manifest["blind_ctr_mean_ps"],
            manifest["blind_ctr_std_ps"],
        )
    else:
        logger.info(
            "Study complete | validation RMSE=%.3f ps | replicas=%d | blind CTR mean=%.3f ± %.3f ps",
            selected_validation_rmse,
            manifest["replica_count"],
            manifest["blind_ctr_mean_ps"],
            manifest["blind_ctr_std_ps"],
        )
    return run_dir
