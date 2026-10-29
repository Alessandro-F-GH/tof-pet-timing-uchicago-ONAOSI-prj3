"""Prepare reusable batch preprocessing and frozen-feature caches without fitting models."""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from waveform_analysis.core.io import atomic_json, canonical_hash
from waveform_analysis.core.config import BatchConfig
from waveform_analysis.data.feature_cache import (
    prepare_frozen_features,
    prepare_frozen_transform,
    uses_frozen_model_input,
)
from waveform_analysis.engine.batch import _runtime_configs
from waveform_analysis.engine.preprocessing import fit_control, prepare_role_dataset
from waveform_analysis.models import get_model

CACHE_MANIFEST_VERSION = 1


def _batch_identity(batch: BatchConfig, configs: list[dict[str, Any]]) -> str:
    """Describe the requested work without binding cache identity to local paths."""
    runs = []
    for config in configs:
        datasets = {}
        for role in ("control", "development", "blind"):
            datasets[role] = {
                key: value
                for key, value in config[role].items()
                if key != "root_file"
            }
        runs.append({
            "run_id": config["run_id"],
            "mode": config["mode"],
            "window_name": config.get("window_name"),
            "window_ns": config["window_ns"],
            "seed": config["seed"],
            "model": config["model"],
            "datasets": datasets,
            "fit": config["fit"],
            "ml_input": config["ml_input"],
            "preprocessing": {
                key: value
                for key, value in config["preprocessing"].items()
                if key not in {"cache_dir", "artifact_root"}
            },
        })
    return canonical_hash({"schema_version": CACHE_MANIFEST_VERSION, "batch": batch.name, "runs": runs})


def _write_manifest(path: Path, payload: dict[str, Any]) -> None:
    atomic_json(path, payload)


def prepare_batch_cache(
    batch: BatchConfig,
    *,
    cache_root: str | Path,
    models: list[str] | tuple[str, ...] | None = None,
    logger=None,
) -> Path:
    """Populate the reusable cache dependencies for every run in a resolved batch.

    This deliberately stops before CV, hyperparameter selection, supervised model
    fitting, blind prediction, bootstrap uncertainty, and report generation.
    """
    if not isinstance(batch, BatchConfig):
        raise TypeError("prepare_batch_cache requires a resolved BatchConfig")

    root = Path(cache_root).expanduser().resolve()
    preprocessing_root = root / "preprocessing"
    artifact_root = root / "artifact_store"
    preprocessing_root.mkdir(parents=True, exist_ok=True)
    artifact_root.mkdir(parents=True, exist_ok=True)
    manifest_path = root / "cache_manifest.json"

    configs = _runtime_configs(batch)
    if models:
        selected = {str(name) for name in models}
        available = {str(config["model"]["name"]) for config in configs}
        unknown = selected - available
        if unknown:
            raise ValueError(
                f"Unknown model(s) requested for cache preparation: {sorted(unknown)}. "
                f"Available models: {sorted(available)}"
            )
        configs = [
            config for config in configs
            if str(config["model"]["name"]) in selected
        ]
    if not configs:
        raise ValueError("Cache preparation selected no runs")
    for config in configs:
        config["preprocessing"]["cache_dir"] = str(preprocessing_root)
        config["preprocessing"]["artifact_root"] = str(artifact_root)

    batch_fingerprint = _batch_identity(batch, configs)
    records: list[dict[str, Any]] = []
    manifest: dict[str, Any] = {
        "schema_version": CACHE_MANIFEST_VERSION,
        "status": "preparing",
        "batch_name": batch.name,
        "batch_fingerprint": batch_fingerprint,
        "layout": {
            "preprocessing": "preprocessing/",
            "artifact_store": "artifact_store/",
        },
        "prepared_runs": records,
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }
    _write_manifest(manifest_path, manifest)

    try:
        for index, config in enumerate(configs, start=1):
            run_id = str(config["run_id"])
            model_name = str(config["model"]["name"])
            spec = get_model(model_name)
            if logger:
                logger.info(
                    "Cache preparation | %d/%d | %s | %s | %s",
                    index, len(configs), config["mode"],
                    config.get("window_name"), model_name,
                )

            control_artifact, _ = fit_control(config, rebuild=False, logger=logger)
            prepared_datasets = {}
            for role in ("control", "development", "blind"):
                _, _, dataset = prepare_role_dataset(
                    config, role, control_artifact, rebuild=False, logger=logger
                )
                prepared_datasets[role] = dataset

            run_record: dict[str, Any] = {
                "run_id": run_id,
                "model": model_name,
                "mode": str(config["mode"]),
                "window_name": config.get("window_name"),
                "datasets": {
                    role: str(dataset.manifest["analysis_protocol_identity"])
                    for role, dataset in prepared_datasets.items()
                },
            }

            if uses_frozen_model_input(spec):
                frozen_transform = prepare_frozen_transform(
                    spec,
                    config["model"]["space"],
                    config,
                    control_dataset=prepared_datasets["control"],
                    development_dataset=prepared_datasets["development"],
                    cache_root=preprocessing_root,
                    logger=logger,
                )
                if frozen_transform is None:
                    raise RuntimeError(
                        f"{model_name} declares frozen model input but produced no transform"
                    )
                run_record["transform_identity"] = frozen_transform.identity
                run_record["fit_role"] = frozen_transform.fit_role
                run_record["feature_matrices"] = {}
                for role in ("development", "blind"):
                    features = prepare_frozen_features(
                        spec,
                        frozen_transform,
                        config,
                        prepared_datasets[role],
                        role=role,
                        cache_root=preprocessing_root,
                        logger=logger,
                    )
                    run_record["feature_matrices"][role] = features.identity

            records.append(run_record)
            manifest["status"] = "running"
            manifest["prepared_runs"] = records
            manifest["updated_at"] = datetime.now(timezone.utc).isoformat()
            _write_manifest(manifest_path, manifest)

        manifest["status"] = "complete"
        manifest["updated_at"] = datetime.now(timezone.utc).isoformat()
        _write_manifest(manifest_path, manifest)
        if logger:
            logger.info(
                "Cache preparation complete | runs=%d | root=%s",
                len(configs), root,
            )
        return manifest_path
    except Exception as exc:
        manifest["status"] = "failed"
        manifest["error"] = f"{type(exc).__name__}: {exc}"
        manifest["prepared_runs"] = records
        manifest["updated_at"] = datetime.now(timezone.utc).isoformat()
        _write_manifest(manifest_path, manifest)
        raise
