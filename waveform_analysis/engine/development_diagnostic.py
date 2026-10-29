"""Backfill training diagnostics by inference on existing prepared caches only."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from waveform_analysis.core.io import canonical_hash
from waveform_analysis.data.dataset import load_prepared_dataset
from waveform_analysis.data.artifact_store import ArtifactStore
from waveform_analysis.data.sample_mask import attach_control_sample_mask
from waveform_analysis.data.storage import RunStore
from waveform_analysis.data.results import resolve_run_config
from waveform_analysis.engine.diagnostics import (
    development_predictions_current,
    diagnostic_random_state,
    save_development_diagnostic,
)
from waveform_analysis.engine.train import (
    load_fitted_model,
    release_training_memory,
    saved_model_complete,
)
from waveform_analysis.models import get_model


def ensure_development_predictions(run: dict[str, Any], *, logger=None, artifact_root=None) -> Path | None:
    """Use saved predictions or a saved model and uniquely matching cache.

    Never fit a model, transform or preprocessing rule, or load raw acquisitions.
    Missing prerequisites leave the diagnostic unavailable and are logged.
    """
    manifest = run["manifest"]
    config = resolve_run_config(run["directory"])
    if artifact_root is not None:
        config["preprocessing"]["artifact_root"] = str(Path(artifact_root).expanduser().resolve())
    store = RunStore(
        run["directory"],
        artifact_root=config["preprocessing"]["artifact_root"],
    )
    fingerprint = manifest.get("stage_fingerprints", {}).get("final_fit")
    if fingerprint and development_predictions_current(store, fingerprint):
        return store.development_predictions_path

    def unavailable(reason: str):
        if logger:
            logger.warning("Development distribution unavailable | %s | %s",
                           manifest["model"], reason)
        return None

    config_path = store.metadata_dir / "config.json"
    if not fingerprint or not config_path.is_file():
        return unavailable("missing final-fit identity or saved configuration")
    spec = get_model(manifest["model"])
    if not saved_model_complete(spec, store.model_dir):
        return unavailable("saved final model is missing or incomplete")
    references = manifest.get("prepared_artifact_refs", {})
    development_ref = references.get("development")
    if not development_ref:
        return unavailable("the run manifest has no prepared development artifact reference")
    artifact_store = ArtifactStore(config["preprocessing"]["artifact_root"])
    try:
        development_reference = artifact_store.resolve(
            development_ref["product_type"], development_ref["product_id"]
        )
        dataset = load_prepared_dataset(development_reference.path)
        control_dataset = None
        if config["mode"] == "timing_to_timing":
            control_ref = references.get("control")
            if not control_ref:
                return unavailable("the run manifest has no prepared control artifact reference")
            control_reference = artifact_store.resolve(
                control_ref["product_type"], control_ref["product_id"]
            )
            control_dataset = load_prepared_dataset(control_reference.path)
            control_dataset = attach_control_sample_mask(
                control_dataset, control_dataset,
                cache_root=Path(config["preprocessing"]["cache_dir"]),
                artifact_root=Path(config["preprocessing"]["artifact_root"]),
            )
        dataset = attach_control_sample_mask(
            dataset, control_dataset,
            cache_root=Path(config["preprocessing"]["cache_dir"]),
            artifact_root=Path(config["preprocessing"]["artifact_root"]),
        )
        expected_protocol = (
            manifest.get("dataset_populations", {}).get("development", {})
            .get("analysis_protocol_identity")
        )
        if (
            expected_protocol is not None
            and dataset.manifest.get("analysis_protocol_identity") != expected_protocol
        ):
            return unavailable("prepared development artifact protocol does not match the run")
    except (KeyError, OSError, ValueError, RuntimeError) as exc:
        return unavailable(f"cannot resolve prepared artifact references: {exc}")
    if logger:
        logger.info("Development distribution | saved-model inference | %s | n=%d",
                    spec.name, dataset.n_events)
    with diagnostic_random_state():
        fitted = load_fitted_model(spec, store.model_dir, run["best"]["parameters"], config)
        try:
            return save_development_diagnostic(store, spec, fitted, dataset, config, fingerprint)
        finally:
            del fitted
            release_training_memory()
