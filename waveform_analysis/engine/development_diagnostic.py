"""Backfill training diagnostics by inference on existing prepared caches only."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from waveform_analysis.core.io import canonical_hash
from waveform_analysis.data.prepared_cache import PreparedCacheLookup
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


def ensure_development_predictions(run: dict[str, Any], *, logger=None) -> Path | None:
    """Use saved predictions or a saved model and uniquely matching cache.

    Never fit a model, transform or preprocessing rule, or load raw acquisitions.
    Missing prerequisites leave the diagnostic unavailable and are logged.
    """
    manifest = run["manifest"]
    config = resolve_run_config(run["directory"])
    store = RunStore(run["directory"])
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
    try:
        dataset = PreparedCacheLookup().find("development", config, manifest)
    except (KeyError, OSError, ValueError, RuntimeError) as exc:
        return unavailable(f"cannot load prepared development cache: {exc}")
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
