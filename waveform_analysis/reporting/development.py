"""Backfill training diagnostics by inference on existing prepared caches only."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from waveform_analysis.data.dataset import load_prepared_dataset
from waveform_analysis.data.storage import RunStore
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
    store = RunStore(run["directory"])
    manifest = run["manifest"]
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
    config = json.loads(config_path.read_text())
    spec = get_model(manifest["model"])
    if not saved_model_complete(spec, store.model_dir):
        return unavailable("saved final model is missing or incomplete")
    control_path = Path(manifest["control_artifact"]) / "manifest.json"
    if not control_path.is_file():
        return unavailable("control metadata needed to identify the cache is missing")
    control = json.loads(control_path.read_text())
    population = manifest.get("dataset_populations", {}).get("development", {})
    expected_population = population.get("population_identity")
    cache_root = Path(config["preprocessing"]["cache_dir"]) / "development_ml" / "prepared"
    matches = []
    for path in cache_root.glob("*/manifest.json"):
        cached = json.loads(path.read_text())
        if (
            cached.get("dataset_role") == "development"
            and cached.get("dataset_source") == config["development"]["root_file"]
            and cached.get("mode") == config["mode"]
            and cached.get("window_ns") == config["window_ns"]
            and cached.get("subsampling") == config["ml_input"]["subsampling"]
            and cached.get("true_tof_ps") == config["development"]["true_tof_ps"]
            and cached.get("control_fingerprint") == control["fingerprint"]
            and (expected_population is None
                 or cached.get("event_population_identity") == expected_population)
        ):
            matches.append(path.parent)
    if len(matches) != 1:
        return unavailable(f"expected one matching prepared development cache, found {len(matches)}")
    dataset = load_prepared_dataset(matches[0])
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
