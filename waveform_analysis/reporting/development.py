"""Backfill training diagnostics by inference on existing prepared caches only."""

from __future__ import annotations

import json
from pathlib import Path, PureWindowsPath
import re
from typing import Any

from waveform_analysis.core.io import canonical_hash
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


def _development_cache_root(config: dict[str, Any], run_directory: Path) -> Path:
    """Resolve saved paths, including a checkout moved after training."""
    configured = Path(config["preprocessing"]["cache_dir"]).expanduser()
    candidates = [configured]
    parts = PureWindowsPath(str(configured)).parts
    anchors = [i for i, part in enumerate(parts) if part.lower() == "waveform_analysis"]
    if anchors:
        suffix = Path(*parts[anchors[-1] + 1:])
        roots = [p for p in run_directory.parents if p.name.lower() == "waveform_analysis"]
        roots.append(Path(__file__).resolve().parents[1])
        candidates.extend(root / suffix for root in roots)
    for candidate in candidates:
        if (candidate / "development_ml" / "prepared").is_dir():
            return candidate
    return configured


def _matches_training_identity(
    cached: dict[str, Any], run: dict[str, Any], config: dict[str, Any], spec: Any,
    control_fingerprint: str | None,
) -> bool:
    """Verify cached preprocessing without requiring the control JSON itself."""
    manifest = run["manifest"]
    population = manifest.get("dataset_populations", {}).get("development", {})
    protocol = population.get("analysis_protocol_identity")
    if protocol is not None:
        return cached.get("analysis_protocol_identity") == protocol
    if control_fingerprint is not None:
        return cached.get("control_fingerprint") == control_fingerprint
    # Control artifact directories are named by the first 16 hash characters.
    prefix = PureWindowsPath(manifest.get("control_artifact", "")).name
    if re.fullmatch(r"[0-9a-f]{16}", prefix):
        return str(cached.get("control_fingerprint", "")).startswith(prefix)
    # Older neural runs already bind the exact development protocol into final_fit.
    if spec.feature_transform is None and spec.selection_method != "ridge_cv":
        if not cached.get("analysis_protocol_identity"):
            return False
        best = run["best"]
        payload = {
            "development": cached["analysis_protocol_identity"],
            "model": config["model"], "candidate_id": best["candidate_id"],
            "parameters": best["parameters"], "seed": config["seed"],
            "frozen_transform": None,
        }
        expected = canonical_hash({"schema_version": manifest["schema_version"],
                                   "stage": "final_fit", "payload": payload})
        return expected == manifest["stage_fingerprints"]["final_fit"]
    return False


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
    control_path = Path(manifest.get("control_artifact", "")) / "manifest.json"
    control_fingerprint = (
        json.loads(control_path.read_text()).get("fingerprint") if control_path.is_file() else None
    )
    population = manifest.get("dataset_populations", {}).get("development", {})
    expected_population = population.get("population_identity")
    cache_root = _development_cache_root(config, store.root) / "development_ml" / "prepared"
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
            and _matches_training_identity(cached, run, config, spec, control_fingerprint)
            and (expected_population is None
                 or cached.get("event_population_identity") == expected_population)
        ):
            matches.append(path.parent)
    if len(matches) != 1:
        return unavailable(
            f"expected one matching prepared development cache, found {len(matches)} in {cache_root}; "
            "retain the prepared train arrays to predict with the saved model"
        )
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
