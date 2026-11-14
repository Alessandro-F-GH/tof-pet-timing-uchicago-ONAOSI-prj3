"""Load saved development predictions and prepared waveforms for diagnostics.

No model inference, feature fitting or access to raw acquisitions.
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from waveform_analysis.data.prepared_cache import PreparedCacheLookup
from waveform_analysis.data.results import read_run, resolve_run_config
from waveform_analysis.data.storage import RunStore
from waveform_analysis.data.view import mode_family


def _logger(output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("saved_prediction_diagnostics")
    logger.handlers.clear()
    logger.setLevel(logging.INFO)
    logger.propagate = False
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    for handler in (logging.StreamHandler(), logging.FileHandler(output / "evaluation.log", mode="w")):
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger


def _prediction_payload(path, dataset, manifest):
    """Align saved predictions with prepared development event IDs."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Saved development predictions missing: {path}")
    with np.load(path, allow_pickle=False) as archive:
        required = ("event_id", "prediction_ps", "final_fit_fingerprint",
                    "analysis_protocol_identity", "dataset_role")
        missing = [name for name in required if name not in archive]
        if missing:
            raise ValueError(f"Development prediction file lacks {missing}: {path}")
        ids = np.asarray(archive["event_id"], dtype=np.int64)
        predictions = np.asarray(archive["prediction_ps"], dtype=np.float64)
        metadata = {key: str(np.asarray(archive[key]).item()) for key in required[2:]}
    expected = np.asarray(dataset.event_index, dtype=np.int64)
    if ids.ndim != 1 or predictions.shape != ids.shape or ids.size != expected.size:
        raise ValueError("Saved prediction/event count differs from prepared Development")
    if np.unique(ids).size != ids.size or np.unique(expected).size != expected.size:
        raise ValueError("Duplicate event IDs prevent unambiguous alignment")
    if metadata["dataset_role"] != "development":
        raise ValueError("Saved predictions must belong to Development")
    final_fit = manifest.get("stage_fingerprints", {}).get("final_fit")
    if not final_fit or metadata["final_fit_fingerprint"] != str(final_fit):
        raise ValueError("Saved predictions do not match final fitted model fingerprint")
    order = np.argsort(ids, kind="stable")
    if not np.array_equal(ids[order], np.sort(expected)):
        raise ValueError("Saved and prepared Development event IDs differ")
    aligned = predictions[order[np.searchsorted(ids[order], expected)]]
    if not np.isfinite(aligned).all():
        raise ValueError("Saved predictions contain non-finite values")
    return aligned


def _load_inputs(args, log):
    run_path = Path(args.model_dir).expanduser().resolve()
    run = read_run(run_path)
    config = resolve_run_config(run_path)
    cache_root = getattr(args, "cache_root", None)
    if cache_root is not None:
        config["preprocessing"]["cache_dir"] = str(Path(cache_root).expanduser().resolve() / "preprocessing")

    log.info("Saved input | run=%s | model=%s | mode=%s",
             run_path, run["manifest"]["model"], config["mode"])
    dataset = PreparedCacheLookup().find("development", config, run["manifest"])
    family = mode_family(config["mode"])
    waves = getattr(dataset, f"{family}_windows")
    if waves is None or waves.ndim != 3 or waves.shape[1] != 2 or waves.shape[0] != dataset.n_events:
        raise ValueError("Expected Development waveforms [events, 2 detectors, samples]")
    store = RunStore(run_path)
    predictions = _prediction_payload(store.development_predictions_path, dataset, run["manifest"])
    keep = np.ones(waves.shape[-1], dtype=bool) if config["mode"] == "energy_to_energy" else dataset.sample_mask
    if keep is None or np.asarray(keep).shape != (waves.shape[-1],):
        raise ValueError("Timing sample mask is missing or has incompatible shape")
    log.info("Prepared waveforms | shape=%s | source=%s", waves.shape, dataset.directory)
    log.info("Saved predictions | events=%d | path=%s", len(predictions), store.development_predictions_path)
    return waves, predictions, np.asarray(keep, dtype=bool), {
        "run": str(run_path),
        "model": run["manifest"]["model"],
        "mode": config["mode"],
        "dataset": str(dataset.directory),
        "prediction_artifact": str(store.development_predictions_path),
        "final_fit_fingerprint": run["manifest"]["stage_fingerprints"]["final_fit"],
        "analysis_protocol_identity": dataset.manifest.get("analysis_protocol_identity"),
        "reference_events": int(dataset.n_events),
    }
