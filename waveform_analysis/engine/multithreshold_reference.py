"""Fixed, target-independent multi-threshold reference for energy batch reports.

A reference is produced once per energy window from the same persisted prepared
cache as the ML models. No threshold search or ML model inference is performed.
Only the scalar offset is calibrated on development labels (as for LED).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from waveform_analysis.core.io import atomic_json, canonical_hash, write_csv
from waveform_analysis.data.prepared_cache import PreparedCacheLookup
from waveform_analysis.data.results import read_run, resolve_run_config
from waveform_analysis.data.view import model_target
from waveform_analysis.data.splits import semantic_seed
from waveform_analysis.engine.multithreshold_led import (
    multithreshold_corrections, validate_threshold_sets,
)
from waveform_analysis.reporting.stats import (
    metric_values, blind_event_bootstrap, paired_model_bootstrap,
)
from waveform_analysis.scripts.plot_conditional_target_heatmaps import baseline_indices


DEFAULT_REFERENCE = {
    "enabled": False,
    "thresholds_mV": [10., 20., 30., 40., 50.],
    "baseline_window_ns": [-1.4, -1.0],
    "crossing_window_ns": [-0.8, 1.8],
    "minimum_coverage": 0.90,
    "chunk_size": 128,
}


def normalize_reference(value):
    from waveform_analysis.core.exceptions import ConfigError
    if value is None:
        value = {}
    if not isinstance(value, dict):
        raise ConfigError("multithreshold_reference must be an object")
    unknown = set(value) - set(DEFAULT_REFERENCE)
    if unknown:
        raise ConfigError(f"Unknown multithreshold_reference fields: {sorted(unknown)}")
    result = {**DEFAULT_REFERENCE, **value}
    if not isinstance(result["enabled"], bool):
        raise ConfigError("multithreshold_reference.enabled must be boolean")
    try:
        groups = validate_threshold_sets([result["thresholds_mV"]])
        result["thresholds_mV"] = list(groups[0])
        for name in ("baseline_window_ns", "crossing_window_ns"):
            pair = tuple(float(v) for v in result[name])
            if len(pair) != 2 or not np.all(np.isfinite(pair)) or pair[0] >= pair[1]:
                raise ValueError(f"{name} must have two increasing finite endpoints")
            result[name] = list(pair)
        coverage = float(result["minimum_coverage"])
        if not 0 < coverage <= 1:
            raise ValueError("minimum_coverage must be within (0,1]")
        result["minimum_coverage"] = coverage
        result["chunk_size"] = int(result["chunk_size"])
        if result["chunk_size"] < 1:
            raise ValueError("chunk_size must be positive")
    except (TypeError, ValueError) as exc:
        raise ConfigError(f"Invalid multithreshold_reference: {exc}") from exc
    return result


def reference_directory(root, window):
    return Path(root) / "references" / "energy" / str(window)


def _calculate(dataset, options):
    waves = dataset.energy_windows
    grid = np.asarray(dataset.energy_time_ps, np.float64)
    transform = dataset.energy_transform
    if waves is None or transform is None:
        raise ValueError("Missing energy prepared waveform or transform")
    if waves.shape != (dataset.n_events, 2, grid.size):
        raise ValueError("Prepared energy waveform shape mismatch")
    bidx = baseline_indices(grid, *options["baseline_window_ns"])
    scale = np.asarray(transform.maximum - transform.minimum, np.float64).reshape(2, -1)
    if scale.shape != (2, 1):
        raise ValueError("Expected one physical amplitude scale per detector")
    correction, diagnostics = multithreshold_corrections(
        waves, grid, threshold_sets=[options["thresholds_mV"]],
        baseline_indices=bidx,
        led_threshold_mV=float(dataset.manifest["fixed_led_threshold_mV"]),
        physical_scale_mV=scale[:, 0],
        crossing_window_ns=options["crossing_window_ns"],
        chunk_size=options["chunk_size"],
    )
    return correction[:, 0], diagnostics


def _check_protocol(dev, blind):
    for key in ("control_mode_fingerprint", "target_definition"):
        if dev.manifest[key] != blind.manifest[key]:
            raise ValueError(f"MT development/blind mismatch: {key}")
    for key in ("fixed_led_threshold_mV", "led_control_mean_ps", "true_tof_ps"):
        if not np.isclose(float(dev.manifest[key]), float(blind.manifest[key]), rtol=0, atol=1e-3):
            raise ValueError(f"MT development/blind mismatch: {key}")


def _save_predictions(path, dataset, raw, offset, options, role):
    target = np.asarray(model_target(dataset, "energy_to_energy"), np.float64)
    valid = np.isfinite(raw) & np.isfinite(target)
    prediction = np.asarray(raw, np.float64) + offset
    prediction[~valid] = np.nan
    np.savez_compressed(
        path, event_id=np.asarray(dataset.event_index, np.int64),
        prediction_ps=prediction, corrected_ps=target-prediction,
        led_residual_ps=target, valid=valid,
        thresholds_mV=np.asarray(options["thresholds_mV"], np.float64),
        development_offset_ps=np.asarray(offset),
        dataset_role=np.asarray(role),
        event_population_identity=np.asarray(dataset.manifest["event_population_identity"]),
        analysis_protocol_identity=np.asarray(dataset.manifest["analysis_protocol_identity"]),
    )
    return valid, target-prediction, target


def run_reference(reference_run, root, options, *, seed, bootstrap_resamples):
    """Compute or reuse an independent energy reference without altering ML results."""
    run = Path(reference_run)
    config = resolve_run_config(run)
    if config["mode"] != "energy_to_energy":
        raise ValueError("Multithreshold reference only supports energy_to_energy")
    manifest = read_run(run)["manifest"]
    loader = PreparedCacheLookup()
    dev = loader.find("development", config, manifest)
    blind = loader.find("blind", config, manifest)
    _check_protocol(dev, blind)
    destination = reference_directory(root, config["window_name"])
    fingerprint = canonical_hash({
        "version": 1, "options": options, "seed": seed,
        "bootstrap_resamples": bootstrap_resamples,
        "cache_development": str(dev.directory),
        "cache_blind": str(blind.directory),
        "dev_protocol": dev.manifest["analysis_protocol_identity"],
        "blind_protocol": blind.manifest["analysis_protocol_identity"],
        "dev_population": dev.manifest["event_population_identity"],
        "blind_population": blind.manifest["event_population_identity"],
        "fit": config["fit"],
    })
    summary_path = destination / "summary.json"
    if summary_path.is_file():
        existing = json.loads(summary_path.read_text(encoding="utf-8"))
        if (existing.get("fingerprint") == fingerprint and
            all((destination / f"{role}_predictions.npz").is_file()
                for role in ("development", "blind"))):
            return existing
    destination.mkdir(parents=True, exist_ok=True)
    raw_dev, _ = _calculate(dev, options)
    y_dev = np.asarray(model_target(dev, "energy_to_energy"), np.float64)
    good = np.isfinite(raw_dev) & np.isfinite(y_dev)
    dev_coverage = float(good.mean())
    if good.sum() < 5 or dev_coverage < options["minimum_coverage"]:
        raise ValueError(f"MT development coverage {dev_coverage:.2%} below minimum")
    # Fixed calibration only: zero-mean residual on development. No target-guided
    # threshold optimization, event filtering, or blind adjustment.
    offset = float(np.mean(y_dev[good] - raw_dev[good]))
    valid_dev, resid_dev, _ = _save_predictions(
        destination / "development_predictions.npz", dev, raw_dev, offset,
        options, "development")
    dev_metrics = metric_values(resid_dev[valid_dev], config["fit"],
                                seed=semantic_seed(seed, "mt-reference", "dev"))
    raw_blind, _ = _calculate(blind, options)
    valid_blind, residual, led = _save_predictions(
        destination / "blind_predictions.npz", blind, raw_blind, offset,
        options, "blind")
    coverage = float(valid_blind.mean())
    if valid_blind.sum() < 5 or coverage < options["minimum_coverage"]:
        raise ValueError(f"MT blind coverage {coverage:.2%} below minimum")
    bootstrap, draws = blind_event_bootstrap(
        residual[valid_blind], led[valid_blind], config["fit"],
        n_resamples=bootstrap_resamples,
        seed=semantic_seed(seed, "mt-reference", "blind"))
    np.savez_compressed(destination / "blind_bootstrap.npz", **draws)
    summary = {
        "method": "fixed_prepared_multithreshold_led", "fingerprint": fingerprint,
        "mode": "energy_to_energy", "window": config["window_name"],
        "thresholds_mV": options["thresholds_mV"],
        "baseline_window_ns": options["baseline_window_ns"],
        "crossing_window_ns": options["crossing_window_ns"],
        "offset_calibration": "development_residual_mean_only",
        "development_offset_ps": offset,
        "development": {"total": int(dev.n_events), "valid": int(valid_dev.sum()),
                        "coverage": dev_coverage, "metrics": dev_metrics},
        "blind": {"total": int(blind.n_events), "valid": int(valid_blind.sum()),
                  "coverage": coverage, "metrics": bootstrap},
        "source_prepared_cache": {"development": str(dev.directory),
                                  "blind": str(blind.directory)},
    }
    atomic_json(summary_path, summary)
    return summary


def load_reference(root, window):
    directory = reference_directory(root, window)
    summary_file = directory / "summary.json"
    if not summary_file.is_file():
        return None
    summary = json.loads(summary_file.read_text(encoding="utf-8"))
    if summary.get("method") != "fixed_prepared_multithreshold_led":
        raise ValueError(f"Unexpected multithreshold reference in {directory}")
    if not (directory / "blind_predictions.npz").is_file():
        raise FileNotFoundError(f"Missing MT blind predictions: {directory}")
    return summary


def compare_models(root, window, group, *, seed, n_resamples):
    """Report paired MT-versus-ML improvements on common valid blind events."""
    from waveform_analysis.data.storage import RunStore
    directory = reference_directory(root, window)
    with np.load(directory / "blind_predictions.npz", allow_pickle=False) as data:
        ids = np.array(data["event_id"])
        residual = np.array(data["corrected_ps"])
        target = np.array(data["led_residual_ps"])
    rows = []
    for run in group:
        model = RunStore(run["directory"]).load_predictions()
        model_ids = np.asarray(model["event_id"], np.int64)
        # Verify matched calibrated targets before any paired comparison.
        from waveform_analysis.reporting.stats import align_by_event_id
        common, mt_led, ml_led = align_by_event_id(
            ids, target, model_ids, model["led_residual_ps"])
        if not np.allclose(mt_led, ml_led, atol=1e-3, rtol=0):
            raise ValueError("MT and ML calibrated LED targets differ")
        fit = resolve_run_config(run["directory"])["fit"]
        row = {"model": run["manifest"]["model"], "n_led_matched": int(len(common))}
        for metric in ("ctr", "rmse"):
            result = paired_model_bootstrap(
                ids, residual, model_ids, model["corrected_ps"], fit,
                metric=metric, n_resamples=n_resamples,
                seed=semantic_seed(seed, "mt-reference", window, metric))
            row[f"{metric}_improvement_vs_mt_ps"] = result["difference_ps"]
            row[f"{metric}_improvement_vs_mt_std_ps"] = result["bootstrap_std_ps"]
            row["n_matched"] = result["n_matched"]
        rows.append(row)
    return rows
