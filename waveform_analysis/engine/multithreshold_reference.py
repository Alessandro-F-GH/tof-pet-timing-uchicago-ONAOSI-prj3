"""Control-selected multi-threshold reference on native filtered energy waves.

Thresholds and global timing offset are selected/fitted once on Control.
Development/Blind use frozen values, independently of ML model fitting.
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
from waveform_analysis.engine.multithreshold_native import (
    CANDIDATE_LEVELS_MV, load_native_energy, select_control_thresholds,
    frozen_native_pickoff,
)
from waveform_analysis.reporting.stats import (
    metric_values, blind_event_bootstrap, paired_model_bootstrap,
)

DEFAULT_REFERENCE = {
    "enabled": False,
    "candidate_levels_mV": list(CANDIDATE_LEVELS_MV),
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
        from waveform_analysis.engine.multithreshold_native import threshold_combinations
        groups = threshold_combinations(result["candidate_levels_mV"])
        result["candidate_levels_mV"] = list(sorted({v for group in groups for v in group}))
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


def _check_protocol(dev, blind):
    for key in ("control_mode_fingerprint", "target_definition"):
        if dev.manifest[key] != blind.manifest[key]:
            raise ValueError(f"MT development/blind mismatch: {key}")
    for key in ("fixed_led_threshold_mV", "led_control_mean_ps", "true_tof_ps"):
        if not np.isclose(float(dev.manifest[key]), float(blind.manifest[key]), rtol=0, atol=1e-3):
            raise ValueError(f"MT development/blind mismatch: {key}")


def _save_predictions(path, dataset, raw_delta_ps, *, control_offset_ps,
                      true_tof_ps, thresholds_mV, role):
    """Express native MT timing as an ML-style correction to the frozen LED."""
    target = np.asarray(model_target(dataset, "energy_to_energy"), np.float64)
    raw = np.asarray(raw_delta_ps, np.float64)
    valid = np.isfinite(raw) & np.isfinite(target)
    # The MT residual is native t0_1-t0_2 minus true TOF and CONTROL offset.
    mt_residual = raw - float(true_tof_ps) - float(control_offset_ps)
    # RunStore's prediction convention is LED target - corrected residual.
    correction = target - mt_residual
    correction[~valid] = np.nan
    residual = target - correction
    np.savez_compressed(
        path, event_id=np.asarray(dataset.event_index, np.int64),
        prediction_ps=correction, corrected_ps=residual,
        led_residual_ps=target, valid=valid,
        thresholds_mV=np.asarray(thresholds_mV, np.float64),
        control_offset_ps=np.asarray(control_offset_ps),
        dataset_role=np.asarray(role),
        event_population_identity=np.asarray(dataset.manifest["event_population_identity"]),
        analysis_protocol_identity=np.asarray(dataset.manifest["analysis_protocol_identity"]),
    )
    return valid, residual, target


def run_reference(reference_run, root, options, *, seed, bootstrap_resamples):
    """Reuse ML populations; select/freeze MT entirely on native Control."""
    run = Path(reference_run)
    config = resolve_run_config(run)
    if config["mode"] != "energy_to_energy":
        raise ValueError("Multithreshold reference only supports energy_to_energy")
    manifest = read_run(run)["manifest"]
    loader = PreparedCacheLookup()
    dev = loader.find("development", config, manifest)
    blind = loader.find("blind", config, manifest)
    _check_protocol(dev, blind)

    native_control = load_native_energy(config, "control")
    native_development = load_native_energy(config, "development")
    native_blind = load_native_energy(config, "blind")
    baseline_window = config["preprocessing"]["selection"]["baseline_window_ns"]
    destination = reference_directory(root, config["window_name"])
    fingerprint = canonical_hash({
        "version": 4, "method": "native_control_selected_multithreshold",
        "options": options, "seed": seed, "bootstrap_resamples": bootstrap_resamples,
        "control_native_fingerprint": native_control.manifest["fingerprint"],
        "development_native_fingerprint": native_development.manifest["fingerprint"],
        "blind_native_fingerprint": native_blind.manifest["fingerprint"],
        "control_true_tof_ps": config["control"]["true_tof_ps"],
        "dev_protocol": dev.manifest["analysis_protocol_identity"],
        "blind_protocol": blind.manifest["analysis_protocol_identity"],
        "dev_population": dev.manifest["event_population_identity"],
        "blind_population": blind.manifest["event_population_identity"],
        "fit": config["fit"], "baseline_window_ns": baseline_window,
    })
    summary_path = destination / "summary.json"
    if summary_path.is_file():
        existing = json.loads(summary_path.read_text(encoding="utf-8"))
        if (existing.get("fingerprint") == fingerprint and
            all((destination / f"{role}_predictions.npz").is_file()
                for role in ("development", "blind"))):
            return existing

    destination.mkdir(parents=True, exist_ok=True)
    choice, scan = select_control_thresholds(
        native_control, config["fit"], levels=options["candidate_levels_mV"],
        minimum_coverage=options["minimum_coverage"],
        seed=semantic_seed(seed, "mt-control", "selection"),
        baseline_window_ns=baseline_window,
        true_tof_ps=float(config["control"]["true_tof_ps"]),
    )
    write_csv(destination / "control_threshold_scan.csv", [
        {**{k: v for k, v in row.items() if k != "thresholds_mV"},
         "thresholds_mV": ",".join(f"{v:g}" for v in row["thresholds_mV"])}
        for row in scan
    ])
    thresholds = choice["thresholds_mV"]
    control_offset = float(choice["offset_ps"])
    true_tof_dev = float(config["development"]["true_tof_ps"])
    true_tof_blind = float(config["blind"]["true_tof_ps"])

    raw_dev, _, _ = frozen_native_pickoff(
        native_development, dev, thresholds, baseline_window_ns=baseline_window)
    valid_dev, resid_dev, _ = _save_predictions(
        destination / "development_predictions.npz", dev, raw_dev,
        control_offset_ps=control_offset, true_tof_ps=true_tof_dev,
        thresholds_mV=thresholds, role="development")
    coverage_dev = float(valid_dev.mean())
    if valid_dev.sum() < 5 or coverage_dev < options["minimum_coverage"]:
        raise ValueError(f"MT development coverage {coverage_dev:.2%} below minimum")
    dev_metrics = metric_values(
        resid_dev[valid_dev], config["fit"],
        seed=semantic_seed(seed, "mt-reference", "development"))

    raw_blind, diagnostics, native_indices = frozen_native_pickoff(
        native_blind, blind, thresholds, baseline_window_ns=baseline_window)
    valid_blind, residual, led = _save_predictions(
        destination / "blind_predictions.npz", blind, raw_blind,
        control_offset_ps=control_offset, true_tof_ps=true_tof_blind,
        thresholds_mV=thresholds, role="blind")

    from waveform_analysis.reporting.multithreshold_diagnostics import save_native_coverage_diagnostic
    diagnostic = save_native_coverage_diagnostic(
        destination, blind, native_blind, native_indices, raw_blind,
        diagnostics, baseline_window_ns=baseline_window, role="blind")

    coverage = float(valid_blind.mean())
    if valid_blind.sum() < 5 or coverage < options["minimum_coverage"]:
        raise ValueError(f"MT blind coverage {coverage:.2%} below minimum")
    bootstrap, draws = blind_event_bootstrap(
        residual[valid_blind], led[valid_blind], config["fit"],
        n_resamples=bootstrap_resamples,
        seed=semantic_seed(seed, "mt-reference", "blind"))
    np.savez_compressed(destination / "blind_bootstrap.npz", **draws)
    summary = {
        "method": "native_control_selected_multithreshold", "fingerprint": fingerprint,
        "mode": "energy_to_energy", "window": config["window_name"],
        "candidate_levels_mV": options["candidate_levels_mV"],
        "thresholds_mV": thresholds,
        "selection_fit_role": "control",
        "control_selection": choice,
        "control_threshold_scan": "control_threshold_scan.csv",
        "baseline_window_ns": baseline_window,
        "offset_calibration": "control_native_mean_residual",
        "control_offset_ps": control_offset,
        "development": {"total": int(dev.n_events), "valid": int(valid_dev.sum()),
                        "coverage": coverage_dev, "metrics": dev_metrics},
        "blind": {"total": int(blind.n_events), "valid": int(valid_blind.sum()),
                  "coverage": coverage, "metrics": bootstrap},
        "coverage_diagnostic": diagnostic,
        "source_native_cache": {
            "control": str(native_control.directory),
            "development": str(native_development.directory),
            "blind": str(native_blind.directory),
        },
        "source_prepared_cache": {
            "development": str(dev.directory), "blind": str(blind.directory),
        },
    }
    atomic_json(summary_path, summary)
    return summary


def load_reference(root, window):
    directory = reference_directory(root, window)
    summary_file = directory / "summary.json"
    if not summary_file.is_file():
        return None
    summary = json.loads(summary_file.read_text(encoding="utf-8"))
    if summary.get("method") != "native_control_selected_multithreshold":
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
