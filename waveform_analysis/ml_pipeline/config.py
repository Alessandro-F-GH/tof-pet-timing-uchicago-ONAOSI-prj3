from __future__ import annotations

import copy, json
from pathlib import Path
from typing import Any

from .common import canonical_hash

CHANNEL_MODES = ("energy_to_energy", "timing_to_timing")


class ConfigError(ValueError):
    pass


def _read(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ConfigError(f"Configuration file not found: {path}") from exc
    except json.JSONDecodeError as exc:
        raise ConfigError(f"Invalid JSON in {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ConfigError(f"Configuration {path} must contain an object")
    return value


def merge(base, override):
    result = copy.deepcopy(base)
    for key, value in override.items():
        result[key] = merge(result[key], value) if isinstance(value, dict) and isinstance(result.get(key), dict) else copy.deepcopy(value)
    return result


def _relative(owner: Path, value):
    path = Path(value).expanduser()
    return (owner.parent / path).resolve() if not path.is_absolute() else path.resolve()


def _resolve(path: Path, stack: tuple[Path, ...] = ()):
    path = path.resolve()
    if path in stack:
        raise ConfigError("Configuration include cycle: " + " -> ".join(map(str, (*stack, path))))
    raw = _read(path)
    result = {}
    refs = raw.get("extends", [])
    refs = [refs] if isinstance(refs, str) else list(refs or [])
    for ref in refs:
        result = merge(result, _resolve(_relative(path, ref), (*stack, path)))
    for key, section in (("data_config", "data"), ("preprocessing_config", "preprocessing")):
        if key in raw:
            module = _read(_relative(path, raw[key]))
            value = module.get(section, module)
            if not isinstance(value, dict):
                raise ConfigError(f"{key} must resolve to an object")
            result = merge(result, {section: value})
    return merge(result, {k: copy.deepcopy(v) for k, v in raw.items() if k not in {"extends", "data_config", "preprocessing_config"}})


def _project_path(root: Path, value):
    path = Path(value).expanduser()
    return str((root / path).resolve() if not path.is_absolute() else path.resolve())


def _load_models(config, root):
    names = config.get("models")
    if not isinstance(names, list) or not names:
        raise ConfigError("models must be a non-empty list")
    model_dir = Path(_project_path(root, config.pop("model_spaces_dir", "config/model_spaces")))
    resolved = {}
    for raw_name in names:
        name = str(raw_name)
        path = model_dir / f"{name}.json"
        model = _read(path)
        if str(model.get("model", name)) != name:
            raise ConfigError(f"Model file {path} must declare model={name!r}")
        resolved[name] = model
    config["models"] = resolved


def mode_family(mode: str) -> str:
    if mode == "energy_to_energy":
        return "energy"
    if mode == "timing_to_timing":
        return "timing"
    raise ConfigError(f"mode must be one of {CHANNEL_MODES}, got {mode!r}")


def validate_config(config):
    required = {"data", "preprocessing", "validation", "standard_methods", "models", "mode", "cfd", "ml_input", "ml_training", "ml_output", "fit", "experiment"}
    missing = sorted(required - set(config))
    if missing:
        raise ConfigError(f"Missing configuration section(s): {missing}")
    mode = str(config["mode"])
    family = mode_family(mode)
    if not isinstance(config["cfd"], bool):
        raise ConfigError("cfd must be true or false")

    experiment = config["experiment"]
    concatenate = bool(experiment.get("concatenate_datasets", False))
    if concatenate:
        if "fixed_led_threshold_mV" not in experiment:
            raise ConfigError("experiment.fixed_led_threshold_mV is required when concatenate_datasets=true")
        fixed_led = float(experiment["fixed_led_threshold_mV"])
        if not np_isfinite_positive(fixed_led):
            raise ConfigError("experiment.fixed_led_threshold_mV must be finite and positive")
        name = str(experiment.get("concatenated_dataset_name", "concatenated")).strip()
        if not name:
            raise ConfigError("experiment.concatenated_dataset_name must not be empty")
        experiment["concatenated_dataset_name"] = name

    validation = config["validation"]
    extra = sorted(set(validation) - {"seed", "test_fraction", "validation_fraction"})
    if extra:
        raise ConfigError(f"Unknown validation option(s): {extra}")
    for key in ("test_fraction", "validation_fraction"):
        if not 0.0 < float(validation[key]) < 0.5:
            raise ConfigError(f"validation.{key} must be in (0, 0.5)")

    ml_input = config["ml_input"]
    if set(ml_input) - {"window_ns", "subsampling"}:
        raise ConfigError("ml_input accepts only window_ns and subsampling")
    if float(ml_input["window_ns"]["end"]) <= float(ml_input["window_ns"]["start"]):
        raise ConfigError("ml_input.window_ns.end must exceed start")
    if int(ml_input.get("subsampling", 1)) <= 0:
        raise ConfigError("ml_input.subsampling must be positive")

    ml_training = config["ml_training"]
    if set(ml_training) != {"target_abs_max_ps"}:
        raise ConfigError("ml_training must contain only target_abs_max_ps")
    ranges = ml_training["target_abs_max_ps"]
    if not isinstance(ranges, list) or not ranges:
        raise ConfigError("ml_training.target_abs_max_ps must be a non-empty list")
    normalized_ranges = []
    for value in ranges:
        if isinstance(value, bool):
            raise ConfigError("ml_training.target_abs_max_ps values must be finite positive numbers")
        limit = float(value)
        if not np_isfinite_positive(limit):
            raise ConfigError("ml_training.target_abs_max_ps values must be finite positive numbers")
        normalized_ranges.append(limit)
    if len(set(normalized_ranges)) != len(normalized_ranges):
        raise ConfigError("ml_training.target_abs_max_ps values must be unique")
    ml_training["target_abs_max_ps"] = normalized_ranges

    ml_output = config["ml_output"]
    if set(ml_output) != {"max_abs_ps"} or float(ml_output["max_abs_ps"]) <= 0:
        raise ConfigError("ml_output must contain one positive max_abs_ps")

    fit = config["fit"]
    allowed_fit = {"min_events", "coverage_fraction", "bin_width_ps", "bootstrap_samples"}
    obsolete_fit = set(fit) - allowed_fit
    if obsolete_fit:
        raise ConfigError(
            f"Unknown/obsolete fit option(s): {sorted(obsolete_fit)}. "
            "Canonical CTR is the Gaussian-equivalent shortest coverage interval; "
            "bin_width_ps is used only for the secondary core-FWHM diagnostic."
        )
    if int(fit.get("min_events", 0)) < 3:
        raise ConfigError("fit.min_events must be an integer >= 3")
    coverage = float(fit.get("coverage_fraction", 0.90))
    if not 0.5 < coverage < 1.0:
        raise ConfigError("fit.coverage_fraction must be in (0.5, 1.0)")
    if float(fit.get("bin_width_ps", 0.0)) <= 0:
        raise ConfigError("fit.bin_width_ps must be positive")
    bootstrap_samples = fit.get("bootstrap_samples")
    if isinstance(bootstrap_samples, bool) or int(bootstrap_samples) != bootstrap_samples or int(bootstrap_samples) < 2:
        raise ConfigError("fit.bootstrap_samples must be an integer >= 2")

    preprocessing = config["preprocessing"]
    for key in ("selection_store_dir", "preprocessed_dir", "prepared_dir", "materialized_window_ns", "selection", "photopeak", "energy"):
        if key not in preprocessing:
            raise ConfigError(f"preprocessing.{key} is required")
    channels = config["data"]["channels"]
    if family == "timing" and not channels.get("timing"):
        raise ConfigError("timing_to_timing requires timing channels")
    for required_family in {"energy", family}:
        if required_family not in preprocessing:
            raise ConfigError(f"preprocessing.{required_family} is required")
        for key in ("trigger_threshold_mV", "vertical_scale_limit_mV"):
            if key not in preprocessing[required_family]:
                raise ConfigError(f"preprocessing.{required_family}.{key} is required")
    if "rising_edge_before_trigger_ns" in preprocessing["energy"]:
        raise ConfigError("preprocessing.energy.rising_edge_before_trigger_ns is obsolete; energy uses the materialized window start to peak")
    if family == "timing" and "rising_edge_before_trigger_ns" not in preprocessing["timing"]:
        raise ConfigError("preprocessing.timing.rising_edge_before_trigger_ns is required")
    if "pulse_duration_mad" in preprocessing["selection"]:
        raise ConfigError("preprocessing.selection.pulse_duration_mad is obsolete; use preprocessing.tot_peak")
    if family == "timing":
        if "tot_peak" not in preprocessing:
            raise ConfigError("preprocessing.tot_peak is required for timing_to_timing")
        tot = preprocessing["tot_peak"]
        required_tot = {"histogram_bin_ns", "search_quantile_min", "smoothing_sigma_bins", "initial_half_width_ns", "iteration_sigma", "max_iterations", "convergence_tolerance_ns", "selection_sigma_low", "selection_sigma_high"}
        missing_tot = sorted(required_tot - set(tot))
        if missing_tot:
            raise ConfigError(f"Missing preprocessing.tot_peak option(s): {missing_tot}")
        if float(tot["histogram_bin_ns"]) <= 0 or float(tot["initial_half_width_ns"]) <= 0 or float(tot["convergence_tolerance_ns"]) <= 0:
            raise ConfigError("preprocessing.tot_peak widths/tolerance must be positive")
        if float(tot["selection_sigma_high"]) <= float(tot["selection_sigma_low"]):
            raise ConfigError("preprocessing.tot_peak.selection_sigma_high must exceed selection_sigma_low")

    noise = preprocessing["selection"]["baseline_noise"]
    if bool(noise.get("enabled", False)) and (len(noise["window_ns"]) != 2 or float(noise["window_ns"][1]) > 0.0):
        raise ConfigError("baseline_noise.window_ns must be [start, end] before the trigger")

    standard = config["standard_methods"]
    if not standard.get("led_thresholds_mV"):
        raise ConfigError("LED threshold list must not be empty")
    if not 0.0 < float(standard.get("led_minimum_crossing_efficiency", 0.95)) <= 1.0:
        raise ConfigError("standard_methods.led_minimum_crossing_efficiency must be in (0, 1]")
    if float(standard.get("led_coincidence_window_ns", 2.0)) <= 0:
        raise ConfigError("standard_methods.led_coincidence_window_ns must be positive")
    if config["cfd"] and not standard.get("cfd_fractions"):
        raise ConfigError("CFD fraction list must not be empty when cfd=true")

    from .models import model_names
    unknown_models = set(config["models"]) - set(model_names())
    if unknown_models:
        raise ConfigError(f"Unregistered model(s): {sorted(unknown_models)}")
    for name, model in config["models"].items():
        training = model.get("training", {}) or {}
        if "selection_metric" in training or "selection_metric" in model:
            raise ConfigError(f"{name}: selection_metric is fixed to validation RMSE and must not be configured")


def np_isfinite_positive(value: float) -> bool:
    import math
    return math.isfinite(float(value)) and float(value) > 0.0


def load_config(path: str | Path, project_root: str | Path | None = None):
    source = Path(path).expanduser().resolve()
    root = Path(project_root).resolve() if project_root else Path(__file__).resolve().parents[1]
    config = _resolve(source)
    _load_models(config, root)
    validate_config(config)
    mode = str(config["mode"])
    if "root_folder" in config["data"]:
        config["data"]["root_folder"] = _project_path(root, config["data"]["root_folder"])
    for key in ("selection_store_dir", "preprocessed_dir", "prepared_dir"):
        config["preprocessing"][key] = str(Path(_project_path(root, config["preprocessing"][key])) / mode)
    config["experiment"]["output_dir"] = _project_path(root, config["experiment"]["output_dir"])
    config["_config_path"] = str(source)
    config["_config_fingerprint"] = canonical_hash({k: v for k, v in config.items() if not str(k).startswith("_")})
    return config


def discover_root_files(config):
    data = config["data"]
    root = Path(data["root_folder"])
    pattern = str(data.get("root_glob", "*.root"))
    files = sorted(root.rglob(pattern) if bool(data.get("recursive", False)) else root.glob(pattern))
    return [p.resolve() for p in files if p.is_file()]


def public_config(config):
    return {k: copy.deepcopy(v) for k, v in config.items() if not str(k).startswith("_")}
