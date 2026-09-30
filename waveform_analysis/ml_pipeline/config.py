from __future__ import annotations

import copy
import json
from dataclasses import dataclass
from pathlib import Path

from .common import canonical_hash

CHANNEL_MODES = ("energy_to_energy", "timing_to_timing")
MODEL_SAVE_POLICIES = ("all", "first", "none")


class ConfigError(ValueError):
    pass


@dataclass(frozen=True)
class BatchConfig:
    name: str
    output_dir: str
    source_path: str
    protocol: dict
    axes: dict
    runs: tuple[dict, ...]


def _read(path):
    path = Path(path)
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ConfigError(f"Configuration file not found: {path}") from exc
    except json.JSONDecodeError as exc:
        raise ConfigError(f"Invalid JSON in {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ConfigError(f"{path} must contain a JSON object")
    return value


def _relative(owner, value):
    path = Path(value).expanduser()
    return (owner.parent / path).resolve() if not path.is_absolute() else path.resolve()


def _project(root, value):
    path = Path(value).expanduser()
    return str((root / path).resolve() if not path.is_absolute() else path.resolve())


def _module(path, key=None):
    raw = _read(path)
    if key and key in raw:
        return copy.deepcopy(raw[key])
    return raw


def _dataset(owner, value, root):
    data = _module(_relative(owner, value)) if isinstance(value, str) else copy.deepcopy(value)
    if not isinstance(data, dict):
        raise ConfigError("dataset must resolve to an object")
    if "root_file" not in data or "true_tof_ps" not in data or "channels" not in data:
        raise ConfigError("dataset requires root_file, true_tof_ps and channels")
    data["root_file"] = _project(root, data["root_file"])
    return data


def _preprocessing(owner, value, root):
    preprocessing = _module(_relative(owner, value), "preprocessing") if isinstance(value, str) else copy.deepcopy(value)
    if not isinstance(preprocessing, dict):
        raise ConfigError("preprocessing_config must resolve to an object")
    if "cache_dir" not in preprocessing:
        preprocessing["cache_dir"] = "processed_data/ml_protocol_v2"
    preprocessing["cache_dir"] = _project(root, preprocessing["cache_dir"])
    return preprocessing


def _model(owner, raw, root):
    if isinstance(raw, str):
        name = raw
        space = _read(root / "config" / "model_spaces" / f"{name}.json")
    elif isinstance(raw, dict):
        name = str(raw.get("name", ""))
        if "space_config" in raw:
            space = _read(_relative(owner, raw["space_config"]))
        elif "space" in raw:
            space = copy.deepcopy(raw["space"])
        else:
            space = _read(root / "config" / "model_spaces" / f"{name}.json")
    else:
        raise ConfigError("model must be a name or object")
    if not name:
        name = str(space.get("model", ""))
    if str(space.get("model", name)) != name:
        raise ConfigError("model-space name mismatch")
    return {"name": name, "space": space}


def _window(raw):
    if isinstance(raw, dict) and set(raw) >= {"start", "end"}:
        return {"start": float(raw["start"]), "end": float(raw["end"])}
    raise ConfigError("window must resolve to one {start,end} interval")


def _fit(raw):
    fit = copy.deepcopy(raw)
    if not isinstance(fit, dict):
        raise ConfigError("fit must be an object")
    if "bootstrap_samples" in fit:
        raise ConfigError("fit.bootstrap_samples is not supported")
    return fit


def _save_models(raw):
    value = str(raw if raw is not None else "all").strip().lower()
    if value not in MODEL_SAVE_POLICIES:
        raise ConfigError(f"save_models must be one of {MODEL_SAVE_POLICIES}")
    return value


def _selection(raw):
    value = copy.deepcopy(raw)
    if not isinstance(value, dict) or set(value) != {"validation_fraction"}:
        raise ConfigError("model_selection must contain only validation_fraction")
    value["validation_fraction"] = float(value["validation_fraction"])
    return value


def _evaluation(raw):
    value = copy.deepcopy(raw)
    required = {"n_replicas", "blind_fraction", "minimum_events_per_split"}
    if not isinstance(value, dict) or set(value) != required:
        raise ConfigError(
            "evaluation must contain exactly n_replicas, blind_fraction and minimum_events_per_split"
        )
    value["n_replicas"] = int(value["n_replicas"])
    value["blind_fraction"] = float(value["blind_fraction"])
    value["minimum_events_per_split"] = int(value["minimum_events_per_split"])
    return value


def validate_config(config):
    required = (
        "reference", "analysis", "preprocessing", "mode", "model", "window_ns",
        "seed", "model_selection", "evaluation", "fit", "ml_input", "ml_output", "output_dir",
    )
    for key in required:
        if key not in config:
            raise ConfigError(f"Missing {key}")

    if config["mode"] not in CHANNEL_MODES:
        raise ConfigError(f"mode must be one of {CHANNEL_MODES}")
    if config["reference"]["channels"] != config["analysis"]["channels"]:
        raise ConfigError("reference and analysis channel definitions must match")
    if config["mode"] == "timing_to_timing" and not config["analysis"]["channels"].get("timing"):
        raise ConfigError("timing mode requires timing channels")

    int(config["seed"])
    validation_fraction = float(config["model_selection"]["validation_fraction"])
    blind_fraction = float(config["evaluation"]["blind_fraction"])
    if not 0.0 < validation_fraction < 1.0:
        raise ConfigError("model_selection.validation_fraction must lie in (0, 1)")
    if not 0.0 < blind_fraction < 1.0:
        raise ConfigError("evaluation.blind_fraction must lie in (0, 1)")
    if validation_fraction + blind_fraction >= 1.0:
        raise ConfigError(
            "validation_fraction + blind_fraction must be < 1 so every replica retains a variable training subset"
        )
    if int(config["evaluation"]["n_replicas"]) < 1:
        raise ConfigError("evaluation.n_replicas must be >= 1")
    if int(config["evaluation"]["minimum_events_per_split"]) < 1:
        raise ConfigError("evaluation.minimum_events_per_split must be >= 1")

    preprocessing = config["preprocessing"]
    for key in ("materialized_window_ns", "energy", "timing", "selection", "photopeak", "tot_peak", "led_selection", "io"):
        if key not in preprocessing:
            raise ConfigError(f"preprocessing.{key} is required")
    clipping = preprocessing["selection"].get("baseline_clipping")
    if not isinstance(clipping, dict) or "margin_mV" not in clipping or float(clipping["margin_mV"]) < 0:
        raise ConfigError("selection.baseline_clipping.margin_mV must be non-negative")
    noise = preprocessing["selection"]["baseline_noise"]
    if len(noise["window_ns"]) != 2 or float(noise["window_ns"][1]) > 0:
        raise ConfigError("baseline window must lie before trigger")
    led = preprocessing["led_selection"]
    if not led.get("thresholds_mV"):
        raise ConfigError("led_selection.thresholds_mV must be non-empty")
    if not 0 < float(led["minimum_crossing_efficiency"]) <= 1:
        raise ConfigError("invalid LED minimum crossing efficiency")
    if float(led["coincidence_window_ns"]) <= 0:
        raise ConfigError("invalid LED coincidence window")

    if config["window_ns"]["end"] <= config["window_ns"]["start"]:
        raise ConfigError("window end must exceed start")
    if int(config["ml_input"].get("subsampling", 1)) <= 0:
        raise ConfigError("ml_input.subsampling must be positive")
    if float(config["ml_output"]["max_abs_ps"]) <= 0:
        raise ConfigError("ml_output.max_abs_ps must be positive")
    if float(config["fit"]["histogram_bin_width_ps"]) <= 0:
        raise ConfigError("invalid fit settings")
    config["save_models"] = _save_models(config.get("save_models", "all"))

    from .models import model_names
    if config["model"]["name"] not in model_names():
        raise ConfigError(f"unregistered model {config['model']['name']}")


def _resolve(source, raw, root):
    required_fields = {
        "reference_dataset", "analysis_dataset", "preprocessing_config", "model",
        "mode", "window", "seed", "model_selection", "evaluation", "fit",
        "ml_output", "output_dir",
    }
    optional_fields = {
        "name", "ml_input", "save_models", "study_name", "run_id", "window_name",
    }
    missing = required_fields - set(raw)
    if missing:
        raise ConfigError(f"Missing study fields: {sorted(missing)}")
    extra = set(raw) - required_fields - optional_fields
    if extra:
        raise ConfigError(f"Unsupported study fields: {sorted(extra)}")

    reference = _dataset(source, raw["reference_dataset"], root)
    analysis = _dataset(source, raw.get("analysis_dataset"), root)
    preprocessing = _preprocessing(source, raw.get("preprocessing_config"), root)
    model = _model(source, raw.get("model"), root)
    window = _window(raw["window"])

    config = {
        "name": str(raw.get("name", source.stem)),
        "reference": reference,
        "analysis": analysis,
        "preprocessing": preprocessing,
        "mode": str(raw["mode"]),
        "model": model,
        "window_ns": window,
        "seed": int(raw["seed"]),
        "model_selection": _selection(raw["model_selection"]),
        "evaluation": _evaluation(raw["evaluation"]),
        "fit": _fit(raw["fit"]),
        "ml_input": copy.deepcopy(raw.get("ml_input", {"subsampling": 1})),
        "ml_output": copy.deepcopy(raw["ml_output"]),
        "output_dir": _project(root, raw["output_dir"]),
        "save_models": _save_models(raw.get("save_models", "all")),
    }
    for key in ("study_name", "run_id", "window_name"):
        if key in raw:
            config[key] = str(raw[key])
    validate_config(config)
    config["_config_path"] = str(source)
    config["_config_fingerprint"] = canonical_hash({k: v for k, v in config.items() if k != "save_models"})
    return config


def load_config(path, project_root=None):
    source = Path(path).expanduser().resolve()
    root = Path(project_root).resolve() if project_root else Path(__file__).resolve().parents[1]
    return _resolve(source, _read(source), root)


def _mode_tag(mode):
    return "energy" if mode == "energy_to_energy" else "timing"


def _excluded(model, mode, window, rules):
    for rule in rules:
        if not isinstance(rule, dict):
            raise ConfigError("sweep.exclude entries must be objects")
        if all(rule.get(key, value) == value for key, value in (("model", model), ("mode", mode), ("window", window))):
            return True
    return False


def _protocol_value(protocol, key, mode):
    value = protocol[key]
    return copy.deepcopy(value[mode]) if isinstance(value, dict) and mode in value else copy.deepcopy(value)


def load_batch_config(path, project_root=None):
    source = Path(path).expanduser().resolve()
    root = Path(project_root).resolve() if project_root else Path(__file__).resolve().parents[1]
    raw = _read(source)

    required = {"name", "reference_dataset", "analysis_dataset", "output_dir", "protocol", "sweep"}
    missing = required - set(raw)
    if missing:
        raise ConfigError(f"Missing batch fields: {sorted(missing)}")
    extra = set(raw) - required - {"save_models"}
    if extra:
        raise ConfigError(f"Unsupported batch fields: {sorted(extra)}")

    protocol = copy.deepcopy(raw["protocol"])
    protocol_required = {
        "seed", "preprocessing_config", "model_selection", "evaluation", "fit", "ml_output"
    }
    if not protocol_required <= set(protocol):
        raise ConfigError(f"protocol requires {sorted(protocol_required)}")
    extra_protocol = set(protocol) - (protocol_required | {"ml_input"})
    if extra_protocol:
        raise ConfigError(f"Unsupported protocol fields: {sorted(extra_protocol)}")

    seed = int(protocol["seed"])
    model_selection = _selection(protocol["model_selection"])
    evaluation = _evaluation(protocol["evaluation"])
    save_models = _save_models(raw.get("save_models", "all"))

    sweep = copy.deepcopy(raw["sweep"])
    extra_sweep = set(sweep) - {"models", "modes", "windows", "exclude"}
    if extra_sweep:
        raise ConfigError(f"Unsupported sweep fields: {sorted(extra_sweep)}")
    models = sweep.get("models")
    modes = sweep.get("modes")
    windows = sweep.get("windows")
    rules = sweep.get("exclude", [])
    if not isinstance(models, list) or not models:
        raise ConfigError("sweep.models must be a non-empty list")
    if not isinstance(modes, list) or not modes:
        raise ConfigError("sweep.modes must be a non-empty list")
    if not isinstance(windows, dict) or not windows:
        raise ConfigError("sweep.windows must be a non-empty object")
    if not isinstance(rules, list):
        raise ConfigError("sweep.exclude must be a list")

    name = str(raw["name"])
    output = Path(_project(root, raw["output_dir"]))
    runs = []
    for model_raw in models:
        model_name = model_raw if isinstance(model_raw, str) else str(model_raw.get("name", ""))
        if not model_name:
            raise ConfigError("every sweep model needs a name")
        for mode_raw in modes:
            mode = str(mode_raw)
            if mode not in CHANNEL_MODES:
                raise ConfigError(f"mode must be one of {CHANNEL_MODES}")
            for window_name, window in windows.items():
                if _excluded(model_name, mode, str(window_name), rules):
                    continue
                run_id = f"{model_name}__{_mode_tag(mode)}__{window_name}"
                item = {
                    "name": f"{name}__{run_id}",
                    "study_name": name,
                    "run_id": run_id,
                    "window_name": str(window_name),
                    "reference_dataset": raw["reference_dataset"],
                    "analysis_dataset": raw["analysis_dataset"],
                    "preprocessing_config": protocol["preprocessing_config"],
                    "model": model_raw,
                    "mode": mode,
                    "window": window,
                    "seed": seed,
                    "model_selection": model_selection,
                    "evaluation": evaluation,
                    "fit": _protocol_value(protocol, "fit", mode),
                    "ml_input": _protocol_value(protocol, "ml_input", mode) if "ml_input" in protocol else {"subsampling": 1},
                    "ml_output": _protocol_value(protocol, "ml_output", mode),
                    "output_dir": str(output / model_name / _mode_tag(mode) / str(window_name)),
                    "save_models": save_models,
                }
                runs.append(_resolve(source, item, root))

    if not runs:
        raise ConfigError("batch sweep produced no runs")

    axes = {
        "models": [m if isinstance(m, str) else str(m.get("name")) for m in models],
        "modes": [str(m) for m in modes],
        "windows": {str(key): _window(value) for key, value in windows.items()},
    }
    normalized_protocol = {
        "seed": seed,
        "preprocessing_config": str(protocol["preprocessing_config"]),
        "model_selection": model_selection,
        "evaluation": evaluation,
        "fit": copy.deepcopy(protocol["fit"]),
        "ml_input": copy.deepcopy(protocol.get("ml_input", {"subsampling": 1})),
        "ml_output": copy.deepcopy(protocol["ml_output"]),
    }
    return BatchConfig(
        name=name,
        output_dir=str(output.resolve()),
        source_path=str(source),
        protocol=normalized_protocol,
        axes=axes,
        runs=tuple(runs),
    )


def public_config(config):
    return {key: value for key, value in config.items() if not str(key).startswith("_")}


def public_batch_config(batch):
    return {
        "name": batch.name,
        "output_dir": batch.output_dir,
        "source_path": batch.source_path,
        "protocol": batch.protocol,
        "axes": batch.axes,
        "save_models": batch.runs[0].get("save_models", "all") if batch.runs else "all",
        "runs": [public_config(config) for config in batch.runs],
    }
