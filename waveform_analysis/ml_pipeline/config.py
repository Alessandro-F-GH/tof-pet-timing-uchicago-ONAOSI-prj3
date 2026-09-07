from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

from .storage import fingerprint

CHANNEL_MODES = ("energy_to_energy", "energy_to_timing", "timing_to_timing")
_DELETED_VALIDATION_KEYS = {
    "strategy", "n_splits", "cv_folds", "n_folds", "nested",
    "outer_folds", "inner_folds", "early_stop_fraction",
}


class ConfigError(ValueError):
    pass


def _read(path: Path) -> dict[str, Any]:
    try:
        with path.open(encoding="utf-8") as stream:
            value = json.load(stream)
    except FileNotFoundError as exc:
        raise ConfigError(f"Configuration file not found: {path}") from exc
    except json.JSONDecodeError as exc:
        raise ConfigError(f"Invalid JSON in {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ConfigError(f"Configuration {path} must contain an object")
    return value


def merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def _relative(owner: Path, value: str | Path) -> Path:
    path = Path(value).expanduser()
    return (owner.parent / path).resolve() if not path.is_absolute() else path.resolve()


def _resolve(path: Path, stack: tuple[Path, ...] = ()) -> dict[str, Any]:
    path = path.resolve()
    if path in stack:
        raise ConfigError("Configuration include cycle: " + " -> ".join(map(str, (*stack, path))))
    raw = _read(path)
    result: dict[str, Any] = {}
    refs = raw.get("extends", [])
    refs = [refs] if isinstance(refs, str) else list(refs or [])
    for ref in refs:
        result = merge(result, _resolve(_relative(path, ref), (*stack, path)))
    for key, section in (("data_config", "data"), ("preprocessing_config", "preprocessing")):
        if key in raw:
            module_path = _relative(path, raw[key])
            module = _read(module_path)
            value = module.get(section, module)
            if not isinstance(value, dict):
                raise ConfigError(f"{key} must resolve to an object")
            result = merge(result, {section: value})
    local = {k: copy.deepcopy(v) for k, v in raw.items() if k not in {"extends", "data_config", "preprocessing_config"}}
    return merge(result, local)


def _project_path(project_root: Path, value: str | Path) -> str:
    path = Path(value).expanduser()
    return str((project_root / path).resolve() if not path.is_absolute() else path.resolve())


def _load_models(config: dict[str, Any], project_root: Path) -> None:
    names = config.get("models", [])
    if not isinstance(names, list) or not names:
        raise ConfigError("models must be a non-empty list of registered model names")
    model_dir = Path(_project_path(project_root, config.pop("model_spaces_dir", "config/model_spaces")))
    models: dict[str, dict[str, Any]] = {}
    for raw_name in names:
        name = str(raw_name)
        path = model_dir / f"{name}.json"
        model = _read(path)
        if str(model.get("model", name)) != name:
            raise ConfigError(f"Model file {path} must declare model={name!r}")
        models[name] = model
    config["models"] = models


def _enabled_modes(config: dict[str, Any]) -> list[str]:
    modes = config.get("modes")
    if not isinstance(modes, dict):
        raise ConfigError("modes must be an object keyed by channel mode")
    unknown = set(modes) - set(CHANNEL_MODES)
    if unknown:
        raise ConfigError(f"Unsupported channel modes: {sorted(unknown)}")
    enabled = [name for name in CHANNEL_MODES if bool((modes.get(name) or {}).get("enabled", False))]
    if not enabled:
        raise ConfigError("At least one mode must be enabled")
    return enabled


def validate_config(config: dict[str, Any]) -> None:
    for key in ("data", "preprocessing", "validation", "standard_methods", "models", "modes", "windows_ns", "experiment"):
        if key not in config:
            raise ConfigError(f"Missing top-level configuration key: {key}")
    validation = config["validation"]
    deleted = sorted(set(validation) & _DELETED_VALIDATION_KEYS)
    if deleted:
        raise ConfigError(f"Removed validation option(s): {deleted}. The pipeline is holdout-only.")
    for key in ("blind_fraction", "validation_fraction"):
        value = float(validation[key])
        if not 0.0 < value < 0.5:
            raise ConfigError(f"validation.{key} must be in (0, 0.5)")
    windows = config["windows_ns"]
    if not isinstance(windows, list) or not windows:
        raise ConfigError("windows_ns must contain at least one window")
    ids: set[str] = set()
    for window in windows:
        if not isinstance(window, dict):
            raise ConfigError("Each windows_ns entry must be an object")
        identifier = str(window["id"])
        if identifier in ids:
            raise ConfigError(f"Duplicate window id: {identifier}")
        ids.add(identifier)
        if float(window["end_ns"]) <= float(window["start_ns"]):
            raise ConfigError(f"Invalid window {identifier}: end_ns must exceed start_ns")
    standard = config["standard_methods"]
    if not standard.get("led_thresholds_mV"):
        raise ConfigError("standard_methods.led_thresholds_mV must not be empty")
    if not standard.get("cfd_fractions"):
        raise ConfigError("standard_methods.cfd_fractions must not be empty")
    from .models import model_names
    unknown_models = set(config["models"]) - set(model_names())
    if unknown_models:
        raise ConfigError(f"Unregistered model(s): {sorted(unknown_models)}")


def load_config(path: str | Path, project_root: str | Path | None = None) -> dict[str, Any]:
    source = Path(path).expanduser().resolve()
    root = Path(project_root).resolve() if project_root else Path(__file__).resolve().parents[1]
    config = _resolve(source)
    config["channel_modes"] = _enabled_modes(config)  # internal current representation used by preprocessing
    _load_models(config, root)
    data = config["data"]
    for key in ("root_folder",):
        if key in data:
            data[key] = _project_path(root, data[key])
    preprocessing = config["preprocessing"]
    for key in ("prepared_dir", "selection_store_dir"):
        if key in preprocessing:
            preprocessing[key] = _project_path(root, preprocessing[key])
    config["experiment"]["output_dir"] = _project_path(root, config["experiment"]["output_dir"])
    config["_config_path"] = str(source)
    config["_config_fingerprint"] = fingerprint({k: v for k, v in config.items() if not str(k).startswith("_")})
    validate_config(config)
    return config


def discover_root_files(config: dict[str, Any]) -> list[Path]:
    data = config["data"]
    root = Path(data["root_folder"])
    pattern = str(data.get("root_glob", "*.root"))
    files = sorted(root.rglob(pattern) if bool(data.get("recursive", False)) else root.glob(pattern))
    return [path.resolve() for path in files if path.is_file()]


def public_config(config: dict[str, Any]) -> dict[str, Any]:
    return {k: copy.deepcopy(v) for k, v in config.items() if not str(k).startswith("_")}
