from __future__ import annotations

import copy
import json
from dataclasses import dataclass
from typing import Any
from .settings import (
    BaselineConfig as BaselineConfig,
    TimingConfig as TimingConfig,
    MLPTrainingConfig as MLPTrainingConfig,
    OnishiTrainingConfig as OnishiTrainingConfig,
    RuntimeConfig,
    XAIConfig,
)
from .exceptions import ConfigError
from pathlib import Path

from waveform_analysis.core.io import canonical_hash

CHANNEL_MODES = ("energy_to_energy", "timing_to_timing")
MODEL_SELECTION_METRICS = ("rmse", "ctr")
DEFAULT_PREDICTION_CHUNK_SIZE = RuntimeConfig.prediction_chunk_size
DEFAULT_PLOT_CONFIG = "config/plots/default.json"


@dataclass(frozen=True)
class BatchConfig:
    name: str
    output_dir: str
    source_path: str
    results: dict[str, Any]
    datasets: dict[str, Any]
    protocol: dict[str, Any]
    axes: dict[str, Any]
    plot_config: dict[str, Any]
    runs: tuple[dict[str, Any], ...]


def _read(path: str | Path) -> dict[str, Any]:
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


def _relative(owner: Path, value: str | Path) -> Path:
    path = Path(value).expanduser()
    return (owner.parent / path).resolve() if not path.is_absolute() else path.resolve()


def _project(root: Path, value: str | Path) -> str:
    path = Path(value).expanduser()
    return str((root / path).resolve() if not path.is_absolute() else path.resolve())


def _module(path: str | Path, key: str | None = None) -> dict[str, Any]:
    raw = _read(path)
    if key and key in raw:
        return copy.deepcopy(raw[key])
    return raw


def _dataset(owner: Path, value: str | dict[str, Any], root: Path) -> dict[str, Any]:
    data = (
        _module(_relative(owner, value))
        if isinstance(value, str)
        else copy.deepcopy(value)
    )
    if not isinstance(data, dict):
        raise ConfigError("dataset must resolve to an object")
    if not {"root_file", "true_tof_ps", "channels"} <= set(data):
        raise ConfigError("dataset requires root_file, true_tof_ps and channels")
    data["root_file"] = _project(root, data["root_file"])
    return data


def _preprocessing(
    owner: Path, value: str | dict[str, Any], root: Path
) -> dict[str, Any]:
    preprocessing = (
        _module(_relative(owner, value), "preprocessing")
        if isinstance(value, str)
        else copy.deepcopy(value)
    )
    if not isinstance(preprocessing, dict):
        raise ConfigError("preprocessing_config must resolve to an object")
    if "cache_dir" not in preprocessing:
        preprocessing["cache_dir"] = "processed_data/ml_protocol_v3"
    preprocessing["cache_dir"] = _project(root, preprocessing["cache_dir"])
    return preprocessing


def _model(owner: Path, raw: str | dict[str, Any], root: Path) -> dict[str, Any]:
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
    optimization = space.get("optimization") or {}
    if "seed" in optimization:
        raise ConfigError(
            "model optimization must not define its own seed; use protocol.seed"
        )
    return {"name": name, "space": space}


def _window(raw: dict[str, Any]) -> dict[str, float]:
    if isinstance(raw, dict) and set(raw) >= {"start", "end"}:
        value = {"start": float(raw["start"]), "end": float(raw["end"])}
        if value["end"] <= value["start"]:
            raise ConfigError("window end must exceed start")
        return value
    raise ConfigError("window must resolve to one {start,end} interval")


def _fit(raw: dict[str, Any]) -> dict[str, Any]:
    fit = copy.deepcopy(raw)
    if not isinstance(fit, dict):
        raise ConfigError("fit must be an object")
    if "bootstrap_samples" in fit:
        raise ConfigError(
            "fit.bootstrap_samples is not supported; use protocol.bootstrap"
        )
    if float(fit["histogram_bin_width_ps"]) <= 0:
        raise ConfigError("fit.histogram_bin_width_ps must be positive")
    return fit


def _runtime(raw: dict[str, Any] | None = None) -> dict[str, int]:
    value = copy.deepcopy(raw or {})
    if not isinstance(value, dict):
        raise ConfigError("runtime must be an object")
    extra = set(value) - {"prediction_chunk_size"}
    if extra:
        raise ConfigError(f"Unsupported runtime fields: {sorted(extra)}")
    chunk = int(value.get("prediction_chunk_size", DEFAULT_PREDICTION_CHUNK_SIZE))
    if chunk < 1:
        raise ConfigError("runtime.prediction_chunk_size must be >= 1")
    return {"prediction_chunk_size": chunk}


def _tolerance(
    raw: float | dict[str, float], *, folds: int, field: str
) -> float | dict[str, float]:
    if isinstance(raw, dict):
        out = {}
        for key, value in raw.items():
            try:
                fold = int(key)
            except (TypeError, ValueError) as exc:
                raise ConfigError(f"{field} mapping keys must be fold counts") from exc
            if fold < 1 or fold >= int(folds):
                raise ConfigError(f"{field} fold keys must lie in [1, folds-1]")
            value = float(value)
            if value < 0:
                raise ConfigError(f"{field} values must be non-negative")
            out[str(fold)] = value
        return out
    value = float(raw)
    if value < 0:
        raise ConfigError(f"{field} must be non-negative")
    return value


def _cross_validation(raw: dict[str, Any]) -> dict[str, Any]:
    value = copy.deepcopy(raw)
    if not isinstance(value, dict):
        raise ConfigError("cross_validation must be an object")
    required = {"folds", "shuffle", "metric", "minimum_events_per_fold", "pruning"}
    if set(value) != required:
        raise ConfigError(f"cross_validation must contain exactly {sorted(required)}")
    folds = int(value["folds"])
    if folds < 2:
        raise ConfigError("cross_validation.folds must be >= 2")
    metric = str(value["metric"]).strip().lower()
    if metric not in MODEL_SELECTION_METRICS:
        raise ConfigError(
            f"cross_validation.metric must be one of {MODEL_SELECTION_METRICS}"
        )
    minimum = int(value["minimum_events_per_fold"])
    if minimum < 1:
        raise ConfigError("cross_validation.minimum_events_per_fold must be >= 1")
    pruning = copy.deepcopy(value["pruning"])
    if not isinstance(pruning, dict):
        raise ConfigError("cross_validation.pruning must be an object")
    required_pruning = {
        "enabled",
        "startup_complete_candidates",
        "min_folds_before_prune",
        "max_degradation_ps",
        "prune_if_worse_than_led",
        "led_max_degradation_ps",
    }
    if set(pruning) != required_pruning:
        raise ConfigError(
            f"cross_validation.pruning must contain exactly {sorted(required_pruning)}"
        )
    startup = int(pruning["startup_complete_candidates"])
    minimum_folds = int(pruning["min_folds_before_prune"])
    if startup < 0:
        raise ConfigError("startup_complete_candidates must be >= 0")
    if bool(pruning["enabled"]) and startup < 1:
        raise ConfigError(
            "enabled pruning requires at least one startup complete candidate"
        )
    if minimum_folds < 1 or minimum_folds >= folds:
        raise ConfigError("min_folds_before_prune must lie in [1, folds-1]")
    max_degradation = _tolerance(
        pruning["max_degradation_ps"], folds=folds, field="max_degradation_ps"
    )
    led_degradation = _tolerance(
        pruning["led_max_degradation_ps"], folds=folds, field="led_max_degradation_ps"
    )
    for field, tolerance in (
        ("max_degradation_ps", max_degradation),
        ("led_max_degradation_ps", led_degradation),
    ):
        if isinstance(tolerance, dict) and bool(pruning["enabled"]):
            missing = [
                str(i) for i in range(minimum_folds, folds) if str(i) not in tolerance
            ]
            if missing:
                raise ConfigError(
                    f"{field} mapping is missing eligible fold counts: {missing}"
                )
    return {
        "folds": folds,
        "shuffle": bool(value["shuffle"]),
        "metric": metric,
        "minimum_events_per_fold": minimum,
        "pruning": {
            "enabled": bool(pruning["enabled"]),
            "startup_complete_candidates": startup,
            "min_folds_before_prune": minimum_folds,
            "max_degradation_ps": max_degradation,
            "prune_if_worse_than_led": bool(pruning["prune_if_worse_than_led"]),
            "led_max_degradation_ps": led_degradation,
        },
    }


def _bootstrap(raw: dict[str, Any]) -> dict[str, Any]:
    value = copy.deepcopy(raw)
    if not isinstance(value, dict) or set(value) != {"n_resamples"}:
        raise ConfigError("bootstrap must contain exactly n_resamples")
    n_resamples = int(value["n_resamples"])
    if n_resamples < 1:
        raise ConfigError("bootstrap.n_resamples must be >= 1")
    return {"n_resamples": n_resamples}


def _xai(raw: dict[str, Any] | None = None) -> dict[str, Any]:
    value = copy.deepcopy(raw or {})
    if not isinstance(value, dict):
        raise ConfigError("xai must be an object")
    extra = set(value) - {"enabled", "group_size_samples", "max_events"}
    if extra:
        raise ConfigError(f"Unsupported xai fields: {sorted(extra)}")
    group_size = int(value.get("group_size_samples", XAIConfig.group_size_samples))
    max_events = int(value.get("max_events", XAIConfig.max_events))
    if group_size < 1 or max_events < 1:
        raise ConfigError("xai group_size_samples and max_events must be >= 1")
    return {
        "enabled": bool(value.get("enabled", XAIConfig.enabled)),
        "group_size_samples": group_size,
        "max_events": max_events,
    }


def _results(raw: dict[str, Any], root: Path) -> dict[str, Any]:
    if not isinstance(raw, dict) or set(raw) != {"root", "folder"}:
        raise ConfigError("results must contain exactly root and folder")
    base = Path(_project(root, raw["root"]))
    folder = str(raw["folder"]).strip().strip("/\\")
    if not folder:
        raise ConfigError("results.folder cannot be empty")
    return {
        "root": str(base.resolve()),
        "folder": folder,
        "directory": str((base / folder).resolve()),
    }


def _mode_tag(mode: str) -> str:
    return "energy" if mode == "energy_to_energy" else "timing"


def _excluded(model: str, mode: str, window: str, rules: list[dict[str, Any]]) -> bool:
    for rule in rules:
        if not isinstance(rule, dict):
            raise ConfigError("sweep.exclude entries must be objects")
        if all(
            rule.get(key, value) == value
            for key, value in (("model", model), ("mode", mode), ("window", window))
        ):
            return True
    return False


def _protocol_value(protocol: dict[str, Any], key: str, mode: str) -> Any:
    value = protocol[key]
    return (
        copy.deepcopy(value[mode])
        if isinstance(value, dict) and mode in value
        else copy.deepcopy(value)
    )


def _validate_dataset_compatibility(
    control: dict[str, Any], development: dict[str, Any], blind: dict[str, Any]
) -> None:
    if not (control["channels"] == development["channels"] == blind["channels"]):
        raise ConfigError(
            "control, development and blind channel definitions must match"
        )


def validate_config(config: dict[str, Any]) -> None:
    required = {
        "control",
        "development",
        "blind",
        "preprocessing",
        "mode",
        "model",
        "window_ns",
        "seed",
        "cross_validation",
        "bootstrap",
        "fit",
        "ml_input",
        "ml_output",
        "runtime",
        "xai",
        "output_dir",
        "save_model",
    }
    missing = required - set(config)
    if missing:
        raise ConfigError(f"Missing resolved study fields: {sorted(missing)}")
    if config["mode"] not in CHANNEL_MODES:
        raise ConfigError(f"mode must be one of {CHANNEL_MODES}")
    _validate_dataset_compatibility(
        config["control"], config["development"], config["blind"]
    )
    if config["mode"] == "timing_to_timing":
        for role in ("control", "development", "blind"):
            if not config[role]["channels"].get("timing"):
                raise ConfigError(f"timing mode requires timing channels in {role}")
    int(config["seed"])
    if int(config["ml_input"].get("subsampling", 1)) <= 0:
        raise ConfigError("ml_input.subsampling must be positive")
    if float(config["ml_output"]["max_abs_ps"]) <= 0:
        raise ConfigError("ml_output.max_abs_ps must be positive")
    preprocessing = config["preprocessing"]
    for key in (
        "materialized_window_ns",
        "energy",
        "timing",
        "selection",
        "photopeak",
        "tot_peak",
        "led_selection",
        "io",
    ):
        if key not in preprocessing:
            raise ConfigError(f"preprocessing.{key} is required")
    from waveform_analysis.models import model_names

    if config["model"]["name"] not in model_names():
        raise ConfigError(f"unregistered model {config['model']['name']}")


def load_batch_config(
    path: str | Path, project_root: str | Path | None = None
) -> BatchConfig:
    source = Path(path).expanduser().resolve()
    root = (
        Path(project_root).resolve()
        if project_root
        else Path(__file__).resolve().parents[1]
    )
    raw = _read(source)
    required = {
        "name",
        "control_dataset",
        "development_dataset",
        "blind_dataset",
        "results",
        "protocol",
        "sweep",
    }
    missing = required - set(raw)
    if missing:
        raise ConfigError(f"Missing batch fields: {sorted(missing)}")
    extra = set(raw) - required - {"save_model", "plot_config"}
    if extra:
        raise ConfigError(f"Unsupported batch fields: {sorted(extra)}")

    datasets = {
        "control": _dataset(source, raw["control_dataset"], root),
        "development": _dataset(source, raw["development_dataset"], root),
        "blind": _dataset(source, raw["blind_dataset"], root),
    }
    _validate_dataset_compatibility(
        datasets["control"], datasets["development"], datasets["blind"]
    )
    results = _results(raw["results"], root)
    plot_path = (
        _relative(source, raw.get("plot_config", DEFAULT_PLOT_CONFIG))
        if "plot_config" in raw
        else root / DEFAULT_PLOT_CONFIG
    )
    plot_config = _read(plot_path)

    protocol = copy.deepcopy(raw["protocol"])
    protocol_required = {
        "seed",
        "preprocessing_config",
        "cross_validation",
        "bootstrap",
        "fit",
        "ml_output",
    }
    if not protocol_required <= set(protocol):
        raise ConfigError(f"protocol requires {sorted(protocol_required)}")
    extra_protocol = set(protocol) - (
        protocol_required | {"ml_input", "runtime", "xai"}
    )
    if extra_protocol:
        raise ConfigError(f"Unsupported protocol fields: {sorted(extra_protocol)}")
    normalized_protocol = {
        "seed": int(protocol["seed"]),
        "preprocessing_config": str(protocol["preprocessing_config"]),
        "cross_validation": _cross_validation(protocol["cross_validation"]),
        "bootstrap": _bootstrap(protocol["bootstrap"]),
        "fit": copy.deepcopy(protocol["fit"]),
        "ml_input": copy.deepcopy(protocol.get("ml_input", {"subsampling": 1})),
        "ml_output": copy.deepcopy(protocol["ml_output"]),
        "runtime": _runtime(protocol.get("runtime")),
        "xai": _xai(protocol.get("xai")),
    }
    preprocessing = _preprocessing(source, protocol["preprocessing_config"], root)

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

    runs = []
    for mode_raw in modes:
        mode = str(mode_raw)
        if mode not in CHANNEL_MODES:
            raise ConfigError(f"mode must be one of {CHANNEL_MODES}")
        for window_name, window_raw in windows.items():
            window = _window(window_raw)
            for model_raw in models:
                model_name = (
                    model_raw
                    if isinstance(model_raw, str)
                    else str(model_raw.get("name", ""))
                )
                if not model_name:
                    raise ConfigError("every sweep model needs a name")
                if _excluded(model_name, mode, str(window_name), rules):
                    continue
                model = _model(source, model_raw, root)
                output_dir = (
                    Path(results["directory"])
                    / _mode_tag(mode)
                    / str(window_name)
                    / model_name
                )
                config = {
                    "name": f"{raw['name']}__{_mode_tag(mode)}__{window_name}__{model_name}",
                    "study_name": str(raw["name"]),
                    "run_id": f"{_mode_tag(mode)}__{window_name}__{model_name}",
                    "window_name": str(window_name),
                    "control": copy.deepcopy(datasets["control"]),
                    "development": copy.deepcopy(datasets["development"]),
                    "blind": copy.deepcopy(datasets["blind"]),
                    "preprocessing": copy.deepcopy(preprocessing),
                    "model": model,
                    "mode": mode,
                    "window_ns": window,
                    "seed": normalized_protocol["seed"],
                    "cross_validation": copy.deepcopy(
                        normalized_protocol["cross_validation"]
                    ),
                    "bootstrap": copy.deepcopy(normalized_protocol["bootstrap"]),
                    "fit": _fit(_protocol_value(protocol, "fit", mode)),
                    "ml_input": _protocol_value(protocol, "ml_input", mode)
                    if "ml_input" in protocol
                    else {"subsampling": 1},
                    "ml_output": _protocol_value(protocol, "ml_output", mode),
                    "runtime": copy.deepcopy(normalized_protocol["runtime"]),
                    "xai": copy.deepcopy(normalized_protocol["xai"]),
                    "output_dir": str(output_dir.resolve()),
                    "batch_output_dir": results["directory"],
                    "plot_config": copy.deepcopy(plot_config),
                    "save_model": bool(raw.get("save_model", True)),
                }
                validate_config(config)
                config["_config_path"] = str(source)
                config["_config_fingerprint"] = canonical_hash(
                    {
                        key: value
                        for key, value in config.items()
                        if key
                        not in {
                            "runtime",
                            "plot_config",
                            "output_dir",
                            "batch_output_dir",
                            "save_model",
                        }
                        and not str(key).startswith("_")
                    }
                )
                runs.append(config)
    if not runs:
        raise ConfigError("batch sweep produced no runs")

    axes = {
        "models": [m if isinstance(m, str) else str(m.get("name")) for m in models],
        "modes": [str(m) for m in modes],
        "windows": {str(key): _window(value) for key, value in windows.items()},
    }
    return BatchConfig(
        name=str(raw["name"]),
        output_dir=results["directory"],
        source_path=str(source),
        results=results,
        datasets=datasets,
        protocol=normalized_protocol,
        axes=axes,
        plot_config=plot_config,
        runs=tuple(runs),
    )


def public_config(config: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in config.items() if not str(key).startswith("_")}


def public_batch_config(batch: BatchConfig) -> dict[str, Any]:
    return {
        "name": batch.name,
        "results": batch.results,
        "datasets": batch.datasets,
        "source_path": batch.source_path,
        "protocol": batch.protocol,
        "axes": batch.axes,
        "save_model": batch.runs[0]["save_model"] if batch.runs else True,
        "runs": [public_config(config) for config in batch.runs],
    }


def load_config(
    path: str | Path, project_root: str | Path | None = None
) -> dict[str, Any]:
    raise ConfigError(
        "single-study configs were removed; use a batch config with control/development/blind roles"
    )


# Retain serialized identities while legacy imports resolve to this module.
from waveform_analysis.core.compat import (
    preserve_legacy_identity as _preserve_legacy_identity,
)

_preserve_legacy_identity(__name__, "waveform_analysis.ml_pipeline.config")
