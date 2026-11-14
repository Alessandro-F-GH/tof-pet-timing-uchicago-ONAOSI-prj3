"""Human-readable paths for immutable published cache directories."""
from __future__ import annotations
import re
from pathlib import Path
from typing import Any
from waveform_analysis.data.cache_store import cache_complete

def safe_component(value: Any) -> str:
    """Convert a scientific label into one safe, readable path component."""
    text = str(value).strip()
    text = re.sub(r"[\\/]+", "_", text)
    text = re.sub(r"[^A-Za-z0-9_.+-]+", "_", text).strip("._")
    if not text:
        raise ValueError(f"Empty cache path component from {value!r}")
    return text


def dataset_name(dataset: dict[str, Any]) -> str:
    """Prefer an explicit dataset label; otherwise use only the ROOT filename."""
    label = dataset.get("name") or dataset.get("dataset_name")
    if not label:
        source = dataset.get("root_file")
        if not source:
            raise ValueError("Dataset cache naming requires root_file or name")
        # Result folders and prepared-cache bundles may move between Windows
        # and POSIX machines. pathlib.Path follows the host OS, so normalize
        # both separator styles before extracting the filename stem.
        label = Path(str(source).replace("\\", "/")).stem
    return safe_component(label)


def mode_name(mode: str) -> str:
    try:
        return {"energy_to_energy": "energy", "timing_to_timing": "timing"}[str(mode)]
    except KeyError as exc:
        raise ValueError(f"Unsupported cache channel mode: {mode}") from exc


def window_name(config: dict[str, Any]) -> str:
    name = config.get("window_name")
    if name:
        return safe_component(name)
    window = config.get("window_ns") or {}
    if {"start", "end"} <= set(window):
        return safe_component(f"{window['start']}_to_{window['end']}ns")
    raise ValueError("Cache naming requires window_name or window_ns")


def cache_root_from_config(config: dict[str, Any], fallback=None) -> Path:
    """Return the explicit preprocessing cache directory."""
    preprocessing = config.get("preprocessing", config)
    directory = preprocessing.get("cache_dir")
    if directory:
        return Path(directory).expanduser().resolve()
    if fallback is not None:
        return Path(fallback).expanduser().resolve()
    raise ValueError("Cannot resolve preprocessing cache directory")


def alias_path(root, *parts) -> Path:
    return Path(root).expanduser().resolve().joinpath(*(safe_component(p) for p in parts))


def control_alias(root, control_dataset, modes) -> Path:
    mode_key = "_".join(sorted(mode_name(mode) for mode in modes))
    return alias_path(root, "control", dataset_name(control_dataset), mode_key)


def selection_alias(root, dataset, control_dataset, mode) -> Path:
    # The selected population depends on both the target dataset and the
    # control dataset whose fitted rules are applied to it.
    return alias_path(
        root, "selection", dataset_name(control_dataset),
        dataset_name(dataset), mode_name(mode),
    )


def native_alias(root, dataset, control_dataset, mode) -> Path:
    return alias_path(root, "native", dataset_name(control_dataset), dataset_name(dataset), mode_name(mode))


def prepared_dataset_alias(root, config, role: str) -> Path:
    return alias_path(
        root, "datasets", dataset_name(config["control"]), dataset_name(config[role]),
        mode_name(config["mode"]), window_name(config), role,
    )


def sample_mask_alias(root, config) -> Path:
    return alias_path(
        root, "sample_masks", dataset_name(config["control"]),
        mode_name(config["mode"]), window_name(config),
    )


def transform_alias(root, config, model_name: str, fit_role: str) -> Path:
    fit_dataset = (config.get(fit_role) or config.get("development")
                   or config.get("control"))
    if fit_dataset is None or "control" not in config:
        raise ValueError("Frozen transform cache naming requires Control and a fit dataset")
    return alias_path(
        root, "transforms", model_name, dataset_name(config["control"]),
        dataset_name(fit_dataset), mode_name(config["mode"]), window_name(config), f"seed_{int(config['seed'])}",
    )


def feature_alias(root, config, model_name: str, dataset, role: str, block="combined") -> Path:
    return alias_path(
        root, "features", model_name, dataset_name(config["control"]),
        dataset_name(dataset), mode_name(config["mode"]), window_name(config), f"seed_{int(config['seed'])}", role, block,
    )


def imputer_alias(
    root, config, *, window_samples, stride_samples, components,
    fit_events, tuning_events, slope_points, seed,
) -> Path:
    settings = (
        f"w{int(window_samples)}_s{int(stride_samples)}_k{int(components)}"
        f"_fit{int(fit_events)}_tune{int(tuning_events)}"
        f"_slope{int(slope_points)}_seed{int(seed)}"
    )
    return alias_path(
        root, "imputers", "hermite_ppca", dataset_name(config["control"]),
        mode_name(config["mode"]), window_name(config), settings,
    )



def alias_exists(path) -> bool:
    return cache_complete(path)


def resolve_alias(path) -> Path:
    return Path(path).expanduser().resolve()
