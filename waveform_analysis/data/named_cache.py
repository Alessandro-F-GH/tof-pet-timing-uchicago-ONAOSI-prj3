"""Human-readable aliases for reusable waveform-analysis cache products.

The underlying ArtifactStore remains content-addressed for integrity and model
references. This module exposes stable, semantic paths so users never need to
know or select a product hash. Existing aliases are trusted by name: choosing
the correct dataset/window/model combination is the caller's responsibility.
"""
from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any


def safe_component(value: Any) -> str:
    """Convert a scientific label into one safe, readable path component."""
    text = str(value).strip()
    text = re.sub(r"[\\/]+", "_", text)
    text = re.sub(r"[^A-Za-z0-9_.+-]+", "_", text)
    text = text.strip("._")
    if not text:
        raise ValueError(f"Empty cache path component from {value!r}")
    return text


def dataset_name(dataset: dict[str, Any]) -> str:
    """Use an explicit dataset label when available, otherwise the ROOT stem."""
    label = dataset.get("name") or dataset.get("dataset_name")
    if not label:
        source = dataset.get("root_file")
        if not source:
            raise ValueError("Dataset cache naming requires root_file or name")
        label = Path(str(source)).stem
    return safe_component(label)


def mode_name(mode: str) -> str:
    value = str(mode)
    if value == "energy_to_energy":
        return "energy"
    if value == "timing_to_timing":
        return "timing"
    raise ValueError(f"Unsupported cache channel mode: {mode}")


def window_name(config: dict[str, Any]) -> str:
    name = config.get("window_name")
    if name:
        return safe_component(name)
    window = config.get("window_ns") or {}
    if {"start", "end"} <= set(window):
        return safe_component(f"{window['start']}_to_{window['end']}ns")
    raise ValueError("Cache naming requires window_name or window_ns")


def cache_root_from_config(config: dict[str, Any], fallback=None) -> Path:
    """Resolve the human-readable cache root from a resolved run configuration."""
    preprocessing = config.get("preprocessing", config)
    named = preprocessing.get("named_cache_root")
    if named:
        return Path(named).expanduser().resolve()
    artifact_root = preprocessing.get("artifact_root")
    if artifact_root:
        artifact_path = Path(artifact_root).expanduser().resolve()
        if artifact_path.name == "artifact_store":
            return artifact_path.parent
    if fallback is not None:
        return Path(fallback).expanduser().resolve()
    cache_dir = preprocessing.get("cache_dir")
    if cache_dir:
        return Path(cache_dir).expanduser().resolve()
    raise ValueError("Cannot resolve named cache root")


def alias_path(root, *parts) -> Path:
    return Path(root).expanduser().resolve().joinpath(*(safe_component(p) for p in parts))


def control_alias(root, control_dataset) -> Path:
    return alias_path(root, "control", dataset_name(control_dataset))


def selection_alias(root, dataset, control_dataset, mode) -> Path:
    return alias_path(
        root, "selection", dataset_name(control_dataset), dataset_name(dataset), mode_name(mode)
    )


def native_alias(root, dataset, control_dataset, mode) -> Path:
    return alias_path(
        root, "native", dataset_name(control_dataset), dataset_name(dataset), mode_name(mode)
    )


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
    fit_dataset = config.get(fit_role, config.get("development", config["control"]))
    return alias_path(
        root, "transforms", model_name, dataset_name(config["control"]),
        dataset_name(fit_dataset), mode_name(config["mode"]), window_name(config),
    )


def feature_alias(root, config, model_name: str, dataset, role: str) -> Path:
    return alias_path(
        root, "features", model_name, dataset_name(config["control"]),
        dataset_name(dataset), mode_name(config["mode"]), window_name(config), role,
    )


def alias_exists(path) -> bool:
    path = Path(path)
    return path.exists() or path.is_symlink()


def resolve_alias(path) -> Path:
    """Resolve a directory symlink or a portable JSON pointer to its product."""
    path = Path(path).expanduser()
    if path.is_symlink():
        return path.resolve()
    pointer = path / ".cache_link.json"
    if pointer.is_file():
        data = json.loads(pointer.read_text(encoding="utf-8"))
        target = Path(data["target"])
        if not target.is_absolute():
            target = path.parent / target
        return target.resolve()
    return path.resolve()


def publish_alias(alias, target, *, replace: bool = False) -> Path:
    """Publish a stable readable alias to an ArtifactStore product.

    On systems where directory symlinks are unavailable, a small relative
    pointer file is used and resolved by resolve_alias().
    """
    alias = Path(alias).expanduser().resolve()
    target = Path(target).expanduser().resolve()
    if not target.is_dir():
        raise FileNotFoundError(f"Cache product directory does not exist: {target}")
    if alias_exists(alias):
        if not replace:
            return alias
        if alias.is_symlink() or alias.is_file():
            alias.unlink()
        else:
            import shutil
            shutil.rmtree(alias)
    alias.parent.mkdir(parents=True, exist_ok=True)
    relative = os.path.relpath(target, alias.parent)
    temporary = alias.with_name(f".{alias.name}.linking")
    try:
        if temporary.exists() or temporary.is_symlink():
            if temporary.is_symlink() or temporary.is_file():
                temporary.unlink()
            else:
                import shutil
                shutil.rmtree(temporary)
        try:
            os.symlink(relative, temporary, target_is_directory=True)
        except (OSError, NotImplementedError):
            temporary.mkdir(parents=True, exist_ok=True)
            (temporary / ".cache_link.json").write_text(
                json.dumps({"target": relative}, indent=2) + "\\n", encoding="utf-8"
            )
        os.replace(temporary, alias)
    except FileExistsError:
        if temporary.is_symlink() or temporary.is_file():
            temporary.unlink(missing_ok=True)
        elif temporary.exists():
            import shutil
            shutil.rmtree(temporary, ignore_errors=True)
    return alias


def product_id_from_alias(path) -> str | None:
    """Read the hidden ArtifactStore ID when an alias points to a stored product."""
    product_manifest = resolve_alias(path) / ".artifact.json"
    if not product_manifest.is_file():
        return None
    try:
        return str(json.loads(product_manifest.read_text(encoding="utf-8"))["product_id"])
    except (OSError, ValueError, KeyError, TypeError):
        return None
