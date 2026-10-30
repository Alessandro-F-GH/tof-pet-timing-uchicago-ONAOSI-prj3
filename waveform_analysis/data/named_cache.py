"""Stable, human-readable cache paths for reusable scientific products.

ArtifactStore hashes remain internal integrity/provenance identifiers. Users and
pipeline stages select products by dataset, channel, window, model and role.
An existing named alias is deliberately reused without comparing recipe hashes:
the caller owns the choice of semantic name.
"""
from __future__ import annotations

from dataclasses import replace
import json
import os
import re
import shutil
import uuid
from pathlib import Path
from typing import Any, Callable


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
        label = Path(str(source).replace("\\\\", "/")).stem
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
    """Resolve the bundle root, not its preprocessing/ or artifact_store child."""
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
        fallback_path = Path(fallback).expanduser().resolve()
        if fallback_path.name == "preprocessing":
            return fallback_path.parent
        return fallback_path
    cache_dir = preprocessing.get("cache_dir")
    if cache_dir:
        cache_path = Path(cache_dir).expanduser().resolve()
        return cache_path.parent if cache_path.name == "preprocessing" else cache_path
    raise ValueError("Cannot resolve named cache root")


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


def legacy_selection_alias(root, dataset, mode) -> Path:
    """Pre-control-qualified selection alias used by earlier cache layouts."""
    return alias_path(root, "selection", dataset_name(dataset), mode_name(mode))


def native_alias(root, dataset, control_dataset, mode) -> Path:
    return alias_path(root, "native", dataset_name(dataset), mode_name(mode))


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


def feature_alias(root, config, model_name: str, dataset, role: str, block="combined") -> Path:
    return alias_path(
        root, "features", model_name, dataset_name(config["control"]),
        dataset_name(dataset), mode_name(config["mode"]), window_name(config), role, block,
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
    path = Path(path)
    return path.exists() or path.is_symlink()


def resolve_alias(path) -> Path:
    """Resolve a directory symlink or a portable JSON pointer."""
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
    """Atomically publish a readable alias without duplicating large payloads."""
    alias = Path(os.path.abspath(Path(alias).expanduser()))
    target = Path(target).expanduser().resolve()
    if not target.is_dir():
        raise FileNotFoundError(f"Cache product directory does not exist: {target}")
    if alias_exists(alias):
        if not replace:
            return alias
        if alias.is_symlink() or alias.is_file():
            alias.unlink()
        else:
            shutil.rmtree(alias)
    alias.parent.mkdir(parents=True, exist_ok=True)
    relative = os.path.relpath(target, alias.parent)
    temporary = alias.with_name(f".{alias.name}.{uuid.uuid4().hex}.linking")
    try:
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
            shutil.rmtree(temporary, ignore_errors=True)
    return alias


def load_named_reference(store, alias, *, expected_type: str | None = None):
    """Resolve an alias to its ArtifactStore reference without hashing large arrays."""
    from waveform_analysis.data.artifact_store import ArtifactCorrupt

    target = resolve_alias(alias)
    manifest_path = target / ".artifact.json"
    if not manifest_path.is_file():
        raise ArtifactCorrupt(f"Named cache entry has no ArtifactStore manifest: {alias}")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        product_type = str(manifest["product_type"])
        product_id = str(manifest["product_id"])
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise ArtifactCorrupt(f"Invalid named cache entry: {alias}") from exc
    if expected_type is not None and product_type != expected_type:
        raise ArtifactCorrupt(
            f"Named cache entry {alias} contains {product_type}, expected {expected_type}"
        )
    reference = store.resolve(product_type, product_id, verify_payloads=False)
    if reference.path.resolve() != target.resolve():
        raise ArtifactCorrupt(f"Named cache alias does not resolve to its registered product: {alias}")
    return reference


def get_or_build_named(
    store,
    recipe,
    alias,
    builder: Callable,
    *,
    metadata=None,
    rebuild: bool = False,
):
    """Load by semantic path first; recipe hashes are only internal fallbacks.

    The builder receives (directory, effective_recipe). The second argument is
    useful for payload metadata that embeds its own internal product reference.
    Explicit rebuilds publish a new immutable product and atomically retarget the
    same readable alias.
    """
    alias = Path(os.path.abspath(Path(alias).expanduser()))
    if alias_exists(alias) and not rebuild:
        return load_named_reference(store, alias, expected_type=recipe.product_type)

    effective_recipe = recipe
    if rebuild:
        parameters = dict(recipe.parameters)
        parameters["_explicit_rebuild"] = uuid.uuid4().hex
        effective_recipe = replace(recipe, parameters=parameters)
    elif not alias_exists(alias):
        # Adopt a valid pre-existing content-addressed product without rebuilding it.
        found = store.lookup(recipe, verify_payloads=False)
        if found.status == "hit":
            publish_alias(alias, found.reference.path)
            return found.reference

    reference = store.get_or_build(
        effective_recipe,
        lambda directory: builder(directory, effective_recipe),
        metadata=metadata,
    )
    publish_alias(alias, reference.path, replace=rebuild)
    return reference


def product_id_from_alias(path) -> str | None:
    """Read the internal ArtifactStore ID for provenance; never needed for lookup."""
    try:
        product_manifest = resolve_alias(path) / ".artifact.json"
        if not product_manifest.is_file():
            return None
        return str(json.loads(product_manifest.read_text(encoding="utf-8"))["product_id"])
    except (OSError, ValueError, KeyError, TypeError):
        return None
