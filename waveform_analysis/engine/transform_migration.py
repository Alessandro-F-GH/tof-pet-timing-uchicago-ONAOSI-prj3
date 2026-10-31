"""Import fitted feature transforms from historical results into a shared cache."""
from __future__ import annotations

import hashlib
import json
import pickle
from pathlib import Path
from typing import Any

import numpy as np

from waveform_analysis.core.artifact_identity import ProductRecipe
from waveform_analysis.core.io import canonical_hash
from waveform_analysis.data.artifact_store import ArtifactStore
from waveform_analysis.data.feature_cache import _CACHE_VERSION, _transform_parameters
from waveform_analysis.data.named_cache import (
    alias_exists,
    load_named_reference,
    publish_alias,
    transform_alias,
)
from waveform_analysis.data.results import discover_runs, resolve_run_config
from waveform_analysis.models import get_model


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _model_directory(run_directory: Path, source_artifact_root: Path | None) -> Path:
    """Resolve a local legacy model first, then its referenced external product."""
    local = run_directory / "model"
    local_transform = local / "feature_transform" / "transform.pkl"
    if local_transform.is_file() and (local / "sample_mask.npy").is_file():
        return local

    reference_path = run_directory / "metadata" / "model_reference.json"
    if reference_path.is_file():
        if source_artifact_root is None:
            if local.is_dir():
                return local
            raise ValueError(
                f"{reference_path} references an external model; supply "
                "--source-artifact-root to resolve it"
            )
        reference = json.loads(reference_path.read_text(encoding="utf-8"))
        return ArtifactStore(source_artifact_root).resolve(
            str(reference["product_type"]), str(reference["product_id"])
        ).path

    if local.is_dir():
        return local
    raise FileNotFoundError(f"No saved model directory found for {run_directory}")


def _transform_source(
    model_directory: Path, source_artifact_root: Path | None
) -> Path:
    """Find the transform pickle, including a separate old feature-transform product."""
    local = model_directory / "feature_transform" / "transform.pkl"
    if local.is_file():
        return local

    model_manifest = model_directory / ".artifact.json"
    if model_manifest.is_file():
        payload = json.loads(model_manifest.read_text(encoding="utf-8"))
        identity = (
            payload.get("recipe", {}).get("parameters", {})
            .get("feature_transform_identity")
        )
        if identity:
            if source_artifact_root is None:
                raise ValueError(
                    f"Model {model_directory} references feature transform {identity}; "
                    "supply --source-artifact-root"
                )
            reference = ArtifactStore(source_artifact_root).resolve(
                "feature-transforms", str(identity)
            )
            candidate = reference.payload_path("transform.pkl")
            if candidate.is_file():
                return candidate

    raise FileNotFoundError(
        f"No fitted feature transform found in {model_directory}; expected "
        "feature_transform/transform.pkl or a resolvable feature-transform reference"
    )


def _training_seed(artifact: Any) -> int:
    """Read the seed from the transform itself, including composite transforms."""
    seeds: set[int] = set()
    candidates = [artifact]
    candidates.extend(
        child for child in (
            getattr(artifact, "multirocket", None),
            getattr(artifact, "rdst", None),
        ) if child is not None
    )
    for candidate in candidates:
        metadata = getattr(candidate, "metadata", None)
        if isinstance(metadata, dict) and metadata.get("training_seed") is not None:
            seeds.add(int(metadata["training_seed"]))
    if not seeds:
        raise ValueError(
            "The saved transform does not record its training_seed; refusing to "
            "invent provenance for an imported fitted transformation"
        )
    if len(seeds) != 1:
        raise ValueError(f"Composite transform contains inconsistent training seeds: {sorted(seeds)}")
    return next(iter(seeds))


def _fit_identity(run_manifest: dict, fit_role: str, artifact_metadata: dict) -> str:
    populations = run_manifest.get("dataset_populations") or {}
    identity = (
        populations.get(fit_role, {}).get("analysis_protocol_identity")
        if isinstance(populations, dict) else None
    )
    if identity is None and fit_role == "control":
        identity = artifact_metadata.get("control_analysis_protocol_identity")
    if not identity:
        raise ValueError(
            f"Historical result does not record the {fit_role!r} analysis protocol "
            "identity; cannot safely bind its fitted transform to the shared cache"
        )
    return str(identity)


def _load_and_validate_source(
    spec, run_directory: Path, model_directory: Path, source_path: Path,
    config: dict, run_manifest: dict,
):
    sample_mask_path = model_directory / "sample_mask.npy"
    if not sample_mask_path.is_file():
        raise FileNotFoundError(f"Saved model sample mask is missing: {sample_mask_path}")
    sample_mask = np.asarray(
        np.load(sample_mask_path, allow_pickle=False), dtype=bool
    )
    if sample_mask.ndim != 1 or not np.any(sample_mask):
        raise ValueError(f"Invalid saved model sample mask: {sample_mask_path}")

    with source_path.open("rb") as stream:
        artifact = pickle.load(stream)
    artifact_metadata = getattr(artifact, "metadata", None)
    if not isinstance(artifact_metadata, dict):
        raise ValueError(f"Fitted transform has no metadata dictionary: {source_path}")

    transform_spec = spec.feature_transform
    fit_role = str(transform_spec.fit_role)
    if fit_role == "deterministic":
        raise ValueError(f"{spec.name} has a deterministic, not fitted, transform")

    parameters = _transform_parameters(spec, config["model"]["space"])
    training_seed = _training_seed(artifact)
    fit_identity = _fit_identity(run_manifest, fit_role, artifact_metadata)
    sample_mask_identity = run_manifest.get("sample_mask_identity")
    expected_after_mask = run_manifest.get("input_samples_after_mask")
    if expected_after_mask is not None and int(expected_after_mask) != int(sample_mask.sum()):
        raise ValueError(
            f"Saved model mask has {int(sample_mask.sum())} samples, but the run "
            f"manifest records {int(expected_after_mask)}"
        )

    if spec.requires_frozen_features:
        time_ps = getattr(artifact, "time_ps", None)
        sample_indices = getattr(artifact, "sample_indices", None)
        if time_ps is None or sample_indices is None:
            raise ValueError(
                f"{spec.name} requires the frozen time grid and sample indices; "
                "the historical transform does not contain them"
            )
        if np.asarray(time_ps).shape != (int(sample_mask.sum()),):
            raise ValueError("Historical transform time grid does not match the saved sample mask")
        if not np.array_equal(
            np.asarray(sample_indices, dtype=np.int64), np.flatnonzero(sample_mask)
        ):
            raise ValueError("Historical transform sample indices differ from the saved model mask")

    existing_control_identity = artifact_metadata.get("control_analysis_protocol_identity")
    if existing_control_identity and str(existing_control_identity) != fit_identity:
        raise ValueError(
            "Historical transform control identity disagrees with the run manifest"
        )

    source_sha256 = _sha256(source_path)
    contract = {
        "model": spec.name,
        "transform": transform_spec.name,
        "parameters": parameters,
        "batch_seed": int(config["seed"]),
        "training_seed": training_seed,
        "fit_role": fit_role,
        "fit_identity": fit_identity,
        "sample_mask_identity": sample_mask_identity,
        "sample_mask": sample_mask.tolist(),
        "input_samples": int(sample_mask.sum()),
        "mode": str(config["mode"]),
        "window_ns": dict(config["window_ns"]),
    }
    recipe = ProductRecipe(
        "feature-transforms",
        {
            "schema_version": _CACHE_VERSION,
            "model": spec.name,
            "transform": transform_spec.name,
            "parameters": parameters,
            "seed": training_seed,
            "fit_role": fit_role,
            "fit_identity": fit_identity,
            "sample_mask_identity": sample_mask_identity,
            "sample_mask": sample_mask.tolist(),
            "input_samples": int(sample_mask.sum()),
            "mode": str(config["mode"]),
            "window_ns": dict(config["window_ns"]),
        },
        implementation=(
            f"{transform_spec.name}:historical-import-v1:{source_sha256}"
        ),
    )
    return artifact, sample_mask, contract, recipe, source_sha256


def _contract_matches(left: dict, right: dict) -> bool:
    return canonical_hash(left) == canonical_hash(right)


def import_fitted_transforms(
    results_root: str | Path,
    *,
    cache_root: str | Path,
    source_artifact_root: str | Path | None = None,
    models: list[str] | tuple[str, ...] | None = None,
    overwrite: bool = False,
    dry_run: bool = False,
    logger=None,
) -> list[dict[str, Any]]:
    """Import saved transforms without fitting them, publishing standard cache aliases.

    The source result tree must use the canonical waveform-results layout. Imported
    aliases are keyed by the same dataset/model-transform/channel/window labels as
    prepare-cache. The embedded contract prevents a cache alias from being reused
    with a different model space, run seed, fit population, or sample mask.
    """
    results_root = Path(results_root).expanduser().resolve()
    cache_root = Path(cache_root).expanduser().resolve()
    source_root = (
        Path(source_artifact_root).expanduser().resolve()
        if source_artifact_root is not None else None
    )
    if source_root is not None and not source_root.is_dir():
        raise FileNotFoundError(f"Source artifact store does not exist: {source_root}")

    runs = discover_runs(results_root, complete=True)
    if not runs:
        raise ValueError(f"No complete canonical model runs found under {results_root}")
    available = {str(run["manifest"]["model"]) for run in runs}
    if models is not None:
        selected = {str(name) for name in models}
        unknown = selected - available
        if unknown:
            raise ValueError(f"Requested models not found in results: {sorted(unknown)}")
        runs = [run for run in runs if str(run["manifest"]["model"]) in selected]

    artifact_root = cache_root / "artifact_store"
    store = None if dry_run else ArtifactStore(artifact_root)
    records: list[dict[str, Any]] = []
    imported_aliases: dict[str, tuple[str, dict]] = {}

    for run in runs:
        run_directory = Path(run["directory"]).resolve()
        run_manifest = run["manifest"]
        config = resolve_run_config(run_directory)
        model_name = str(config["model"]["name"])
        spec = get_model(model_name)
        transform_spec = spec.feature_transform
        if transform_spec is None or transform_spec.save is None:
            continue
        if str(transform_spec.fit_role) == "deterministic":
            continue

        model_directory = _model_directory(run_directory, source_root)
        source_path = _transform_source(model_directory, source_root)
        artifact, sample_mask, contract, recipe, source_sha256 = _load_and_validate_source(
            spec, run_directory, model_directory, source_path, config, run_manifest
        )
        alias = transform_alias(
            cache_root, config, transform_spec.name, str(transform_spec.fit_role)
        )
        alias_key = str(alias)
        previous = imported_aliases.get(alias_key)
        if previous is not None:
            previous_sha, previous_contract = previous
            if previous_sha != source_sha256 or not _contract_matches(previous_contract, contract):
                raise ValueError(
                    f"Multiple historical runs map to the same transform alias but contain "
                    f"different fitted transforms: {alias}. Select one run/model or use "
                    "separate cache roots."
                )
            records.append({
                "run_id": config["run_id"], "model": model_name,
                "transform": transform_spec.name, "status": "duplicate_source",
                "alias": str(alias),
            })
            continue
        imported_aliases[alias_key] = (source_sha256, contract)

        if dry_run:
            records.append({
                "run_id": config["run_id"], "model": model_name,
                "transform": transform_spec.name, "status": "dry_run",
                "source": source_path.name, "source_sha256": source_sha256,
                "alias": str(alias),
            })
            continue

        if alias_exists(alias):
            existing = load_named_reference(
                store, alias, expected_type="feature-transforms"
            )
            existing_metadata = existing.manifest.get("metadata", {})
            existing_contract = existing_metadata.get("transform_contract")
            if (
                existing_metadata.get("source_sha256") == source_sha256
                and isinstance(existing_contract, dict)
                and _contract_matches(existing_contract, contract)
            ):
                records.append({
                    "run_id": config["run_id"], "model": model_name,
                    "transform": transform_spec.name, "status": "already_imported",
                    "product_id": existing.product_id, "alias": str(alias),
                })
                continue
            if not overwrite:
                raise FileExistsError(
                    f"Transform alias already points to a different product: {alias}. "
                    "Use --overwrite to retarget this alias to the historical fitted transform."
                )

        def build(directory, effective_recipe=None):
            metadata = getattr(artifact, "metadata")
            metadata["transform_fit_role"] = contract["fit_role"]
            metadata["input_samples"] = contract["input_samples"]
            metadata["transform_identity"] = (
                effective_recipe or recipe
            ).product_id
            metadata["control_analysis_protocol_identity"] = contract["fit_identity"]
            with (directory / "transform.pkl").open("wb") as stream:
                pickle.dump(artifact, stream, protocol=pickle.HIGHEST_PROTOCOL)
                stream.flush()

        reference = store.get_or_build(
            recipe,
            build,
            metadata={
                "model": model_name,
                "transform": transform_spec.name,
                "fit_role": contract["fit_role"],
                "imported_from_historical_results": True,
                "source_sha256": source_sha256,
                "source_run_id": str(config["run_id"]),
                "source_transform_filename": source_path.name,
                "transform_contract": contract,
            },
        )
        publish_alias(alias, reference.path, replace=bool(overwrite))
        records.append({
            "run_id": config["run_id"], "model": model_name,
            "transform": transform_spec.name, "status": "imported",
            "product_id": reference.product_id, "alias": str(alias),
        })
        if logger:
            logger.info(
                "Imported fitted transform | %s | %s | id=%s",
                model_name, alias, reference.product_id[:12],
            )

    if not records:
        raise ValueError(
            "No fitted feature transformations were found in the selected complete runs"
        )
    return records
