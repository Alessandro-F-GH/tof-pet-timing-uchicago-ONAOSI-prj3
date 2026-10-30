"""Migrate fitted feature transforms from legacy run folders into a shared cache.

No transformations are refitted. Each legacy pickle is imported into the
content-addressed ArtifactStore using the recipe derived from its saved run
configuration and prepared control/development datasets.
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

from waveform_analysis.data.prepared_cache import PreparedCacheLookup
from waveform_analysis.data.results import discover_runs
from waveform_analysis.data.storage import RunStore
from waveform_analysis.data.feature_cache import (
    prepare_frozen_transform,
    uses_frozen_model_input,
)
from waveform_analysis.engine.cache_preparation import _runtime_configs
from waveform_analysis.engine.preprocessing import fit_control, prepare_role_dataset
from waveform_analysis.models import get_model


def _legacy_transform_path(run, spec, config):
    """Find the transform beside the saved model, including artifact-backed models."""
    store = RunStore(
        run["directory"],
        artifact_root=config["preprocessing"].get("artifact_root"),
    )
    model_dir = store.model_dir
    candidate = model_dir / "feature_transform" / "transform.pkl"
    if candidate.is_file():
        return candidate
    # Some older runs stored the complete model directory locally.
    candidate = Path(run["directory"]) / "model" / "feature_transform" / "transform.pkl"
    if candidate.is_file():
        return candidate
    return None


def migrate_feature_transforms(
    results_root,
    *,
    cache_root,
    models=None,
    dry_run=False,
    logger=None,
):
    results_root = Path(results_root).expanduser().resolve()
    cache_root = Path(cache_root).expanduser().resolve()
    artifact_root = cache_root / "artifact_store"
    preprocessing_root = cache_root / "preprocessing"

    runs = discover_runs(results_root, complete=False)
    if not runs:
        raise ValueError(f"No result runs found under {results_root}")
    selected_models = None if models is None else {str(x) for x in models}
    lookup = PreparedCacheLookup()
    report = {"imported": [], "already_available": [], "skipped": [], "failed": []}

    for index, run in enumerate(runs, start=1):
        directory = Path(run["directory"])
        manifest = run["manifest"]
        model_name = str(manifest["model"])
        if selected_models is not None and model_name not in selected_models:
            continue
        try:
            config = __import__(
                "waveform_analysis.data.results", fromlist=["resolve_run_config"]
            ).resolve_run_config(directory)
            # Resolve legacy prepared datasets and model references using the
            # source paths recorded in the run configuration.
            spec = get_model(model_name)
            if not uses_frozen_model_input(spec):
                report["skipped"].append({"run": str(directory), "reason": "model has no fitted feature transform"})
                continue

            source = _legacy_transform_path(run, spec, config)
            if source is None:
                raise FileNotFoundError(
                    f"No legacy feature_transform/transform.pkl found for {directory}"
                )
            control = lookup.find("control", config, manifest)
            development = lookup.find("development", config, manifest)
            if control is None or development is None:
                raise FileNotFoundError(
                    "Prepared control/development datasets are not available in the source cache"
                )
            # Only redirect artifact creation after the source datasets/model have
            # been resolved. This keeps migration read-only with respect to old caches.
            config["preprocessing"]["cache_dir"] = str(preprocessing_root)
            config["preprocessing"]["artifact_root"] = str(artifact_root)

            if logger:
                logger.info(
                    "Transform migration | %d/%d | %s | %s | %s",
                    index, len(runs), model_name, config.get("window_name"), source,
                )
            if dry_run:
                report["skipped"].append({
                    "run": str(directory), "source": str(source),
                    "reason": "dry-run; no changes made",
                })
                continue

            from waveform_analysis.data.named_cache import transform_alias
            alias = transform_alias(
                cache_root, config, spec.feature_transform.name,
                spec.feature_transform.fit_role,
            )
            was_present = alias.exists() or alias.is_symlink()
            frozen = prepare_frozen_transform(
                spec,
                config["model"]["space"],
                config,
                control_dataset=control,
                development_dataset=development,
                cache_root=preprocessing_root,
                logger=logger,
                legacy_transform_path=source,
            )
            if frozen is None:
                raise RuntimeError("Expected a fitted transformation but none was returned")
            # Resolve the just-published semantic alias to verify the artifact is usable.
            from waveform_analysis.data.artifact_store import ArtifactStore
            from waveform_analysis.data.named_cache import load_named_reference
            reference = load_named_reference(
                ArtifactStore(artifact_root), alias, expected_type="feature-transforms"
            )
            if reference.product_id != frozen.identity:
                raise RuntimeError("Published transform identity does not match loaded transform")
            item = {
                "run": str(directory),
                "source": str(source),
                "transform": spec.feature_transform.name,
                "product_id": reference.product_id,
                "alias": str(alias),
            }
            report["already_available" if was_present else "imported"].append(item)
        except Exception as exc:
            item = {"run": str(directory), "error": f"{type(exc).__name__}: {exc}"}
            report["failed"].append(item)
            if logger:
                logger.exception("Transform migration failed | %s", directory)

    return report


def main():
    parser = argparse.ArgumentParser(
        description="Import fitted transforms from old results into a shared cache."
    )
    parser.add_argument("--results", required=True, help="Root folder containing legacy/canonical results")
    parser.add_argument("--cache-root", required=True, help="Destination shared cache bundle")
    parser.add_argument("--models", nargs="+", help="Optional model-name filter")
    parser.add_argument("--dry-run", action="store_true", help="Inspect sources without importing")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )
    logger = logging.getLogger("migrate_feature_transforms")
    report = migrate_feature_transforms(
        args.results,
        cache_root=args.cache_root,
        models=args.models,
        dry_run=args.dry_run,
        logger=logger,
    )
    for key, items in report.items():
        print(f"{key}: {len(items)}")
        for item in items:
            print(f"  {item}")
    if report["failed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
