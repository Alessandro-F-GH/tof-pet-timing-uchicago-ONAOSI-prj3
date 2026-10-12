"""Recompute XAI from saved batch models and verified prepared caches only."""
from __future__ import annotations

import csv
import json
from pathlib import Path, PureWindowsPath
import re
from typing import Any

from waveform_analysis.core.io import atomic_json, canonical_hash
from waveform_analysis.data.dataset import PreparedDataset, load_prepared_dataset
from waveform_analysis.data.splits import semantic_seed
from waveform_analysis.data.storage import RunStore
from waveform_analysis.engine.diagnostics import diagnostic_random_state
from waveform_analysis.engine.train import load_fitted_model, release_training_memory, saved_model_complete
from waveform_analysis.engine.xai import XAI_METHOD_VERSION, temporal_occlusion_importance
from waveform_analysis.models import get_model
from waveform_analysis.reporting.plotting import plot_run_xai


def _read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _cache_root(config: dict[str, Any], run: Path, override: Path | None) -> Path:
    if override is not None:
        return override.expanduser().resolve()
    configured = Path(config["preprocessing"]["cache_dir"]).expanduser()
    candidates = [configured]
    parts = PureWindowsPath(str(configured)).parts
    anchors = [i for i, part in enumerate(parts) if part.lower() == "waveform_analysis"]
    if anchors:
        suffix = Path(*parts[anchors[-1] + 1:])
        roots = [parent for parent in run.parents if parent.name.lower() == "waveform_analysis"]
        roots.append(Path(__file__).resolve().parents[1])
        candidates.extend(root / suffix for root in roots)
    for candidate in candidates:
        if all((candidate / f"{role}_ml" / "prepared").is_dir() for role in ("control", "blind")):
            return candidate.resolve()
    return configured.resolve()


class PreparedCacheLookup:
    """Index metadata once and reuse memory-mapped datasets across saved models."""

    def __init__(self):
        self.index: dict[tuple[Path, str], list[tuple[Path, dict[str, Any]]]] = {}
        self.datasets: dict[Path, PreparedDataset] = {}

    def find(self, root: Path, role: str, config: dict[str, Any], manifest: dict[str, Any]) -> PreparedDataset:
        key = root, role
        if key not in self.index:
            entries = []
            for path in (root / f"{role}_ml" / "prepared").glob("*/manifest.json"):
                try:
                    entries.append((path.parent, _read(path)))
                except (OSError, ValueError):
                    continue
            self.index[key] = entries
        population = manifest.get("dataset_populations", {}).get(role, {})
        expected_protocol = population.get("analysis_protocol_identity")
        expected_population = population.get("population_identity")
        control_prefix = PureWindowsPath(manifest.get("control_artifact", "")).name
        matches = []
        for path, cached in self.index[key]:
            if not (
                cached.get("dataset_role") == role
                and cached.get("dataset_source") == config[role]["root_file"]
                and cached.get("mode") == config["mode"]
                and cached.get("window_ns") == config["window_ns"]
                and cached.get("subsampling") == config["ml_input"]["subsampling"]
                and cached.get("true_tof_ps") == config[role]["true_tof_ps"]
                and (expected_population is None or cached.get("event_population_identity") == expected_population)
            ):
                continue
            if expected_protocol is not None:
                verified = cached.get("analysis_protocol_identity") == expected_protocol
            else:
                verified = (re.fullmatch(r"[0-9a-f]{16}", control_prefix) is not None
                            and str(cached.get("control_fingerprint", "")).startswith(control_prefix))
            if verified:
                matches.append(path)
        if len(matches) != 1:
            raise ValueError(
                f"Expected one verified {role} cache, found {len(matches)} in "
                f"{root / f'{role}_ml' / 'prepared'}. Retain the original prepared "
                "control/blind arrays or supply their location with --cache-root."
            )
        path = matches[0]
        if path not in self.datasets:
            # Read legacy full-grid arrays without changing their input policy;
            # saved models supply their original masks and fitted transforms.
            self.datasets[path] = load_prepared_dataset(path, diagnostic_legacy=True)
        return self.datasets[path]


def _runs(root: Path) -> list[RunStore]:
    index = root / "tables" / "runs.csv"
    indices = [index] if index.is_file() else sorted(root.glob("**/tables/runs.csv"))
    legacy = root / "runs.csv"
    if not indices and legacy.is_file():
        indices = [legacy]
    if not indices:
        raise FileNotFoundError(f"No batch tables/runs.csv found in {root}")
    stores = {}
    for path in indices:
        batch_root = path.parent.parent if path.parent.name == "tables" else path.parent
        with path.open(encoding="utf-8", newline="") as stream:
            for row in csv.DictReader(stream):
                directory = (batch_root / row["path"]).resolve()
                store = RunStore(directory)
                if store.manifest_path.is_file() and _read(store.manifest_path).get("status") == "complete":
                    stores[directory] = store
    if not stores:
        raise ValueError(f"No complete saved-model runs found in {root}")
    return list(stores.values())


def rerun_xai(
    results: str | Path, *, models: list[str] | None = None,
    cache_root: Path | None = None, group_size_samples: int | None = None,
    max_events: int | None = None, logger=None,
) -> list[Path]:
    """Recompute XAI and model-level plots, preserving training and blind metrics.

    This explicit command runs even if the original batch disabled XAI. It never
    fits a model, preprocessing rule or feature transform, and never reads raw
    acquisitions. It respects each checkpoint's original mask/output limits.
    Each new artifact records the exact control cohort and explanation settings.
    """
    if any(v is not None and v < 1 for v in (group_size_samples, max_events)):
        raise ValueError("XAI group size and max events must be positive")
    stores = _runs(Path(results).expanduser().resolve())
    available = {_read(store.manifest_path)["model"] for store in stores}
    if models is not None and set(models) - available:
        raise ValueError(f"Requested models not found: {sorted(set(models) - available)}")
    lookup = PreparedCacheLookup()
    outputs, failures = [], []
    for store in stores:
        manifest = _read(store.manifest_path)
        if models is not None and manifest["model"] not in models:
            continue
        try:
            config = _read(store.metadata_dir / "config.json")
            settings = {**config["xai"], "enabled": True}
            if group_size_samples is not None:
                settings["group_size_samples"] = int(group_size_samples)
            if max_events is not None:
                settings["max_events"] = int(max_events)
            best = _read(store.best_path)
            spec = get_model(manifest["model"])
            if not saved_model_complete(spec, store.model_dir):
                raise FileNotFoundError("Saved final model or frozen transform is missing/incomplete")
            root = _cache_root(config, store.root, cache_root)
            control = lookup.find(root, "control", config, manifest)
            blind = lookup.find(root, "blind", config, manifest)
            final_fit = manifest["stage_fingerprints"]["final_fit"]
            payload = {
                "final_fit": final_fit, "blind": blind.manifest["analysis_protocol_identity"],
                "xai": settings, "seed": config["seed"], "method_version": XAI_METHOD_VERSION,
                "control": control.manifest["analysis_protocol_identity"],
                "formulation": spec.estimator_formulation,
            }
            fingerprint = canonical_hash({"schema_version": manifest["schema_version"],
                                          "stage": "xai", "payload": payload})
            if logger:
                logger.info("XAI rerun | saved model | %s | %s | %s", spec.name, config["mode"], config.get("window_name"))
            with diagnostic_random_state():
                fitted = load_fitted_model(spec, store.model_dir, best["parameters"], config)
                try:
                    scores = temporal_occlusion_importance(
                        spec, fitted, blind, config["mode"], control_dataset=control,
                        cache_root=root, group_size_samples=settings["group_size_samples"],
                        max_events=settings["max_events"],
                        seed=semantic_seed(config["seed"], spec.name, "xai"),
                    )
                finally:
                    del fitted
                    release_training_memory()
            store.save_xai(**scores)
            metadata = {"method": "grouped_temporal_occlusion", "replacement_method": "control_sample_mean",
                        "method_version": XAI_METHOD_VERSION, "formulation": spec.estimator_formulation,
                        "control_analysis_protocol_identity": control.manifest["analysis_protocol_identity"],
                        "control_n_events": control.n_events, "settings": settings}
            # Preserve other stages, including older saved-model input policies.
            state = store.read_json(store.state_path, {"schema_version": manifest["schema_version"], "stages": {}})
            state.setdefault("stages", {})["xai"] = {"status": "complete", "fingerprint": fingerprint,
                                                     "metadata": metadata}
            atomic_json(store.state_path, state)
            manifest["stage_fingerprints"]["xai"] = fingerprint
            store.write_manifest(manifest)
            atomic_json(store.metadata_dir / "xai.json", {**metadata, "fingerprint": fingerprint})
            plot_run_xai(store.root, config["plot_config"])
            outputs.append(store.xai_path)
        except Exception as exc:
            failures.append(f"{store.root}: {exc}")
            if logger:
                logger.error("XAI unavailable | %s | %s", store.root, exc)
    if failures:
        raise RuntimeError("Some XAI reruns failed; successful artifacts were retained:\n" + "\n".join(failures))
    return outputs
