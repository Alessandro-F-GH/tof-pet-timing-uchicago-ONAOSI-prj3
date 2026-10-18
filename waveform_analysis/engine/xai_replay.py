"""Recompute XAI from saved batch models and verified prepared caches only."""
from __future__ import annotations

import csv
import json
from pathlib import Path, PureWindowsPath
import re
from typing import Any

from waveform_analysis.core.io import atomic_json
from waveform_analysis.core.dependencies import stage_fingerprint
from waveform_analysis.data.dataset import PreparedDataset, load_prepared_dataset
from waveform_analysis.data.splits import semantic_seed
from waveform_analysis.data.storage import RunStore
from waveform_analysis.data.results import discover_runs, read_run_manifest, resolve_run_config
from waveform_analysis.engine.diagnostics import diagnostic_random_state
from waveform_analysis.engine.train import load_fitted_model, release_training_memory, saved_model_complete
from waveform_analysis.engine.xai import model_importance, model_xai_method, model_xai_version
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
            self.datasets[path] = load_prepared_dataset(path)
        return self.datasets[path]


def _runs(root: Path) -> list[RunStore]:
    runs = discover_runs(root)
    if not runs:
        raise ValueError(f"No complete canonical saved-model runs found in {root}")
    return [RunStore(r["directory"]) for r in runs]


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
    available = {read_run_manifest(store.root)["model"] for store in stores}
    if models is not None and set(models) - available:
        raise ValueError(f"Requested models not found: {sorted(set(models) - available)}")
    lookup = PreparedCacheLookup()
    outputs, failures = [], []
    for store in stores:
        manifest = read_run_manifest(store.root)
        if models is not None and manifest["model"] not in models:
            continue
        try:
            spec = get_model(manifest["model"])
            if model_xai_method(spec) is None:
                if logger:
                    logger.info("XAI | skipped for %s", spec.name)
                continue
            config = resolve_run_config(store.root)
            settings = {**config["xai"], "enabled": True}
            if group_size_samples is not None:
                settings["group_size_samples"] = int(group_size_samples)
            if max_events is not None:
                settings["max_events"] = int(max_events)
            best = _read(store.best_path)
            if not saved_model_complete(spec, store.model_dir):
                raise FileNotFoundError("Saved final model or frozen transform is missing/incomplete")
            root = _cache_root(config, store.root, cache_root)
            coefficient_xai = model_xai_method(spec) == "shapelet_coefficients"
            control = None if coefficient_xai else lookup.find(root, "control", config, manifest)
            blind = None if coefficient_xai else lookup.find(root, "blind", config, manifest)
            control_identity = (manifest["dataset_populations"]["control"]["analysis_protocol_identity"]
                                if coefficient_xai else control.manifest["analysis_protocol_identity"])
            blind_identity = (manifest["dataset_populations"]["blind"]["analysis_protocol_identity"]
                              if coefficient_xai else blind.manifest["analysis_protocol_identity"])
            final_fit = manifest["stage_fingerprints"]["final_fit"]
            payload = {
                "final_fit": final_fit, "blind": blind_identity,
                "xai": settings, "seed": config["seed"], "method_version": model_xai_version(spec),
                "control": control_identity,
                "formulation": spec.estimator_formulation,
            }
            fingerprint = stage_fingerprint("xai", payload)
            if logger:
                logger.info("XAI rerun | saved model | %s | %s | %s", spec.name, config["mode"], config.get("window_name"))
            with diagnostic_random_state():
                fitted = load_fitted_model(spec, store.model_dir, best["parameters"], config)
                try:
                    scores = model_importance(
                        spec, fitted, blind, config["mode"], control_dataset=control,
                        cache_root=root, group_size_samples=settings["group_size_samples"],
                        max_events=settings["max_events"],
                        seed=semantic_seed(config["seed"], spec.name, "xai"),
                    )
                finally:
                    del fitted
                    release_training_memory()
            store.save_xai(**scores)
            metadata = {"method": model_xai_method(spec), "replacement_method": "none" if coefficient_xai else "control_sample_mean",
                        "method_version": model_xai_version(spec), "formulation": spec.estimator_formulation,
                        "control_analysis_protocol_identity": control_identity,
                        "control_n_events": manifest["dataset_populations"]["control"]["n_selected"] if coefficient_xai else control.n_events, "settings": settings}
            # Preserve other stages, including older saved-model input policies.
            store.mark_stage("xai", fingerprint, metadata=metadata)
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
