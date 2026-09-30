from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .common import atomic_json, canonical_hash
from .splits import ResamplingSplit, make_resampling_split, semantic_seed
from .stats import ctr_estimate, rmse_ps


@dataclass(frozen=True)
class SharedReplicaArtifacts:
    directory: Path
    split: ResamplingSplit
    led_ps: np.ndarray
    led_ctr_ps: float
    led_rmse_ps: float


def _atomic_npz(path: Path, **arrays) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            np.savez_compressed(stream, **arrays)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def _safe(text: object) -> str:
    value = "".join(ch if ch.isalnum() or ch in "-_." else "-" for ch in str(text)).strip("-")
    return value or "value"


class ExperimentArtifactStore:
    """Batch-level storage for artifacts that do not depend on the ML model."""

    def __init__(self, root: str | Path):
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)

    def _population_dir(self, dataset, config) -> Path:
        protocol_identity = str(dataset.manifest.get("analysis_protocol_identity") or dataset.manifest["analysis_population_identity"])
        mode = "energy" if config["mode"] == "energy_to_energy" else "timing"
        window = config.get("window_name") or f"{float(config['window_ns']['start']):g}_{float(config['window_ns']['end']):g}ns"
        return self.root / "populations" / f"{mode}__{_safe(window)}__{protocol_identity[:12]}"

    def prepare_replica(self, dataset, config, seed: int, target: np.ndarray) -> SharedReplicaArtifacts:
        protocol_identity = str(dataset.manifest.get("analysis_protocol_identity") or dataset.manifest["analysis_population_identity"])
        event_population_identity = str(dataset.manifest.get("event_population_identity") or dataset.manifest.get("analysis_population_identity"))
        population_dir = self._population_dir(dataset, config)
        population_dir.mkdir(parents=True, exist_ok=True)

        population_manifest = {
            "schema_version": 1,
            "analysis_protocol_identity": protocol_identity,
            "event_population_identity": event_population_identity,
            "prepared_dataset": str(Path(dataset.directory).resolve()),
            "analysis_source": dataset.manifest.get("analysis_source"),
            "mode": config["mode"],
            "window_name": config.get("window_name"),
            "window_ns": config["window_ns"],
            "n_events": int(dataset.n_events),
        }
        manifest_path = population_dir / "manifest.json"
        if manifest_path.is_file():
            existing = json.loads(manifest_path.read_text(encoding="utf-8"))
            for key in ("analysis_protocol_identity", "event_population_identity", "n_events"):
                if existing.get(key) != population_manifest.get(key):
                    raise RuntimeError(f"Shared population artifact mismatch for {key}: {population_dir}")
        else:
            atomic_json(manifest_path, population_manifest)

        resampling = config["resampling"]
        resampling_identity = canonical_hash({
            "policy": resampling.get("policy", "repeated_holdout"),
            "event_population_identity": event_population_identity,
            "validation_fraction": float(resampling["validation_fraction"]),
            "test_fraction": float(resampling["test_fraction"]),
            "minimum_events_per_split": int(resampling["minimum_events_per_split"]),
        })
        replica_dir = population_dir / "resampling" / resampling_identity[:16] / f"seed_{int(seed)}"
        replica_dir.mkdir(parents=True, exist_ok=True)

        split_path = replica_dir / "split.npz"
        if split_path.is_file():
            with np.load(split_path) as data:
                split = ResamplingSplit(np.asarray(data["train"], dtype=np.int64), np.asarray(data["validation"], dtype=np.int64), np.asarray(data["test"], dtype=np.int64))
            split.validate(dataset.n_events)
        else:
            split = make_resampling_split(
                dataset.n_events,
                analysis_identity=event_population_identity,
                resampling_seed=int(seed),
                validation_fraction=float(resampling["validation_fraction"]),
                test_fraction=float(resampling["test_fraction"]),
            )
            minimum = int(resampling["minimum_events_per_split"])
            if min(len(split.train), len(split.validation), len(split.test)) < minimum:
                raise RuntimeError(f"Resampling seed {seed} violates minimum_events_per_split={minimum}")
            _atomic_npz(split_path, train=split.train, validation=split.validation, test=split.test)

        replica_manifest = {
            "schema_version": 1,
            "seed": int(seed),
            "analysis_protocol_identity": protocol_identity,
            "event_population_identity": event_population_identity,
            "resampling_identity": resampling_identity,
            "n_train": int(len(split.train)),
            "n_validation": int(len(split.validation)),
            "n_test": int(len(split.test)),
            "split_path": str(split_path.resolve()),
        }
        atomic_json(replica_dir / "manifest.json", replica_manifest)

        target = np.asarray(target, dtype=np.float64)
        led_ps = np.asarray(target[split.test], dtype=np.float64)
        event_index = np.asarray(dataset.event_index[split.test], dtype=np.int64)
        reference_path = replica_dir / "blind_reference.npz"
        if reference_path.is_file():
            with np.load(reference_path) as data:
                saved_event_index = np.asarray(data["event_index"], dtype=np.int64)
                saved_led = np.asarray(data["led_ps"], dtype=np.float64)
            if not np.array_equal(saved_event_index, event_index) or not np.array_equal(saved_led, led_ps):
                raise RuntimeError(f"Shared blind reference does not match resolved population: {reference_path}")
        else:
            _atomic_npz(reference_path, event_index=event_index, led_ps=led_ps)

        fit_identity = canonical_hash(config["fit"])
        baseline_path = replica_dir / f"baseline_{fit_identity[:16]}.json"
        if baseline_path.is_file():
            baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
        else:
            point = ctr_estimate(led_ps, config["fit"], seed=semantic_seed(int(seed), "shared_led_baseline", fit_identity), bootstrap=False)
            baseline = {"fit_identity": fit_identity, "n": int(led_ps.size), "led_ctr_ps": float(point.ctr_ps), "led_rmse_ps": float(rmse_ps(led_ps))}
            atomic_json(baseline_path, baseline)

        return SharedReplicaArtifacts(directory=replica_dir.resolve(), split=split, led_ps=led_ps, led_ctr_ps=float(baseline["led_ctr_ps"]), led_rmse_ps=float(baseline["led_rmse_ps"]))
