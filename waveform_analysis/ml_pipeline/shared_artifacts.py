from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .artifact_naming import prefer_existing, replica_tag
from .common import atomic_json, canonical_hash
from .splits import FixedValidationSplit, ReplicaSplit, make_fixed_validation_split, make_replica_split
from .stats import ctr_estimate, rmse_ps


SAMPLING_PROTOCOL = "fixed_validation_excluded_from_replicas_v3"


@dataclass(frozen=True)
class SharedValidationArtifacts:
    directory: Path
    split: FixedValidationSplit
    sampling_identity: str


@dataclass(frozen=True)
class SharedReplicaArtifacts:
    directory: Path
    split: ReplicaSplit
    led_ps: np.ndarray
    led_ctr_ps: float
    led_rmse_ps: float
    sampling_identity: str


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
    """Batch-level storage for model-independent tuning and replica artifacts."""

    def __init__(self, root: str | Path):
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)

    def _population_dir(self, dataset, config) -> Path:
        protocol_identity = str(dataset.manifest["analysis_protocol_identity"])
        mode = "energy" if config["mode"] == "energy_to_energy" else "timing"
        window = config.get("window_name") or (
            f"{float(config['window_ns']['start']):g}_{float(config['window_ns']['end']):g}ns"
        )
        name = f"{mode}__{_safe(window)}__{protocol_identity[:12]}"
        return prefer_existing(
            self.root / "pop" / name,
            self.root / "populations" / name,
        )

    def _sampling_identity(self, dataset, config) -> str:
        return canonical_hash({
            "sampling_protocol": SAMPLING_PROTOCOL,
            "event_population_identity": str(dataset.manifest["event_population_identity"]),
            "batch_seed": int(config["seed"]),
            "validation_fraction": float(config["model_selection"]["validation_fraction"]),
            "blind_fraction": float(config["evaluation"]["blind_fraction"]),
            "minimum_events_per_split": int(config["evaluation"]["minimum_events_per_split"]),
        })

    def prepare_fixed_validation(self, dataset, config) -> SharedValidationArtifacts:
        protocol_identity = str(dataset.manifest["analysis_protocol_identity"])
        event_identity = str(dataset.manifest["event_population_identity"])
        sampling_identity = self._sampling_identity(dataset, config)
        population_dir = self._population_dir(dataset, config)
        sampling_dir = prefer_existing(
            population_dir / "s" / sampling_identity[:16],
            population_dir / "sampling" / sampling_identity[:16],
        )
        sampling_dir.mkdir(parents=True, exist_ok=True)

        population_manifest = {
            "schema_version": 3,
            "analysis_protocol_identity": protocol_identity,
            "event_population_identity": event_identity,
            "prepared_dataset": str(Path(dataset.directory).resolve()),
            "analysis_source": dataset.manifest.get("analysis_source"),
            "mode": config["mode"],
            "window_name": config.get("window_name"),
            "window_ns": config["window_ns"],
            "n_events": int(dataset.n_events),
        }
        population_manifest_path = population_dir / "manifest.json"
        if population_manifest_path.is_file():
            existing = json.loads(population_manifest_path.read_text(encoding="utf-8"))
            for key in ("analysis_protocol_identity", "event_population_identity", "n_events"):
                if existing.get(key) != population_manifest.get(key):
                    raise RuntimeError(f"Shared population artifact mismatch for {key}: {population_dir}")
        else:
            atomic_json(population_manifest_path, population_manifest)

        split = make_fixed_validation_split(
            dataset.n_events,
            analysis_identity=event_identity,
            batch_seed=int(config["seed"]),
            validation_fraction=float(config["model_selection"]["validation_fraction"]),
        )
        minimum = int(config["evaluation"]["minimum_events_per_split"])
        if min(len(split.tuning_train), len(split.validation)) < minimum:
            raise RuntimeError(
                f"Fixed model-selection split violates minimum_events_per_split={minimum}"
            )

        split_path = prefer_existing(sampling_dir / "validation.npz", sampling_dir / "fixed_validation.npz")
        if split_path.is_file():
            with np.load(split_path) as data:
                saved = FixedValidationSplit(
                    tuning_train=np.asarray(data["tuning_train"], dtype=np.int64),
                    validation=np.asarray(data["validation"], dtype=np.int64),
                    seed=int(data["seed"]),
                )
            saved.validate(dataset.n_events)
            if (
                saved.seed != split.seed
                or not np.array_equal(saved.tuning_train, split.tuning_train)
                or not np.array_equal(saved.validation, split.validation)
            ):
                raise RuntimeError(f"Shared fixed validation split mismatch: {split_path}")
            split = saved
        else:
            _atomic_npz(
                split_path,
                tuning_train=split.tuning_train,
                validation=split.validation,
                seed=np.asarray(split.seed, dtype=np.int64),
            )

        manifest = {
            "schema_version": 3,
            "sampling_protocol": SAMPLING_PROTOCOL,
            "sampling_identity": sampling_identity,
            "batch_seed": int(config["seed"]),
            "fixed_validation_seed": int(split.seed),
            "validation_fraction": float(config["model_selection"]["validation_fraction"]),
            "blind_fraction": float(config["evaluation"]["blind_fraction"]),
            "n_events": int(dataset.n_events),
            "n_tuning_train": int(len(split.tuning_train)),
            "n_validation": int(len(split.validation)),
            "replica_pool_excludes_validation": True,
            "split_path": str(split_path.resolve()),
        }
        atomic_json(sampling_dir / "manifest.json", manifest)
        return SharedValidationArtifacts(
            directory=sampling_dir.resolve(),
            split=split,
            sampling_identity=sampling_identity,
        )

    def prepare_replica(
        self,
        dataset,
        config,
        replica_index: int,
        target: np.ndarray,
        fixed: SharedValidationArtifacts | None = None,
    ) -> SharedReplicaArtifacts:
        fixed = fixed or self.prepare_fixed_validation(dataset, config)
        split = make_replica_split(
            dataset.n_events,
            fixed.split,
            analysis_identity=str(dataset.manifest["event_population_identity"]),
            batch_seed=int(config["seed"]),
            replica_index=int(replica_index),
            blind_fraction=float(config["evaluation"]["blind_fraction"]),
        )
        minimum = int(config["evaluation"]["minimum_events_per_split"])
        if min(len(split.train), len(split.test)) < minimum:
            raise RuntimeError(
                f"Replica {replica_index} violates minimum_events_per_split={minimum}"
            )
        if set(map(int, split.train)) & set(map(int, fixed.split.validation)):
            raise AssertionError("fixed validation must never enter replica training")
        if set(map(int, split.test)) & set(map(int, fixed.split.validation)):
            raise AssertionError("fixed validation must never enter replica blind evaluation")

        replica_dir = prefer_existing(
            fixed.directory / "r" / replica_tag(replica_index, split.seed),
            fixed.directory / "replicas" / f"replica_{int(replica_index):03d}_seed_{int(split.seed)}",
        )
        replica_dir.mkdir(parents=True, exist_ok=True)
        split_path = replica_dir / "split.npz"
        if split_path.is_file():
            with np.load(split_path) as data:
                saved_train = np.asarray(data["train"], dtype=np.int64)
                saved_test = np.asarray(data["test"], dtype=np.int64)
                saved_seed = int(data["seed"])
            if (
                saved_seed != split.seed
                or not np.array_equal(saved_train, split.train)
                or not np.array_equal(saved_test, split.test)
            ):
                raise RuntimeError(f"Shared replica split mismatch: {split_path}")
        else:
            _atomic_npz(
                split_path,
                train=split.train,
                test=split.test,
                seed=np.asarray(split.seed, dtype=np.int64),
            )

        target = np.asarray(target, dtype=np.float64)
        led_ps = np.asarray(target[split.test], dtype=np.float64)
        event_index = np.asarray(dataset.event_index[split.test], dtype=np.int64)
        reference_path = prefer_existing(replica_dir / "blind.npz", replica_dir / "blind_reference.npz")
        if reference_path.is_file():
            with np.load(reference_path) as data:
                saved_event_index = np.asarray(data["event_index"], dtype=np.int64)
                saved_led = np.asarray(data["led_ps"], dtype=np.float64)
            if not np.array_equal(saved_event_index, event_index) or not np.array_equal(saved_led, led_ps):
                raise RuntimeError(f"Shared blind reference mismatch: {reference_path}")
        else:
            _atomic_npz(reference_path, event_index=event_index, led_ps=led_ps)

        fit_identity = canonical_hash(config["fit"])
        baseline_path = prefer_existing(
            replica_dir / f"base_{fit_identity[:16]}.json",
            replica_dir / f"baseline_{fit_identity[:16]}.json",
        )
        if baseline_path.is_file():
            baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
        else:
            point = ctr_estimate(led_ps, config["fit"], seed=split.seed, bootstrap=False)
            baseline = {
                "fit_identity": fit_identity,
                "n": int(led_ps.size),
                "led_ctr_ps": float(point.ctr_ps),
                "led_rmse_ps": float(rmse_ps(led_ps)),
            }
            atomic_json(baseline_path, baseline)

        replica_manifest = {
            "schema_version": 3,
            "sampling_protocol": SAMPLING_PROTOCOL,
            "replica_index": int(replica_index),
            "seed": int(split.seed),
            "sampling_identity": fixed.sampling_identity,
            "n_train": int(len(split.train)),
            "n_test": int(len(split.test)),
            "n_excluded_validation": int(len(fixed.split.validation)),
            "fixed_validation_excluded": True,
            "split_path": str(split_path.resolve()),
        }
        atomic_json(replica_dir / "manifest.json", replica_manifest)

        return SharedReplicaArtifacts(
            directory=replica_dir.resolve(),
            split=split,
            led_ps=led_ps,
            led_ctr_ps=float(baseline["led_ctr_ps"]),
            led_rmse_ps=float(baseline["led_rmse_ps"]),
            sampling_identity=fixed.sampling_identity,
        )
