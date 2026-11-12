"""Shared cross-validation products backed by the canonical artifact store."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from waveform_analysis.core.artifact_identity import ProductRecipe
from waveform_analysis.data.artifact_store import ArtifactStore
from waveform_analysis.data.splits import (
    cv_split_from_assignment,
    fold_assignment,
    make_cv_split,
    semantic_seed,
)
from waveform_analysis.reporting.stats import metric_values
from waveform_analysis.data.view import model_target


CV_ARTIFACT_VERSION = 2


@dataclass(frozen=True)
class SharedCVArtifacts:
    directory: Path
    split: object
    fold_metrics: dict[int, dict]
    fingerprint: str


def _write_npz(path, **arrays):
    with Path(path).open("wb") as stream:
        np.savez_compressed(stream, **arrays)
        stream.flush()


class ExperimentArtifactStore:
    """Own CV split assignments and LED fold metrics as one immutable product."""

    def __init__(self, root):
        self.store = ArtifactStore(root)

    def prepare_cv(self, dataset, config):
        cv = config["cross_validation"]
        population_id = str(dataset.manifest["event_population_identity"])
        parameters = {
            "version": CV_ARTIFACT_VERSION,
            "event_population_identity": population_id,
            "mode": config["mode"],
            "window_ns": config["window_ns"],
            "seed": int(config["seed"]),
            "folds": int(cv["folds"]),
            "shuffle": bool(cv["shuffle"]),
            "fit": config["fit"],
            "minimum_events_per_fold": int(cv["minimum_events_per_fold"]),
        }
        recipe = ProductRecipe(
            "cross-validation",
            parameters,
            parents=(population_id,),
            implementation=f"cv-split-and-led-metrics-v{CV_ARTIFACT_VERSION}",
        )

        def build(directory):
            split = make_cv_split(
                dataset.n_events,
                population_identity=population_id,
                batch_seed=int(config["seed"]),
                n_folds=int(cv["folds"]),
                shuffle=bool(cv["shuffle"]),
            )
            minimum = int(cv["minimum_events_per_fold"])
            too_small = [
                fold.fold_id for fold in split.folds
                if len(fold.validation) < minimum
            ]
            if too_small:
                raise RuntimeError(
                    f"CV validation folds {too_small} violate minimum_events_per_fold={minimum}"
                )
            target = np.asarray(model_target(dataset, config["mode"]), dtype=np.float64)
            metrics = {}
            for fold in split.folds:
                validation = np.asarray(fold.validation, dtype=np.int64)
                score = metric_values(
                    target[validation],
                    config["fit"],
                    seed=semantic_seed(split.seed, "led", int(fold.fold_id)),
                )
                metrics[int(fold.fold_id)] = {
                    "fold_id": int(fold.fold_id),
                    "n_validation": int(validation.size),
                    "led_ctr_ps": float(score["ctr_ps"]),
                    "led_rmse_ps": float(score["rmse_ps"]),
                }
            _write_npz(
                directory / "folds.npz",
                assignment=fold_assignment(split),
                seed=np.asarray(split.seed, dtype=np.int64),
                shuffle=np.asarray(int(split.shuffle), dtype=np.int8),
            )
            (directory / "led_folds.json").write_text(
                json.dumps({str(key): value for key, value in metrics.items()},
                           sort_keys=True, indent=2) + "\n",
                encoding="utf-8",
            )

        reference = self.store.get_or_build(
            recipe,
            build,
            metadata={
                "event_population_identity": population_id,
                "folds": int(cv["folds"]),
                "seed": int(config["seed"]),
            },
        )
        with np.load(reference.payload_path("folds.npz"), allow_pickle=False) as data:
            split = cv_split_from_assignment(
                np.asarray(data["assignment"], dtype=np.int64),
                seed=int(data["seed"]),
                shuffle=bool(int(data["shuffle"])),
            )
        raw_metrics = json.loads(
            reference.payload_path("led_folds.json").read_text(encoding="utf-8")
        )
        metrics = {int(key): value for key, value in raw_metrics.items()}
        return SharedCVArtifacts(reference.path, split, metrics, reference.product_id)
