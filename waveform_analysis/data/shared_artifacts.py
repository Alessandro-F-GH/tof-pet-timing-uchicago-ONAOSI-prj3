"""Shared CV folds and LED reference metrics saved under readable run paths."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from waveform_analysis.data.splits import (
    cv_split_from_assignment,
    fold_assignment,
    make_cv_split,
)
from waveform_analysis.data.cache_store import get_or_build
from waveform_analysis.data.named_cache import dataset_name, safe_component
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


class SharedCVCache:
    """Persist and reload common CV folds and LED reference metrics."""

    def __init__(self, root):
        self.root = Path(root).resolve()

    def prepare_cv(self, dataset, config):
        cv = config["cross_validation"]
        population_id = str(dataset.manifest["event_population_identity"])
        window = str(config.get("window_name") or (
            f"{config['window_ns']['start']}_to_{config['window_ns']['end']}ns"
        ))
        cv_label = f"folds_{int(cv['folds'])}_shuffle_{int(bool(cv['shuffle']))}"
        directory = (
            self.root / "cv"
            / dataset_name(config["control"])
            / dataset_name(config["development"])
            / safe_component(config["mode"])
            / safe_component(window)
            / cv_label
            / f"seed_{int(config['seed'])}"
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
                    seed=split.seed,
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
            (directory / "population.json").write_text(
                json.dumps({
                    "event_population_identity": population_id,
                    "n_events": int(dataset.n_events),
                    "n_folds": int(cv["folds"]),
                    "shuffle": bool(cv["shuffle"]),
                    "seed": int(config["seed"]),
                }, indent=2) + "\n", encoding="utf-8"
            )
            (directory / "led_folds.json").write_text(
                json.dumps({str(key): value for key, value in metrics.items()},
                           sort_keys=True, indent=2) + "\n",
                encoding="utf-8",
            )

        get_or_build(self.root, directory, build)
        folds_file = directory / "folds.npz"
        metrics_file = directory / "led_folds.json"
        provenance_file = directory / "population.json"
        provenance = json.loads(provenance_file.read_text(encoding="utf-8"))
        # The directory already identifies the named datasets, mode, window
        # and CV definition. Validate the operational CV settings explicitly.
        # A provenance hash may change after moving the ROOT data between
        # operating systems; it must not prevent reusing the stored folds.
        expected = {
            "n_events": int(dataset.n_events),
            "n_folds": int(cv["folds"]),
            "shuffle": bool(cv["shuffle"]),
            "seed": int(config["seed"]),
        }
        differences = {
            key: {"cached": provenance.get(key), "expected": value}
            for key, value in expected.items()
            if provenance.get(key) != value
        }
        if differences:
            details = "; ".join(
                f"{key}: cached={values['cached']!r}, expected={values['expected']!r}"
                for key, values in differences.items()
            )
            raise ValueError(
                f"Shared CV cache incompatible | path={directory} | {details}. "
                "Check the dataset selection and CV settings before reusing this cache."
            )

        with np.load(folds_file, allow_pickle=False) as data:
            split = cv_split_from_assignment(
                np.asarray(data["assignment"], dtype=np.int64),
                seed=int(data["seed"]),
                shuffle=bool(int(data["shuffle"])),
            )
        if len(fold_assignment(split)) != int(dataset.n_events) or len(split.folds) != int(cv["folds"]):
            raise ValueError("Shared CV cache differs from the Development population or fold configuration")
        raw_metrics = json.loads(
            metrics_file.read_text(encoding="utf-8")
        )
        metrics = {int(key): value for key, value in raw_metrics.items()}
        return SharedCVArtifacts(directory, split, metrics, directory.relative_to(self.root).as_posix())
