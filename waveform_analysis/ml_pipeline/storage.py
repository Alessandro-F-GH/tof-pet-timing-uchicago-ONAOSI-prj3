from __future__ import annotations

import csv
import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

from .common import atomic_json, canonical_hash, canonical_json


def fingerprint(value: Any) -> str:
    return canonical_hash(value)


class RunStore:
    def __init__(self, root: str | Path, *, overwrite: bool = False) -> None:
        self.root = Path(root).resolve()
        if overwrite and self.root.exists():
            shutil.rmtree(self.root)
        if self.root.exists() and any(self.root.iterdir()):
            raise FileExistsError(
                f"Run directory is not empty: {self.root}. Use --overwrite for a new run."
            )
        self.root.mkdir(parents=True, exist_ok=True)
        for name in ("models", "artifacts", "search", "splits"):
            (self.root / name).mkdir(exist_ok=True)

    def write_manifest(self, value: dict[str, Any]) -> None:
        atomic_json(self.root / "manifest.json", value)

    def write_results(self, rows: list[dict[str, Any]]) -> None:
        fields = list(dict.fromkeys(key for row in rows for key in row))
        target = self.root / "results.csv"
        fd, temporary = tempfile.mkstemp(prefix=".results.", suffix=".csv", dir=self.root)
        try:
            with os.fdopen(fd, "w", encoding="utf-8", newline="") as stream:
                if fields:
                    writer = csv.DictWriter(stream, fieldnames=fields)
                    writer.writeheader()
                    writer.writerows(rows)
            os.replace(temporary, target)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)

    def save_split(self, dataset: str, split: Any) -> Path:
        target = self.root / "splits" / f"{dataset}.npz"
        np.savez_compressed(
            target,
            development=np.asarray(split.development, dtype=np.int64),
            blind=np.asarray(split.blind, dtype=np.int64),
            training=np.asarray(split.training, dtype=np.int64),
            validation=np.asarray(split.validation, dtype=np.int64),
        )
        return target

    def save_search(self, dataset: str, mode: str, name: str, value: dict[str, Any]) -> Path:
        target = self.root / "search" / dataset / mode / f"{name}.json"
        atomic_json(target, value)
        return target

    def model_dir(self, dataset: str, mode: str, model: str) -> Path:
        target = self.root / "models" / dataset / mode / model
        target.mkdir(parents=True, exist_ok=True)
        return target

    def save_residuals(self, dataset: str, mode: str, method: str, values: np.ndarray) -> Path:
        target = self.root / "artifacts" / dataset / mode / f"{method}_blind_residuals_ps.npy"
        target.parent.mkdir(parents=True, exist_ok=True)
        np.save(target, np.asarray(values, dtype=np.float64))
        return target

    def save_xai(
        self,
        dataset: str,
        mode: str,
        model: str,
        *,
        time_ps: np.ndarray,
        importance: np.ndarray,
        example_pair_mV: np.ndarray,
    ) -> Path:
        target = self.root / "artifacts" / dataset / mode / f"{model}_xai.npz"
        target.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            target,
            time_ps=np.asarray(time_ps, dtype=np.float64),
            importance=np.asarray(importance, dtype=np.float64),
            example_pair_mV=np.asarray(example_pair_mV, dtype=np.float32),
        )
        return target
