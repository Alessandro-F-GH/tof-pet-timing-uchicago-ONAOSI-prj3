from __future__ import annotations

import csv
import os
import shutil
import tempfile
from pathlib import Path

import numpy as np

from .common import atomic_json


RESULT_FIELDS = (
    "phase",
    "replica_index",
    "seed",
    "model",
    "estimator_formulation",
    "mode",
    "window_start_ns",
    "window_end_ns",
    "population_identity",
    "event_population_identity",
    "analysis_protocol_identity",
    "sampling_identity",
    "candidate_id",
    "selected",
    "ctr_ps",
    "uncorrected_ctr_ps",
    "improvement_ps",
    "improvement_percent",
    "rmse_ps",
    "uncorrected_rmse_ps",
    "rmse_improvement_ps",
    "rmse_improvement_percent",
    "n",
    "train_n",
    "swap_rmse_ps",
)


class RunStore:
    def __init__(self, root, *, overwrite=False, resume=False):
        self.root = Path(root).resolve()
        if overwrite and resume:
            raise ValueError("overwrite and resume are mutually exclusive")
        if overwrite and self.root.exists():
            shutil.rmtree(self.root)
        if self.root.exists() and any(self.root.iterdir()) and not resume:
            raise FileExistsError(f"Run directory is not empty: {self.root}")
        self.root.mkdir(parents=True, exist_ok=True)
        self.resume = bool(resume)

    @property
    def results_path(self):
        return self.root / "results.csv"

    def write_manifest(self, value):
        atomic_json(self.root / "manifest.json", value)

    def write_resolved_config(self, value):
        atomic_json(self.root / "resolved_config.json", value)

    def write_candidates(self, value):
        atomic_json(self.root / "candidates.json", value)

    def write_selected_hyperparameters(self, value):
        atomic_json(self.root / "selected_hyperparameters.json", value)

    def read_results(self):
        if not self.results_path.is_file():
            return []
        with self.results_path.open("r", encoding="utf-8", newline="") as stream:
            return list(csv.DictReader(stream))

    def _atomic_rows(self, rows):
        fd, tmp = tempfile.mkstemp(prefix=".results.", suffix=".csv", dir=self.root)
        try:
            with os.fdopen(fd, "w", encoding="utf-8", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=RESULT_FIELDS)
                writer.writeheader()
                for row in rows:
                    writer.writerow({key: row.get(key, "") for key in RESULT_FIELDS})
            os.replace(tmp, self.results_path)
        finally:
            if os.path.exists(tmp):
                os.unlink(tmp)

    def upsert_result(self, row):
        rows = self.read_results()
        key = (
            str(row["phase"]),
            str(row.get("replica_index", "")),
            str(row["candidate_id"]),
        )
        rows = [
            existing
            for existing in rows
            if (
                str(existing.get("phase", "")),
                str(existing.get("replica_index", "")),
                str(existing.get("candidate_id", "")),
            )
            != key
        ]
        rows.append(dict(row))

        def sort_key(value):
            phase_order = 0 if value.get("phase") == "hyperparameter_validation" else 1
            replica = int(value["replica_index"]) if str(value.get("replica_index", "")).strip() else 0
            return phase_order, replica, str(value.get("candidate_id", ""))

        rows.sort(key=sort_key)
        self._atomic_rows(rows)

    def has_result(self, phase, candidate_id, replica_index=None):
        wanted_replica = "" if replica_index is None else str(int(replica_index))
        return any(
            row.get("phase") == phase
            and row.get("candidate_id") == candidate_id
            and str(row.get("replica_index", "")) == wanted_replica
            for row in self.read_results()
        )

    def blind_residuals_path(self, seed, candidate_id):
        return self.root / "blind_residuals" / f"seed_{int(seed)}_{candidate_id}.npz"

    def save_blind_residuals(self, seed, candidate_id, corrected_ps):
        path = self.blind_residuals_path(seed, candidate_id)
        path.parent.mkdir(exist_ok=True)
        np.savez_compressed(path, corrected_ps=np.asarray(corrected_ps, np.float64))
        return path

    def model_dir(self, replica_index, seed, candidate_id):
        path = (
            self.root
            / "models"
            / f"replica_{int(replica_index):03d}_seed_{int(seed)}"
            / candidate_id
        )
        path.mkdir(parents=True, exist_ok=True)
        return path
