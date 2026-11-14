from __future__ import annotations
import csv, json, os, shutil, tempfile
from pathlib import Path
import numpy as np
from waveform_analysis.core.io import atomic_json, write_csv



def _read_csv(path):
    if not Path(path).is_file():
        return []
    with Path(path).open("r", encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def _atomic_npz(path, **arrays):
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


class RunStore:
    def __init__(self, root, *, artifact_root=None):
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.artifact_root = None

    @property
    def metadata_dir(self):
        return self.root / "metadata"

    @property
    def manifest_path(self):
        return self.metadata_dir / "manifest.json"

    @property
    def state_path(self):
        return self.metadata_dir / "state.json"

    @property
    def tables_dir(self):
        return self.root / "tables"

    @property
    def artifacts_dir(self):
        return self.root / "artifacts"

    @property
    def folds_path(self):
        return self.tables_dir / "folds.csv"

    @property
    def cv_path(self):
        return self.tables_dir / "cv.csv"

    @property
    def candidates_path(self):
        return self.metadata_dir / "candidates.json"

    @property
    def best_path(self):
        return self.metadata_dir / "best.json"

    @property
    def final_fit_path(self):
        return self.metadata_dir / "final_fit.json"

    @property
    def blind_path(self):
        return self.metadata_dir / "blind.json"

    @property
    def bootstrap_path(self):
        return self.metadata_dir / "bootstrap.json"

    @property
    def bootstrap_draws_path(self):
        return self.artifacts_dir / "bootstrap.npz"

    @property
    def predictions_path(self):
        return self.artifacts_dir / "predictions.npz"

    @property
    def development_predictions_path(self):
        return self.artifacts_dir / "development_predictions.npz"

    @property
    def xai_path(self):
        return self.artifacts_dir / "xai.npz"

    @property
    def model_dir(self):
        return self.root / "model"

    def save_model_artifact(self, spec, fitted, parameters, *, final_fit_fingerprint):
        from waveform_analysis.engine.train import save_model
        save_model(spec, fitted, self.model_dir, parameters,
                   include_feature_transform=True)
        return self.model_dir

    @property
    def plots_dir(self):
        return self.root / "plots"

    def write_manifest(self, v):
        from waveform_analysis.data.results import write_run_manifest
        write_run_manifest(self.root, v)

    def write_resolved_config(self, v):
        from waveform_analysis.data.results import write_run_config
        write_run_config(v, managed_runs=v.get("_managed_runs"))

    def write_candidates(self, v):
        atomic_json(self.candidates_path, v)

    def write_best(self, v):
        atomic_json(self.best_path, v)

    def write_final_fit(self, v):
        atomic_json(self.final_fit_path, v)

    def write_blind(self, v):
        atomic_json(self.blind_path, v)

    def write_bootstrap(self, v):
        atomic_json(self.bootstrap_path, v)

    def read_json(self, path, default=None):
        try:
            return json.loads(Path(path).read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return default

    def read_state(self):
        v = self.read_json(self.state_path, {})
        if not isinstance(v, dict):
            return {'stages': {}}
        v.setdefault("stages", {})
        return v

    def stage_status(self, stage, fingerprint):
        item = self.read_state()["stages"].get(str(stage))
        if not isinstance(item, dict):
            return "missing"
        if item.get("fingerprint") != str(fingerprint):
            return "stale"
        return "complete" if item.get("status") == "complete" else "incomplete"

    def mark_stage(self, stage, fingerprint, *, status="complete", metadata=None):
        state = self.read_state()
        state["stages"][stage] = {
            "status": str(status),
            "fingerprint": str(fingerprint),
            "metadata": dict(metadata or {}),
        }
        atomic_json(self.state_path, state)

    def _stage_output_paths(self):
        return {
            "cv": [self.folds_path, self.cv_path, self.candidates_path,
                   self.root / "optuna.db", self.root / "optuna_sampler.pkl"],
            "selection": [self.best_path],
            "final_fit": [self.final_fit_path, self.model_dir,
                          self.development_predictions_path],
            "blind": [self.blind_path, self.predictions_path],
            "bootstrap": [self.bootstrap_path, self.bootstrap_draws_path],
            "xai": [self.xai_path],
            "plots": [self.plots_dir],
        }

    def _remove_stage_outputs(self, stage):
        for path in self._stage_output_paths()[stage]:
            if path.is_dir():
                shutil.rmtree(path, ignore_errors=True)
            elif path.exists():
                path.unlink()

    def invalidate_stage(self, stage):
        """Invalidate one stage while retaining downstream records until parent checks."""
        from waveform_analysis.core.dependencies import INVALIDATES

        if stage not in INVALIDATES:
            raise ValueError(f"Unknown pipeline stage: {stage}")
        self._remove_stage_outputs(stage)
        state = self.read_state()
        state["stages"].pop(stage, None)
        atomic_json(self.state_path, state)

    def invalidate_from(self, stage):
        """Explicitly invalidate a stage and all descendants in the dependency graph."""
        from waveform_analysis.core.dependencies import INVALIDATES

        if stage not in INVALIDATES:
            raise ValueError(f"Unknown pipeline stage: {stage}")
        for current in INVALIDATES[stage]:
            self._remove_stage_outputs(current)
        state = self.read_state()
        for current in INVALIDATES[stage]:
            state["stages"].pop(current, None)
        atomic_json(self.state_path, state)

    def read_fold_rows(self, candidate_id=None):
        rows = _read_csv(self.folds_path)
        return (
            rows
            if candidate_id is None
            else [r for r in rows if r.get("candidate_id") == str(candidate_id)]
        )

    def upsert_fold(self, row):
        key = (str(row["candidate_id"]), int(row["fold_id"]))
        rows = [
            r
            for r in _read_csv(self.folds_path)
            if (str(r.get("candidate_id")), int(r.get("fold_id", -1))) != key
        ]
        rows.append(dict(row))
        rows.sort(
            key=lambda r: (str(r.get("candidate_id", "")), int(r.get("fold_id", 0)))
        )
        write_csv(self.folds_path, rows)

    def read_candidate_rows(self):
        return _read_csv(self.cv_path)

    def candidate_row(self, candidate_id):
        return next(
            (
                r
                for r in self.read_candidate_rows()
                if r.get("candidate_id") == str(candidate_id)
            ),
            None,
        )

    def upsert_candidate(self, row):
        rows = [
            r
            for r in self.read_candidate_rows()
            if r.get("candidate_id") != str(row["candidate_id"])
        ]
        rows.append(dict(row))
        rows.sort(key=lambda r: str(r.get("candidate_id", "")))
        write_csv(self.cv_path, rows)

    def save_predictions(
        self, *, event_id, prediction_ps, corrected_ps, led_residual_ps,
        dataset_role="blind", metadata=None,
    ):
        if dataset_role not in {"blind", "development"}:
            raise ValueError(f"Unsupported prediction dataset role: {dataset_role}")
        payload = {
            "event_id": np.asarray(event_id, dtype=np.int64),
            "prediction_ps": np.asarray(prediction_ps, dtype=np.float64),
            "corrected_ps": np.asarray(corrected_ps, dtype=np.float64),
            "led_residual_ps": np.asarray(led_residual_ps, dtype=np.float64),
            **(metadata or {}),
        }
        path = (self.predictions_path if dataset_role == "blind"
                else self.development_predictions_path)
        _atomic_npz(path, **payload)
        return path

    def load_predictions(self):
        with np.load(self.predictions_path) as data:
            return {k: np.asarray(data[k]) for k in data.files}

    def save_bootstrap_draws(self, draws, *, fingerprint=None, blind_fingerprint=None):
        arrays = {k: np.asarray(v, dtype=np.float64) for k, v in draws.items()}
        _atomic_npz(self.bootstrap_draws_path, **arrays)
        return self.bootstrap_draws_path

    def save_xai(self, **arrays):
        _atomic_npz(self.xai_path, **arrays)
        return self.xai_path
