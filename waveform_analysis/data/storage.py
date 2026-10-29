from __future__ import annotations
import csv, json, os, shutil, tempfile
from pathlib import Path
import numpy as np
from waveform_analysis.core.io import atomic_json, write_csv
from waveform_analysis.core.artifact_identity import ProductRecipe
from waveform_analysis.data.artifact_store import ArtifactStore



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
        self.artifact_root = (
            Path(artifact_root).expanduser().resolve()
            if artifact_root is not None else None
        )
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
        return self.artifacts_dir / "pred.npz"

    @property
    def development_predictions_path(self):
        return self.artifacts_dir / "development_pred.npz"

    @property
    def xai_path(self):
        return self.artifacts_dir / "xai.npz"

    @property
    def model_reference_path(self):
        return self.metadata_dir / "model_reference.json"

    @property
    def model_dir(self):
        """Resolve the immutable model artifact, or return the absent local staging path."""
        if self.model_reference_path.is_file() and self.artifact_root is not None:
            reference = self.read_json(self.model_reference_path, {})
            try:
                return ArtifactStore(self.artifact_root).resolve(
                    reference["product_type"], reference["product_id"]
                ).path
            except (KeyError, ValueError, RuntimeError):
                return self.root / "model"
        return self.root / "model"

    def save_model_artifact(self, spec, fitted, parameters, *, final_fit_fingerprint):
        """Persist fitted weights once outside result views and record a stable reference."""
        if self.artifact_root is None:
            raise RuntimeError("An artifact_root is required to persist model artifacts")
        from waveform_analysis.engine.train import save_model
        compatibility = {
            "python_major_minor": f"{__import__('sys').version_info.major}.{__import__('sys').version_info.minor}",
            "numpy_version": np.__version__,
            "artifact_format": spec.artifact_format,
        }
        transform_identity = (
            getattr(fitted.feature_transform, "identity", None)
            if fitted.feature_transform is not None else None
        )
        recipe = ProductRecipe(
            "final-models",
            {
                "model": spec.name,
                "parameters": dict(parameters or {}),
                "final_fit_fingerprint": str(final_fit_fingerprint),
                "sample_mask": np.asarray(fitted.sample_mask, dtype=bool).tolist()
                    if fitted.sample_mask is not None else None,
                "feature_transform_identity": transform_identity,
                "output_max_abs_ps": fitted.output_max_abs_ps,
                "training_events": fitted.metadata.get("training_events"),
            },
            parents=(str(final_fit_fingerprint),) + (
                (str(transform_identity),) if transform_identity else ()
            ),
            implementation=f"{spec.name}:{spec.artifact_format}:model-v1",
            compatibility=compatibility,
        )
        store = ArtifactStore(self.artifact_root)
        reference = store.get_or_build(
            recipe,
            lambda directory: save_model(
                spec, fitted, directory, parameters, include_feature_transform=False
            ),
            metadata={
                "model": spec.name,
                "final_fit_fingerprint": str(final_fit_fingerprint),
                "pinned": True,
            },
        )
        atomic_json(self.model_reference_path, {
            "product_type": reference.product_type,
            "product_id": reference.product_id,
        })
        return reference

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

    def invalidate_from(self, stage):
        from waveform_analysis.core.dependencies import INVALIDATES
        files = {
            "cv": [
                self.folds_path,
                self.cv_path,
                self.candidates_path,
                self.root / "optuna.db",
                self.root / "optuna_sampler.pkl",
            ],
            "selection": [self.best_path],
            "final_fit": [self.final_fit_path, self.model_reference_path, self.development_predictions_path],
            "blind": [self.blind_path, self.predictions_path],
            "bootstrap": [self.bootstrap_path, self.bootstrap_draws_path],
            "xai": [self.xai_path],
            "plots": [self.plots_dir],
        }
        for current in INVALIDATES[stage]:
            for path in files[current]:
                if path.is_dir():
                    shutil.rmtree(path, ignore_errors=True)
                elif path.exists():
                    path.unlink()
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
        path = (self.predictions_path if dataset_role == "blind"
                else self.development_predictions_path)
        _atomic_npz(
            path,
            event_id=np.asarray(event_id, dtype=np.int64),
            prediction_ps=np.asarray(prediction_ps, dtype=np.float64),
            corrected_ps=np.asarray(corrected_ps, dtype=np.float64),
            led_residual_ps=np.asarray(led_residual_ps, dtype=np.float64),
            **(metadata or {}),
        )
        return path

    def load_predictions(self):
        with np.load(self.predictions_path) as data:
            return {k: np.asarray(data[k]) for k in data.files}

    def save_bootstrap_draws(self, draws):
        _atomic_npz(
            self.bootstrap_draws_path,
            **{k: np.asarray(v, dtype=np.float64) for k, v in draws.items()},
        )

    def save_xai(self, **arrays):
        _atomic_npz(self.xai_path, **arrays)
        return self.xai_path
