from __future__ import annotations

import json
from pathlib import Path

from . import report_engine as _report
from .common import canonical_hash
from .models import get_model, model_names
from .study import _SCHEMA_VERSION as CURRENT_RESULT_SCHEMA


_REQUIRED_MANIFEST_FIELDS = {
    "analysis",
    "model",
    "mode",
    "window_ns",
    "analysis_protocol_identity",
    "sampling_identity",
}


def _validated_manifest(run_dir: Path) -> dict:
    manifest_path = Path(run_dir) / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    schema = int(manifest.get("schema_version", 0))
    if schema != CURRENT_RESULT_SCHEMA:
        raise RuntimeError(
            f"Unsupported result schema {schema} in {run_dir}; "
            f"current code requires schema {CURRENT_RESULT_SCHEMA}"
        )
    missing = sorted(_REQUIRED_MANIFEST_FIELDS - set(manifest))
    if missing:
        raise RuntimeError(
            f"Incomplete result manifest in {run_dir}; missing fields: {missing}"
        )
    return manifest


def _model_identity(manifest: dict, run_dir: Path) -> tuple[str, str]:
    model = str(manifest["model"])
    if model not in model_names():
        raise RuntimeError(f"Unregistered model in {run_dir}: {model!r}")
    formulation = str(get_model(model).estimator_formulation).strip().lower()
    if formulation not in {"shared", "direct"}:
        raise RuntimeError(
            f"Invalid estimator formulation in {run_dir}: {formulation!r} "
            f"for model {model!r}"
        )
    return model, formulation


def collect_results(paths):
    run_dirs = []
    for path in paths:
        run_dirs.extend(_report._expand_result_path(path))

    unique = []
    seen = set()
    for path in run_dirs:
        resolved = Path(path).resolve()
        token = str(resolved)
        if token not in seen:
            seen.add(token)
            unique.append(resolved)

    records = []
    for run_dir in unique:
        manifest = _validated_manifest(run_dir)
        analysis = manifest["analysis"]
        dataset_key = canonical_hash(analysis)
        model, formulation = _model_identity(manifest, run_dir)
        mode = str(manifest["mode"])
        window = manifest["window_ns"]
        window_name = manifest.get("window_name")
        study_name = str(manifest.get("study_name") or manifest.get("name") or run_dir.name)
        protocol_identity = str(manifest["analysis_protocol_identity"])
        sampling_identity = str(manifest["sampling_identity"])
        shared = manifest.get("shared_replicas") or {}
        timings = _report.replica_wall_times(run_dir)

        for row in _report._read_rows(run_dir / "results.csv"):
            if row.get("phase") != "replica":
                continue
            replica_index = int(row["replica_index"])
            seed = int(row["seed"])
            records.append(
                {
                    "study": study_name,
                    "source_run": str(run_dir),
                    "dataset_key": dataset_key,
                    "dataset": str(analysis.get("root_file", "")),
                    "population_identity": protocol_identity,
                    "sampling_identity": sampling_identity,
                    "model": model,
                    "estimator_formulation": formulation,
                    "mode": mode,
                    "window": _report._window_label(
                        float(window["start"]),
                        float(window["end"]),
                        window_name,
                    ),
                    "window_start_ns": float(window["start"]),
                    "window_end_ns": float(window["end"]),
                    "replica_index": replica_index,
                    "seed": seed,
                    "candidate_id": str(row["candidate_id"]),
                    "shared_replica": str(shared.get(str(replica_index), "")),
                    "ctr_ps": _report._finite(row.get("ctr_ps")),
                    "led_ctr_ps": _report._finite(row.get("uncorrected_ctr_ps")),
                    "improvement_ps": _report._finite(row.get("improvement_ps")),
                    "improvement_percent": _report._finite(row.get("improvement_percent")),
                    "rmse_ps": _report._finite(row.get("rmse_ps")),
                    "led_rmse_ps": _report._finite(row.get("uncorrected_rmse_ps")),
                    "rmse_improvement_ps": _report._finite(row.get("rmse_improvement_ps")),
                    "rmse_improvement_percent": _report._finite(
                        row.get("rmse_improvement_percent")
                    ),
                    "replica_wall_time_s": _report._finite(timings.get(replica_index)),
                }
            )

    if not records:
        raise RuntimeError("No replica rows were found in the supplied studies")
    return records


def generate_report(paths, output_dir, *, logger=None, report_config=None):
    original = _report.collect_results
    _report.collect_results = collect_results
    try:
        return _report.generate_report(
            paths,
            output_dir,
            logger=logger,
            report_config=report_config,
        )
    finally:
        _report.collect_results = original


batch_result_dirs = _report.batch_result_dirs


def __getattr__(name):
    return getattr(_report, name)
