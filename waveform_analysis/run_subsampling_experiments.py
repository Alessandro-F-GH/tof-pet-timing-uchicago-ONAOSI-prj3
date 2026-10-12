"""Run existing ML-input subsampling sweeps without changing the scientific pipeline.

Examples (run from the repository root)::

    python -m waveform_analysis.run_subsampling_experiments --dry-run
    python -m waveform_analysis.run_subsampling_experiments --config \
        waveform_analysis/config/experiments/subsampling.json
    python -m waveform_analysis.run_subsampling_experiments --boards FBK \
        --window-ns -1.5 2.0 --subsampling 1 2 4 8 \
        --models direct_linear_ridge shared_linear_ridge

Preprocessing and LED calibration always use the original, full-resolution ROOT
acquisitions. Only cropped ML inputs use ``ml_input.subsampling``. Sample-based
kernels and XAI groups keep their configured sizes, so their physical spans grow
with the factor. Requested window boundaries stay fixed; the existing native-grid
crop/stride rules determine the actual retained samples and event population.

Each board/factor has a normal, independently resumable batch and report. Native
preprocessing caches remain shared. Small fingerprint-validated CV artifacts and
preprocessing diagnostics are copied between factor batches, avoiding repeated
reference fits/plots without symlinks, monkeypatching, or sharing trained models.
Model spaces, parameter ordering, seeds, preprocessing and evaluation settings are
inherited from the referenced batch. Linear Ridge keeps its existing RidgeCV path.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
from dataclasses import dataclass, replace
import json
import math
import os
from pathlib import Path
import shutil
import sys
import tempfile
from typing import Any

if __package__ in {None, ""}:
    script_directory = Path(__file__).resolve().parent
    # Direct execution must not let waveform_analysis/signal shadow stdlib signal.
    sys.path[:] = [entry for entry in sys.path
                   if Path(entry or os.getcwd()).resolve() != script_directory]
    sys.path.insert(0, str(script_directory.parent))

from waveform_analysis.core.config import BatchConfig, load_batch_config, public_batch_config
from waveform_analysis.core.io import atomic_json, canonical_hash, source_signature, write_csv
from waveform_analysis.core.logging import configure_logging
from waveform_analysis.data.preparation import _input_offsets
from waveform_analysis.data.root_io import energy_sampling_interval_s
from waveform_analysis.engine.batch import run_batch
from waveform_analysis.engine.control_preprocessing import CONTROL_ARTIFACT_VERSION

PROJECT = Path(__file__).resolve().parent
DEFAULT_CONFIG = PROJECT / "config/experiments/subsampling.json"
EXPERIMENT_VERSION = 1


@dataclass(frozen=True)
class Experiment:
    """Validated wrapper settings; paths are absolute and windows are in ns."""

    source: Path
    name: str
    window_name: str
    window_ns: dict[str, float]
    factors: tuple[int, ...]
    output: Path
    boards: dict[str, Path]


@dataclass(frozen=True)
class FactorBatch:
    board: str
    factor: int
    config_path: Path
    raw: dict[str, Any]
    batch: BatchConfig


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def _relative(owner: Path, value: str) -> Path:
    path = Path(value).expanduser()
    return (owner.parent / path).resolve() if not path.is_absolute() else path.resolve()


def load_experiment(args: argparse.Namespace) -> Experiment:
    """Read the dedicated config and apply explicit CLI overrides."""
    source = args.config.expanduser().resolve()
    raw = _read(source)
    required = {"name", "window_name", "window_ns", "subsampling", "output_dir", "boards"}
    if set(raw) != required:
        raise ValueError(f"Experiment config must contain exactly {sorted(required)}")
    window = raw["window_ns"] if args.window_ns is None else dict(zip(("start", "end"), args.window_ns))
    if not isinstance(window, dict) or set(window) != {"start", "end"}:
        raise ValueError("window_ns must contain start and end")
    window = {key: float(value) for key, value in window.items()}
    if not all(math.isfinite(v) for v in window.values()) or window["end"] <= window["start"]:
        raise ValueError("Physical window must have finite boundaries with end > start")
    factors = raw["subsampling"] if args.subsampling is None else args.subsampling
    if (not isinstance(factors, list) or not factors
            or any(type(k) is not int or k < 1 for k in factors)
            or len(set(factors)) != len(factors)):
        raise ValueError("subsampling must be a nonempty list of distinct positive integers")
    boards = raw["boards"]
    if not isinstance(boards, dict) or not boards or set(boards) - {"UC", "FBK"}:
        raise ValueError("boards must map UC and/or FBK to existing batch configs")
    if args.boards is not None:
        unknown = set(args.boards) - set(boards)
        if unknown:
            raise ValueError(f"Boards not configured: {sorted(unknown)}")
        boards = {board: path for board, path in boards.items() if board in args.boards}
    name, window_name = str(raw["name"]).strip(), str(raw["window_name"]).strip()
    if not name or not window_name or window_name in {".", ".."} or any(c in window_name for c in "/\\"):
        raise ValueError("name must be nonempty and window_name must be a single path component")
    output = Path(args.output_dir or raw["output_dir"]).expanduser()
    output = (PROJECT / output).resolve() if not output.is_absolute() else output.resolve()
    return Experiment(source, name, window_name, window, tuple(factors), output,
                      {board: _relative(source, path) for board, path in boards.items()})


def _template(source: Path, models: list[str] | None) -> dict[str, Any]:
    """Preserve batch settings and resolve includes before relocating its JSON."""
    raw = _read(source)
    for key in ("control_dataset", "development_dataset", "blind_dataset"):
        if isinstance(raw[key], str):
            raw[key] = str(_relative(source, raw[key]))
    preprocessing = raw["protocol"]["preprocessing_config"]
    if isinstance(preprocessing, str):
        raw["protocol"]["preprocessing_config"] = str(_relative(source, preprocessing))
    if "plot_config" in raw:
        raw["plot_config"] = str(_relative(source, raw["plot_config"]))
    available = raw["sweep"]["models"]
    for model in available:
        if isinstance(model, dict) and "space_config" in model:
            model["space_config"] = str(_relative(source, model["space_config"]))
    if models is not None:
        names = [model if isinstance(model, str) else model["name"] for model in available]
        unknown = set(models) - set(names)
        if unknown:
            raise ValueError(f"Models not present in {source}: {sorted(unknown)}")
        raw["sweep"]["models"] = [model for model, name in zip(available, names) if name in models]
    return raw


def _write_config(path: Path, raw: dict[str, Any]) -> None:
    """Atomically save JSON without sorting keys, preserving grid candidate order."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(raw, stream, indent=2, allow_nan=False)
            stream.write("\n")
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def build_batches(experiment: Experiment, models: list[str] | None = None) -> list[FactorBatch]:
    """Resolve all factor batches up front, with no persistent writes or data reads."""
    plans = []
    with tempfile.TemporaryDirectory(prefix="pet-subsampling-") as temporary:
        for board, source in experiment.boards.items():
            template = _template(source, models)
            for factor in experiment.factors:
                raw = deepcopy(template)
                raw["name"] = f"{experiment.name}__{board}__s{factor}"
                raw["sweep"]["windows"] = {experiment.window_name: experiment.window_ns}
                raw["protocol"]["ml_input"] = {"subsampling": factor}
                raw["results"] = {"root": str(experiment.output), "folder": f"{board}/subsampling_{factor}"}
                scratch = Path(temporary) / f"{board}_{factor}.json"
                _write_config(scratch, raw)
                resolved = load_batch_config(scratch, PROJECT)
                snapshot = experiment.output / "configs" / board / f"subsampling_{factor}.json"
                runs = tuple({**run, "_config_path": str(snapshot)} for run in resolved.runs)
                resolved = replace(resolved, source_path=str(snapshot), runs=runs)
                plans.append(FactorBatch(board, factor, snapshot, raw, resolved))
    return plans


def _reuse_small_artifacts(source: Path, destination: Path) -> None:
    """Copy only CV reference artifacts and diagnostics; never models or predictions.

    The existing consumers validate each copied artifact's fingerprint before use.
    Copying instead of linking also allows independent invalidation on Windows.
    """
    for name in ("artifacts", "preprocessing"):
        directory = source / name
        if directory.is_dir():
            shutil.copytree(directory, destination / name, dirs_exist_ok=True)


def _shared_scope(plan: FactorBatch) -> str:
    """Bind reusable references to the full-resolution calibration and acquisitions.

    Control changes can shift the LED target even if selected event IDs stay the
    same. The factor is excluded, but control provenance must never be excluded.
    """
    return canonical_hash({
        "version": EXPERIMENT_VERSION, "control_artifact_version": CONTROL_ARTIFACT_VERSION,
        "datasets": plan.batch.datasets,
        "sources": {role: source_signature(Path(dataset["root_file"]))
                    for role, dataset in plan.batch.datasets.items()},
        "preprocessing": plan.batch.runs[0]["preprocessing"],
        "window_ns": plan.batch.runs[0]["window_ns"], "window_name": plan.batch.runs[0]["window_name"],
        "seed": plan.batch.protocol["seed"], "cv": plan.batch.protocol["cross_validation"],
        "fits": {run["mode"]: run["fit"] for run in plan.batch.runs},
    })


def _sampling_intervals(plans: list[FactorBatch]) -> dict[tuple[str, str], float]:
    """Validate inputs/windows once per acquisition/channel pair before training."""
    seen: dict[tuple[str, tuple[int, int]], float] = {}
    intervals = {}
    for plan in plans:
        # Dataset/window settings are shared by all models in each mode.
        representatives = {run["mode"]: run for run in plan.batch.runs}
        for run in representatives.values():
            family = "energy" if run["mode"] == "energy_to_energy" else "timing"
            for role in ("control", "development", "blind"):
                dataset = run[role]
                channels = tuple(dataset["channels"][family])
                key = (dataset["root_file"], channels)
                if key not in seen:
                    # This existing helper reads only the first entry's time metadata.
                    seen[key] = energy_sampling_interval_s(Path(key[0]), channels)
                _input_offsets(seen[key], run["window_ns"], plan.factor)
                if role == "development":
                    intervals[plan.board, run["mode"]] = seen[key]
    return intervals


def _summary_rows(plan: FactorBatch, intervals: dict[tuple[str, str], float]) -> list[dict[str, Any]]:
    """Read persisted central metrics without inference, fitting or bootstrapping."""
    rows = []
    for run in plan.batch.runs:
        metadata = Path(run["output_dir"]) / "metadata"
        manifest, best, blind = (_read(metadata / name) for name in ("manifest.json", "best.json", "blind.json"))
        dt_ns = intervals[plan.board, run["mode"]] * 1e9
        population = manifest["dataset_populations"]
        rows.append({
            "board": plan.board, "mode": run["mode"], "model": run["model"]["name"],
            "window": run["window_name"], "start_ns": run["window_ns"]["start"],
            "end_ns": run["window_ns"]["end"], "subsampling": plan.factor,
            "native_interval_ns": dt_ns, "ml_interval_ns": dt_ns * plan.factor,
            "ml_sampling_rate_GSa_s": 1.0 / (dt_ns * plan.factor),
            "n_control": population["control"]["n_selected"],
            "n_development": population["development"]["n_selected"],
            "n_blind": population["blind"]["n_selected"],
            "selection_method": manifest["selection_method"],
            "validation_ctr_ps": best["validation_ctr_mean_ps"],
            "validation_rmse_ps": best["validation_rmse_mean_ps"],
            "ridge_internal_cv_mse_ps2": best.get("internal_cv_mse_ps2"),
            "blind_ctr_ps": blind["ctr_ps"], "blind_rmse_ps": blind["rmse_ps"],
            "led_ctr_ps": blind["led_ctr_ps"], "led_rmse_ps": blind["led_rmse_ps"],
            "led_control_mean_ps": manifest["led_control_mean_ps"],
            "result_directory": run["output_dir"],
        })
    return rows


def run_experiment(experiment: Experiment, plans: list[FactorBatch]) -> Path:
    """Execute ordinary batches sequentially, sharing only compatible reusable work."""
    intervals = _sampling_intervals(plans)
    logger = configure_logging()
    summaries: dict[str, list[dict[str, Any]]] = {}
    scopes = {board: _shared_scope(next(plan for plan in plans if plan.board == board))
              for board in experiment.boards}
    manifest = {
        "version": EXPERIMENT_VERSION, "name": experiment.name,
        "source": str(experiment.source), "window_ns": experiment.window_ns,
        "window_name": experiment.window_name, "subsampling": list(experiment.factors),
        "preprocessing_sampling": "full_resolution", "subsampling_stage": "ml_input_only",
        "sample_based_model_and_xai_settings": "unchanged",
        "shared_cache_scopes": scopes,
        "batches": [{"board": plan.board, "subsampling": plan.factor,
                     "config": str(plan.config_path), "directory": plan.batch.output_dir,
                     "fingerprint": canonical_hash(public_batch_config(plan.batch)),
                     "status": "pending"} for plan in plans],
    }
    path = experiment.output / "experiment.json"
    atomic_json(path, manifest)
    for item, plan in zip(manifest["batches"], plans):
        shared = experiment.output / "shared" / plan.board / scopes[plan.board]
        batch_root = Path(plan.batch.output_dir)
        try:
            _write_config(plan.config_path, plan.raw)
            stamp = batch_root / "sampling_references.json"
            previous = _read(stamp) if stamp.is_file() else {}
            if previous.get("scope") != scopes[plan.board]:
                # Only rebuildable CV references live here; fitted models stay in
                # their model scopes and retain the normal batch confirmation.
                shutil.rmtree(batch_root / "artifacts", ignore_errors=True)
            _reuse_small_artifacts(shared, batch_root)
            atomic_json(stamp, {"scope": scopes[plan.board]})
            item["status"] = "running"
            atomic_json(path, manifest)
            logger.info("Subsampling experiment | %s | factor=%d | window=[%g, %g] ns",
                        plan.board, plan.factor, experiment.window_ns["start"], experiment.window_ns["end"])
            run_batch(plan.batch, logger=logger)
            _reuse_small_artifacts(batch_root, shared)
            summaries.setdefault(plan.board, []).extend(_summary_rows(plan, intervals))
            write_csv(experiment.output / "tables" / f"subsampling_{plan.board}.csv", summaries[plan.board])
            item["status"] = "complete"
            atomic_json(path, manifest)
        except (Exception, KeyboardInterrupt, SystemExit) as exc:
            item["status"] = "failed"
            item["error"] = str(exc)
            atomic_json(path, manifest)
            raise
    return experiment.output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--window-ns", nargs=2, type=float, metavar=("START", "END"))
    parser.add_argument("--subsampling", nargs="+", type=int, metavar="FACTOR")
    parser.add_argument("--boards", nargs="+", choices=("UC", "FBK"))
    parser.add_argument("--models", nargs="+", metavar="MODEL")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--dry-run", action="store_true", help="Validate/print batches without raw-data reads or persistent writes")
    args = parser.parse_args()
    try:
        experiment = load_experiment(args)
        plans = build_batches(experiment, args.models)
    except (ValueError, KeyError, TypeError, OSError) as exc:
        parser.error(str(exc))
    if args.dry_run:
        for plan in plans:
            print(f"{plan.board} | subsampling={plan.factor} | window={experiment.window_ns} | "
                  f"{len(plan.batch.runs)} models/modes | {plan.batch.output_dir}")
        return
    print(run_experiment(experiment, plans))


if __name__ == "__main__":
    main()
