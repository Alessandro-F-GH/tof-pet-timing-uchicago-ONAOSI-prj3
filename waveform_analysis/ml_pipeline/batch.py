from __future__ import annotations

import json
import shutil
from pathlib import Path

from .common import atomic_json, canonical_hash, write_csv
from .config import BatchConfig
from .control_preprocessing import fit_control_artifact
from .data import preprocess_selected
from .event_selection import apply_selection_rules
from .preprocessing_plots import plot_materialized_event
from .study import run_study


_SELECTION_DIAGNOSTIC_FILES = (
    "photopeak_selection.png",
    "baseline_noise.png",
    "baseline_clipping.png",
    "timing_tot_selection.png",
    "selection_summary.csv",
)


def _log_gpu_status(logger):
    if logger is None:
        return
    try:
        import torch
    except Exception as exc:
        logger.warning("GPU check | PyTorch unavailable | %s", exc)
        return
    if not torch.cuda.is_available():
        logger.info("GPU check | CUDA unavailable | PyTorch=%s | using CPU", torch.__version__)
        return
    try:
        device = torch.device("cuda:0")
        probe = torch.ones(8, device=device)
        result = (probe * 2.0).sum()
        torch.cuda.synchronize(device)
        if float(result.item()) != 16.0:
            raise RuntimeError("unexpected CUDA smoke-test result")
        props = torch.cuda.get_device_properties(device)
        logger.info(
            "GPU check | CUDA OK | device=%s | capability=%d.%d | memory=%.1f GiB | PyTorch=%s | CUDA=%s",
            torch.cuda.get_device_name(device),
            props.major,
            props.minor,
            props.total_memory / (1024**3),
            torch.__version__,
            torch.version.cuda,
        )
    except Exception as exc:
        logger.warning("GPU check | CUDA detected but smoke test failed | %s", exc)


def _run_index(batch: BatchConfig):
    root = Path(batch.output_dir).resolve()
    rows = []
    for config in batch.runs:
        path = Path(config["output_dir"]).resolve()
        try:
            relative = str(path.relative_to(root))
        except ValueError:
            relative = str(path)
        rows.append(
            {
                "run_id": config.get("run_id", config["name"]),
                "name": config["name"],
                "model": config["model"]["name"],
                "mode": config["mode"],
                "window": config.get("window_name", ""),
                "window_start_ns": config["window_ns"]["start"],
                "window_end_ns": config["window_ns"]["end"],
                "save_models": config["save_models"],
                "status": "pending",
                "path": relative,
                "config_fingerprint": config["_config_fingerprint"],
            }
        )
    return rows


def _existing_study_state(config):
    path = Path(config["output_dir"]).resolve()
    if not path.exists() or not any(path.iterdir()):
        return "missing"

    manifest_path = path / "manifest.json"
    if not manifest_path.is_file():
        return "unreadable"

    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return "unreadable"

    if manifest.get("config_fingerprint") != config["_config_fingerprint"]:
        return "mismatch"
    if manifest.get("status") == "complete":
        return "matching_complete"
    return "matching_incomplete"


def _write_batch_state(batch, rows, status):
    root = Path(batch.output_dir).resolve()
    root.mkdir(parents=True, exist_ok=True)
    manifest = {
        "schema_version": 5,
        "name": batch.name,
        "status": status,
        "source_config": batch.source_path,
        "protocol": batch.protocol,
        "axes": batch.axes,
        "save_models": batch.runs[0]["save_models"],
        "statistical_unit": "replica",
        "hyperparameter_selection": "one fixed validation split, RMSE selection, performed once before replicas",
        "fixed_validation_used_in_replicas": False,
        "pairing_rule": "same analysis protocol + sampling identity + mode + window + replica index",
        "shared_artifact_root": str((root / "artifacts").resolve()),
        "preprocessing_diagnostics_root": str((root / "preprocessing").resolve()),
        "runs": rows,
    }
    atomic_json(root / "manifest.json", manifest)
    write_csv(root / "runs.csv", rows)


def _control_protocol(configs):
    by_mode = {}
    for config in configs:
        mode = str(config["mode"])
        fit = dict(config["fit"])
        if mode in by_mode and canonical_hash(by_mode[mode]) != canonical_hash(fit):
            raise ValueError(
                f"All runs in one batch must use one control CTR fit definition per mode; mismatch for {mode}"
            )
        by_mode[mode] = fit
    return by_mode, tuple(sorted(by_mode))


def _runtime_configs(configs, batch_root, rebuild_preprocessing=False):
    fit_by_mode, modes = _control_protocol(configs)
    artifact_root = str((Path(batch_root).resolve() / "artifacts").resolve())
    output = []
    seen_modes = set()
    seen_populations = set()
    for index, config in enumerate(configs):
        runtime = dict(config)
        runtime["_batch_artifact_root"] = artifact_root
        runtime["_control_fit_by_mode"] = {
            key: dict(value) for key, value in fit_by_mode.items()
        }
        runtime["_control_modes"] = modes
        mode = str(config["mode"])
        population_key = (
            mode,
            float(config["window_ns"]["start"]),
            float(config["window_ns"]["end"]),
            canonical_hash(config.get("ml_input", {})),
        )
        runtime["_rebuild_control"] = bool(rebuild_preprocessing and index == 0)
        runtime["_rebuild_analysis"] = bool(
            rebuild_preprocessing and mode not in seen_modes
        )
        runtime["_rebuild_prepared"] = bool(
            rebuild_preprocessing and population_key not in seen_populations
        )
        seen_modes.add(mode)
        seen_populations.add(population_key)
        output.append(runtime)
    return output


def _publish_batch_selection_diagnostics(
    config,
    batch_root,
    *,
    force=False,
    logger=None,
):
    """Publish cached selection diagnostics and one materialized event for this mode."""
    mode = str(config["mode"])
    output_dir = Path(batch_root).resolve() / "preprocessing" / mode
    metadata_path = output_dir / "selection_manifest.json"

    control, _ = fit_control_artifact(
        config["reference"]["root_file"],
        config["reference"],
        config["preprocessing"],
        config["fit"],
        fit_by_mode=config.get("_control_fit_by_mode"),
        modes=config.get("_control_modes") or [mode],
        cache_root=config["preprocessing"]["cache_dir"],
        rebuild=False,
        logger=None,
    )
    selection = apply_selection_rules(
        config["analysis"]["root_file"],
        config["analysis"],
        config["preprocessing"],
        control["selection_rules"],
        mode,
        cache_dir=Path(config["preprocessing"]["cache_dir"]) / "analysis_selection",
        rebuild=False,
        logger=None,
    )

    if metadata_path.is_file() and not force:
        try:
            existing = json.loads(metadata_path.read_text(encoding="utf-8"))
            if existing.get("fingerprint") == selection.manifest.get("fingerprint"):
                if logger:
                    logger.info(
                        "Batch preprocessing diagnostics already current | mode=%s | %s",
                        mode,
                        output_dir,
                    )
                return output_dir
        except (json.JSONDecodeError, OSError):
            pass

    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    copied = []
    for name in _SELECTION_DIAGNOSTIC_FILES:
        source = Path(selection.directory) / name
        if source.is_file():
            shutil.copy2(source, output_dir / name)
            copied.append(name)

    family = str(selection.manifest["family"])
    native = preprocess_selected(
        config["analysis"]["root_file"],
        selection,
        config["analysis"],
        config["preprocessing"],
        mode,
        cache_dir=Path(config["preprocessing"]["cache_dir"]) / "analysis_native",
        rebuild=False,
        logger=None,
    )
    waveform_name = f"{family}_selected_event_waveform.png"
    waveform_path = plot_materialized_event(
        native,
        family,
        config["analysis"]["channels"][family],
        output_dir / waveform_name,
        f"Selected {family}-channel event",
    )
    if waveform_path is not None:
        copied.append(waveform_name)

    atomic_json(
        metadata_path,
        {
            "fingerprint": selection.manifest["fingerprint"],
            "source_selection_artifact": str(Path(selection.directory).resolve()),
            "analysis_source": str(Path(config["analysis"]["root_file"]).resolve()),
            "mode": mode,
            "family": family,
            "n_raw": selection.manifest.get("n_raw"),
            "n_selected": selection.manifest.get("n_selected"),
            "stage_counts": selection.manifest.get("stage_counts"),
            "files": copied,
            "selection_rules_fingerprint": selection.manifest.get("rules_fingerprint"),
        },
    )
    if logger:
        logger.info(
            "Batch preprocessing diagnostics published | mode=%s | files=%d | %s",
            mode,
            len(copied),
            output_dir,
        )
    return output_dir


def run_batch(
    batch,
    *,
    overwrite=False,
    resume=False,
    rebuild_preprocessing=False,
    logger=None,
):
    if not isinstance(batch, BatchConfig):
        raise TypeError("run_batch requires a resolved BatchConfig")

    configs = _runtime_configs(
        list(batch.runs),
        batch.output_dir,
        rebuild_preprocessing=rebuild_preprocessing,
    )
    outputs = []
    total = len(configs)
    _log_gpu_status(logger)
    rows = _run_index(batch)

    root = Path(batch.output_dir).resolve()
    _write_batch_state(batch, rows, "running")

    published_modes = set()
    for index, config in enumerate(configs, 1):
        if logger:
            logger.info("Batch run %d/%d | %s", index, total, config["name"])
        try:
            study_overwrite = bool(overwrite)
            study_resume = bool(resume)

            if overwrite:
                state = _existing_study_state(config)
                if state == "matching_complete":
                    output = Path(config["output_dir"]).resolve()
                    outputs.append(output)
                    rows[index - 1]["status"] = "complete"
                    _write_batch_state(batch, rows, "running")
                    if logger:
                        logger.info(
                            "Batch reused | %s | matching resolved configuration | %s",
                            config["name"],
                            output,
                        )

                    mode = str(config["mode"])
                    if mode not in published_modes:
                        _publish_batch_selection_diagnostics(
                            config,
                            root,
                            force=bool(rebuild_preprocessing),
                            logger=logger,
                        )
                        published_modes.add(mode)
                    continue

                if state == "matching_incomplete":
                    study_overwrite = False
                    study_resume = True
                    if logger:
                        logger.info(
                            "Batch selective resume | %s | matching configuration with incomplete study",
                            config["name"],
                        )
                elif state in {"mismatch", "unreadable"}:
                    study_overwrite = True
                    study_resume = False
                    if logger:
                        reason = (
                            "resolved configuration changed"
                            if state == "mismatch"
                            else "existing study metadata is unreadable"
                        )
                        logger.info(
                            "Batch selective overwrite | %s | %s",
                            config["name"],
                            reason,
                        )
                else:
                    study_overwrite = False
                    study_resume = False

            output = run_study(
                config,
                overwrite=study_overwrite,
                resume=study_resume,
                rebuild_preprocessing=False,
            )
            outputs.append(output)

            mode = str(config["mode"])
            if mode not in published_modes:
                _publish_batch_selection_diagnostics(
                    config,
                    root,
                    force=bool(rebuild_preprocessing),
                    logger=logger,
                )
                published_modes.add(mode)

            rows[index - 1]["status"] = "complete"
            _write_batch_state(batch, rows, "running")
            if logger:
                logger.info("Batch completed | %s | %s", config["name"], output)
        except Exception:
            rows[index - 1]["status"] = "failed"
            _write_batch_state(batch, rows, "failed")
            if logger:
                logger.exception("Batch failed | %s", config["name"])
            raise

    _write_batch_state(batch, rows, "complete")
    return outputs