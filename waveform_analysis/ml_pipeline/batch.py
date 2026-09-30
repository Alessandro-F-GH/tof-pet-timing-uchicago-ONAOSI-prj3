from __future__ import annotations

import shutil
from pathlib import Path

from .common import atomic_json, canonical_hash, write_csv
from .config import BatchConfig
from .study import run_study


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
        rows.append({
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
        })
    return rows


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
        runtime["_control_fit_by_mode"] = {key: dict(value) for key, value in fit_by_mode.items()}
        runtime["_control_modes"] = modes
        mode = str(config["mode"])
        population_key = (
            mode,
            float(config["window_ns"]["start"]),
            float(config["window_ns"]["end"]),
            canonical_hash(config.get("ml_input", {})),
        )
        runtime["_rebuild_control"] = bool(rebuild_preprocessing and index == 0)
        runtime["_rebuild_analysis"] = bool(rebuild_preprocessing and mode not in seen_modes)
        runtime["_rebuild_prepared"] = bool(
            rebuild_preprocessing and population_key not in seen_populations
        )
        seen_modes.add(mode)
        seen_populations.add(population_key)
        output.append(runtime)
    return output


def run_batch(batch, *, overwrite=False, resume=False, rebuild_preprocessing=False, logger=None):
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
    if overwrite and root.exists():
        shutil.rmtree(root)
    _write_batch_state(batch, rows, "running")

    for index, config in enumerate(configs, 1):
        if logger:
            logger.info("Batch run %d/%d | %s", index, total, config["name"])
        try:
            output = run_study(
                config,
                overwrite=overwrite,
                resume=resume,
                rebuild_preprocessing=False,
            )
            outputs.append(output)
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
