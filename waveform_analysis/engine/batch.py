from __future__ import annotations
import json, shutil
from pathlib import Path
from waveform_analysis.core.io import atomic_json, canonical_hash, write_csv
from waveform_analysis.core.config import BatchConfig, public_batch_config
from waveform_analysis.engine.preprocessing import (
    fit_control,
    publish_preprocessing_diagnostics,
)
from waveform_analysis.data.storage import RunStore
from waveform_analysis.engine.study import run_study
from waveform_analysis.engine.xai import XAI_METHOD_VERSION
from waveform_analysis.core.ridge import LINEAR_RIDGE_MODELS, RIDGE_CV_VERSION

BATCH_SCHEMA_VERSION = 11


def _read_json(path):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return None


def _mode_fit_map(batch):
    result = {}
    for config in batch.runs:
        mode = str(config["mode"])
        fit = dict(config["fit"])
        if mode in result and canonical_hash(result[mode]) != canonical_hash(fit):
            raise ValueError(
                f"one batch requires one CTR fit definition per mode; mismatch for {mode}"
            )
        result[mode] = fit
    return result


def _runtime_configs(batch):
    fit_by_mode = _mode_fit_map(batch)
    out = []
    all_modes = tuple(sorted(fit_by_mode))
    for config in batch.runs:
        item = dict(config)
        item["_control_modes"] = all_modes
        item["_control_fit_by_mode"] = {
            mode: dict(fit_by_mode[mode]) for mode in all_modes
        }
        out.append(item)
    return out


def _planner_fingerprints(config):
    pre = canonical_hash(
        {
            "control": config["control"],
            "preprocessing": config["preprocessing"],
            "mode": config["mode"],
            "fit": config["fit"],
        }
    )
    dev = canonical_hash(
        {
            "preprocessing": pre,
            "development": config["development"],
            "window": config["window_ns"],
            "ml_input": config["ml_input"],
        }
    )
    cv_dependencies = {
        "development": dev,
        "cross_validation": config["cross_validation"],
        "model": config["model"],
        "seed": config["seed"],
    }
    if config["model"]["name"] in LINEAR_RIDGE_MODELS:
        cv_dependencies.pop("cross_validation")
        cv_dependencies["selection_method"] = "ridge_cv"
        cv_dependencies["ridge_cv_version"] = RIDGE_CV_VERSION
    cv = canonical_hash(cv_dependencies)
    final = canonical_hash(
        {"cv": cv, "development": dev, "model": config["model"], "seed": config["seed"]}
    )
    blind = canonical_hash(
        {
            "final_fit": final,
            "preprocessing": pre,
            "blind": config["blind"],
            "window": config["window_ns"],
            "ml_input": config["ml_input"],
            "fit": config["fit"],
        }
    )
    bootstrap = canonical_hash(
        {"blind": blind, "bootstrap": config["bootstrap"], "seed": config["seed"]}
    )
    xai = canonical_hash(
        {
            "final_fit": final,
            "blind": blind,
            "xai": config["xai"],
            "seed": config["seed"],
            "method_version": XAI_METHOD_VERSION,
        }
    )
    plots = canonical_hash({"plot_config": config["plot_config"]})
    return {
        "preprocessing": pre,
        "development": dev,
        "cv": cv,
        "final_fit": final,
        "blind": blind,
        "bootstrap": bootstrap,
        "xai": xai,
        "plots": plots,
    }


def _run_state(config, previous):
    path = Path(config["output_dir"])
    current = _planner_fingerprints(config)
    if not path.exists() or not any(path.iterdir()):
        return "run", "all", current
    RunStore(path)
    manifest = _read_json(path / "metadata" / "manifest.json")
    if manifest is not None and int(manifest.get("schema_version", -1)) != 51:
        return "rebuild", "incompatible_schema", current
    old = (previous or {}).get(config["run_id"])
    if not isinstance(old, dict):
        return "rebuild", "unknown_dependencies", current
    if (
        old.get("preprocessing") != current["preprocessing"]
        or old.get("development") != current["development"]
        or old.get("cv") != current["cv"]
    ):
        return "rebuild", "cv", current
    for stage in ("final_fit", "blind", "bootstrap", "xai", "plots"):
        if old.get(stage) != current[stage]:
            return "resume", stage, current
    if manifest is not None and manifest.get("status") == "complete":
        return "keep", "complete", current
    return "resume", "incomplete", current


def _plan_batch(batch):
    root = Path(batch.output_dir).resolve()
    previous_manifest = _read_json(root / "manifest.json") or {}
    previous = (
        previous_manifest.get("planner_fingerprints", {})
        if int(previous_manifest.get("schema_version", -1)) == BATCH_SCHEMA_VERSION
        else {}
    )
    studies = []
    current = {}
    for config in batch.runs:
        action, stage, fingerprints = _run_state(config, previous)
        current[config["run_id"]] = fingerprints
        studies.append(
            {
                "config": config,
                "action": action,
                "stage": stage,
                "path": Path(config["output_dir"]).resolve(),
            }
        )
    return {
        "studies": studies,
        "planner_fingerprints": current,
        "destructive": any(i["action"] == "rebuild" for i in studies),
    }


def _print_batch_plan(batch, plan):
    print(f"Batch plan | {batch.name}")
    for item in plan["studies"]:
        config = item["config"]
        suffix = (
            "" if item["stage"] in {"all", "complete"} else f" from {item['stage']}"
        )
        print(
            f"  {item['action'].upper():7s} {config['mode']} / {config.get('window_name')} / {config['model']['name']}{suffix}"
        )
    print(
        f"  {('REMAKE' if any(i['action'] != 'keep' for i in plan['studies']) else 'KEEP'):7s} report"
    )


def _confirm(plan):
    if not plan["destructive"]:
        return
    print(
        "\nOne or more incompatible/CV-invalidated model result scopes will be rebuilt."
    )
    try:
        answer = input("Proceed? [y/N] ").strip().lower()
    except EOFError as exc:
        raise RuntimeError(
            "Batch requires destructive confirmation in an interactive terminal"
        ) from exc
    if answer not in {"y", "yes"}:
        raise SystemExit("Batch cancelled; no existing results were modified.")


def _apply(plan):
    for item in plan["studies"]:
        if item["action"] == "rebuild" and item["path"].exists():
            shutil.rmtree(item["path"])


def _rows(batch, plan, statuses=None):
    statuses = statuses or {}
    root = Path(batch.output_dir).resolve()
    return [
        {
            "run_id": i["config"]["run_id"],
            "model": i["config"]["model"]["name"],
            "mode": i["config"]["mode"],
            "window": i["config"].get("window_name"),
            "status": statuses.get(i["config"]["run_id"], "pending"),
            "action": i["action"],
            "path": str(Path(i["config"]["output_dir"]).resolve().relative_to(root)),
        }
        for i in plan["studies"]
    ]


def _write(batch, plan, status, statuses=None):
    root = Path(batch.output_dir).resolve()
    root.mkdir(parents=True, exist_ok=True)
    rows = _rows(batch, plan, statuses)
    atomic_json(
        root / "manifest.json",
        {
            "schema_version": BATCH_SCHEMA_VERSION,
            "name": batch.name,
            "status": status,
            "source_config": batch.source_path,
            "results": batch.results,
            "datasets": batch.datasets,
            "protocol": batch.protocol,
            "axes": batch.axes,
            "statistical_units": {
                "validation": "development_cv_fold",
                "blind_uncertainty": "blind_event_bootstrap",
            },
            "preprocessing_fit_role": "control",
            "model_selection_role": "development",
            "blind_role": "one_time_final_evaluation",
            "single_configured_seed": int(batch.protocol["seed"]),
            "planner_fingerprints": plan["planner_fingerprints"],
            "runs": rows,
        },
    )
    atomic_json(root / "config.json", public_batch_config(batch))
    atomic_json(root / "plots.json", batch.plot_config)
    write_csv(root / "tables" / "runs.csv", rows)


def run_batch(batch, *, logger=None):
    if not isinstance(batch, BatchConfig):
        raise TypeError("run_batch requires a resolved BatchConfig")
    root = Path(batch.output_dir).resolve()
    legacy_runs = root / "runs.csv"
    organized_runs = root / "tables" / "runs.csv"
    if legacy_runs.is_file() and not organized_runs.exists():
        organized_runs.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(legacy_runs), str(organized_runs))
    plan = _plan_batch(batch)
    _print_batch_plan(batch, plan)
    _confirm(plan)
    _apply(plan)
    configs = {c["run_id"]: c for c in _runtime_configs(batch)}
    statuses = {}
    outputs = []
    _write(batch, plan, "running", statuses)
    published = set()
    try:
        for item in plan["studies"]:
            config = configs[item["config"]["run_id"]]
            run_id = config["run_id"]
            output = (
                Path(config["output_dir"]).resolve()
                if item["action"] == "keep"
                else run_study(config, logger=logger)
            )
            outputs.append(output)
            statuses[run_id] = "complete"
            _write(batch, plan, "running", statuses)
            mode = str(config["mode"])
            if mode not in published:
                control, _ = fit_control(config, rebuild=False, logger=None)
                publish_preprocessing_diagnostics(
                    config, "control", control, batch.output_dir, logger=logger
                )
                publish_preprocessing_diagnostics(
                    config, "development", control, batch.output_dir, logger=logger
                )
                published.add(mode)
        for mode in sorted({str(c["mode"]) for c in configs.values()}):
            config = next(c for c in configs.values() if str(c["mode"]) == mode)
            control, _ = fit_control(config, rebuild=False, logger=None)
            publish_preprocessing_diagnostics(
                config, "blind", control, batch.output_dir, logger=logger
            )
        from waveform_analysis.reporting.report import generate_report

        report_manifest = Path(batch.output_dir).resolve() / "report" / "manifest.json"
        if not (
            all(i["action"] == "keep" for i in plan["studies"])
            and report_manifest.is_file()
        ):
            generate_report(batch.output_dir, logger=logger)
        _write(batch, plan, "complete", statuses)
        return outputs
    except Exception:
        _write(batch, plan, "failed", statuses)
        raise


# Retain serialized identities while legacy imports resolve to this module.
from waveform_analysis.core.compat import (
    preserve_legacy_identity as _preserve_legacy_identity,
)

_preserve_legacy_identity(__name__, "waveform_analysis.ml_pipeline.batch")
