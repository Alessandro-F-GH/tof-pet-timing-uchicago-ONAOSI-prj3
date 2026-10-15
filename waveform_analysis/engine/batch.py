from __future__ import annotations
import json, shutil
from pathlib import Path
from waveform_analysis.core.io import atomic_json, canonical_hash, write_csv
from waveform_analysis.core.dependencies import planner_fingerprints as _planner_fingerprints
from waveform_analysis.core.config import BatchConfig
from waveform_analysis.data.results import write_run_config
from waveform_analysis.engine.preprocessing import (
    fit_control,
    publish_preprocessing_diagnostics,
)
from waveform_analysis.engine.study import run_study


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
    prepared = {}
    all_modes = tuple(sorted(fit_by_mode))
    for config in batch.runs:
        item = dict(config)
        item['_board'] = next((p for p in Path(batch.results['folder']).parts if p in {'UC', 'FBK'}), None)
        item["_prepared_datasets"] = prepared
        item["_control_modes"] = all_modes
        item["_control_fit_by_mode"] = {
            mode: dict(fit_by_mode[mode]) for mode in all_modes
        }
        out.append(item)
    return out



def _run_state(config, previous):
    path = Path(config["output_dir"])
    current = _planner_fingerprints(config)
    if not path.exists() or not any(path.iterdir()):
        return "run", "all", current
    manifest = _read_json(path / "metadata" / "manifest.json")
    if manifest is not None:
        from waveform_analysis.data.results import batch_root
        batch_root(path)  # Old layouts must be explicitly converted, never rebuilt implicitly.
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
    previous = previous_manifest.get("planner_fingerprints", {})
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
    rebuilding = {i['path'] for i in plan['studies'] if i['action'] == 'rebuild'}
    for scope in {path.parent for path in rebuilding}:
        owners = {p.parent.parent.resolve() for p in scope.glob('*/metadata/config.json')}
        if owners <= rebuilding:
            (scope / 'population.json').unlink(missing_ok=True)
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
            "status": status,
            "planner_fingerprints": plan["planner_fingerprints"],
        },
    )
    write_csv(root / "tables" / "runs.csv", rows)


def run_batch(batch, *, logger=None):
    if not isinstance(batch, BatchConfig):
        raise TypeError("run_batch requires a resolved BatchConfig")
    root = Path(batch.output_dir).resolve()
    plan = _plan_batch(batch)
    configs = {c["run_id"]: c for c in _runtime_configs(batch)}
    managed = [c["output_dir"] for c in configs.values()]
    for config in configs.values():
        write_run_config(config, managed_runs=managed, validate_only=True)
    _print_batch_plan(batch, plan)
    _confirm(plan)
    _apply(plan)
    for config in configs.values():
        config["_managed_runs"] = managed
        write_run_config(config, managed_runs=managed)
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
        from waveform_analysis.reporting.latex_tables import DATASET_TABLE_VERSION
        from waveform_analysis.engine.diagnostics import DEVELOPMENT_DISTRIBUTION_VERSION

        report_manifest = Path(batch.output_dir).resolve() / "report" / "manifest.json"
        if not (
            all(i["action"] == "keep" for i in plan["studies"])
            and report_manifest.is_file()
            and (_read_json(report_manifest) or {}).get("dataset_table_version")
            == DATASET_TABLE_VERSION
            and (_read_json(report_manifest) or {}).get("development_distribution_version")
            == DEVELOPMENT_DISTRIBUTION_VERSION
            and (_read_json(report_manifest) or {}).get("blind_boxplot_version") == 1
        ):
            generate_report(
                batch.output_dir,
                logger=logger,
                reuse_numeric=all(i["action"] == "keep" for i in plan["studies"]),
            )
        _write(batch, plan, "complete", statuses)
        return outputs
    except Exception:
        _write(batch, plan, "failed", statuses)
        raise
