from __future__ import annotations
import copy, json, time
from pathlib import Path
from waveform_analysis.core.io import atomic_json, canonical_hash, write_csv
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
        item = copy.deepcopy(config)
        item['_board'] = next((p for p in Path(batch.results['folder']).parts if p in {'UC', 'FBK'}), None)
        item["_prepared_datasets"] = prepared
        item["_control_modes"] = all_modes
        item["_control_fit_by_mode"] = {
            mode: dict(fit_by_mode[mode]) for mode in all_modes
        }
        out.append(item)
    return out



def _run_state(config, previous=None):
    """Resume existing result files; never infer invalidation from hashes."""
    from waveform_analysis.data.storage import RunStore
    from waveform_analysis.engine.train import saved_model_complete
    from waveform_analysis.models import get_model

    path = Path(config["output_dir"])
    if not path.exists() or not any(path.iterdir()):
        return "run", "all", None
    manifest = _read_json(path / "metadata" / "manifest.json")
    if manifest is None or manifest.get("status") != "complete":
        return "resume", "incomplete", None
    store = RunStore(path)
    checks = (
        ("cv", store.cv_path),
        ("selection", store.best_path),
        ("final_fit", store.final_fit_path),
        ("blind", store.predictions_path),
        ("development_diagnostic", store.development_predictions_path),
        ("bootstrap", store.bootstrap_path),
        ("bootstrap", store.bootstrap_draws_path),
    )
    for stage, filename in checks:
        if not filename.is_file():
            return "resume", stage, None
    if config.get("save_model", True):
        if not saved_model_complete(get_model(config["model"]["name"]), store.model_dir):
            return "resume", "final_fit", None
    return "keep", "complete", None


def _plan_batch(batch):
    studies = []
    for config in batch.runs:
        action, stage, _ = _run_state(config)
        studies.append({
            "config": config,
            "action": action,
            "stage": stage,
            "path": Path(config["output_dir"]).resolve(),
        })
    return {"studies": studies, "destructive": False}


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
    print("BATCH START | name=%s | runs=%d | results=%s" %
          (batch.name, len(plan["studies"]), batch.output_dir), flush=True)
    for config in configs.values():
        config["_managed_runs"] = managed
        write_run_config(config, managed_runs=managed)
    statuses = {}
    outputs = []
    _write(batch, plan, "running", statuses)
    published = set()
    try:
        for run_number, item in enumerate(plan["studies"], 1):
            config = configs[item["config"]["run_id"]]
            run_id = config["run_id"]
            started = time.monotonic()
            print("BATCH RUN START | %d/%d | %s | action=%s | stage=%s" %
                  (run_number, len(plan["studies"]), run_id,
                   item["action"], item["stage"]), flush=True)
            try:
                output = (
                    Path(config["output_dir"]).resolve()
                    if item["action"] == "keep"
                    else run_study(config, logger=logger)
                )
            except Exception as exc:
                print(
                    "BATCH RUN FAILED | %d/%d | %s | action=%s | stage=%s "
                    "| elapsed=%.1f s | error=%s: %s" %
                    (run_number, len(plan["studies"]), run_id, item["action"],
                     item["stage"], time.monotonic() - started,
                     type(exc).__name__, exc),
                    flush=True,
                )
                raise
            outputs.append(output)
            statuses[run_id] = "complete"
            print("BATCH RUN DONE | %d/%d | %s | elapsed=%.1f s" %
                  (run_number, len(plan["studies"]), run_id,
                   time.monotonic()-started), flush=True)
            _write(batch, plan, "running", statuses)
            mode = str(config["mode"])
            if mode not in published:
                preprocessing_started = time.monotonic()
                print("BATCH PREPROCESSING START | mode=%s | control" % mode, flush=True)
                try:
                    control, _ = fit_control(config, rebuild=False, logger=logger)
                    print("BATCH PREPROCESSING CONTROL FIT DONE | mode=%s | elapsed=%.1f s" %
                          (mode, time.monotonic() - preprocessing_started), flush=True)
                    for role in ("control", "development"):
                        role_started = time.monotonic()
                        print("BATCH PREPROCESSING START | mode=%s | role=%s" %
                              (mode, role), flush=True)
                        publish_preprocessing_diagnostics(
                            config, role, control, batch.output_dir, logger=logger
                        )
                        print("BATCH PREPROCESSING DONE | mode=%s | role=%s | elapsed=%.1f s" %
                              (mode, role, time.monotonic() - role_started), flush=True)
                except Exception as exc:
                    print("BATCH PREPROCESSING FAILED | mode=%s | elapsed=%.1f s | error=%s: %s" %
                          (mode, time.monotonic() - preprocessing_started,
                           type(exc).__name__, exc), flush=True)
                    raise
                published.add(mode)
        for mode in sorted({str(c["mode"]) for c in configs.values()}):
            config = next(c for c in configs.values() if str(c["mode"]) == mode)
            blind_started = time.monotonic()
            print("BATCH PREPROCESSING START | mode=%s | role=blind" % mode, flush=True)
            try:
                control, _ = fit_control(config, rebuild=False, logger=logger)
                publish_preprocessing_diagnostics(
                    config, "blind", control, batch.output_dir, logger=logger
                )
                print("BATCH PREPROCESSING DONE | mode=%s | role=blind | elapsed=%.1f s" %
                      (mode, time.monotonic() - blind_started), flush=True)
            except Exception as exc:
                print("BATCH PREPROCESSING FAILED | mode=%s | role=blind | error=%s: %s" %
                      (mode, type(exc).__name__, exc), flush=True)
                raise
        mt_options = batch.protocol.get("multithreshold_reference", {})
        reference_fingerprints = {}
        if mt_options.get("enabled", False):
            from waveform_analysis.engine.multithreshold_reference import run_reference
            energy_by_window = {}
            for item in plan["studies"]:
                config = item["config"]
                if config["mode"] == "energy_to_energy":
                    energy_by_window.setdefault(config["window_name"], config["output_dir"])
            for window, reference_run in sorted(energy_by_window.items()):
                summary = run_reference(
                    reference_run, batch.output_dir, mt_options,
                    seed=int(batch.protocol["seed"]),
                    bootstrap_resamples=int(batch.protocol["bootstrap"]["n_resamples"]),
                )
                reference_fingerprints[window] = summary["fingerprint"]
                print("BATCH MT CONTROL SELECTED | window=%s | thresholds=%s mV | control CTR=%.3f ps | control coverage=%.2f%%" %
                      (window, summary["thresholds_mV"],
                       summary["control_selection"]["ctr_ps"],
                       100 * summary["control_selection"]["coverage"]), flush=True)
                print("BATCH MT REFERENCE | window=%s | blind CTR=%.3f ps | coverage=%.2f%%" %
                      (window, summary["blind"]["metrics"]["ctr_ps"],
                       100 * summary["blind"]["coverage"]), flush=True)
        atomic_json(Path(batch.output_dir) / "references" / "settings.json", {
            "enabled": bool(mt_options.get("enabled", False)),
            "fingerprints": reference_fingerprints,
        })
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
            and (_read_json(report_manifest) or {}).get("multithreshold_reference_fingerprints", {}) == reference_fingerprints
            and (
                not mt_options.get("enabled", False)
                or (
                    (_read_json(report_manifest) or {}).get("multithreshold_reference_version") == 1
                    and (_read_json(report_manifest) or {}).get("multithreshold_reference_fingerprints") == reference_fingerprints
                )
            )
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
