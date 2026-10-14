"""Group saved results without importing models, migrating sources or inference.

Run with ``python -m waveform_analysis.scripts.group_results --help`` or invoke
this file directly. Optional summary plots use the repository reporting stack.
"""
from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from dataclasses import dataclass
import hashlib
import json
import math
import ntpath
from pathlib import Path
import re
import shutil
import tempfile
from typing import Any

TOOL = "waveform_analysis.group_results"
ROLES = ("control", "development", "blind")
PLOT_TYPES = {".png", ".pdf", ".svg", ".jpg", ".jpeg", ".eps", ".webp"}
METADATA = ("manifest", "config", "best", "blind", "bootstrap", "final_fit")


def _json(path):
    if not path.is_file():
        return {}
    value = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def _saved(root, name):
    modern = root / "metadata" / f"{name}.json"
    return modern if modern.is_file() else root / f"{name}.json"


def _canonical(value):
    if isinstance(value, dict):
        return {k: _canonical(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_canonical(v) for v in value]
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    return value


def _digest(value):
    value = _canonical(value)
    data = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(data.encode()).hexdigest()[:12]


def _safe(value):
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("._") or "unknown"


def _number(value):
    value = float(value)
    if not math.isfinite(value):
        raise ValueError("Non-finite voltage or window boundary")
    return f"{value:g}"


def _path_key(value):
    # Lexical Windows normalization works on Linux too. Never resolve a saved
    # acquisition path against the machine doing the grouping.
    value = str(value).replace("\\", "/")
    if re.match(r"^[A-Za-z]:/", value) or value.startswith("//"):
        return ntpath.normpath(value).replace("\\", "/").casefold()
    import posixpath
    return posixpath.normpath(value)


def _inside(path, parent):
    return path == parent or parent in path.parents


def _batch(root):
    for parent in root.parents:
        indexed = (parent / "tables/runs.csv").is_file() or (parent / "runs.csv").is_file()
        config = _json(parent / "config.json")
        if indexed or ("protocol" in config and ("sweep" in config or "axes" in config)):
            return parent, config, _json(parent / "manifest.json")
    return root.parent, {}, {}


def _subsampling(manifest, config, batch_config, batch_manifest):
    candidates = [manifest.get("subsampling")]
    for value in (manifest, config, batch_config.get("protocol", {}), batch_manifest.get("protocol", {})):
        candidates.append(value.get("ml_input", {}).get("subsampling"))
    found = set()
    for value in candidates:
        if value is not None:
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"Invalid subsampling: {value!r}")
            found.add(value)
    if len(found) != 1:
        raise ValueError(f"Missing or conflicting subsampling: {sorted(found)}")
    return found.pop()


def _datasets(manifest, config, batch_config, aliases):
    datasets = {}
    for role in ROLES:
        value = manifest.get(f"{role}_dataset") or config.get(role) or batch_config.get("datasets", {}).get(role) or {}
        if isinstance(value, str):
            value = {"root_file": value}
        source = value.get("root_file") or ((manifest.get("dataset_populations") or {}).get(role) or {}).get("source")
        if not source:
            raise ValueError(f"Missing {role} dataset source")
        key = _path_key(source)
        datasets[role] = {
            "source": aliases.get(key, key), "channels": value.get("channels"),
            "true_tof_ps": value.get("true_tof_ps"),
        }
    return datasets


def _board_voltage(root, manifest, config, batch_config, batch_manifest, datasets):
    labels = [str(root), str(config.get("batch_output_dir", ""))]
    for value in (config, batch_config, batch_manifest):
        labels.append(str(value.get("results", {}).get("folder", "")))
    boards = set()
    for label in labels:
        boards.update(re.findall(r"(?:^|[/\\])(UC|FBK)(?=$|[/\\])", label.upper()))
    if len(boards) > 1:
        raise ValueError(f"Conflicting board labels: {sorted(boards)}")
    voltages = set()
    for role in ROLES:
        saved = ((manifest.get("dataset_populations") or {}).get(role) or {}).get("bias_voltage_V") or []
        if isinstance(saved, (int, float)):
            saved = [saved]
        voltages.update(_number(v) for v in saved)
    if not voltages:
        for value in datasets.values():
            voltages.update(_number(v) for v in re.findall(r"(?<![A-Za-z0-9])(\d+(?:\.\d+)?)V", value["source"], re.I))
    voltage = "-".join(sorted(voltages, key=float)) + "V" if voltages else "unknownV"
    return next(iter(boards), "UNKNOWN"), voltage


@dataclass
class SavedRun:
    root: Path
    batch: Path
    id: str
    model: str
    mode: str
    window: tuple[float, float]
    subsampling: int
    datasets: dict
    board: str
    voltage: str
    metadata: dict
    metric_fit: dict
    group: str = ""

    @property
    def leaf(self):
        return f"{_safe(self.mode)}/window_{_number(self.window[0])}_{_number(self.window[1])}ns"

    @property
    def selection_key(self):
        return f"{self.group}/{self.leaf}/{self.model}"

    def describe(self):
        return {"id": self.id, "source": str(self.root), "batch": str(self.batch),
                "model": self.model, "group": self.group, "mode": self.mode,
                "window_ns": list(self.window), "subsampling": self.subsampling,
                "selection_key": self.selection_key}


def discover(sources, output, aliases=None):
    aliases = {_path_key(k): v for k, v in (aliases or {}).items()}
    runs, skipped, seen = [], [], set()
    for source in sources:
        source = Path(source).expanduser().resolve()
        if not source.is_dir():
            raise ValueError(f"Source directory does not exist: {source}")
        for path in sorted(source.rglob("manifest.json")):
            if _inside(path, output):
                continue
            root = path.parent.parent if path.parent.name == "metadata" else path.parent
            root = root.resolve()
            if root in seen or any((p / "grouping_manifest.json").is_file() for p in (root, *root.parents)):
                continue
            manifest = _json(_saved(root, "manifest"))
            model = manifest.get("model")
            if not model:
                continue  # Batch, preprocessing and report manifests are not runs.
            seen.add(root)
            if manifest.get("status") != "complete":
                skipped.append({"source": str(root), "reason": "run is not complete"})
                continue
            try:
                if not isinstance(model, str):
                    raise ValueError("Model name must be a string")
                metadata = {name: _json(_saved(root, name)) for name in METADATA}
                config = metadata["config"]
                batch, batch_config, batch_manifest = _batch(root)
                subsampling = _subsampling(manifest, config, batch_config, batch_manifest)
                datasets = _datasets(manifest, config, batch_config, aliases)
                window = manifest.get("window_ns") or config.get("window_ns")
                if not isinstance(window, dict):
                    raise ValueError("Missing numeric window_ns")
                window = tuple(float(window[key]) for key in ("start", "end"))
                if not all(math.isfinite(v) for v in window) or window[1] <= window[0]:
                    raise ValueError("Invalid numeric window_ns")
                mode = manifest.get("mode") or config.get("mode")
                if mode not in {"energy_to_energy", "timing_to_timing"}:
                    raise ValueError("Missing or unsupported waveform mode")
                board, voltage = _board_voltage(root, manifest, config, batch_config, batch_manifest, datasets)
                fit = config.get("fit") or batch_config.get("protocol", {}).get("fit", {})
                fit = fit.get(mode, fit) if isinstance(fit, dict) else {}
                runs.append(SavedRun(root, batch, _digest(str(root)), model, mode, window,
                                     subsampling, datasets, board, voltage, metadata, fit))
            except (ValueError, KeyError, TypeError) as exc:
                skipped.append({"source": str(root), "reason": str(exc)})
    families = defaultdict(set)
    for run in runs:
        families[f"{run.board}_{run.voltage}_sub{run.subsampling}"].add(_digest(run.datasets))
    for run in runs:
        base = f"{run.board}_{run.voltage}_sub{run.subsampling}"
        run.group = base if len(families[base]) == 1 else f"{base}_{_digest(run.datasets)}"
    return sorted(runs, key=lambda r: (r.group, r.leaf, r.model, str(r.root))), skipped


def inventory(runs, skipped, selected=()):
    selected = set(selected)
    unknown = selected - {run.id for run in runs}
    if unknown:
        raise ValueError(f"Unknown selected run IDs: {sorted(unknown)}")
    by_key = defaultdict(list)
    for run in runs:
        by_key[run.selection_key].append(run)
    choices, conflicts = [], []
    for key, candidates in sorted(by_key.items()):
        chosen = [run for run in candidates if run.id in selected]
        if len(candidates) == 1:
            choices.extend(candidates)
        elif len(chosen) == 1:
            choices.extend(chosen)
        else:
            conflicts.append({"selection_key": key, "candidates": [r.describe() for r in candidates]})
    return {"runs": [r.describe() for r in runs], "skipped": skipped,
            "conflicts": conflicts, "selected": [r.describe() for r in choices]}, choices


def _write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def _csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _copy_tree(source, destination, suffixes):
    count = 0
    if source.is_dir():
        for path in sorted(source.rglob("*")):
            if path.is_file() and not path.is_symlink() and path.suffix.lower() in suffixes:
                target = destination / path.relative_to(source)
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(path, target)
                count += 1
    return count


def _summary(run):
    m, blind, bootstrap, best = (run.metadata[key] for key in ("manifest", "blind", "bootstrap", "best"))
    row = {**run.describe(), "window_ns": json.dumps(list(run.window)), "n_events": blind.get("n_events"),
           "schema_version": m.get("schema_version"), "sample_mask_policy": m.get("sample_mask_fit_role", "unknown"),
           "sample_mask_identity": m.get("sample_mask_identity", "unknown"),
           "selection_method": m.get("selection_method", "unknown")}
    for metric in ("ctr", "rmse"):
        row[f"{metric}_ps"] = blind.get(f"{metric}_ps")
        row[f"{metric}_std_ps"] = bootstrap.get(f"{metric}_bootstrap_std_ps")
        row[f"led_{metric}_ps"] = blind.get(f"led_{metric}_ps")
        row[f"validation_{metric}_ps"] = best.get(f"validation_{metric}_mean_ps")
    return row


def _compatibility(runs):
    if len(runs) == 1:
        return []
    reasons = []
    for role in ROLES:
        identities = [((r.metadata["manifest"].get("dataset_populations") or {}).get(role) or {}).get("population_identity") for r in runs]
        if not all(identities) or len(set(identities)) != 1:
            reasons.append(f"{role} event population missing or different")
    fits = [r.metric_fit for r in runs]
    if not all(fits) or len({_digest(fit) for fit in fits}) != 1:
        reasons.append("metric definitions missing or different")
    calibrations = [(r.metadata["manifest"].get("led_centering_fit_role"),
                     r.metadata["manifest"].get("led_control_mean_ps")) for r in runs]
    if any(role != "control" or not isinstance(mean, (float, int)) or not math.isfinite(mean) for role, mean in calibrations) or len(set(calibrations)) != 1:
        reasons.append("control LED calibration missing or different")
    # Model-input masks deliberately do not participate in these checks.
    return reasons


def _summary_plots(directory, members, cfg):
    import matplotlib
    matplotlib.use("Agg")
    # Direct file invocation must also locate the repository's reporting package.
    import sys
    repository = str(Path(__file__).resolve().parents[2])
    if repository not in sys.path:
        sys.path.insert(0, repository)
    from waveform_analysis.reporting.metric_rows import blind_metric_rows
    from waveform_analysis.reporting.model_labels import model_label
    from waveform_analysis.reporting.plotting import blind_metric_bar, output_path
    from waveform_analysis.reporting.latex_tables import export_blind_metric_table

    skipped = []
    for metric in ("ctr", "rmse"):
        group, led_values = [], []
        for run in members:
            manifest = dict(run.metadata["manifest"])
            formulation = manifest.get("estimator_formulation")
            if formulation not in ("direct", "shared"):
                code = model_label(run.model)[0]
                formulation = "direct" if code.startswith("D-") else "shared" if code.startswith("S-") else None
            value = run.metadata["blind"].get(f"{metric}_ps")
            led = run.metadata["blind"].get(f"led_{metric}_ps")
            if formulation is None or not all(isinstance(v, (int, float)) and math.isfinite(v) for v in (value, led)):
                skipped.append(f"{metric}: {run.model} missing metric, LED reference or formulation")
                break
            manifest["estimator_formulation"] = formulation
            bootstrap = dict(run.metadata["bootstrap"])
            error = bootstrap.get(f"{metric}_bootstrap_std_ps")
            bootstrap[f"{metric}_bootstrap_std_ps"] = error if isinstance(error, (int, float)) and error >= 0 else float("nan")
            group.append({"manifest": manifest, "blind": run.metadata["blind"], "bootstrap": bootstrap})
            led_values.append(float(led))
        else:
            if not all(math.isclose(v, led_values[0], rel_tol=1e-9, abs_tol=1e-9) for v in led_values):
                skipped.append(f"{metric}: saved LED reference differs between runs")
                continue
            rows = blind_metric_rows(group, metric)
            _csv(directory / f"blind_{metric}.csv", rows)
            export_blind_metric_table(
                directory / f"blind_{metric}.tex", rows, led_values[0], metric=metric,
                mode=members[0].mode, window=members[0].leaf.split("/")[-1], report_id=members[0].group,
            )
            blind_metric_bar(rows, output_path(directory, f"blind_{metric}", cfg), cfg,
                             metric=metric, led_value=led_values[0])
    return skipped


def group_results(sources, output, *, selected=(), aliases=None, check=False,
                  overwrite=False, summary_plots=True, copy_predictions=False, plot_config=None):
    output = Path(output).expanduser().resolve()
    if aliases is not None and (not isinstance(aliases, dict) or not all(
        isinstance(k, str) and isinstance(v, str) and v for k, v in aliases.items()
    )):
        raise ValueError("Dataset aliases must map paths to nonempty string labels")
    runs, skipped = discover(sources, output, aliases)
    plan, choices = inventory(runs, skipped, selected)
    if check:
        return plan
    if not runs:
        raise ValueError("No complete runs with sufficient grouping metadata found")
    if plan["conflicts"]:
        raise ValueError("Duplicate models require explicit --select RUN_ID choices. Run --check to list candidates.")
    if any(_inside(output, run.root) or _inside(run.root, output) for run in runs):
        raise ValueError("Output must not overlap a source run")
    if output.exists() and (not overwrite or _json(output / "grouping_manifest.json").get("tool") != TOOL):
        raise ValueError("Output exists. --overwrite only replaces a previous output created by this script.")
    output.parent.mkdir(parents=True, exist_ok=True)
    cfg = _json(Path(plot_config).expanduser().resolve() if plot_config else
                Path(__file__).resolve().parents[1] / "config/plots/default.json") if summary_plots else None
    if summary_plots and not cfg:
        raise ValueError("Plot configuration is missing or empty")
    stage = Path(tempfile.mkdtemp(prefix=f".{output.name}-", dir=output.parent))
    try:
        groups = defaultdict(list)
        for run in choices:
            groups[(run.group, run.leaf)].append(run)
        leaves = []
        for (group, leaf), members in sorted(groups.items()):
            directory = stage / group / leaf
            directory.mkdir(parents=True)
            rows, missing = [], []
            for run in members:
                name = _safe(run.model)
                if sum(_safe(r.model).casefold() == name.casefold() for r in members) > 1 or name.upper() in {"CON", "PRN", "AUX", "NUL"}:
                    name += f"__{run.id}"
                target = directory / "runs" / name
                for name in METADATA:
                    source = _saved(run.root, name)
                    if source.is_file():
                        destination = target / "metadata" / f"{name}.json"
                        destination.parent.mkdir(parents=True, exist_ok=True)
                        shutil.copy2(source, destination)
                    else:
                        missing.append({"run": run.id, "missing": f"{name}.json"})
                count = _copy_tree(run.root / "plots", target / "plots", PLOT_TYPES)
                # Legacy runs may store figures directly in the run directory.
                for path in run.root.iterdir():
                    if path.is_file() and not path.is_symlink() and path.suffix.lower() in PLOT_TYPES:
                        destination = target / "plots" / path.name
                        destination.parent.mkdir(parents=True, exist_ok=True)
                        shutil.copy2(path, destination)
                        count += 1
                if not count:
                    missing.append({"run": run.id, "missing": "saved plots"})
                _copy_tree(run.root / "tables", target / "tables", {".csv", ".tex", ".json"})
                if copy_predictions:
                    for name in ("pred.npz", "bootstrap.npz", "development_pred.npz"):
                        source = run.root / "artifacts" / name
                        if not source.is_file():
                            source = run.root / name
                        if source.is_file():
                            destination = target / "artifacts" / name
                            destination.parent.mkdir(parents=True, exist_ok=True)
                            shutil.copy2(source, destination)
                rows.append(_summary(run))
            _csv(directory / "summary.csv", rows)
            _csv(directory / "sources.csv", [r.describe() for r in members])
            reasons = _compatibility(members)
            if summary_plots and not reasons:
                reasons.extend(_summary_plots(directory, members, cfg))
            info = {"group": group, "leaf": leaf, "subsampling": members[0].subsampling,
                    "datasets": members[0].datasets, "runs": [r.describe() for r in members],
                    "missing": missing, "comparison_plot_skip_reasons": reasons,
                    "summary_plots_requested": summary_plots}
            _write_json(directory / "group.json", info)
            leaves.append(info)
        if cfg is not None:
            _write_json(stage / "plots.json", cfg)
        _csv(stage / "sources.csv", [r.describe() for r in choices])
        _write_json(stage / "grouping_manifest.json", {"tool": TOOL, "version": 1, "groups": leaves,
                                                      "skipped": skipped, "dataset_aliases": aliases or {}})
        # Build fully before replacing a previous generated report. Roll back if
        # publishing the staged directory fails; never mutate source runs.
        backup = None
        if output.exists():
            if not overwrite or _json(output / "grouping_manifest.json").get("tool") != TOOL:
                raise ValueError("Output appeared during grouping; refusing to replace it")
            backup = Path(tempfile.mkdtemp(prefix=f".{output.name}-backup-", dir=output.parent))
            backup.rmdir()
            output.rename(backup)
        try:
            stage.rename(output)
        except Exception:
            if backup is not None:
                backup.rename(output)
            raise
        if backup is not None:
            shutil.rmtree(backup)
    finally:
        if stage.exists():
            shutil.rmtree(stage)
    return {**plan, "output": str(output), "groups": leaves}



def _print_plan(plan):
    runs, conflicts = plan["runs"], plan["conflicts"]
    groups = defaultdict(list)
    for run in runs:
        groups[run["group"]].append(run)
    print(f"{len(runs)} runs | {len(groups)} groups | {len(conflicts)} choices needed | {len(plan['skipped'])} skipped")
    for group, members in sorted(groups.items()):
        windows = {(r["mode"], tuple(r["window_ns"])) for r in members}
        print(f"  {group}: {len(members)} runs, {len(windows)} mode/window combinations")
    for index, conflict in enumerate(conflicts, 1):
        first = conflict["candidates"][0]
        start, end = first["window_ns"]
        print(f"\nChoice {index}: {first['group']} | {first['mode']} | {_number(start)} to {_number(end)} ns | {first['model']}")
        print("  Select                     Batch / run")
        for run in conflict["candidates"]:
            batch, source = Path(run["batch"]), Path(run["source"])
            try:
                label = f"{batch.name}/{source.relative_to(batch).as_posix()}"
            except ValueError:
                label = str(source)
            print(f"  --select {run['id']}  {label}")
        # Keep long shared prefixes out of each candidate row.
        parents = sorted({str(Path(r["batch"]).parent) for r in conflict["candidates"]})
        print("  Batch location: " + "; ".join(parents))
    if conflicts:
        print("\nChoose ONE --select ID per block above and append those options to your command.")
        print("Unique models are included automatically. No output has been written.")
    elif "output" in plan:
        print(f"Saved to: {plan['output']}")
    elif runs:
        print("Ready. Remove --check to write the grouped results.")
    else:
        print("No groupable runs found.")
    if plan["skipped"]:
        print("\nSkipped:")
        for item in plan["skipped"]:
            print(f"  {item['source']}: {item['reason']}")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sources", nargs="+", type=Path, required=True, help="Batch folders or parents to scan recursively")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--select", action="append", default=[], metavar="RUN_ID", help="Choose a run from a duplicate-model conflict; repeat as needed")
    parser.add_argument("--selection", type=Path, help='JSON file containing {"select": ["RUN_ID", ...]}')
    parser.add_argument("--dataset-aliases", type=Path, help="JSON mapping old/new acquisition paths to explicit dataset labels")
    parser.add_argument("--check", action="store_true", help="Show a compact inventory and duplicate choices without writing output")
    parser.add_argument("--json", action="store_true", help="Print full machine-readable JSON instead of the compact summary")
    parser.add_argument("--overwrite", action="store_true", help="Replace only a prior output created by this script")
    parser.add_argument("--no-summary-plots", action="store_true")
    parser.add_argument("--plot-config", type=Path, help="Reporting plot configuration JSON; defaults to config/plots/default.json")
    parser.add_argument("--copy-predictions", action="store_true", help="Copy saved prediction arrays without loading or recomputing them")
    args = parser.parse_args(argv)
    try:
        for path in (args.selection, args.dataset_aliases):
            if path is not None and not path.is_file():
                raise ValueError(f"Configuration file does not exist: {path}")
        configured = _json(args.selection).get("select", []) if args.selection else []
        if not isinstance(configured, list):
            raise ValueError("selection.select must be a list of run-ID strings")
        selected = args.select + configured
        if not all(isinstance(value, str) for value in selected):
            raise ValueError("Selections must be run-ID strings")
        aliases = _json(args.dataset_aliases) if args.dataset_aliases else None
        result = group_results(args.sources, args.output, selected=selected,
                               aliases=aliases, check=True)
        blocked = bool(result["conflicts"] or not result["runs"])
        if not args.check and not blocked:
            result = group_results(args.sources, args.output, selected=selected,
                                   aliases=aliases, overwrite=args.overwrite,
                                   summary_plots=not args.no_summary_plots, copy_predictions=args.copy_predictions,
                                   plot_config=args.plot_config)
        if args.json:
            print(json.dumps(result, indent=2, allow_nan=False))
        else:
            _print_plan(result)
        return 2 if blocked else 0
    except (ValueError, OSError) as exc:
        parser.exit(2, f"Error: {exc}\n")


if __name__ == "__main__":
    raise SystemExit(main())
