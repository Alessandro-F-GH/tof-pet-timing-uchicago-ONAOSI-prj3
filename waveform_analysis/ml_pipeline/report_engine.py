from __future__ import annotations

import csv
import itertools
import json
import logging
import re
import shutil
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from .common import atomic_json, canonical_hash, write_csv
from .config import BatchConfig
from .reporting_config import formulation_style, load_reporting_config, plot_style, rc_params
from .stats import paired_replica_difference

REPORT_BOOTSTRAP_REPLICATES = 5000
REPORT_CONFIDENCE = 0.90
_LOG_TIMESTAMP_FORMAT = "%Y-%m-%d %H:%M:%S,%f"
_REPLICA_START = re.compile(r"^(?P<time>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3}).*\| Replica (?P<replica>\d+)/\d+ \|")
_REPLICA_RESULT = re.compile(r"^(?P<time>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3}).*\| Replica result \| replica=(?P<replica>\d+) \|")


def _finite(value):
    try:
        value = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return value if np.isfinite(value) else float("nan")


def _safe(value):
    text = re.sub(r"[^A-Za-z0-9_.-]+", "-", str(value)).strip("-")
    return text or "value"


def _window_label(start, end, name=None):
    return str(name) if name else f"{start:g}_{end:g}ns"


def _window_key(row):
    return str(row["window"]), float(row["window_start_ns"]), float(row["window_end_ns"])


def _context_key(row):
    return (
        str(row["study"]),
        str(row["dataset_key"]),
        str(row["dataset"]),
        str(row["population_identity"]),
        str(row["sampling_identity"]),
    )


def _report_context_key(row):
    return str(row["study"]), str(row["dataset_key"]), str(row["dataset"])


def _group(rows, key_fn):
    groups = defaultdict(list)
    for row in rows:
        groups[key_fn(row)].append(row)
    return groups


def _read_rows(path):
    with Path(path).open("r", encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def _expand_result_path(path):
    path = Path(path).expanduser().resolve()
    manifest_path = path / "manifest.json"
    if (path / "results.csv").is_file() and manifest_path.is_file():
        return [path]

    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        runs = manifest.get("runs")
        if isinstance(runs, list):
            result = []
            for run in runs:
                child = Path(run["path"])
                child = child if child.is_absolute() else path / child
                if (child / "results.csv").is_file() and (child / "manifest.json").is_file():
                    result.append(child.resolve())
            if result:
                return result

    if path.is_dir():
        discovered = sorted(
            candidate.parent.resolve()
            for candidate in path.rglob("results.csv")
            if (candidate.parent / "manifest.json").is_file()
        )
        if discovered:
            return discovered

    raise FileNotFoundError(f"No study results found in {path}")


def batch_result_dirs(batch: BatchConfig):
    return [Path(config["output_dir"]).resolve() for config in batch.runs]


def replica_wall_times(run_dir):
    path = Path(run_dir) / "study.log"
    if not path.is_file():
        return {}
    pending = {}
    completed = {}
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        match = _REPLICA_START.search(line)
        if match:
            pending[int(match.group("replica"))] = datetime.strptime(match.group("time"), _LOG_TIMESTAMP_FORMAT)
            continue
        match = _REPLICA_RESULT.search(line)
        if not match:
            continue
        replica = int(match.group("replica"))
        start = pending.pop(replica, None)
        if start is None:
            continue
        finish = datetime.strptime(match.group("time"), _LOG_TIMESTAMP_FORMAT)
        duration = (finish - start).total_seconds()
        if duration >= 0:
            completed[replica] = float(duration)
    return completed


def collect_results(paths):
    run_dirs = []
    for path in paths:
        run_dirs.extend(_expand_result_path(path))
    unique = []
    seen = set()
    for path in run_dirs:
        if str(path) not in seen:
            seen.add(str(path))
            unique.append(path)
    records = []
    for run_dir in unique:
        manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
        if int(manifest.get("schema_version", 0)) != 40:
            raise RuntimeError(f"Unsupported result schema in {run_dir}")
        analysis = manifest["analysis"]
        dataset_key = canonical_hash(analysis)
        model = str(manifest["model"])
        formulation = str(manifest["estimator_formulation"]).strip().lower()
        if formulation not in {"shared", "direct"}:
            raise RuntimeError(f"Invalid estimator formulation in {run_dir}: {formulation}")
        mode = str(manifest["mode"])
        window = manifest["window_ns"]
        window_name = manifest.get("window_name")
        study_name = str(manifest.get("study_name") or manifest["name"])
        protocol_identity = str(manifest["analysis_protocol_identity"])
        sampling_identity = str(manifest["sampling_identity"])
        shared = manifest.get("shared_replicas") or {}
        timings = replica_wall_times(run_dir)
        for row in _read_rows(run_dir / "results.csv"):
            if row.get("phase") != "replica":
                continue
            replica_index = int(row["replica_index"])
            seed = int(row["seed"])
            records.append({
                "study": study_name,
                "source_run": str(run_dir),
                "dataset_key": dataset_key,
                "dataset": str(analysis.get("root_file", "")),
                "population_identity": protocol_identity,
                "sampling_identity": sampling_identity,
                "model": model,
                "estimator_formulation": formulation,
                "mode": mode,
                "window": _window_label(float(window["start"]), float(window["end"]), window_name),
                "window_start_ns": float(window["start"]),
                "window_end_ns": float(window["end"]),
                "replica_index": replica_index,
                "seed": seed,
                "candidate_id": str(row["candidate_id"]),
                "shared_replica": str(shared.get(str(replica_index), "")),
                "ctr_ps": _finite(row.get("ctr_ps")),
                "led_ctr_ps": _finite(row.get("uncorrected_ctr_ps")),
                "improvement_ps": _finite(row.get("improvement_ps")),
                "improvement_percent": _finite(row.get("improvement_percent")),
                "rmse_ps": _finite(row.get("rmse_ps")),
                "led_rmse_ps": _finite(row.get("uncorrected_rmse_ps")),
                "rmse_improvement_ps": _finite(row.get("rmse_improvement_ps")),
                "rmse_improvement_percent": _finite(row.get("rmse_improvement_percent")),
                "replica_wall_time_s": _finite(timings.get(replica_index)),
            })
    if not records:
        raise RuntimeError("No replica rows were found in the supplied studies")
    return records


def _mean_std(values):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if not values.size:
        return float("nan"), float("nan"), 0
    return float(np.mean(values)), float(np.std(values, ddof=1)) if values.size > 1 else 0.0, int(values.size)


def _paired_reference_stats(rows, candidate_key, reference_key, seed_payload):
    candidate_mean, candidate_std, n = _mean_std([row[candidate_key] for row in rows])
    reference_mean, reference_std, _ = _mean_std([row[reference_key] for row in rows])
    differences = np.asarray([row[reference_key] - row[candidate_key] for row in rows], dtype=float)
    diff_mean, diff_std, _ = _mean_std(differences)
    percentages = np.asarray([
        100.0 * (row[reference_key] - row[candidate_key]) / row[reference_key]
        if np.isfinite(row[reference_key]) and row[reference_key] != 0.0 else float("nan")
        for row in rows
    ], dtype=float)
    pct_mean, pct_std, _ = _mean_std(percentages)
    reference = np.asarray([row[reference_key] for row in rows], float)
    candidate = np.asarray([row[candidate_key] for row in rows], float)
    finite = np.isfinite(reference) & np.isfinite(candidate)
    if np.any(finite):
        paired = paired_replica_difference(
            reference[finite], candidate[finite],
            seed=int(canonical_hash(seed_payload)[:8], 16),
            n_bootstrap=REPORT_BOOTSTRAP_REPLICATES,
            confidence=REPORT_CONFIDENCE,
        )
        ci_low, ci_high = paired.ci_low, paired.ci_high
    else:
        ci_low = ci_high = float("nan")
    return {
        "candidate_mean": candidate_mean, "candidate_std": candidate_std,
        "reference_mean": reference_mean, "reference_std": reference_std,
        "difference_mean": diff_mean, "difference_std": diff_std,
        "difference_percent_mean": pct_mean, "difference_percent_std": pct_std,
        "ci_low": ci_low, "ci_high": ci_high, "n": n,
    }


def study_summary(records):
    groups = _group(records, lambda row: (*_context_key(row), row["model"], row["estimator_formulation"], row["mode"], *_window_key(row)))
    output = []
    for key, rows in sorted(groups.items()):
        study, dataset_key, dataset, population_identity, sampling_identity, model, formulation, mode, window, start, end = key
        seed_base = {
            "study": study, "dataset": dataset_key, "population": population_identity,
            "sampling": sampling_identity, "model": model, "mode": mode, "window": [start, end],
        }
        ctr = _paired_reference_stats(rows, "ctr_ps", "led_ctr_ps", {**seed_base, "comparison": "led_vs_ml_ctr"})
        rmse = _paired_reference_stats(rows, "rmse_ps", "led_rmse_ps", {**seed_base, "comparison": "led_vs_ml_rmse"})
        time_mean, time_std, n_timed = _mean_std([row["replica_wall_time_s"] for row in rows])
        output.append({
            "study": study, "dataset_key": dataset_key, "dataset": dataset,
            "population_identity": population_identity, "sampling_identity": sampling_identity,
            "model": model, "estimator_formulation": formulation, "mode": mode,
            "window": window, "window_start_ns": start, "window_end_ns": end,
            "n_replicas": ctr["n"], "ctr_mean_ps": ctr["candidate_mean"], "ctr_std_ps": ctr["candidate_std"],
            "led_ctr_mean_ps": ctr["reference_mean"], "led_ctr_std_ps": ctr["reference_std"],
            "paired_led_improvement_mean_ps": ctr["difference_mean"], "paired_led_improvement_std_ps": ctr["difference_std"],
            "paired_led_improvement_ci_low_ps": ctr["ci_low"], "paired_led_improvement_ci_high_ps": ctr["ci_high"],
            "paired_led_improvement_bootstrap_confidence": REPORT_CONFIDENCE,
            "paired_led_improvement_mean_percent": ctr["difference_percent_mean"], "paired_led_improvement_std_percent": ctr["difference_percent_std"],
            "rmse_mean_ps": rmse["candidate_mean"], "rmse_std_ps": rmse["candidate_std"],
            "led_rmse_mean_ps": rmse["reference_mean"], "led_rmse_std_ps": rmse["reference_std"],
            "paired_led_rmse_improvement_mean_ps": rmse["difference_mean"], "paired_led_rmse_improvement_std_ps": rmse["difference_std"],
            "paired_led_rmse_improvement_ci_low_ps": rmse["ci_low"], "paired_led_rmse_improvement_ci_high_ps": rmse["ci_high"],
            "paired_led_rmse_improvement_bootstrap_confidence": REPORT_CONFIDENCE,
            "paired_led_rmse_improvement_mean_percent": rmse["difference_percent_mean"], "paired_led_rmse_improvement_std_percent": rmse["difference_percent_std"],
            "replica_wall_time_mean_s": time_mean, "replica_wall_time_std_s": time_std, "n_timed_replicas": n_timed,
        })
    return output


def _unique_replica_map(rows):
    by_replica = _group(rows, lambda row: row["replica_index"])
    if any(len(values) != 1 for values in by_replica.values()):
        return None
    return {replica: values[0] for replica, values in by_replica.items()}


def _paired_model_metric(map_a, map_b, replicas, metric_key, seed_payload):
    a = np.asarray([map_a[index][metric_key] for index in replicas], float)
    b = np.asarray([map_b[index][metric_key] for index in replicas], float)
    finite = np.isfinite(a) & np.isfinite(b)
    if not np.any(finite):
        return None
    return paired_replica_difference(a[finite], b[finite], seed=int(canonical_hash(seed_payload)[:8], 16), n_bootstrap=REPORT_BOOTSTRAP_REPLICATES, confidence=REPORT_CONFIDENCE)


def paired_model_comparisons(records):
    groups = _group(records, lambda row: (*_context_key(row), row["mode"], *_window_key(row)))
    output = []
    for key, rows in sorted(groups.items()):
        study, dataset_key, dataset, population_identity, sampling_identity, mode, window, start, end = key
        by_model = _group(rows, lambda row: row["model"])
        for model_a, model_b in itertools.combinations(sorted(by_model), 2):
            map_a = _unique_replica_map(by_model[model_a])
            map_b = _unique_replica_map(by_model[model_b])
            if map_a is None or map_b is None:
                continue
            replicas = sorted(set(map_a) & set(map_b))
            if not replicas:
                continue
            seed_base = {"study": study, "dataset": dataset_key, "population": population_identity, "sampling": sampling_identity, "mode": mode, "window": [start, end], "models": [model_a, model_b]}
            ctr = _paired_model_metric(map_a, map_b, replicas, "ctr_ps", {**seed_base, "metric": "ctr"})
            rmse = _paired_model_metric(map_a, map_b, replicas, "rmse_ps", {**seed_base, "metric": "rmse"})
            if ctr is None:
                continue
            output.append({
                "study": study, "dataset_key": dataset_key, "dataset": dataset,
                "population_identity": population_identity, "sampling_identity": sampling_identity,
                "mode": mode, "window": window, "window_start_ns": start, "window_end_ns": end,
                "reference_model": model_a, "reference_formulation": map_a[replicas[0]]["estimator_formulation"],
                "candidate_model": model_b, "candidate_formulation": map_b[replicas[0]]["estimator_formulation"],
                "n_paired_replicas": ctr.n_pairs,
                "reference_ctr_mean_ps": ctr.reference_mean, "candidate_ctr_mean_ps": ctr.candidate_mean,
                "ctr_difference_reference_minus_candidate_ps": ctr.difference_mean, "ctr_difference_std_ps": ctr.difference_std,
                "ctr_paired_bootstrap_ci_low_ps": ctr.ci_low, "ctr_paired_bootstrap_ci_high_ps": ctr.ci_high,
                "paired_bootstrap_confidence": REPORT_CONFIDENCE,
                "reference_rmse_mean_ps": float("nan") if rmse is None else rmse.reference_mean,
                "candidate_rmse_mean_ps": float("nan") if rmse is None else rmse.candidate_mean,
                "rmse_difference_reference_minus_candidate_ps": float("nan") if rmse is None else rmse.difference_mean,
                "rmse_difference_std_ps": float("nan") if rmse is None else rmse.difference_std,
                "rmse_paired_bootstrap_ci_low_ps": float("nan") if rmse is None else rmse.ci_low,
                "rmse_paired_bootstrap_ci_high_ps": float("nan") if rmse is None else rmse.ci_high,
            })
    return output


def _plot_group_key(row):
    return (*_context_key(row), row["mode"], *_window_key(row))


def _mode_group_key(row):
    return (*_report_context_key(row), str(row["mode"]))


def _multiple_report_contexts(rows):
    return len({_report_context_key(row) for row in rows}) > 1


def _context_dir(output_dir, report_context, multiple_contexts):
    path = Path(output_dir)
    if multiple_contexts:
        study, dataset_key, _ = report_context
        path = path / _safe(study) / _safe(dataset_key[:10])
    return path


def _window_output_path(output_dir, strict_context, mode, window, multiple_contexts):
    return _context_dir(output_dir, strict_context[:3], multiple_contexts) / _safe(mode) / f"{_safe(window)}.png"


def _formulation_order(row):
    return 0 if row["estimator_formulation"] == "shared" else 1, row["model"]


def _formulation_handles(reporting, formulations):
    handles = []
    for formulation in ("shared", "direct"):
        if formulation not in formulations:
            continue
        style = formulation_style(reporting, formulation)
        handles.append(Line2D([0], [0], marker=style["marker"], color="none", markerfacecolor=style["color"], markeredgecolor=style["color"], label=style["label"], markersize=7))
    return handles


def _apply_axes_style(ax, reporting, *, grid_axis="y"):
    style = reporting["global"]
    if bool(style.get("grid", True)):
        ax.grid(True, axis=grid_axis, alpha=float(style["grid_alpha"]), linestyle=style["grid_linestyle"])
        ax.set_axisbelow(True)


def _save(fig, path, reporting, *, tight=True):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if tight:
        fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def _reference_value(rows, mean_key, std_key):
    means = np.asarray([row[mean_key] for row in rows if np.isfinite(row[mean_key])], float)
    stds = np.asarray([row[std_key] for row in rows if np.isfinite(row[std_key])], float)
    if not means.size:
        return float("nan"), float("nan")
    if not np.allclose(means, means[0], rtol=1e-7, atol=1e-9):
        raise RuntimeError(f"Inconsistent reference values across models: {means.tolist()}")
    if stds.size and not np.allclose(stds, stds[0], rtol=1e-7, atol=1e-9):
        raise RuntimeError(f"Inconsistent reference spread across models: {stds.tolist()}")
    return float(means[0]), float(stds[0]) if stds.size else float("nan")


def plot_metric_comparisons(summary, output_dir, *, metric, reporting=None):
    reporting = load_reporting_config() if reporting is None else reporting
    if metric == "ctr":
        mean_key, std_key, ref_mean_key, ref_std_key, ylabel, title_metric = "ctr_mean_ps", "ctr_std_ps", "led_ctr_mean_ps", "led_ctr_std_ps", "Blind CTR [ps]", "CTR"
    elif metric == "rmse":
        mean_key, std_key, ref_mean_key, ref_std_key, ylabel, title_metric = "rmse_mean_ps", "rmse_std_ps", "led_rmse_mean_ps", "led_rmse_std_ps", "Blind RMSE [ps]", "RMSE"
    else:
        raise ValueError(f"Unsupported metric: {metric}")
    style = plot_style(reporting, "comparison")
    groups = _group(summary, _plot_group_key)
    multiple_contexts = _multiple_report_contexts(summary)
    outputs = []
    for key, rows in sorted(groups.items()):
        context = key[:5]
        mode, window, start, end = key[5:]
        rows = sorted(rows, key=_formulation_order)
        categories = ["LED", *[row["model"] for row in rows]]
        width = max(float(style["figure_width_min"]), float(style["width_per_category"]) * len(categories))
        fig, ax = plt.subplots(figsize=(width, float(style["figure_height"])))
        reference = reporting["reference"]
        ref_mean, ref_std = _reference_value(rows, ref_mean_key, ref_std_key)
        if np.isfinite(ref_mean):
            if metric == "ctr":
                ax.axhline(
                    ref_mean,
                    color=reference["color"],
                    linestyle=reference["linestyle"],
                    linewidth=float(reporting["global"]["line_width"]),
                )
            ax.errorbar([0], [ref_mean], yerr=[ref_std] if np.isfinite(ref_std) else None, fmt=reference["marker"], color=reference["color"], capsize=float(reporting["global"]["error_capsize"]), markersize=float(style["marker_size"]))
        for index, row in enumerate(rows, start=1):
            family = formulation_style(reporting, row["estimator_formulation"])
            ax.errorbar([index], [row[mean_key]], yerr=[row[std_key]], fmt=family["marker"], color=family["color"], capsize=float(reporting["global"]["error_capsize"]), markersize=float(style["marker_size"]))
        ax.set_xticks(range(len(categories)), categories, rotation=float(style["x_label_rotation"]), ha="right")
        ax.set_ylabel(ylabel)
        ax.set_title(f"{title_metric} model comparison | {mode} | {window} [{start:g}, {end:g}] ns")
        formulations = {row["estimator_formulation"] for row in rows}
        reference_handle = Line2D(
            [0], [0],
            marker=reference["marker"],
            color=reference["color"] if metric == "ctr" else "none",
            linestyle=reference["linestyle"] if metric == "ctr" else "None",
            markerfacecolor=reference["color"],
            markeredgecolor=reference["color"],
            label=reference["label"],
            markersize=7,
        )
        ax.legend(handles=[reference_handle, *_formulation_handles(reporting, formulations)])
        _apply_axes_style(ax, reporting)
        outputs.append(_save(fig, _window_output_path(output_dir, context, mode, window, multiple_contexts), reporting))
    return outputs


def plot_ctr_comparisons(summary, output_dir, reporting=None):
    return plot_metric_comparisons(summary, output_dir, metric="ctr", reporting=reporting)


def plot_rmse_comparisons(summary, output_dir, reporting=None):
    return plot_metric_comparisons(summary, output_dir, metric="rmse", reporting=reporting)


def plot_led_improvements(summary, output_dir, reporting=None):
    reporting = load_reporting_config() if reporting is None else reporting
    style = plot_style(reporting, "comparison")
    groups = _group(summary, _plot_group_key)
    multiple_contexts = _multiple_report_contexts(summary)
    outputs = []
    for key, rows in sorted(groups.items()):
        context = key[:5]
        mode, window, start, end = key[5:]
        rows = sorted(rows, key=_formulation_order)
        width = max(float(style["figure_width_min"]), float(style["width_per_category"]) * len(rows))
        fig, ax = plt.subplots(figsize=(width, float(style["figure_height"])))
        for index, row in enumerate(rows):
            family = formulation_style(reporting, row["estimator_formulation"])
            ax.errorbar([index], [row["paired_led_improvement_mean_ps"]], yerr=[row["paired_led_improvement_std_ps"]], fmt=family["marker"], color=family["color"], capsize=float(reporting["global"]["error_capsize"]), markersize=float(style["marker_size"]))
        reference = reporting["reference"]
        ax.axhline(0.0, color=reference["color"], linestyle=reference["linestyle"])
        ax.set_xticks(range(len(rows)), [row["model"] for row in rows], rotation=float(style["x_label_rotation"]), ha="right")
        ax.set_ylabel("Paired CTR improvement, LED - ML [ps]")
        ax.set_title(f"Paired LED improvement | {mode} | {window} [{start:g}, {end:g}] ns")
        ax.legend(handles=_formulation_handles(reporting, {row["estimator_formulation"] for row in rows}))
        _apply_axes_style(ax, reporting)
        outputs.append(_save(fig, _window_output_path(output_dir, context, mode, window, multiple_contexts), reporting))
    return outputs


def plot_rmse_ctr_correlation(summary, output_dir, reporting=None):
    reporting = load_reporting_config() if reporting is None else reporting
    style = plot_style(reporting, "scatter")
    groups = _group(summary, _plot_group_key)
    multiple_contexts = _multiple_report_contexts(summary)
    outputs = []
    for key, rows in sorted(groups.items()):
        context = key[:5]
        mode, window, start, end = key[5:]
        points = [row for row in sorted(rows, key=_formulation_order) if np.isfinite(row["ctr_mean_ps"]) and np.isfinite(row["rmse_mean_ps"])]
        if not points:
            continue
        fig, ax = plt.subplots(figsize=tuple(style["figsize"]))
        for row in points:
            family = formulation_style(reporting, row["estimator_formulation"])
            ax.scatter([row["ctr_mean_ps"]], [row["rmse_mean_ps"]], s=float(style["marker_size"]), marker=family["marker"], color=family["color"])
            ax.annotate(row["model"], (row["ctr_mean_ps"], row["rmse_mean_ps"]), xytext=tuple(style["annotation_offset"]), textcoords="offset points", fontsize=float(reporting["global"]["annotation_size"]))
        ctr_values = np.asarray([row["ctr_mean_ps"] for row in points], float)
        rmse_values = np.asarray([row["rmse_mean_ps"] for row in points], float)
        correlation = float("nan")
        if len(points) >= 2 and np.std(ctr_values) > 0 and np.std(rmse_values) > 0:
            correlation = float(np.corrcoef(ctr_values, rmse_values)[0, 1])
        label = f"Pearson r={correlation:.3f}" if np.isfinite(correlation) else "Pearson r=n/a"
        ax.set_xlabel("Blind CTR [ps]")
        ax.set_ylabel("Blind RMSE [ps]")
        ax.set_title(f"RMSE vs CTR | {mode} | {window} [{start:g}, {end:g}] ns\n{label}")
        ax.legend(handles=_formulation_handles(reporting, {row["estimator_formulation"] for row in points}))
        _apply_axes_style(ax, reporting, grid_axis="both")
        outputs.append(_save(fig, _window_output_path(output_dir, context, mode, window, multiple_contexts), reporting))
    return outputs


def _unique_reference_values(rows, metric_key):
    unique = {}
    for row in rows:
        value = float(row[metric_key])
        if np.isfinite(value):
            unique[_window_key(row)] = value
    return np.asarray(list(unique.values()), dtype=float)


def plot_window_model_comparisons(summary, output_dir, *, metric, reporting=None):
    reporting = load_reporting_config() if reporting is None else reporting
    if metric == "ctr":
        mean_key, std_key, led_key, ylabel, title_metric = "ctr_mean_ps", "ctr_std_ps", "led_ctr_mean_ps", "Blind CTR [ps]", "CTR"
    elif metric == "rmse":
        mean_key, std_key, led_key, ylabel, title_metric = "rmse_mean_ps", "rmse_std_ps", "led_rmse_mean_ps", "Blind RMSE [ps]", "RMSE"
    else:
        raise ValueError(f"Unsupported metric: {metric}")
    style = plot_style(reporting, "bar")
    groups = _group(summary, _mode_group_key)
    multiple_contexts = _multiple_report_contexts(summary)
    outputs = []
    for key, rows in sorted(groups.items()):
        report_context = key[:3]
        mode = key[3]
        models = sorted({row["model"] for row in rows}, key=lambda model: _formulation_order(next(row for row in rows if row["model"] == model)))
        windows = sorted({_window_key(row) for row in rows}, key=lambda item: (item[1], item[2], item[0]))
        if not models or len(windows) < 2:
            continue
        width = max(float(style["figure_width_min"]), float(style["width_per_category"]) * len(models))
        fig, ax = plt.subplots(figsize=(width, float(style["figure_height"])))
        x = np.arange(len(models), dtype=float)
        bar_width = float(style["group_width"]) / len(windows)
        hatches = list(style["window_hatches"])
        by_model_window = {(row["model"], _window_key(row)): row for row in rows}
        model_formulation = {row["model"]: row["estimator_formulation"] for row in rows}
        for window_index, window_key in enumerate(windows):
            offset = (window_index - (len(windows) - 1) / 2.0) * bar_width
            for model_index, model in enumerate(models):
                row = by_model_window.get((model, window_key))
                if row is None or not np.isfinite(row[mean_key]):
                    continue
                family = formulation_style(reporting, model_formulation[model])
                xpos = x[model_index] + offset
                ax.bar(
                    xpos,
                    row[mean_key],
                    width=bar_width,
                    yerr=row[std_key] if np.isfinite(row[std_key]) else None,
                    capsize=float(reporting["global"]["error_capsize"]),
                    color=family["color"],
                    hatch=hatches[window_index % len(hatches)],
                    edgecolor=style["edge_color"],
                    linewidth=float(style["edge_line_width"]),
                )
                if metric == "ctr":
                    error = row[std_key] if np.isfinite(row[std_key]) else 0.0
                    ax.annotate(
                        f"{row[mean_key]:.0f} ps",
                        (xpos, row[mean_key] + error),
                        xytext=(0, 4),
                        textcoords="offset points",
                        ha="center",
                        va="bottom",
                        fontsize=float(reporting["global"]["annotation_size"]),
                    )
        led_values = _unique_reference_values(rows, led_key)
        reference = reporting["reference"]
        if led_values.size:
            ax.axhline(float(np.median(led_values)), color=reference["color"], linestyle=reference["linestyle"])
        ax.set_xticks(x, models, rotation=25, ha="right")
        ax.set_ylabel(ylabel)
        ax.set_title(f"{title_metric} across waveform windows | {mode}")
        handles = [*_formulation_handles(reporting, set(model_formulation.values())), *[
            Patch(facecolor="white", edgecolor=style["edge_color"], hatch=hatches[index % len(hatches)], label=f"{window[0]} [{window[1]:g}, {window[2]:g}] ns")
            for index, window in enumerate(windows)
        ]]
        if led_values.size:
            handles.append(Line2D([0], [0], color=reference["color"], linestyle=reference["linestyle"], label="Median LED reference"))
        ax.legend(handles=handles)
        _apply_axes_style(ax, reporting)
        path = _context_dir(output_dir, report_context, multiple_contexts) / _safe(mode) / f"{metric}.png"
        outputs.append(_save(fig, path, reporting))
    return outputs


def best_by_formulation(summary):
    groups = _group(summary, _plot_group_key)
    output = []
    for key, rows in sorted(groups.items()):
        context = key[:5]
        mode, window, start, end = key[5:]
        selected = {}
        for formulation in ("shared", "direct"):
            candidates = [row for row in rows if row["estimator_formulation"] == formulation and np.isfinite(row["ctr_mean_ps"])]
            if candidates:
                selected[formulation] = min(candidates, key=lambda row: row["ctr_mean_ps"])
        base = {
            "study": context[0], "dataset_key": context[1], "dataset": context[2],
            "population_identity": context[3], "sampling_identity": context[4],
            "mode": mode, "window": window, "window_start_ns": start, "window_end_ns": end,
        }
        for formulation in ("shared", "direct"):
            row = selected.get(formulation)
            base.update({
                f"best_{formulation}_model": "" if row is None else row["model"],
                f"best_{formulation}_ctr_mean_ps": float("nan") if row is None else row["ctr_mean_ps"],
                f"best_{formulation}_ctr_std_ps": float("nan") if row is None else row["ctr_std_ps"],
                f"best_{formulation}_rmse_mean_ps": float("nan") if row is None else row["rmse_mean_ps"],
                f"best_{formulation}_replica_wall_time_mean_s": float("nan") if row is None else row["replica_wall_time_mean_s"],
            })
        shared_ctr, direct_ctr = base["best_shared_ctr_mean_ps"], base["best_direct_ctr_mean_ps"]
        base["direct_minus_shared_ctr_ps"] = float(direct_ctr - shared_ctr) if np.isfinite(shared_ctr) and np.isfinite(direct_ctr) else float("nan")
        output.append(base)
    return output


def plot_best_models_by_mode(summary, output_dir, reporting=None):
    reporting = load_reporting_config() if reporting is None else reporting
    style = plot_style(reporting, "scatter")
    groups = _group(summary, _mode_group_key)
    multiple_contexts = _multiple_report_contexts(summary)
    outputs = []
    for key, rows in sorted(groups.items()):
        report_context = key[:3]
        mode = key[3]
        points = [row for row in rows if np.isfinite(row["ctr_mean_ps"])]
        if not points:
            continue
        windows = sorted({_window_key(row) for row in points}, key=lambda item: (item[1], item[2], item[0]))
        models = sorted(
            {row["model"] for row in points},
            key=lambda model: _formulation_order(next(row for row in points if row["model"] == model)),
        )
        window_positions = {window: index for index, window in enumerate(windows)}
        if len(models) == 1:
            model_offsets = {models[0]: 0.0}
        else:
            model_offsets = dict(zip(models, np.linspace(-0.28, 0.28, len(models))))
        fig, ax = plt.subplots(figsize=tuple(style["figsize"]))
        for row in sorted(points, key=lambda item: (_window_key(item)[1], _window_key(item)[2], _formulation_order(item))):
            family = formulation_style(reporting, row["estimator_formulation"])
            x = window_positions[_window_key(row)] + model_offsets[row["model"]]
            ax.scatter([x], [row["ctr_mean_ps"]], s=float(style["marker_size"]), marker=family["marker"], color=family["color"])
            if np.isfinite(row["ctr_std_ps"]):
                ax.errorbar(
                    [x], [row["ctr_mean_ps"]], yerr=[row["ctr_std_ps"]], fmt="none",
                    ecolor=family["color"], capsize=float(reporting["global"]["error_capsize"]),
                    linewidth=float(reporting["global"]["line_width"]),
                )
            ax.annotate(
                row["model"], (x, row["ctr_mean_ps"]),
                xytext=tuple(style["annotation_offset"]), textcoords="offset points",
                fontsize=float(reporting["global"]["annotation_size"]),
            )
        labels = [f"{window[0]}\n[{window[1]:g}, {window[2]:g}] ns" for window in windows]
        ax.set_xticks(range(len(windows)), labels)
        ax.set_xlabel("Waveform window")
        ax.set_ylabel("Blind CTR [ps]")
        ax.set_title(f"Model and waveform-window comparison | {mode}")
        ax.legend(handles=_formulation_handles(reporting, {row["estimator_formulation"] for row in points}))
        _apply_axes_style(ax, reporting)
        path = _context_dir(output_dir, report_context, multiple_contexts) / f"{_safe(mode)}.png"
        outputs.append(_save(fig, path, reporting))
    return outputs


def pareto_frontier(points, *, x_key, y_key):
    finite = [index for index, point in enumerate(points) if np.isfinite(point[x_key]) and np.isfinite(point[y_key])]
    frontier = []
    for index in finite:
        x, y = float(points[index][x_key]), float(points[index][y_key])
        dominated = any(
            other != index
            and float(points[other][x_key]) <= x
            and float(points[other][y_key]) <= y
            and (float(points[other][x_key]) < x or float(points[other][y_key]) < y)
            for other in finite
        )
        if not dominated:
            frontier.append(index)
    return frontier


def plot_ctr_vs_time(summary, output_dir, reporting=None):
    reporting = load_reporting_config() if reporting is None else reporting
    style = plot_style(reporting, "pareto")
    scatter_style = plot_style(reporting, "scatter")
    groups = _group(summary, _plot_group_key)
    multiple_contexts = _multiple_report_contexts(summary)
    outputs = []
    for key, rows in sorted(groups.items()):
        context = key[:5]
        mode, window, start, end = key[5:]
        points = [row for row in sorted(rows, key=_formulation_order) if np.isfinite(row["ctr_mean_ps"]) and np.isfinite(row["replica_wall_time_mean_s"]) and row["replica_wall_time_mean_s"] > 0]
        if not points:
            continue
        frontier_indices = pareto_frontier(points, x_key="replica_wall_time_mean_s", y_key="ctr_mean_ps")
        fig, ax = plt.subplots(figsize=tuple(style["figsize"]))
        for row in points:
            family = formulation_style(reporting, row["estimator_formulation"])
            ax.scatter([row["replica_wall_time_mean_s"]], [row["ctr_mean_ps"]], s=float(style["marker_size"]), marker=family["marker"], color=family["color"])
            ax.annotate(row["model"], (row["replica_wall_time_mean_s"], row["ctr_mean_ps"]), xytext=tuple(scatter_style["annotation_offset"]), textcoords="offset points", fontsize=float(reporting["global"]["annotation_size"]))
        if frontier_indices:
            frontier = sorted([points[index] for index in frontier_indices], key=lambda row: row["replica_wall_time_mean_s"])
            ax.plot([row["replica_wall_time_mean_s"] for row in frontier], [row["ctr_mean_ps"] for row in frontier], color=style["frontier_line_color"], linestyle=style["frontier_line_style"], linewidth=float(style["frontier_line_width"]))
        ax.set_xscale("log")
        ax.set_xlabel("Mean replica wall time [s] (log scale)")
        ax.set_ylabel("Blind CTR [ps]")
        ax.set_title(f"CTR vs computation time | {mode} | {window} [{start:g}, {end:g}] ns")
        handles = _formulation_handles(reporting, {row["estimator_formulation"] for row in points})
        handles.append(Line2D([0], [0], color=style["frontier_line_color"], linestyle=style["frontier_line_style"], linewidth=float(style["frontier_line_width"]), label="Pareto frontier"))
        ax.legend(handles=handles)
        _apply_axes_style(ax, reporting, grid_axis="both")
        outputs.append(_save(fig, _window_output_path(output_dir, context, mode, window, multiple_contexts), reporting))
    return outputs


def _load_model_output(record, cache):
    key = record["source_run"], record["seed"], record["candidate_id"]
    if key in cache:
        return cache[key]
    residual_path = Path(record["source_run"]) / "blind_residuals" / f"seed_{int(record['seed'])}_{record['candidate_id']}.npz"
    if not residual_path.is_file() or not record["shared_replica"]:
        return None
    with np.load(residual_path) as data:
        corrected = np.asarray(data["corrected_ps"], float)
    reference_path = Path(record["shared_replica"]) / "blind_reference.npz"
    if not reference_path.is_file():
        return None
    with np.load(reference_path) as data:
        event_index = np.asarray(data["event_index"], np.int64)
        led = np.asarray(data["led_ps"], float)
    if corrected.shape != led.shape or event_index.size != led.size:
        raise RuntimeError(f"Blind output/reference size mismatch: {residual_path}")
    cache[key] = event_index, np.asarray(led - corrected, float)
    return cache[key]


def _replica_output_correlation(a, b):
    event_a, out_a = a
    event_b, out_b = b
    common, ia, ib = np.intersect1d(event_a, event_b, assume_unique=False, return_indices=True)
    if common.size < 2:
        return float("nan")
    x, y = np.asarray(out_a[ia], float), np.asarray(out_b[ib], float)
    finite = np.isfinite(x) & np.isfinite(y)
    x, y = x[finite], y[finite]
    if x.size < 2 or np.std(x) == 0 or np.std(y) == 0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def _fisher_mean(values):
    values = np.asarray(values, float)
    values = values[np.isfinite(values)]
    if not values.size:
        return float("nan")
    return float(np.tanh(np.mean(np.arctanh(np.clip(values, -0.999999, 0.999999)))))


def _write_matrix(path, labels, matrix):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["model", *labels])
        for label, row in zip(labels, matrix):
            writer.writerow([label, *row])
    return path


def model_output_correlations(records, output_dir, plot_dir=None, reporting=None):
    reporting = load_reporting_config() if reporting is None else reporting
    table_dir = Path(output_dir)
    plot_dir = table_dir if plot_dir is None else Path(plot_dir)
    groups = _group(records, _plot_group_key)
    multiple_contexts = _multiple_report_contexts(records)
    style = plot_style(reporting, "correlation_heatmap")
    cache, long_rows, outputs = {}, [], []
    for key, rows in sorted(groups.items()):
        context = key[:5]
        mode, window, start, end = key[5:]
        by_model = _group(rows, lambda row: row["model"])
        model_formulation = {model: values[0]["estimator_formulation"] for model, values in by_model.items()}
        models = sorted(by_model, key=lambda model: (0 if model_formulation[model] == "shared" else 1, model))
        if len(models) < 2:
            continue
        matrix = np.eye(len(models), dtype=float)
        counts = np.zeros((len(models), len(models)), dtype=int)
        for i, model_a in enumerate(models):
            counts[i, i] = len(by_model[model_a])
            map_a = _unique_replica_map(by_model[model_a])
            if map_a is None:
                continue
            for j in range(i + 1, len(models)):
                model_b = models[j]
                map_b = _unique_replica_map(by_model[model_b])
                if map_b is None:
                    continue
                replica_rs = []
                for replica_index in sorted(set(map_a) & set(map_b)):
                    output_a = _load_model_output(map_a[replica_index], cache)
                    output_b = _load_model_output(map_b[replica_index], cache)
                    if output_a is None or output_b is None:
                        continue
                    value = _replica_output_correlation(output_a, output_b)
                    if np.isfinite(value):
                        replica_rs.append(value)
                value = _fisher_mean(replica_rs)
                matrix[i, j] = matrix[j, i] = value
                counts[i, j] = counts[j, i] = len(replica_rs)
                long_rows.append({
                    "study": context[0], "dataset_key": context[1], "population_identity": context[3], "sampling_identity": context[4],
                    "mode": mode, "window": window, "window_start_ns": start, "window_end_ns": end,
                    "model_a": model_a, "formulation_a": model_formulation[model_a], "model_b": model_b, "formulation_b": model_formulation[model_b],
                    "pearson_r_fisher_mean": value, "n_paired_replicas": len(replica_rs),
                })
        table_base = _context_dir(table_dir, context[:3], multiple_contexts) / _safe(mode)
        plot_base = _context_dir(plot_dir, context[:3], multiple_contexts) / _safe(mode)
        matrix_path = _write_matrix(table_base / f"{_safe(window)}.csv", models, matrix)
        count_path = _write_matrix(table_base / f"{_safe(window)}_n.csv", models, counts)
        min_width, min_height = style["min_figsize"]
        per_width, per_height = style["per_model"]
        fig, ax = plt.subplots(figsize=(max(float(min_width), float(per_width) * len(models)), max(float(min_height), float(per_height) * len(models))), constrained_layout=True)
        image = ax.imshow(matrix, vmin=float(style["vmin"]), vmax=float(style["vmax"]), cmap=style["cmap"])
        ax.set_xticks(range(len(models)), models, rotation=float(style["x_label_rotation"]), ha="right")
        ax.set_yticks(range(len(models)), models)
        ax.set_title(f"Model output correlation | {mode} | {window}")
        colorbar = fig.colorbar(image, ax=ax, fraction=float(style["colorbar_fraction"]), pad=float(style["colorbar_pad"]), shrink=float(style["colorbar_shrink"]))
        colorbar.set_label("Pearson r (Fisher-z mean across replicas)")
        for i in range(len(models)):
            for j in range(len(models)):
                if np.isfinite(matrix[i, j]):
                    ax.text(j, i, f"{matrix[i, j]:.2f}", ha="center", va="center", fontsize=float(style["cell_text_size"]))
        shared_count = sum(model_formulation[model] == "shared" for model in models)
        if 0 < shared_count < len(models):
            boundary = shared_count - 0.5
            ax.axvline(boundary, color=style["separator_color"], linewidth=float(style["separator_line_width"]))
            ax.axhline(boundary, color=style["separator_color"], linewidth=float(style["separator_line_width"]))
        plot_path = _save(fig, plot_base / f"{_safe(window)}.png", reporting, tight=False)
        outputs.extend([matrix_path, count_path, plot_path])
    if long_rows:
        long_path = table_dir / "model_output_correlations.csv"
        write_csv(long_path, long_rows)
        outputs.append(long_path)
    return outputs, long_rows


def _copy_preprocessing_diagnostics(source_dir, destination_dir):
    source = Path(source_dir).expanduser().resolve()
    destination = Path(destination_dir).resolve()
    if not source.is_dir():
        raise FileNotFoundError(f"Preprocessing folder not found: {source}")
    if destination.exists():
        shutil.rmtree(destination)
    shutil.copytree(source, destination)
    return destination


def _prepare_report_layout(output_dir):
    root = Path(output_dir).expanduser().resolve()
    for name in ("tables", "plots"):
        path = root / name
        if path.exists():
            shutil.rmtree(path)
    layout = {
        "root": root,
        "tables": root / "tables",
        "correlation_tables": root / "tables" / "correlations",
        "ctr_plots": root / "plots" / "ctr",
        "rmse_plots": root / "plots" / "rmse",
        "led_improvement_plots": root / "plots" / "led_improvement",
        "correlation_plots": root / "plots" / "correlations",
        "rmse_ctr_plots": root / "plots" / "rmse_vs_ctr",
        "ctr_time_plots": root / "plots" / "ctr_vs_time",
        "window_plots": root / "plots" / "window_comparison",
        "best_model_plots": root / "plots" / "best_model",
        "preprocessing": root / "preprocessing",
    }
    for key, path in layout.items():
        if key != "root":
            path.mkdir(parents=True, exist_ok=True)
    root.mkdir(parents=True, exist_ok=True)
    return layout


def generate_report(paths, output_dir, *, preprocessing_dir=None, logger=None, report_config=None):
    log = logger or logging.getLogger("waveform-report")
    reporting = load_reporting_config(report_config)
    layout = _prepare_report_layout(output_dir)
    root = layout["root"]
    preprocessing_source = None
    if preprocessing_dir is not None:
        preprocessing_source = Path(preprocessing_dir).expanduser().resolve()
        _copy_preprocessing_diagnostics(preprocessing_source, layout["preprocessing"])

    records = collect_results(paths)
    summary = study_summary(records)
    comparisons = paired_model_comparisons(records)
    best_rows = best_by_formulation(summary)
    summary_path = layout["tables"] / "study_summary.csv"
    write_csv(summary_path, summary)
    outputs = [summary_path]
    if comparisons:
        comparison_path = layout["tables"] / "paired_model_comparisons.csv"
        write_csv(comparison_path, comparisons)
        outputs.append(comparison_path)
    if best_rows:
        best_path = layout["tables"] / "best_by_formulation.csv"
        write_csv(best_path, best_rows)
        outputs.append(best_path)
    resolved_style_path = root / "reporting_config_resolved.json"
    atomic_json(resolved_style_path, reporting)
    outputs.append(resolved_style_path)
    with plt.rc_context(rc_params(reporting)):
        outputs.extend(plot_ctr_comparisons(summary, layout["ctr_plots"], reporting))
        outputs.extend(plot_rmse_comparisons(summary, layout["rmse_plots"], reporting))
        outputs.extend(plot_led_improvements(summary, layout["led_improvement_plots"], reporting))
        outputs.extend(plot_rmse_ctr_correlation(summary, layout["rmse_ctr_plots"], reporting))
        outputs.extend(plot_ctr_vs_time(summary, layout["ctr_time_plots"], reporting))
        outputs.extend(plot_window_model_comparisons(summary, layout["window_plots"], metric="ctr", reporting=reporting))
        outputs.extend(plot_window_model_comparisons(summary, layout["window_plots"], metric="rmse", reporting=reporting))
        outputs.extend(plot_best_models_by_mode(summary, layout["best_model_plots"], reporting))
        correlation_outputs, correlation_rows = model_output_correlations(records, layout["correlation_tables"], plot_dir=layout["correlation_plots"], reporting=reporting)
        outputs.extend(correlation_outputs)
    manifest = {
        "schema_version": 6,
        "sources": sorted({row["source_run"] for row in records}),
        "preprocessing_source": None if preprocessing_source is None else str(preprocessing_source),
        "n_replica_rows": len(records), "n_summary_rows": len(summary),
        "n_paired_model_comparisons": len(comparisons), "n_model_output_correlations": len(correlation_rows),
        "statistical_unit": "replica", "fit_bootstrap": False, "paired_bootstrap_unit": "replica",
        "pairing_rule": "same study + dataset + analysis protocol + sampling identity + mode + window + replica index",
        "window_comparison_grouping": "same study + dataset + mode; population and sampling identities may differ because they are window-specific",
        "mode_pooling": False, "formulation_field": "estimator_formulation", "formulation_classes": ["shared", "direct"],
        "timing_metric": "replica wall time parsed from study.log between replica start and Replica result",
        "timing_scope": "fit + blind prediction/evaluation + diagnostics + residual/model serialization performed before Replica result",
        "pareto_objectives": ["minimize mean replica wall time", "minimize blind CTR"],
        "reporting_config": "reporting_config_resolved.json",
        "layout": {key: str(path.relative_to(root)) for key, path in layout.items() if key != "root"},
        "plots_and_tables": [str(Path(path).resolve().relative_to(root)) for path in outputs],
    }
    atomic_json(root / "manifest.json", manifest)
    log.info("Report complete | sources=%d | summary=%d | paired=%d | output-correlations=%d | %s", len(manifest["sources"]), len(summary), len(comparisons), len(correlation_rows), root)
    return root
