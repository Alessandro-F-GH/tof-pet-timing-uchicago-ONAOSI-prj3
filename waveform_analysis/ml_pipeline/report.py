from __future__ import annotations

import csv
import itertools
import json
import logging
import re
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from .common import atomic_json, canonical_hash, write_csv
from .config import BatchConfig
from .stats import paired_replica_difference

REPORT_BOOTSTRAP_REPLICATES = 5000
REPORT_CONFIDENCE = 0.90


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
                if (child / "results.csv").is_file():
                    result.append(child.resolve())
            if result:
                return result
    raise FileNotFoundError(f"No study results found in {path}")


def batch_result_dirs(batch: BatchConfig):
    return [Path(cfg["output_dir"]).resolve() for cfg in batch.runs]


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
        analysis = manifest.get("analysis") or {}
        dataset_key = canonical_hash(analysis)
        model = str(manifest["model"])
        mode = str(manifest["mode"])
        window = manifest["window_ns"]
        window_name = manifest.get("window_name")
        study_name = str(manifest.get("study_name") or manifest.get("name") or run_dir.parent.name)
        population_identity = str(manifest.get("analysis_population_identity", ""))
        for row in _read_rows(run_dir / "results.csv"):
            if row.get("stage") != "blind":
                continue
            records.append({
                "study": study_name,
                "source_run": str(run_dir),
                "dataset_key": dataset_key,
                "dataset": str(analysis.get("root_file", "")),
                "population_identity": population_identity or str(row.get("population_identity", "")),
                "model": model,
                "mode": mode,
                "window": _window_label(float(window["start"]), float(window["end"]), window_name),
                "window_start_ns": float(window["start"]),
                "window_end_ns": float(window["end"]),
                "seed": int(row["seed"]),
                "ctr_ps": _finite(row.get("ctr_ps")),
                "led_ctr_ps": _finite(row.get("uncorrected_ctr_ps")),
                "improvement_ps": _finite(row.get("improvement_ps")),
                "improvement_percent": _finite(row.get("improvement_percent")),
                "rmse_ps": _finite(row.get("rmse_ps")),
                "led_rmse_ps": _finite(row.get("uncorrected_rmse_ps")),
                "rmse_improvement_ps": _finite(row.get("rmse_improvement_ps")),
            })
    if not records:
        raise RuntimeError("No blind replica rows were found in the supplied studies")
    return records


def _mean_std(values):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if not values.size:
        return float("nan"), float("nan"), 0
    return float(np.mean(values)), float(np.std(values, ddof=1)) if values.size > 1 else 0.0, int(values.size)


def study_summary(records):
    groups = defaultdict(list)
    for row in records:
        key = (
            row["study"], row["dataset_key"], row["dataset"], row["population_identity"],
            row["model"], row["mode"], row["window"], row["window_start_ns"], row["window_end_ns"],
        )
        groups[key].append(row)
    output = []
    for key, rows in sorted(groups.items()):
        study, dataset_key, dataset, population_identity, model, mode, window, start, end = key
        ctr_mean, ctr_std, n = _mean_std([r["ctr_ps"] for r in rows])
        led_mean, led_std, _ = _mean_std([r["led_ctr_ps"] for r in rows])
        imp_mean, imp_std, _ = _mean_std([r["improvement_ps"] for r in rows])
        valid_pairs = [r for r in rows if np.isfinite(r["led_ctr_ps"]) and np.isfinite(r["ctr_ps"])]
        if valid_pairs:
            paired = paired_replica_difference(
                np.asarray([r["led_ctr_ps"] for r in valid_pairs], float),
                np.asarray([r["ctr_ps"] for r in valid_pairs], float),
                seed=int(canonical_hash({"study": study, "dataset": dataset_key, "population": population_identity, "model": model, "mode": mode, "window": [start, end], "comparison": "led_vs_ml"})[:8], 16),
                n_bootstrap=REPORT_BOOTSTRAP_REPLICATES,
                confidence=REPORT_CONFIDENCE,
            )
            imp_ci_low, imp_ci_high = paired.ci_low, paired.ci_high
        else:
            imp_ci_low = imp_ci_high = float("nan")
        imp_pct_mean, imp_pct_std, _ = _mean_std([r["improvement_percent"] for r in rows])
        rmse_mean, rmse_std, _ = _mean_std([r["rmse_ps"] for r in rows])
        output.append({
            "study": study, "dataset_key": dataset_key, "dataset": dataset,
            "population_identity": population_identity, "model": model, "mode": mode, "window": window,
            "window_start_ns": start, "window_end_ns": end, "n_replicas": n,
            "ctr_mean_ps": ctr_mean, "ctr_std_ps": ctr_std,
            "led_ctr_mean_ps": led_mean, "led_ctr_std_ps": led_std,
            "paired_led_improvement_mean_ps": imp_mean, "paired_led_improvement_std_ps": imp_std,
            "paired_led_improvement_ci_low_ps": imp_ci_low, "paired_led_improvement_ci_high_ps": imp_ci_high,
            "paired_led_improvement_bootstrap_confidence": REPORT_CONFIDENCE,
            "paired_led_improvement_mean_percent": imp_pct_mean, "paired_led_improvement_std_percent": imp_pct_std,
            "rmse_mean_ps": rmse_mean, "rmse_std_ps": rmse_std,
        })
    return output


def _unique_seed_map(rows):
    by_seed = defaultdict(list)
    for row in rows:
        by_seed[row["seed"]].append(row)
    if any(len(values) != 1 for values in by_seed.values()):
        return None
    return {seed: values[0] for seed, values in by_seed.items()}


def paired_model_comparisons(records):
    groups = defaultdict(list)
    for row in records:
        key = (
            row["dataset_key"], row["dataset"], row["population_identity"], row["mode"],
            row["window"], row["window_start_ns"], row["window_end_ns"],
        )
        groups[key].append(row)
    output = []
    for key, rows in sorted(groups.items()):
        dataset_key, dataset, population_identity, mode, window, start, end = key
        by_model = defaultdict(list)
        for row in rows:
            by_model[row["model"]].append(row)
        for model_a, model_b in itertools.combinations(sorted(by_model), 2):
            map_a = _unique_seed_map(by_model[model_a]); map_b = _unique_seed_map(by_model[model_b])
            if map_a is None or map_b is None:
                continue
            seeds = sorted(set(map_a) & set(map_b))
            if not seeds:
                continue
            a = np.asarray([map_a[s]["ctr_ps"] for s in seeds], float)
            b = np.asarray([map_b[s]["ctr_ps"] for s in seeds], float)
            summary = paired_replica_difference(
                a, b,
                seed=int(canonical_hash({"dataset": dataset_key, "mode": mode, "window": [start, end], "models": [model_a, model_b]})[:8], 16),
                n_bootstrap=REPORT_BOOTSTRAP_REPLICATES, confidence=REPORT_CONFIDENCE,
            )
            output.append({
                "dataset_key": dataset_key, "dataset": dataset, "population_identity": population_identity,
                "mode": mode, "window": window, "window_start_ns": start, "window_end_ns": end,
                "reference_model": model_a, "candidate_model": model_b, "n_paired_replicas": summary.n_pairs,
                "reference_ctr_mean_ps": summary.reference_mean, "candidate_ctr_mean_ps": summary.candidate_mean,
                "ctr_difference_reference_minus_candidate_ps": summary.difference_mean,
                "ctr_difference_std_ps": summary.difference_std,
                "paired_bootstrap_ci_low_ps": summary.ci_low, "paired_bootstrap_ci_high_ps": summary.ci_high,
                "paired_bootstrap_confidence": REPORT_CONFIDENCE,
            })
    return output


def _group_summary_for_plot(summary):
    groups = defaultdict(list)
    for row in summary:
        groups[(row["mode"], row["window"], row["window_start_ns"], row["window_end_ns"])].append(row)
    return groups


def _safe(value):
    text = re.sub(r"[^A-Za-z0-9_.-]+", "-", str(value)).strip("-")
    return text or "value"


def _save(fig, path):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True); fig.tight_layout(); fig.savefig(path); plt.close(fig); return path


def plot_led_improvements(summary, output_dir):
    outputs = []
    for (mode, window, start, end), rows in sorted(_group_summary_for_plot(summary).items()):
        models = sorted({row["model"] for row in rows}); studies = sorted({row["study"] for row in rows})
        x_lookup = {model: i for i, model in enumerate(models)}
        fig, ax = plt.subplots(figsize=(max(6.0, 1.2 * len(models)), 4.5))
        offsets = np.linspace(-0.18, 0.18, max(1, len(studies))); offset_lookup = {study: offsets[i] for i, study in enumerate(studies)}
        for row in rows:
            x = x_lookup[row["model"]] + offset_lookup[row["study"]]
            ax.errorbar([x], [row["paired_led_improvement_mean_ps"]], yerr=[row["paired_led_improvement_std_ps"]], fmt="o", capsize=3, label=row["study"] if row["model"] == models[0] else None)
        ax.axhline(0.0, linestyle="--", linewidth=1)
        ax.set_xticks(range(len(models)), models, rotation=25, ha="right")
        ax.set_ylabel("Paired CTR improvement, LED - ML [ps]")
        ax.set_title(f"Paired LED improvement | {mode} | {window} [{start:g}, {end:g}] ns")
        if len(studies) > 1: ax.legend(title="Study")
        outputs.append(_save(fig, Path(output_dir) / f"paired_led_improvement__{_safe(mode)}__{_safe(window)}.png"))
    return outputs


def plot_ctr_comparisons(summary, output_dir):
    outputs = []
    for (mode, window, start, end), rows in sorted(_group_summary_for_plot(summary).items()):
        models = sorted({row["model"] for row in rows}); studies = sorted({row["study"] for row in rows})
        x_lookup = {model: i for i, model in enumerate(models)}
        fig, ax = plt.subplots(figsize=(max(6.0, 1.2 * len(models)), 4.5))
        offsets = np.linspace(-0.18, 0.18, max(1, len(studies))); offset_lookup = {study: offsets[i] for i, study in enumerate(studies)}
        for row in rows:
            x = x_lookup[row["model"]] + offset_lookup[row["study"]]
            ax.errorbar([x], [row["ctr_mean_ps"]], yerr=[row["ctr_std_ps"]], fmt="o", capsize=3, label=row["study"] if row["model"] == models[0] else None)
        ax.set_xticks(range(len(models)), models, rotation=25, ha="right")
        ax.set_ylabel("Blind CTR [ps]"); ax.set_title(f"Model comparison | {mode} | {window} [{start:g}, {end:g}] ns")
        if len(studies) > 1: ax.legend(title="Study")
        outputs.append(_save(fig, Path(output_dir) / f"ctr_comparison__{_safe(mode)}__{_safe(window)}.png"))
    return outputs


def generate_report(paths, output_dir, *, logger=None):
    log = logger or logging.getLogger("waveform-report")
    output_dir = Path(output_dir).expanduser().resolve(); output_dir.mkdir(parents=True, exist_ok=True)
    records = collect_results(paths); summary = study_summary(records); comparisons = paired_model_comparisons(records)
    write_csv(output_dir / "study_summary.csv", summary)
    if comparisons: write_csv(output_dir / "paired_model_comparisons.csv", comparisons)
    led_plots = plot_led_improvements(summary, output_dir); ctr_plots = plot_ctr_comparisons(summary, output_dir)
    manifest = {
        "schema_version": 1, "sources": sorted({row["source_run"] for row in records}),
        "n_blind_replica_rows": len(records), "n_summary_rows": len(summary), "n_paired_model_comparisons": len(comparisons),
        "statistical_unit": "repeated_holdout_replica", "fit_bootstrap": False,
        "paired_bootstrap_unit": "replica_seed",
        "paired_bootstrap_allowed_only_for": "same analysis dataset + population + mode + window",
        "mode_pooling": False, "plots": [str(path) for path in led_plots + ctr_plots],
    }
    atomic_json(output_dir / "manifest.json", manifest)
    log.info("Report complete | sources=%d | summary rows=%d | paired comparisons=%d | %s", len(manifest["sources"]), len(summary), len(comparisons), output_dir)
    return output_dir
