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
    return [Path(config["output_dir"]).resolve() for config in batch.runs]


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
        mode = str(manifest["mode"])
        window = manifest["window_ns"]
        window_name = manifest.get("window_name")
        study_name = str(manifest.get("study_name") or manifest["name"])
        protocol_identity = str(manifest["analysis_protocol_identity"])
        sampling_identity = str(manifest["sampling_identity"])
        shared = manifest.get("shared_replicas") or {}

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
            })
    if not records:
        raise RuntimeError("No replica rows were found in the supplied studies")
    return records


def _mean_std(values):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if not values.size:
        return float("nan"), float("nan"), 0
    return (
        float(np.mean(values)),
        float(np.std(values, ddof=1)) if values.size > 1 else 0.0,
        int(values.size),
    )


def study_summary(records):
    groups = defaultdict(list)
    for row in records:
        key = (
            row["study"], row["dataset_key"], row["dataset"], row["population_identity"],
            row["sampling_identity"], row["model"], row["mode"], row["window"],
            row["window_start_ns"], row["window_end_ns"],
        )
        groups[key].append(row)

    output = []
    for key, rows in sorted(groups.items()):
        (
            study, dataset_key, dataset, population_identity, sampling_identity,
            model, mode, window, start, end,
        ) = key
        ctr_mean, ctr_std, n = _mean_std([row["ctr_ps"] for row in rows])
        led_mean, led_std, _ = _mean_std([row["led_ctr_ps"] for row in rows])
        imp_mean, imp_std, _ = _mean_std([row["improvement_ps"] for row in rows])
        valid = [row for row in rows if np.isfinite(row["led_ctr_ps"]) and np.isfinite(row["ctr_ps"])]
        if valid:
            paired = paired_replica_difference(
                np.asarray([row["led_ctr_ps"] for row in valid], float),
                np.asarray([row["ctr_ps"] for row in valid], float),
                seed=int(canonical_hash({
                    "study": study, "dataset": dataset_key, "population": population_identity,
                    "sampling": sampling_identity, "model": model, "mode": mode,
                    "window": [start, end], "comparison": "led_vs_ml",
                })[:8], 16),
                n_bootstrap=REPORT_BOOTSTRAP_REPLICATES,
                confidence=REPORT_CONFIDENCE,
            )
            ci_low, ci_high = paired.ci_low, paired.ci_high
        else:
            ci_low = ci_high = float("nan")
        imp_pct_mean, imp_pct_std, _ = _mean_std([row["improvement_percent"] for row in rows])
        rmse_mean, rmse_std, _ = _mean_std([row["rmse_ps"] for row in rows])
        output.append({
            "study": study,
            "dataset_key": dataset_key,
            "dataset": dataset,
            "population_identity": population_identity,
            "sampling_identity": sampling_identity,
            "model": model,
            "mode": mode,
            "window": window,
            "window_start_ns": start,
            "window_end_ns": end,
            "n_replicas": n,
            "ctr_mean_ps": ctr_mean,
            "ctr_std_ps": ctr_std,
            "led_ctr_mean_ps": led_mean,
            "led_ctr_std_ps": led_std,
            "paired_led_improvement_mean_ps": imp_mean,
            "paired_led_improvement_std_ps": imp_std,
            "paired_led_improvement_ci_low_ps": ci_low,
            "paired_led_improvement_ci_high_ps": ci_high,
            "paired_led_improvement_bootstrap_confidence": REPORT_CONFIDENCE,
            "paired_led_improvement_mean_percent": imp_pct_mean,
            "paired_led_improvement_std_percent": imp_pct_std,
            "rmse_mean_ps": rmse_mean,
            "rmse_std_ps": rmse_std,
        })
    return output


def _unique_replica_map(rows):
    by_replica = defaultdict(list)
    for row in rows:
        by_replica[row["replica_index"]].append(row)
    if any(len(values) != 1 for values in by_replica.values()):
        return None
    return {replica: values[0] for replica, values in by_replica.items()}


def paired_model_comparisons(records):
    groups = defaultdict(list)
    for row in records:
        key = (
            row["dataset_key"], row["dataset"], row["population_identity"],
            row["sampling_identity"], row["mode"], row["window"],
            row["window_start_ns"], row["window_end_ns"],
        )
        groups[key].append(row)

    output = []
    for key, rows in sorted(groups.items()):
        (
            dataset_key, dataset, population_identity, sampling_identity,
            mode, window, start, end,
        ) = key
        by_model = defaultdict(list)
        for row in rows:
            by_model[row["model"]].append(row)
        for model_a, model_b in itertools.combinations(sorted(by_model), 2):
            map_a = _unique_replica_map(by_model[model_a])
            map_b = _unique_replica_map(by_model[model_b])
            if map_a is None or map_b is None:
                continue
            replicas = sorted(set(map_a) & set(map_b))
            if not replicas:
                continue
            a = np.asarray([map_a[index]["ctr_ps"] for index in replicas], float)
            b = np.asarray([map_b[index]["ctr_ps"] for index in replicas], float)
            summary = paired_replica_difference(
                a,
                b,
                seed=int(canonical_hash({
                    "dataset": dataset_key, "population": population_identity,
                    "sampling": sampling_identity, "mode": mode,
                    "window": [start, end], "models": [model_a, model_b],
                })[:8], 16),
                n_bootstrap=REPORT_BOOTSTRAP_REPLICATES,
                confidence=REPORT_CONFIDENCE,
            )
            output.append({
                "dataset_key": dataset_key,
                "dataset": dataset,
                "population_identity": population_identity,
                "sampling_identity": sampling_identity,
                "mode": mode,
                "window": window,
                "window_start_ns": start,
                "window_end_ns": end,
                "reference_model": model_a,
                "candidate_model": model_b,
                "n_paired_replicas": summary.n_pairs,
                "reference_ctr_mean_ps": summary.reference_mean,
                "candidate_ctr_mean_ps": summary.candidate_mean,
                "ctr_difference_reference_minus_candidate_ps": summary.difference_mean,
                "ctr_difference_std_ps": summary.difference_std,
                "paired_bootstrap_ci_low_ps": summary.ci_low,
                "paired_bootstrap_ci_high_ps": summary.ci_high,
                "paired_bootstrap_confidence": REPORT_CONFIDENCE,
            })
    return output


def _group_summary_for_plot(summary):
    groups = defaultdict(list)
    for row in summary:
        groups[(row["mode"], row["window"], row["window_start_ns"], row["window_end_ns"])].append(row)
    return groups


def _save(fig, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return path


def plot_led_improvements(summary, output_dir):
    outputs = []
    for (mode, window, start, end), rows in sorted(_group_summary_for_plot(summary).items()):
        models = sorted({row["model"] for row in rows})
        studies = sorted({row["study"] for row in rows})
        x_lookup = {model: index for index, model in enumerate(models)}
        offsets = np.linspace(-0.18, 0.18, max(1, len(studies)))
        offset_lookup = {study: offsets[index] for index, study in enumerate(studies)}
        fig, ax = plt.subplots(figsize=(max(6.0, 1.2 * len(models)), 4.5))
        for row in rows:
            x = x_lookup[row["model"]] + offset_lookup[row["study"]]
            ax.errorbar(
                [x],
                [row["paired_led_improvement_mean_ps"]],
                yerr=[row["paired_led_improvement_std_ps"]],
                fmt="o",
                capsize=3,
                label=row["study"] if row["model"] == models[0] else None,
            )
        ax.axhline(0.0, linestyle="--", linewidth=1)
        ax.set_xticks(range(len(models)), models, rotation=25, ha="right")
        ax.set_ylabel("Paired CTR improvement, LED - ML [ps]")
        ax.set_title(f"Paired LED improvement | {mode} | {window} [{start:g}, {end:g}] ns")
        if len(studies) > 1:
            ax.legend(title="Study")
        outputs.append(
            _save(fig, Path(output_dir) / f"paired_led_improvement__{_safe(mode)}__{_safe(window)}.png")
        )
    return outputs


def plot_ctr_comparisons(summary, output_dir):
    outputs = []
    for (mode, window, start, end), rows in sorted(_group_summary_for_plot(summary).items()):
        models = sorted({row["model"] for row in rows})
        studies = sorted({row["study"] for row in rows})
        x_lookup = {model: index for index, model in enumerate(models)}
        offsets = np.linspace(-0.18, 0.18, max(1, len(studies)))
        offset_lookup = {study: offsets[index] for index, study in enumerate(studies)}
        fig, ax = plt.subplots(figsize=(max(6.0, 1.2 * len(models)), 4.5))
        for row in rows:
            x = x_lookup[row["model"]] + offset_lookup[row["study"]]
            ax.errorbar(
                [x], [row["ctr_mean_ps"]], yerr=[row["ctr_std_ps"]],
                fmt="o", capsize=3,
                label=row["study"] if row["model"] == models[0] else None,
            )
        ax.set_xticks(range(len(models)), models, rotation=25, ha="right")
        ax.set_ylabel("Blind CTR [ps]")
        ax.set_title(f"Model comparison | {mode} | {window} [{start:g}, {end:g}] ns")
        if len(studies) > 1:
            ax.legend(title="Study")
        outputs.append(
            _save(fig, Path(output_dir) / f"ctr_comparison__{_safe(mode)}__{_safe(window)}.png")
        )
    return outputs


def _load_model_output(record, cache):
    key = (record["source_run"], record["seed"], record["candidate_id"])
    if key in cache:
        return cache[key]
    residual_path = (
        Path(record["source_run"])
        / "blind_residuals"
        / f"seed_{int(record['seed'])}_{record['candidate_id']}.npz"
    )
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
    output = np.asarray(led - corrected, float)
    cache[key] = (event_index, output)
    return cache[key]


def _replica_output_correlation(a, b):
    event_a, out_a = a
    event_b, out_b = b
    common, ia, ib = np.intersect1d(event_a, event_b, assume_unique=False, return_indices=True)
    if common.size < 2:
        return float("nan")
    x = np.asarray(out_a[ia], float)
    y = np.asarray(out_b[ib], float)
    finite = np.isfinite(x) & np.isfinite(y)
    x = x[finite]
    y = y[finite]
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


def model_output_correlations(records, output_dir):
    groups = defaultdict(list)
    for row in records:
        key = (
            row["dataset_key"], row["population_identity"], row["sampling_identity"],
            row["mode"], row["window"], row["window_start_ns"], row["window_end_ns"],
        )
        groups[key].append(row)

    cache = {}
    long_rows = []
    outputs = []
    for key, rows in sorted(groups.items()):
        dataset_key, population_identity, sampling_identity, mode, window, start, end = key
        by_model = defaultdict(list)
        for row in rows:
            by_model[row["model"]].append(row)
        models = sorted(by_model)
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
                    r = _replica_output_correlation(output_a, output_b)
                    if np.isfinite(r):
                        replica_rs.append(r)
                value = _fisher_mean(replica_rs)
                matrix[i, j] = matrix[j, i] = value
                counts[i, j] = counts[j, i] = len(replica_rs)
                long_rows.append({
                    "dataset_key": dataset_key,
                    "population_identity": population_identity,
                    "sampling_identity": sampling_identity,
                    "mode": mode,
                    "window": window,
                    "window_start_ns": start,
                    "window_end_ns": end,
                    "model_a": model_a,
                    "model_b": model_b,
                    "pearson_r_fisher_mean": value,
                    "n_paired_replicas": len(replica_rs),
                })

        stem = f"model_output_correlation__{_safe(mode)}__{_safe(window)}__{population_identity[:10]}"
        matrix_path = _write_matrix(Path(output_dir) / f"{stem}.csv", models, matrix)
        count_path = _write_matrix(Path(output_dir) / f"{stem}__n_replicas.csv", models, counts)
        fig, ax = plt.subplots(figsize=(max(5.5, 0.9 * len(models)), max(4.8, 0.8 * len(models))))
        image = ax.imshow(matrix, vmin=-1.0, vmax=1.0, cmap="coolwarm")
        ax.set_xticks(range(len(models)), models, rotation=35, ha="right")
        ax.set_yticks(range(len(models)), models)
        ax.set_title(f"Model output correlation | {mode} | {window}")
        fig.colorbar(image, ax=ax, label="Pearson r (Fisher-z mean across replicas)")
        for i in range(len(models)):
            for j in range(len(models)):
                if np.isfinite(matrix[i, j]):
                    ax.text(j, i, f"{matrix[i, j]:.2f}", ha="center", va="center", fontsize=8)
        plot_path = _save(fig, Path(output_dir) / f"{stem}.png")
        outputs.extend([matrix_path, count_path, plot_path])

    if long_rows:
        long_path = Path(output_dir) / "model_output_correlations.csv"
        write_csv(long_path, long_rows)
        outputs.append(long_path)
    return outputs, long_rows


def generate_report(paths, output_dir, *, logger=None):
    log = logger or logging.getLogger("waveform-report")
    output_dir = Path(output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    records = collect_results(paths)
    summary = study_summary(records)
    comparisons = paired_model_comparisons(records)
    write_csv(output_dir / "study_summary.csv", summary)
    if comparisons:
        write_csv(output_dir / "paired_model_comparisons.csv", comparisons)

    led_plots = plot_led_improvements(summary, output_dir)
    ctr_plots = plot_ctr_comparisons(summary, output_dir)
    correlation_outputs, correlation_rows = model_output_correlations(records, output_dir)
    outputs = led_plots + ctr_plots + correlation_outputs

    manifest = {
        "schema_version": 2,
        "sources": sorted({row["source_run"] for row in records}),
        "n_replica_rows": len(records),
        "n_summary_rows": len(summary),
        "n_paired_model_comparisons": len(comparisons),
        "n_model_output_correlations": len(correlation_rows),
        "statistical_unit": "replica",
        "fit_bootstrap": False,
        "paired_bootstrap_unit": "replica",
        "pairing_rule": "same dataset + analysis protocol + sampling identity + mode + window + replica index",
        "mode_pooling": False,
        "plots_and_tables": [str(path) for path in outputs],
    }
    atomic_json(output_dir / "manifest.json", manifest)
    log.info(
        "Report complete | sources=%d | summary=%d | paired=%d | output-correlations=%d | %s",
        len(manifest["sources"]),
        len(summary),
        len(comparisons),
        len(correlation_rows),
        output_dir,
    )
    return output_dir
