from __future__ import annotations

import json
import logging
import re
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from .common import atomic_json, canonical_hash, write_csv
from .report import (
    REPORT_BOOTSTRAP_REPLICATES,
    REPORT_CONFIDENCE,
    collect_results,
    study_summary,
)
from .stats import paired_replica_difference


def _finite_values(values):
    values = np.asarray(values, dtype=float)
    return values[np.isfinite(values)]


def _mean_std(values):
    values = _finite_values(values)
    if not values.size:
        return float("nan"), float("nan")
    return (
        float(np.mean(values)),
        float(np.std(values, ddof=1)) if values.size > 1 else 0.0,
    )


def _safe(value):
    text = re.sub(r"[^A-Za-z0-9_.-]+", "-", str(value)).strip("-")
    return text or "value"


def _save(fig, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return path


def _summary_key(row):
    return (
        row["study"],
        row["dataset_key"],
        row["dataset"],
        row["population_identity"],
        row["sampling_identity"],
        row["model"],
        row["mode"],
        row["window"],
        float(row["window_start_ns"]),
        float(row["window_end_ns"]),
    )


def _extend_summary_with_rmse_reference(records, summary):
    grouped = defaultdict(list)
    for row in records:
        grouped[_summary_key(row)].append(row)

    extended = []
    for base in summary:
        row = dict(base)
        members = grouped[_summary_key(row)]
        led_rmse_mean, led_rmse_std = _mean_std(
            [member["led_rmse_ps"] for member in members]
        )
        improvements = np.asarray(
            [member["led_rmse_ps"] - member["rmse_ps"] for member in members],
            dtype=float,
        )
        improvement_mean, improvement_std = _mean_std(improvements)
        percentages = np.asarray(
            [
                100.0 * (member["led_rmse_ps"] - member["rmse_ps"]) / member["led_rmse_ps"]
                if np.isfinite(member["led_rmse_ps"]) and member["led_rmse_ps"] != 0.0
                else float("nan")
                for member in members
            ],
            dtype=float,
        )
        improvement_pct_mean, improvement_pct_std = _mean_std(percentages)

        led_values = np.asarray([member["led_rmse_ps"] for member in members], float)
        model_values = np.asarray([member["rmse_ps"] for member in members], float)
        finite = np.isfinite(led_values) & np.isfinite(model_values)
        if np.any(finite):
            paired = paired_replica_difference(
                led_values[finite],
                model_values[finite],
                seed=int(
                    canonical_hash(
                        {
                            "study": row["study"],
                            "dataset": row["dataset_key"],
                            "population": row["population_identity"],
                            "sampling": row["sampling_identity"],
                            "model": row["model"],
                            "mode": row["mode"],
                            "window": [row["window_start_ns"], row["window_end_ns"]],
                            "comparison": "led_vs_ml_rmse",
                        }
                    )[:8],
                    16,
                ),
                n_bootstrap=REPORT_BOOTSTRAP_REPLICATES,
                confidence=REPORT_CONFIDENCE,
            )
            ci_low, ci_high = paired.ci_low, paired.ci_high
        else:
            ci_low = ci_high = float("nan")

        row.update(
            {
                "led_rmse_mean_ps": led_rmse_mean,
                "led_rmse_std_ps": led_rmse_std,
                "paired_led_rmse_improvement_mean_ps": improvement_mean,
                "paired_led_rmse_improvement_std_ps": improvement_std,
                "paired_led_rmse_improvement_ci_low_ps": ci_low,
                "paired_led_rmse_improvement_ci_high_ps": ci_high,
                "paired_led_rmse_improvement_bootstrap_confidence": REPORT_CONFIDENCE,
                "paired_led_rmse_improvement_mean_percent": improvement_pct_mean,
                "paired_led_rmse_improvement_std_percent": improvement_pct_std,
            }
        )
        extended.append(row)
    return extended


def _group_mode_window(summary):
    groups = defaultdict(list)
    for row in summary:
        groups[
            (
                row["mode"],
                row["window"],
                float(row["window_start_ns"]),
                float(row["window_end_ns"]),
            )
        ].append(row)
    return groups


def _study_colors(studies):
    colors = plt.rcParams["axes.prop_cycle"].by_key().get("color", [])
    if not colors:
        return {study: None for study in studies}
    return {study: colors[index % len(colors)] for index, study in enumerate(studies)}


def _study_rmse_reference(rows, study):
    selected = [row for row in rows if row["study"] == study]
    means = _finite_values([row["led_rmse_mean_ps"] for row in selected])
    stds = _finite_values([row["led_rmse_std_ps"] for row in selected])
    if not means.size:
        return float("nan"), float("nan")
    if not np.allclose(means, means[0], rtol=1e-7, atol=1e-9):
        raise RuntimeError(
            f"Inconsistent LED RMSE reference across models for study {study}: {means.tolist()}"
        )
    if stds.size and not np.allclose(stds, stds[0], rtol=1e-7, atol=1e-9):
        raise RuntimeError(
            f"Inconsistent LED RMSE spread across models for study {study}: {stds.tolist()}"
        )
    return float(means[0]), float(stds[0]) if stds.size else float("nan")


def plot_rmse_comparisons(summary, output_dir):
    outputs = []
    for (mode, window, start, end), rows in sorted(_group_mode_window(summary).items()):
        models = sorted({row["model"] for row in rows})
        studies = sorted({row["study"] for row in rows})
        colors = _study_colors(studies)
        categories = ["LED", *models]
        x_lookup = {name: index for index, name in enumerate(categories)}
        offsets = np.linspace(-0.18, 0.18, max(1, len(studies)))
        offset_lookup = {study: offsets[index] for index, study in enumerate(studies)}
        fig, ax = plt.subplots(figsize=(max(6.5, 1.2 * len(categories)), 4.5))

        for study in studies:
            led_mean, led_std = _study_rmse_reference(rows, study)
            if np.isfinite(led_mean):
                ax.errorbar(
                    [x_lookup["LED"] + offset_lookup[study]],
                    [led_mean],
                    yerr=[led_std] if np.isfinite(led_std) else None,
                    fmt="s",
                    capsize=3,
                    color=colors[study],
                    label=study if len(studies) > 1 else "LED reference",
                )

        for row in rows:
            x = x_lookup[row["model"]] + offset_lookup[row["study"]]
            ax.errorbar(
                [x],
                [row["rmse_mean_ps"]],
                yerr=[row["rmse_std_ps"]],
                fmt="o",
                capsize=3,
                color=colors[row["study"]],
            )

        ax.set_xticks(range(len(categories)), categories, rotation=25, ha="right")
        ax.set_ylabel("Blind RMSE [ps]")
        ax.set_title(
            f"Model comparison with LED reference | {mode} | {window} [{start:g}, {end:g}] ns"
        )
        if len(studies) > 1:
            ax.legend(title="Study")
        outputs.append(
            _save(
                fig,
                Path(output_dir) / f"rmse_comparison__{_safe(mode)}__{_safe(window)}.png",
            )
        )
    return outputs


def _mean_finite(values):
    values = _finite_values(values)
    return float(np.mean(values)) if values.size else float("nan")


def plot_rmse_ctr_correlation(summary, output_dir):
    outputs = []
    for (mode, window, start, end), rows in sorted(_group_mode_window(summary).items()):
        by_model = defaultdict(list)
        for row in rows:
            by_model[row["model"]].append(row)

        points = []
        for model in sorted(by_model):
            model_rows = by_model[model]
            ctr = _mean_finite([row["ctr_mean_ps"] for row in model_rows])
            rmse = _mean_finite([row["rmse_mean_ps"] for row in model_rows])
            if np.isfinite(ctr) and np.isfinite(rmse):
                points.append((model, ctr, rmse))
        if not points:
            continue

        ctr_values = np.asarray([point[1] for point in points], float)
        rmse_values = np.asarray([point[2] for point in points], float)
        fig, ax = plt.subplots(figsize=(6.2, 5.0))
        ax.scatter(ctr_values, rmse_values, s=55)
        for model, ctr, rmse in points:
            ax.annotate(model, (ctr, rmse), xytext=(5, 5), textcoords="offset points", fontsize=8)

        correlation = float("nan")
        if len(points) >= 2 and np.std(ctr_values) > 0 and np.std(rmse_values) > 0:
            correlation = float(np.corrcoef(ctr_values, rmse_values)[0, 1])
        correlation_label = (
            f"Pearson r={correlation:.3f}" if np.isfinite(correlation) else "Pearson r=n/a"
        )
        ax.set_xlabel("Blind CTR [ps]")
        ax.set_ylabel("Blind RMSE [ps]")
        ax.set_title(
            f"RMSE vs CTR | {mode} | {window} [{start:g}, {end:g}] ns\n{correlation_label}"
        )
        outputs.append(
            _save(
                fig,
                Path(output_dir) / f"rmse_vs_ctr__{_safe(mode)}__{_safe(window)}.png",
            )
        )
    return outputs


def _window_key(row):
    return (row["window"], float(row["window_start_ns"]), float(row["window_end_ns"]))


def _unique_led_reference_values(rows, metric_key):
    unique = {}
    for row in rows:
        key = (
            row["study"],
            row["dataset_key"],
            row["population_identity"],
            row["sampling_identity"],
            row["mode"],
            *_window_key(row),
        )
        value = float(row[metric_key])
        if np.isfinite(value):
            unique[key] = value
    return np.asarray(list(unique.values()), dtype=float)


def plot_window_model_comparisons(summary, output_dir, *, metric):
    if metric == "ctr":
        mean_key = "ctr_mean_ps"
        std_key = "ctr_std_ps"
        led_key = "led_ctr_mean_ps"
        ylabel = "Blind CTR [ps]"
        title_metric = "CTR"
    elif metric == "rmse":
        mean_key = "rmse_mean_ps"
        std_key = "rmse_std_ps"
        led_key = "led_rmse_mean_ps"
        ylabel = "Blind RMSE [ps]"
        title_metric = "RMSE"
    else:
        raise ValueError(f"Unsupported metric: {metric}")

    by_mode = defaultdict(list)
    for row in summary:
        by_mode[row["mode"]].append(row)

    outputs = []
    for mode, rows in sorted(by_mode.items()):
        models = sorted({row["model"] for row in rows})
        windows = sorted({_window_key(row) for row in rows}, key=lambda item: (item[1], item[2], item[0]))
        if not models or len(windows) < 2:
            continue

        x = np.arange(len(models), dtype=float)
        width = 0.8 / len(windows)
        fig, ax = plt.subplots(figsize=(max(7.0, 1.25 * len(models)), 4.8))

        for index, (window, start, end) in enumerate(windows):
            subset = [row for row in rows if _window_key(row) == (window, start, end)]
            by_model = defaultdict(list)
            for row in subset:
                by_model[row["model"]].append(row)
            values = np.asarray(
                [
                    _mean_finite([row[mean_key] for row in by_model.get(model, [])])
                    for model in models
                ],
                float,
            )
            errors = np.asarray(
                [
                    _mean_finite([row[std_key] for row in by_model.get(model, [])])
                    for model in models
                ],
                float,
            )
            offset = (index - (len(windows) - 1) / 2.0) * width
            ax.bar(
                x + offset,
                values,
                width=width,
                yerr=np.where(np.isfinite(errors), errors, 0.0),
                capsize=2.5,
                label=f"{window} [{start:g}, {end:g}] ns",
            )

        led_values = _unique_led_reference_values(rows, led_key)
        if led_values.size:
            led_median = float(np.median(led_values))
            ax.axhline(
                led_median,
                linestyle="--",
                linewidth=1.2,
                label=f"Median LED {title_metric}: {led_median:.2f} ps",
            )

        ax.set_xticks(x, models, rotation=25, ha="right")
        ax.set_ylabel(ylabel)
        ax.set_title(f"{title_metric} across waveform windows | {mode}")
        ax.legend(title="Window / reference")
        outputs.append(
            _save(fig, Path(output_dir) / f"{metric}_by_window__{_safe(mode)}.png")
        )
    return outputs


def augment_report(paths, report_root, *, logger=None):
    log = logger or logging.getLogger("waveform-report")
    root = Path(report_root).expanduser().resolve()
    records = collect_results(paths)
    summary = _extend_summary_with_rmse_reference(records, study_summary(records))

    # Keep the canonical study table as the single summary source, now with RMSE reference statistics.
    write_csv(root / "tables" / "study_summary.csv", summary)

    rmse_dir = root / "plots" / "rmse"
    rmse_ctr_dir = root / "plots" / "rmse_ctr_correlation"
    window_ctr_dir = root / "plots" / "window_comparison" / "ctr"
    window_rmse_dir = root / "plots" / "window_comparison" / "rmse"

    outputs = []
    outputs.extend(plot_rmse_comparisons(summary, rmse_dir))
    outputs.extend(plot_rmse_ctr_correlation(summary, rmse_ctr_dir))
    outputs.extend(plot_window_model_comparisons(summary, window_ctr_dir, metric="ctr"))
    outputs.extend(plot_window_model_comparisons(summary, window_rmse_dir, metric="rmse"))

    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["schema_version"] = max(4, int(manifest.get("schema_version", 0)))
    manifest["rmse_plots_include_led_reference"] = True
    manifest["rmse_ctr_scatter_unit"] = "one point per model for each mode and window"
    manifest["window_comparison_led_reference"] = (
        "median uncorrected LED metric across unique study/window references for each mode"
    )
    layout = manifest.setdefault("layout", {})
    layout.update(
        {
            "rmse_plots": "plots/rmse",
            "rmse_ctr_plots": "plots/rmse_ctr_correlation",
            "window_ctr_plots": "plots/window_comparison/ctr",
            "window_rmse_plots": "plots/window_comparison/rmse",
        }
    )
    listed = list(manifest.get("plots_and_tables", []))
    listed.append("tables/study_summary.csv")
    listed.extend(str(path.resolve().relative_to(root)) for path in outputs)
    manifest["plots_and_tables"] = list(dict.fromkeys(listed))
    atomic_json(manifest_path, manifest)

    log.info(
        "Extended report metrics complete | rmse=%d | rmse-ctr=%d | window-comparisons=%d | %s",
        len(list(rmse_dir.glob("*.png"))),
        len(list(rmse_ctr_dir.glob("*.png"))),
        len(list((root / "plots" / "window_comparison").rglob("*.png"))),
        root,
    )
    return outputs
