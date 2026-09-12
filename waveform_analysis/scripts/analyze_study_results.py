from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from waveform_analysis.ml_pipeline.common import voltage_from_name
from waveform_analysis.ml_pipeline.reporting import LABELS, MODEL_ORDER


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Summarize a completed waveform study: selected LED threshold, blind-test CTR by voltage, "
            "and validation CTR sensitivity to the symmetric training-target filter."
        )
    )
    parser.add_argument("--run-dir", type=Path, required=True, help="Completed study directory")
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Destination directory (default: <run-dir>/analysis_summary)",
    )
    parser.add_argument(
        "--plot-format",
        choices=("pdf", "png"),
        default="pdf",
        help="Plot file format (default: pdf)",
    )
    return parser


def _float(value: Any, default: float = float("nan")) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(f"Missing study results: {path}")
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def _read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as stream:
        value = json.load(stream)
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return value


def _ordered_models(methods: set[str]) -> list[str]:
    models = [name for name in MODEL_ORDER if name in methods and name not in {"led", "cfd"}]
    models.extend(sorted(methods - set(MODEL_ORDER) - {"led", "cfd"}))
    return models


def _row_voltage(row: dict[str, Any]) -> float:
    value = _float(row.get("voltage_V"))
    return value if np.isfinite(value) else voltage_from_name(str(row.get("dataset", "")))


def _dataset_voltage(dataset: str, rows: list[dict[str, Any]]) -> float:
    values = [_row_voltage(row) for row in rows if row.get("dataset") == dataset]
    finite = [value for value in values if np.isfinite(value)]
    return float(np.median(finite)) if finite else voltage_from_name(dataset)


def _led_threshold_from_selection(rows: list[dict[str, Any]], dataset: str) -> float:
    row = next(
        (
            item
            for item in rows
            if item.get("dataset") == dataset
            and item.get("method") == "led"
            and item.get("stage") == "development_selection"
        ),
        None,
    )
    if row is None:
        return float("nan")
    try:
        parameters = json.loads(row.get("parameters_json") or "{}")
    except json.JSONDecodeError:
        return float("nan")
    return _float(parameters.get("threshold_mV"))


def _led_threshold_from_manifest(manifest: dict[str, Any], dataset: str, mode: str) -> float:
    dataset_info = (manifest.get("datasets") or {}).get(dataset) or {}
    thresholds = dataset_info.get("led_threshold_mV") or {}
    family = str(mode).split("_to_", 1)[0]
    if isinstance(thresholds, dict):
        return _float(thresholds.get(family))
    return _float(thresholds)


def _metric_row(rows: list[dict[str, Any]], dataset: str, method: str) -> dict[str, Any] | None:
    return next(
        (
            row
            for row in rows
            if row.get("dataset") == dataset
            and row.get("method") == method
            and row.get("stage") == "test"
        ),
        None,
    )


def _study_summary(run: Path) -> tuple[dict[str, Any], list[dict[str, Any]], list[str]]:
    rows = _read_csv(run / "csv" / "results.csv")
    manifest = _read_json(run / "manifest.json")
    mode = str(manifest.get("mode") or (manifest.get("config") or {}).get("mode") or "")
    methods = {row.get("method", "") for row in rows if row.get("stage") == "test"}
    models = _ordered_models(methods)
    datasets = sorted(
        {row["dataset"] for row in rows if row.get("stage") == "test" and row.get("dataset")},
        key=lambda name: (_dataset_voltage(name, rows) if np.isfinite(_dataset_voltage(name, rows)) else math.inf, name),
    )

    summary = []
    for dataset in datasets:
        threshold = _led_threshold_from_selection(rows, dataset)
        if not np.isfinite(threshold):
            threshold = _led_threshold_from_manifest(manifest, dataset, mode)
        led = _metric_row(rows, dataset, "led")
        if led is None:
            continue
        model_metrics = {}
        for model in models:
            row = _metric_row(rows, dataset, model)
            if row is not None:
                model_metrics[model] = {
                    "ctr_ps": _float(row.get("ctr_ps")),
                    "uncertainty_ps": _float(row.get("ctr_uncertainty_ps")),
                }
        summary.append(
            {
                "dataset": dataset,
                "voltage_V": _dataset_voltage(dataset, rows),
                "led_threshold_mV": threshold,
                "led": {
                    "ctr_ps": _float(led.get("ctr_ps")),
                    "uncertainty_ps": _float(led.get("ctr_uncertainty_ps")),
                },
                "models": model_metrics,
            }
        )
    if not summary:
        raise ValueError("No blind-test LED rows were found in results.csv")
    return manifest, summary, models


def _escape_latex(value: str) -> str:
    replacements = {
        "\\": r"\textbackslash{}",
        "_": r"\_",
        "%": r"\%",
        "&": r"\&",
        "#": r"\#",
        "{": r"\{",
        "}": r"\}",
    }
    return "".join(replacements.get(char, char) for char in str(value))


def _measurement(value: float, uncertainty: float) -> str:
    if not np.isfinite(value):
        return "--"
    if not np.isfinite(uncertainty) or uncertainty <= 0:
        return f"{value:.1f}"
    return f"{value:.1f} $\\pm$ {uncertainty:.1f}"


def _write_latex_table(path: Path, manifest: dict[str, Any], summary: list[dict[str, Any]], models: list[str]) -> None:
    mode = str(manifest.get("mode") or (manifest.get("config") or {}).get("mode") or "study")
    columns = ["Bias voltage [V]", "LED threshold [mV]", "LED CTR [ps]"] + [
        f"{LABELS.get(model, model)} CTR [ps]" for model in models
    ]
    alignment = "r" * len(columns)
    lines = [
        "% Requires \\usepackage{booktabs}",
        "\\begin{table}[htbp]",
        "\\centering",
        f"\\caption{{Blind-test CTR summary for {_escape_latex(mode.replace('_', ' '))}.}}",
        "\\label{tab:study_ctr_summary}",
        f"\\begin{{tabular}}{{{alignment}}}",
        "\\toprule",
        " & ".join(_escape_latex(item) for item in columns) + r" \\",
        "\\midrule",
    ]
    for item in summary:
        voltage = item["voltage_V"]
        threshold = item["led_threshold_mV"]
        row = [
            f"{voltage:g}" if np.isfinite(voltage) else _escape_latex(item["dataset"]),
            f"{threshold:g}" if np.isfinite(threshold) else "--",
            _measurement(item["led"]["ctr_ps"], item["led"]["uncertainty_ps"]),
        ]
        for model in models:
            metric = item["models"].get(model)
            row.append("--" if metric is None else _measurement(metric["ctr_ps"], metric["uncertainty_ps"]))
        lines.append(" & ".join(row) + r" \\")
    lines.extend(["\\bottomrule", "\\end{tabular}", "\\end{table}", ""])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def _summary_plot(path: Path, manifest: dict[str, Any], summary: list[dict[str, Any]], models: list[str]) -> None:
    import matplotlib.pyplot as plt

    finite = [item for item in summary if np.isfinite(item["voltage_V"])]
    if not finite:
        return
    voltages = np.asarray([item["voltage_V"] for item in finite], dtype=float)
    thresholds = np.asarray([item["led_threshold_mV"] for item in finite], dtype=float)
    order = np.argsort(voltages)
    voltages = voltages[order]
    thresholds = thresholds[order]
    finite = [finite[index] for index in order]

    fig, axes = plt.subplots(2, 1, figsize=(8.8, 7.0), sharex=True, height_ratios=(0.8, 2.0))
    threshold_ax, ctr_ax = axes

    good_threshold = np.isfinite(thresholds)
    if np.any(good_threshold):
        threshold_ax.plot(voltages[good_threshold], thresholds[good_threshold], marker="o")
    threshold_ax.set_ylabel("LED threshold [mV]")
    threshold_ax.grid(True, alpha=0.2)

    methods = ["led", *models]
    for method in methods:
        values = []
        errors = []
        for item in finite:
            metric = item["led"] if method == "led" else item["models"].get(method)
            values.append(float("nan") if metric is None else metric["ctr_ps"])
            errors.append(float("nan") if metric is None else metric["uncertainty_ps"])
        values = np.asarray(values, dtype=float)
        errors = np.asarray(errors, dtype=float)
        mask = np.isfinite(values)
        if not np.any(mask):
            continue
        safe_errors = np.where(np.isfinite(errors[mask]), errors[mask], 0.0)
        ctr_ax.errorbar(
            voltages[mask],
            values[mask],
            yerr=safe_errors,
            marker="o",
            capsize=3,
            label=LABELS.get(method, method),
        )

    mode = str(manifest.get("mode") or (manifest.get("config") or {}).get("mode") or "")
    ctr_ax.set_xlabel("Bias voltage [V]")
    ctr_ax.set_ylabel("Blind-test CTR [ps]")
    ctr_ax.set_title(mode.replace("_", " "))
    ctr_ax.grid(True, alpha=0.2)
    ctr_ax.legend()
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def _target_filter_curves(run: Path, datasets: list[str], models: list[str]) -> dict[str, dict[str, dict[str, Any]]]:
    output: dict[str, dict[str, dict[str, Any]]] = {}
    for dataset in datasets:
        dataset_output: dict[str, dict[str, Any]] = {}
        for model in models:
            path = run / "search" / dataset / f"{model}.json"
            if not path.is_file():
                continue
            search = _read_json(path)
            by_filter: dict[float, list[float]] = defaultdict(list)
            for row in search.get("candidates") or []:
                if row.get("error"):
                    continue
                candidate = row.get("candidate") or {}
                limit = _float(candidate.get("target_abs_max_ps"))
                score = _float(row.get("score"))
                if np.isfinite(limit) and np.isfinite(score):
                    by_filter[limit].append(score)
            if not by_filter:
                continue
            limits = sorted(by_filter)
            scores = [min(by_filter[limit]) for limit in limits]
            best = search.get("best") or {}
            selected_limit = _float((best.get("candidate") or {}).get("target_abs_max_ps"))
            selected_score = _float(best.get("score"))
            dataset_output[model] = {
                "limits_ps": np.asarray(limits, dtype=float),
                "validation_ctr_ps": np.asarray(scores, dtype=float),
                "selected_limit_ps": selected_limit,
                "selected_ctr_ps": selected_score,
            }
        if dataset_output:
            output[dataset] = dataset_output
    return output


def _target_filter_plot(
    path: Path,
    summary: list[dict[str, Any]],
    models: list[str],
    curves: dict[str, dict[str, dict[str, Any]]],
) -> None:
    import matplotlib.pyplot as plt

    datasets = [item["dataset"] for item in summary if item["dataset"] in curves]
    if not datasets:
        return
    ncols = min(2, len(datasets))
    nrows = int(math.ceil(len(datasets) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(7.0 * ncols, 4.8 * nrows), squeeze=False, sharey=True)
    flat_axes = list(axes.ravel())

    model_handles: dict[str, Any] = {}
    colors = plt.rcParams["axes.prop_cycle"].by_key().get("color", [])
    model_colors = {
        model: (colors[index % len(colors)] if colors else None)
        for index, model in enumerate(models)
    }
    for ax, dataset in zip(flat_axes, datasets):
        item = next(row for row in summary if row["dataset"] == dataset)
        for model in models:
            curve = curves[dataset].get(model)
            if curve is None:
                continue
            line, = ax.plot(
                curve["limits_ps"],
                curve["validation_ctr_ps"],
                marker="o",
                color=model_colors[model],
                label=LABELS.get(model, model),
            )
            model_handles.setdefault(model, line)
            if np.isfinite(curve["selected_limit_ps"]) and np.isfinite(curve["selected_ctr_ps"]):
                ax.scatter(
                    [curve["selected_limit_ps"]],
                    [curve["selected_ctr_ps"]],
                    marker="*",
                    s=120,
                    color=line.get_color(),
                    edgecolors="black",
                    linewidths=0.5,
                    zorder=5,
                )
        voltage = item["voltage_V"]
        ax.set_title(f"{voltage:g} V" if np.isfinite(voltage) else dataset)
        ax.set_xlabel(r"Training target filter $|y_{target}| \leq T$ [ps]")
        ax.set_ylabel("Validation CTR [ps]")
        ax.grid(True, alpha=0.2)

    for ax in flat_axes[len(datasets):]:
        ax.remove()

    if model_handles:
        ordered = [model for model in models if model in model_handles]
        fig.suptitle("Validation CTR vs training-target filter", y=0.995)
        fig.legend(
            [model_handles[model] for model in ordered],
            [LABELS.get(model, model) for model in ordered],
            loc="upper center",
            bbox_to_anchor=(0.5, 0.955),
            ncol=max(1, min(4, len(ordered))),
            frameon=False,
        )
        fig.text(
            0.5,
            0.015,
            "Each point is the best validation CTR over the model's other hyperparameters; ★ marks the globally selected candidate.",
            ha="center",
            va="bottom",
            fontsize=9,
        )
    fig.tight_layout(rect=(0, 0.05, 1, 0.89))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def analyze(run_dir: Path, output_dir: Path | None = None, plot_format: str = "pdf") -> list[Path]:
    run = run_dir.resolve()
    output = (output_dir or (run / "analysis_summary")).resolve()
    manifest, summary, models = _study_summary(run)

    table_path = output / "study_summary.tex"
    voltage_plot = output / f"study_summary_vs_voltage.{plot_format}"
    target_plot = output / f"validation_ctr_vs_target_filter.{plot_format}"

    _write_latex_table(table_path, manifest, summary, models)
    _summary_plot(voltage_plot, manifest, summary, models)
    curves = _target_filter_curves(run, [item["dataset"] for item in summary], models)
    _target_filter_plot(target_plot, summary, models, curves)

    generated = [table_path]
    if voltage_plot.is_file():
        generated.append(voltage_plot)
    if target_plot.is_file():
        generated.append(target_plot)
    return generated


def main() -> None:
    args = _parser().parse_args()
    for path in analyze(args.run_dir, args.output_dir, args.plot_format):
        print(path)


if __name__ == "__main__":
    main()
