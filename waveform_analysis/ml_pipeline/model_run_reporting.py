from __future__ import annotations

import json
from itertools import combinations
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from .common import voltage_from_name, write_csv
from .model_output_reporting import _model_output, _pearson
from .plot_style import (
    DOUBLE_COLUMN,
    LABELS,
    finish_voltage_axis,
    model_style,
    paper_context,
    plot_voltage_series,
    save_figure,
)
from .reporting import plot_ctr_vs_voltage, read_results


def _manifest(run: Path) -> dict[str, Any]:
    path = run / "manifest.json"
    if not path.is_file():
        raise FileNotFoundError(f"Model-study manifest not found: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if value.get("status") != "complete":
        raise ValueError(f"Model study is not complete: {run}")
    if value.get("experiment_type") != "model_study":
        raise ValueError(f"Expected model_study run, got {value.get('experiment_type')!r}: {run}")
    if not value.get("model"):
        raise ValueError(f"Model-study manifest has no model name: {run}")
    return value


def _subrun(root: Path, manifest: dict[str, Any], window: str) -> Path:
    local = root / window
    if local.is_dir():
        return local.resolve()
    configured = (manifest.get("subruns") or {}).get(window)
    if configured:
        candidate = Path(str(configured)).expanduser()
        if candidate.is_dir():
            return candidate.resolve()
    raise FileNotFoundError(f"Missing window subrun {window!r}: {local}")


def _row_voltage(row: dict[str, Any]) -> float:
    try:
        value = float(row.get("voltage_V", float("nan")))
    except (TypeError, ValueError):
        value = float("nan")
    if np.isfinite(value):
        return value
    return float(voltage_from_name(str(row.get("dataset", ""))))


def _method_rows(rows: list[dict[str, Any]], method: str) -> list[dict[str, Any]]:
    return sorted(
        [row for row in rows if row.get("method") == method],
        key=_row_voltage,
    )


def _voltage_signature(rows: list[dict[str, Any]], method: str) -> tuple[float, ...]:
    values = [_row_voltage(row) for row in _method_rows(rows, method)]
    if not values or not all(np.isfinite(value) for value in values):
        raise ValueError(f"Missing or invalid voltage values for method {method!r}")
    rounded = tuple(round(float(value), 9) for value in values)
    if len(set(rounded)) != len(rounded):
        raise ValueError(f"Duplicate voltage results for method {method!r}: {rounded}")
    return rounded


def _row_at_voltage(
    rows: list[dict[str, Any]],
    method: str,
    voltage: float,
) -> dict[str, Any]:
    matches = [
        row
        for row in rows
        if row.get("method") == method
        and np.isfinite(_row_voltage(row))
        and np.isclose(_row_voltage(row), voltage, rtol=0.0, atol=1e-9)
    ]
    if len(matches) != 1:
        raise ValueError(
            f"Expected one {method!r} result at {voltage:g} V, found {len(matches)}"
        )
    return matches[0]


def _result_difference(
    reference_row: dict[str, Any],
    method_row: dict[str, Any],
) -> tuple[float, float, float, float]:
    reference = float(reference_row["ctr_ps"])
    method = float(method_row["ctr_ps"])
    reference_unc = float(reference_row.get("ctr_uncertainty_ps", float("nan")))
    method_unc = float(method_row.get("ctr_uncertainty_ps", float("nan")))

    delta = reference - method
    delta_unc = (
        float(np.hypot(reference_unc, method_unc))
        if np.isfinite(reference_unc) and np.isfinite(method_unc)
        else float("nan")
    )

    if not np.isfinite(reference) or reference <= 0 or not np.isfinite(method):
        return delta, delta_unc, float("nan"), float("nan")

    relative = 100.0 * delta / reference
    if np.isfinite(reference_unc) and np.isfinite(method_unc):
        d_ref = 100.0 * method / (reference * reference)
        d_method = -100.0 / reference
        relative_unc = float(
            np.hypot(d_ref * reference_unc, d_method * method_unc)
        )
    else:
        relative_unc = float("nan")
    return delta, delta_unc, relative, relative_unc


def _test_event_index(run: Path, dataset: str) -> np.ndarray:
    path = run / "splits" / f"{dataset}.npz"
    if not path.is_file():
        raise FileNotFoundError(f"Missing split file required for output correlation: {path}")
    with np.load(path) as split:
        if "test_event_index" not in split:
            raise ValueError(f"Split file has no test_event_index: {path}")
        values = np.asarray(split["test_event_index"], dtype=np.int64).reshape(-1)
    if np.unique(values).size != values.size:
        raise ValueError(f"Blind event IDs are not unique: {path}")
    return values


def _aligned_model_outputs(
    run_a: Path,
    dataset_a: str,
    model_a: str,
    run_b: Path,
    dataset_b: str,
    model_b: str,
) -> tuple[np.ndarray, np.ndarray]:
    output_a = _model_output(run_a, dataset_a, model_a, "test")
    output_b = _model_output(run_b, dataset_b, model_b, "test")
    if output_a is None or output_b is None:
        missing = []
        if output_a is None:
            missing.append(f"{model_a}:{run_a / 'artifacts' / dataset_a}")
        if output_b is None:
            missing.append(f"{model_b}:{run_b / 'artifacts' / dataset_b}")
        raise FileNotFoundError(
            "Missing blind model-output artifact(s): " + ", ".join(missing)
        )

    ids_a = _test_event_index(run_a, dataset_a)
    ids_b = _test_event_index(run_b, dataset_b)
    if output_a.size != ids_a.size or output_b.size != ids_b.size:
        raise ValueError(
            "Model-output length does not match persisted blind event IDs: "
            f"{model_a}={output_a.size}/{ids_a.size}, "
            f"{model_b}={output_b.size}/{ids_b.size}"
        )
    if ids_a.size != ids_b.size or set(ids_a.tolist()) != set(ids_b.tolist()):
        raise ValueError(
            f"Blind event identities differ for {model_a} and {model_b}; "
            "model-output correlation would not be paired"
        )
    if np.array_equal(ids_a, ids_b):
        return output_a, output_b

    position_b = {int(event_id): index for index, event_id in enumerate(ids_b)}
    reorder_b = np.asarray([position_b[int(event_id)] for event_id in ids_a], dtype=np.int64)
    return output_a, output_b[reorder_b]


def _plot_improvement(
    output: Path,
    window: str,
    rows: list[dict[str, Any]],
    *,
    relative: bool,
) -> Path | None:
    if not rows:
        return None
    fig, ax = plt.subplots(figsize=DOUBLE_COLUMN)
    models = list(dict.fromkeys(str(row["model"]) for row in rows))
    all_voltage = sorted({float(row["voltage_V"]) for row in rows})
    for index, model in enumerate(models):
        subset = sorted(
            [row for row in rows if row["model"] == model],
            key=lambda row: float(row["voltage_V"]),
        )
        x = np.asarray([float(row["voltage_V"]) for row in subset], dtype=float)
        if relative:
            y = np.asarray([float(row["improvement_percent"]) for row in subset])
            e = np.asarray([float(row["uncertainty_percent"]) for row in subset])
        else:
            y = np.asarray([float(row["improvement_ps"]) for row in subset])
            e = np.asarray([float(row["uncertainty_ps"]) for row in subset])
        plot_voltage_series(
            ax,
            x,
            y,
            errors=e,
            label=LABELS.get(model, model),
            style=model_style(model, index),
        )
    finish_voltage_axis(
        ax,
        all_voltage,
        ylabel="Improvement [%]" if relative else "Improvement [ps]",
        zero_line=True,
    )
    fig.tight_layout()
    target = save_figure(
        fig,
        output / (
            f"relative_improvement_vs_led_{window}.pdf"
            if relative
            else f"improvement_vs_led_{window}.pdf"
        ),
    )
    plt.close(fig)
    return target


def _plot_pairwise_model_improvement(
    output: Path,
    window: str,
    rows: list[dict[str, Any]],
) -> list[Path]:
    generated: list[Path] = []
    pairs = list(
        dict.fromkeys((str(row["model_a"]), str(row["model_b"])) for row in rows)
    )
    for model_a, model_b in pairs:
        subset = sorted(
            [
                row
                for row in rows
                if row["window"] == window
                and row["model_a"] == model_a
                and row["model_b"] == model_b
            ],
            key=lambda row: float(row["voltage_V"]),
        )
        if not subset:
            continue
        x = np.asarray([float(row["voltage_V"]) for row in subset], dtype=float)
        y = np.asarray(
            [float(row["model_a_improvement_over_model_b_percent"]) for row in subset],
            dtype=float,
        )
        e = np.asarray(
            [float(row["uncertainty_percent"]) for row in subset],
            dtype=float,
        )
        fig, ax = plt.subplots(figsize=DOUBLE_COLUMN)
        plot_voltage_series(
            ax,
            x,
            y,
            errors=e,
            style=model_style(model_a),
        )
        finish_voltage_axis(
            ax,
            x,
            ylabel="Improvement [%]",
            legend=False,
            zero_line=True,
        )
        fig.tight_layout()
        target = save_figure(
            fig,
            output / f"{model_a}_vs_{model_b}_{window}.pdf",
        )
        plt.close(fig)
        generated.append(target)
    return generated


def _plot_model_output_correlation(
    output: Path,
    window: str,
    rows: list[dict[str, Any]],
) -> Path | None:
    subset_window = [row for row in rows if row["window"] == window]
    if not subset_window:
        return None
    fig, ax = plt.subplots(figsize=DOUBLE_COLUMN)
    pairs = list(
        dict.fromkeys(
            (str(row["model_a"]), str(row["model_b"]))
            for row in subset_window
        )
    )
    all_voltages = sorted({float(row["voltage_V"]) for row in subset_window})
    for index, (model_a, model_b) in enumerate(pairs):
        pair_rows = sorted(
            [
                row
                for row in subset_window
                if row["model_a"] == model_a and row["model_b"] == model_b
            ],
            key=lambda row: float(row["voltage_V"]),
        )
        plot_voltage_series(
            ax,
            [float(row["voltage_V"]) for row in pair_rows],
            [float(row["pearson_r"]) for row in pair_rows],
            label=f"{LABELS.get(model_a, model_a)} vs {LABELS.get(model_b, model_b)}",
            style=model_style(model_a, index),
        )
    finish_voltage_axis(ax, all_voltages, ylabel="Correlation [–]")
    ax.set_ylim(-1.02, 1.02)
    fig.tight_layout()
    target = save_figure(fig, output / f"model_output_correlation_{window}.pdf")
    plt.close(fig)
    return target


def compare_model_runs(
    run_dirs: list[str | Path],
    output_dir: str | Path,
) -> list[Path]:
    roots = [Path(path).expanduser().resolve() for path in run_dirs]
    if len(roots) < 2:
        raise ValueError("compare-runs requires at least two completed model studies")

    manifests = [_manifest(root) for root in roots]
    models = [str(manifest["model"]) for manifest in manifests]
    if len(set(models)) != len(models):
        raise ValueError(f"Each compared run must contain a different model; got {models}")

    windows = list((manifests[0].get("windows_ns") or {}).keys())
    if not windows:
        raise ValueError("Compared model studies contain no windows")
    for manifest in manifests[1:]:
        if manifest.get("windows_ns") != manifests[0].get("windows_ns"):
            raise ValueError("Compared model studies use different window definitions")

    output = Path(output_dir).expanduser().resolve()
    plot_dir = output / "plots"
    csv_dir = output / "csv"
    plot_dir.mkdir(parents=True, exist_ok=True)
    csv_dir.mkdir(parents=True, exist_ok=True)

    generated: list[Path] = []
    all_improvement_rows: list[dict[str, Any]] = []
    pairwise_rows: list[dict[str, Any]] = []
    correlation_rows: list[dict[str, Any]] = []

    with paper_context():
        for window in windows:
            subruns = [
                _subrun(root, manifest, window)
                for root, manifest in zip(roots, manifests)
            ]
            subrun_by_model = dict(zip(models, subruns))
            result_sets = [
                [row for row in read_results(subrun) if row.get("stage") == "test"]
                for subrun in subruns
            ]

            voltage_sets = [
                _voltage_signature(rows, model)
                for rows, model in zip(result_sets, models)
            ]
            if any(values != voltage_sets[0] for values in voltage_sets[1:]):
                raise ValueError(
                    f"{window}: compared runs contain different voltage sets: "
                    f"{dict(zip(models, voltage_sets))}"
                )
            voltages = list(voltage_sets[0])

            for rows, model in zip(result_sets, models):
                led_voltages = _voltage_signature(rows, "led")
                if led_voltages != voltage_sets[0]:
                    raise ValueError(
                        f"{window}/{model}: LED and model results use different voltages"
                    )

            combined_rows: list[dict[str, Any]] = []
            combined_rows.extend(_method_rows(result_sets[0], "led"))
            for model, rows in zip(models, result_sets):
                combined_rows.extend(_method_rows(rows, model))

            plot_ctr_vs_voltage(
                plot_dir,
                combined_rows,
                generated,
                filename=f"ctr_vs_voltage_{window}.pdf",
            )

            window_improvement: list[dict[str, Any]] = []
            for voltage in voltages:
                model_rows_at_voltage: dict[str, dict[str, Any]] = {}

                for model, rows in zip(models, result_sets):
                    led_row = _row_at_voltage(rows, "led", voltage)
                    model_row = _row_at_voltage(rows, model, voltage)
                    model_rows_at_voltage[model] = model_row

                    delta, delta_unc, relative, relative_unc = _result_difference(
                        led_row,
                        model_row,
                    )
                    row = {
                        "window": window,
                        "dataset": str(model_row.get("dataset", "")),
                        "voltage_V": float(voltage),
                        "model": model,
                        "led_ctr_ps": float(led_row["ctr_ps"]),
                        "model_ctr_ps": float(model_row["ctr_ps"]),
                        "improvement_ps": delta,
                        "uncertainty_ps": delta_unc,
                        "improvement_percent": relative,
                        "uncertainty_percent": relative_unc,
                        "uncertainty_method": "independent_propagation",
                    }
                    window_improvement.append(row)
                    all_improvement_rows.append(row)

                for model_a, model_b in combinations(models, 2):
                    row_a = model_rows_at_voltage[model_a]
                    row_b = model_rows_at_voltage[model_b]
                    delta, delta_unc, relative, relative_unc = _result_difference(
                        row_b,
                        row_a,
                    )
                    pairwise_rows.append(
                        {
                            "window": window,
                            "dataset_a": str(row_a.get("dataset", "")),
                            "dataset_b": str(row_b.get("dataset", "")),
                            "voltage_V": float(voltage),
                            "model_a": model_a,
                            "model_b": model_b,
                            "model_a_ctr_ps": float(row_a["ctr_ps"]),
                            "model_b_ctr_ps": float(row_b["ctr_ps"]),
                            "model_a_improvement_over_model_b_ps": delta,
                            "uncertainty_ps": delta_unc,
                            "model_a_improvement_over_model_b_percent": relative,
                            "uncertainty_percent": relative_unc,
                            "uncertainty_method": "independent_propagation",
                        }
                    )

                    dataset_a = str(row_a.get("dataset", ""))
                    dataset_b = str(row_b.get("dataset", ""))
                    output_a, output_b = _aligned_model_outputs(
                        subrun_by_model[model_a],
                        dataset_a,
                        model_a,
                        subrun_by_model[model_b],
                        dataset_b,
                        model_b,
                    )
                    correlation, n = _pearson(output_a, output_b)
                    correlation_rows.append(
                        {
                            "window": window,
                            "dataset_a": dataset_a,
                            "dataset_b": dataset_b,
                            "voltage_V": float(voltage),
                            "model_a": model_a,
                            "model_b": model_b,
                            "pearson_r": correlation,
                            "n": n,
                            "population": "blind/test",
                            "alignment": "test_event_index",
                        }
                    )

            absolute_path = _plot_improvement(
                plot_dir,
                window,
                window_improvement,
                relative=False,
            )
            relative_path = _plot_improvement(
                plot_dir,
                window,
                window_improvement,
                relative=True,
            )
            correlation_path = _plot_model_output_correlation(
                plot_dir,
                window,
                correlation_rows,
            )
            if absolute_path is not None:
                generated.append(absolute_path)
            if relative_path is not None:
                generated.append(relative_path)
            if correlation_path is not None:
                generated.append(correlation_path)
            generated.extend(
                _plot_pairwise_model_improvement(
                    plot_dir,
                    window,
                    pairwise_rows,
                )
            )

    write_csv(csv_dir / "improvement_vs_led.csv", all_improvement_rows)
    write_csv(csv_dir / "pairwise_model_comparison.csv", pairwise_rows)
    write_csv(csv_dir / "model_output_correlations.csv", correlation_rows)

    comparison_manifest = {
        "schema_version": 3,
        "type": "model_run_comparison",
        "runs": [str(root) for root in roots],
        "models": models,
        "windows_ns": manifests[0]["windows_ns"],
        "comparison_basis": "persisted blind/test result rows only",
        "checks": ["matching window definitions", "matching voltage sets"],
        "dataset_identity_checked": False,
        "event_identity_checked": False,
        "residual_identity_checked": False,
        "model_output_correlation": {
            "metric": "Pearson r",
            "population": "blind/test",
            "event_identity_checked": True,
            "alignment": "test_event_index",
        },
        "uncertainty_method": "independent propagation from persisted CTR uncertainties",
    }
    (output / "manifest.json").write_text(
        json.dumps(comparison_manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return generated
