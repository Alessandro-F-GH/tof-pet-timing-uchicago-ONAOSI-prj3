from __future__ import annotations

import csv
import json
from itertools import combinations
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from utils_fit import fit_ctr_ps

from .common import voltage_from_name
from .plot_style import (
    DOUBLE_COLUMN,
    LABELS,
    clean_axis,
    model_style,
    paper_context,
    save_figure,
    set_voltage_ticks,
)
from .reporting import plot_ctr_vs_voltage, read_results
from .splits import semantic_seed


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
    configured = (manifest.get("subruns") or {}).get(window)
    if configured:
        candidate = Path(str(configured)).expanduser()
        if candidate.is_dir():
            return candidate.resolve()
    candidate = root / window
    if not candidate.is_dir():
        raise FileNotFoundError(f"Missing window subrun {window!r}: {candidate}")
    return candidate.resolve()


def _test_event_index(subrun: Path, dataset: str) -> np.ndarray:
    path = subrun / "splits" / f"{dataset}.npz"
    if not path.is_file():
        raise FileNotFoundError(f"Missing persisted split file: {path}")
    with np.load(path) as split:
        if "test_event_index" not in split:
            raise ValueError(f"Split file has no test_event_index: {path}")
        return np.asarray(split["test_event_index"], dtype=np.int64)


def _residual(subrun: Path, dataset: str, method: str) -> np.ndarray:
    path = subrun / "artifacts" / dataset / f"{method}_test_residuals_ps.npy"
    if not path.is_file():
        raise FileNotFoundError(f"Missing blind residuals: {path}")
    return np.asarray(np.load(path), dtype=np.float64).reshape(-1)


def _paired_difference(
    reference: np.ndarray,
    method: np.ndarray,
    fit_config: dict[str, Any],
    *,
    samples: int,
    seed: int,
) -> tuple[float, float, float, float, int]:
    reference = np.asarray(reference, dtype=np.float64).reshape(-1)
    method = np.asarray(method, dtype=np.float64).reshape(-1)
    if reference.shape != method.shape:
        raise ValueError("Paired residual arrays have different shapes")
    finite = np.isfinite(reference) & np.isfinite(method)
    reference = reference[finite]
    method = method[finite]
    if reference.size < 2:
        raise ValueError("Paired comparison requires at least two finite blind events")

    ref_ctr = float(fit_ctr_ps(reference, fit_config, bootstrap=False).ctr_ps)
    method_ctr = float(fit_ctr_ps(method, fit_config, bootstrap=False).ctr_ps)
    delta = ref_ctr - method_ctr
    relative = 100.0 * delta / ref_ctr if ref_ctr > 0 else float("nan")

    rng = np.random.default_rng(int(seed))
    delta_boot: list[float] = []
    relative_boot: list[float] = []
    for _ in range(int(samples)):
        idx = rng.integers(0, reference.size, size=reference.size)
        try:
            ref_b = float(fit_ctr_ps(reference[idx], fit_config, bootstrap=False).ctr_ps)
            method_b = float(fit_ctr_ps(method[idx], fit_config, bootstrap=False).ctr_ps)
        except ValueError:
            continue
        if not (np.isfinite(ref_b) and ref_b > 0 and np.isfinite(method_b)):
            continue
        value = ref_b - method_b
        delta_boot.append(value)
        relative_boot.append(100.0 * value / ref_b)

    delta_unc = float(np.std(delta_boot, ddof=1)) if len(delta_boot) > 1 else float("nan")
    relative_unc = (
        float(np.std(relative_boot, ddof=1))
        if len(relative_boot) > 1
        else float("nan")
    )
    return delta, delta_unc, relative, relative_unc, len(delta_boot)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


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
        ax.errorbar(
            x,
            y,
            yerr=np.where(np.isfinite(e), e, 0.0),
            capsize=2.5,
            label=LABELS.get(model, model),
            **model_style(model, index),
        )
    ax.axhline(0.0, color="#7F7F7F", linestyle=":", linewidth=0.9)
    set_voltage_ticks(ax, all_voltage)
    ax.set_xlabel("Voltage [V]")
    ax.set_ylabel("Improvement [%]" if relative else "Improvement [ps]")
    ax.legend(loc="best", ncol=2)
    clean_axis(ax, grid="y")
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
        ax.errorbar(
            x,
            y,
            yerr=np.where(np.isfinite(e), e, 0.0),
            capsize=2.5,
            **model_style(model_a),
        )
        ax.axhline(0.0, color="#7F7F7F", linestyle=":", linewidth=0.9)
        set_voltage_ticks(ax, x)
        ax.set_xlabel("Voltage [V]")
        ax.set_ylabel("Improvement [%]")
        clean_axis(ax, grid="y")
        fig.tight_layout()
        target = save_figure(
            fig,
            output / f"{model_a}_vs_{model_b}_{window}.pdf",
        )
        plt.close(fig)
        generated.append(target)
    return generated


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

    signatures = [str((manifest.get("compatibility") or {}).get("signature", "")) for manifest in manifests]
    if not signatures[0] or len(set(signatures)) != 1:
        raise ValueError(
            "Model-study compatibility signatures differ. "
            "Use runs with identical data, preprocessing, split policy, LED threshold, "
            "CTR settings, and profile-defined windows."
        )

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

    fit_config = dict((manifests[0].get("config") or {}).get("fit") or {})
    samples = int(fit_config.get("bootstrap_samples", 0))
    seed = int(((manifests[0].get("config") or {}).get("validation") or {}).get("seed", 0))

    generated: list[Path] = []
    all_improvement_rows: list[dict[str, Any]] = []
    pairwise_rows: list[dict[str, Any]] = []

    with paper_context():
        for window in windows:
            subruns = [
                _subrun(root, manifest, window)
                for root, manifest in zip(roots, manifests)
            ]
            result_sets = [
                [row for row in read_results(subrun) if row.get("stage") == "test"]
                for subrun in subruns
            ]
            dataset_sets = [
                {str(row["dataset"]) for row in rows if row.get("method") == "led"}
                for rows in result_sets
            ]
            if any(value != dataset_sets[0] for value in dataset_sets[1:]):
                raise ValueError(f"{window}: compared runs contain different blind datasets")
            datasets = sorted(dataset_sets[0], key=voltage_from_name)

            combined_rows: list[dict[str, Any]] = []
            first_led = {
                str(row["dataset"]): row
                for row in result_sets[0]
                if row.get("method") == "led"
            }
            combined_rows.extend(first_led.values())
            for model, rows in zip(models, result_sets):
                model_rows = [row for row in rows if row.get("method") == model]
                if {str(row["dataset"]) for row in model_rows} != set(datasets):
                    raise ValueError(f"{window}/{model}: missing model blind results")
                combined_rows.extend(model_rows)

            plot_ctr_vs_voltage(
                plot_dir,
                combined_rows,
                generated,
                filename=f"ctr_vs_voltage_{window}.pdf",
            )

            window_improvement: list[dict[str, Any]] = []
            for dataset in datasets:
                event_ids = [
                    _test_event_index(subrun, dataset)
                    for subrun in subruns
                ]
                for other in event_ids[1:]:
                    if not np.array_equal(event_ids[0], other):
                        raise ValueError(
                            f"{window}/{dataset}: blind event identities/order differ across runs"
                        )

                led_residuals = [
                    _residual(subrun, dataset, "led")
                    for subrun in subruns
                ]
                for other in led_residuals[1:]:
                    if not np.allclose(
                        led_residuals[0],
                        other,
                        rtol=0.0,
                        atol=1e-9,
                        equal_nan=True,
                    ):
                        raise ValueError(
                            f"{window}/{dataset}: LED residuals differ across runs"
                        )

                voltage = float(voltage_from_name(dataset))
                model_residuals: dict[str, np.ndarray] = {}
                for model, subrun in zip(models, subruns):
                    residual = _residual(subrun, dataset, model)
                    model_residuals[model] = residual
                    delta, delta_unc, relative, relative_unc, successful = _paired_difference(
                        led_residuals[0],
                        residual,
                        fit_config,
                        samples=samples,
                        seed=semantic_seed(
                            seed,
                            window,
                            dataset,
                            model,
                            "compare_runs_vs_led",
                        ),
                    )
                    row = {
                        "window": window,
                        "dataset": dataset,
                        "voltage_V": voltage,
                        "model": model,
                        "improvement_ps": delta,
                        "uncertainty_ps": delta_unc,
                        "improvement_percent": relative,
                        "uncertainty_percent": relative_unc,
                        "paired_bootstrap_successful": successful,
                    }
                    window_improvement.append(row)
                    all_improvement_rows.append(row)

                for model_a, model_b in combinations(models, 2):
                    # Positive means model_a has lower CTR than model_b.
                    delta, delta_unc, relative, relative_unc, successful = _paired_difference(
                        model_residuals[model_b],
                        model_residuals[model_a],
                        fit_config,
                        samples=samples,
                        seed=semantic_seed(
                            seed,
                            window,
                            dataset,
                            model_a,
                            model_b,
                            "compare_runs_pairwise",
                        ),
                    )
                    pairwise_rows.append(
                        {
                            "window": window,
                            "dataset": dataset,
                            "voltage_V": voltage,
                            "model_a": model_a,
                            "model_b": model_b,
                            "model_a_improvement_over_model_b_ps": delta,
                            "uncertainty_ps": delta_unc,
                            "model_a_improvement_over_model_b_percent": relative,
                            "uncertainty_percent": relative_unc,
                            "paired_bootstrap_successful": successful,
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
            if absolute_path is not None:
                generated.append(absolute_path)
            if relative_path is not None:
                generated.append(relative_path)
            generated.extend(
                _plot_pairwise_model_improvement(
                    plot_dir,
                    window,
                    pairwise_rows,
                )
            )

    _write_csv(csv_dir / "improvement_vs_led.csv", all_improvement_rows)
    _write_csv(csv_dir / "pairwise_model_comparison.csv", pairwise_rows)

    comparison_manifest = {
        "schema_version": 1,
        "type": "model_run_comparison",
        "runs": [str(root) for root in roots],
        "models": models,
        "windows_ns": manifests[0]["windows_ns"],
        "compatibility_signature": signatures[0],
        "paired_population": "blind test event identities verified across runs",
    }
    (output / "manifest.json").write_text(
        json.dumps(comparison_manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return generated
