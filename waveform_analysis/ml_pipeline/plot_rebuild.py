from __future__ import annotations

import csv
import json
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import matplotlib.pyplot as plt

from .analyses import plot_threshold_scan
from .latex_tables import make_latex_tables
from .model_output_reporting import make_model_output_reports
from .plot_style import (
    DOUBLE_COLUMN,
    LABELS,
    clean_axis,
    model_style,
    paper_context,
    save_figure,
    set_voltage_ticks,
)
from .reporting import (
    make_plots,
    plot_ctr_vs_voltage,
    plot_improvement_vs_led,
    read_results,
)


def _reset(directory: Path) -> None:
    if directory.is_dir():
        shutil.rmtree(directory)


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def rebuild_study_plots(
    run_dir: str | Path,
    output_dir: str | Path | None = None,
    *,
    latex_tables: bool = False,
) -> list[Path]:
    """Recreate every ordinary-study/report plot from persisted artifacts only."""
    run = Path(run_dir).resolve()
    if not (run / "manifest.json").is_file():
        raise FileNotFoundError(f"Study manifest not found: {run / 'manifest.json'}")
    if not (run / "csv" / "results.csv").is_file():
        raise FileNotFoundError(f"Study results not found: {run / 'csv' / 'results.csv'}")

    destination = run if output_dir is None else Path(output_dir).expanduser().resolve()
    plot_root = destination / "plots"
    _reset(plot_root)

    paths: list[Path] = []
    paths.extend(make_plots(run, plot_root))
    paths.extend(
        make_model_output_reports(
            run,
            plot_root / "model_output_diagnostics",
            labels=LABELS,
        )
    )
    if latex_tables:
        table_root = destination / "latex_tables"
        _reset(table_root)
        paths.extend(make_latex_tables(run, table_root))
    return paths


def _subrun_path(root: Path, manifest: dict[str, Any], window: str) -> Path:
    configured = (manifest.get("subruns") or {}).get(window)
    if configured:
        candidate = Path(str(configured)).expanduser()
        if candidate.is_dir():
            return candidate.resolve()
    return (root / window).resolve()


def rebuild_model_comparison_root_plots(
    run_dir: str | Path,
    output_dir: str | Path | None = None,
) -> list[Path]:
    """Recreate model-comparison root plots from persisted CSV/results only."""
    run = Path(run_dir).resolve()
    manifest_path = run / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Experiment manifest not found: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    destination = run if output_dir is None else Path(output_dir).expanduser().resolve()
    plot_root = destination / "plots"
    plot_root.mkdir(parents=True, exist_ok=True)

    paired_rows = _read_csv(run / "csv" / "paired_model_comparison.csv")
    generated: list[Path] = []

    with paper_context():
        for window in ("onishi", "wide"):
            subrun = _subrun_path(run, manifest, window)
            results_path = subrun / "csv" / "results.csv"
            if results_path.is_file():
                test_rows = [
                    row for row in read_results(subrun)
                    if row.get("stage") == "test"
                ]
                plot_ctr_vs_voltage(
                    plot_root,
                    test_rows,
                    generated,
                    filename=f"ctr_vs_voltage_{window}.pdf",
                )
                sub_manifest = json.loads(
                    (subrun / "manifest.json").read_text(encoding="utf-8")
                )
                plot_improvement_vs_led(
                    subrun,
                    plot_root,
                    test_rows,
                    sub_manifest,
                    generated,
                    filename=f"improvement_vs_led_{window}.pdf",
                )

            subset = sorted(
                [
                    row for row in paired_rows
                    if str(row.get("window")) == window
                ],
                key=lambda row: float(row["voltage_V"]),
            )
            if not subset:
                continue
            x = np.asarray(
                [float(row["voltage_V"]) for row in subset],
                dtype=float,
            )
            improvement = np.asarray(
                [
                    float(row["mlp_improvement_over_onishi_percent"])
                    for row in subset
                ],
                dtype=float,
            )
            uncertainty = np.asarray(
                [
                    float(row["paired_bootstrap_uncertainty_percent"])
                    for row in subset
                ],
                dtype=float,
            )
            fig, ax = plt.subplots(figsize=DOUBLE_COLUMN)
            ax.errorbar(
                x,
                improvement,
                yerr=np.where(np.isfinite(uncertainty), uncertainty, 0.0),
                capsize=2.5,
                **model_style("mlp"),
            )
            ax.axhline(0.0, color="#7F7F7F", linestyle=":", linewidth=0.9)
            set_voltage_ticks(ax, x)
            ax.set_xlabel("Voltage [V]")
            ax.set_ylabel("Improvement [%]")
            clean_axis(ax, grid="y")
            fig.tight_layout()
            target = save_figure(
                fig,
                plot_root / f"paired_model_improvement_{window}.pdf",
            )
            plt.close(fig)
            generated.append(target)

    return generated


def rebuild_experiment_plots(
    run_dir: str | Path,
    output_dir: str | Path | None = None,
    *,
    latex_tables: bool = False,
) -> list[Path]:
    """Recreate plots for standard, model-comparison, or threshold-scan runs."""
    run = Path(run_dir).resolve()
    manifest_path = run / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Run manifest not found: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    experiment_type = str(
        manifest.get("experiment_type")
        or ((manifest.get("config") or {}).get("experiment") or {}).get("type", "standard")
    ).lower()

    if experiment_type == "model_comparison":
        destination = run if output_dir is None else Path(output_dir).expanduser().resolve()
        _reset(destination / "plots")
        paths: list[Path] = []
        for window in ("onishi", "wide"):
            subrun = _subrun_path(run, manifest, window)
            sub_destination = (
                subrun
                if output_dir is None
                else destination / window
            )
            paths.extend(
                rebuild_study_plots(
                    subrun,
                    sub_destination,
                    latex_tables=latex_tables,
                )
            )
        paths.extend(rebuild_model_comparison_root_plots(run, destination))
        return paths

    if experiment_type == "threshold_scan":
        destination = run if output_dir is None else Path(output_dir).expanduser().resolve()
        _reset(destination / "plots")
        rows = _read_csv(run / "csv" / "threshold_scan.csv")
        if not rows:
            raise FileNotFoundError(
                f"Threshold-scan results not found: {run / 'csv' / 'threshold_scan.csv'}"
            )
        return plot_threshold_scan(destination, rows)

    return rebuild_study_plots(
        run,
        output_dir,
        latex_tables=latex_tables,
    )
