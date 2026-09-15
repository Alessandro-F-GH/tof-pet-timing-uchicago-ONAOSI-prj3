from __future__ import annotations

import csv
import json
import shutil
from pathlib import Path
from typing import Any

from .analyses import plot_threshold_scan
from .latex_tables import make_latex_tables
from .model_output_reporting import make_model_output_reports
from .plot_style import LABELS
from .reporting import make_plots


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


def rebuild_experiment_plots(
    run_dir: str | Path,
    output_dir: str | Path | None = None,
    *,
    latex_tables: bool = False,
) -> list[Path]:
    """Recreate plots for standard, model-study, or threshold-scan runs."""
    run = Path(run_dir).resolve()
    manifest_path = run / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Run manifest not found: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    experiment_type = str(
        manifest.get("experiment_type")
        or ((manifest.get("config") or {}).get("experiment") or {}).get("type", "standard")
    ).lower()

    if experiment_type == "model_study":
        destination = run if output_dir is None else Path(output_dir).expanduser().resolve()
        paths: list[Path] = []
        windows = list(
            (manifest.get("windows_ns") or (manifest.get("subruns") or {})).keys()
        )
        for window in windows:
            subrun = _subrun_path(run, manifest, window)
            sub_destination = subrun if output_dir is None else destination / window
            paths.extend(
                rebuild_study_plots(
                    subrun,
                    sub_destination,
                    latex_tables=latex_tables,
                )
            )
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
