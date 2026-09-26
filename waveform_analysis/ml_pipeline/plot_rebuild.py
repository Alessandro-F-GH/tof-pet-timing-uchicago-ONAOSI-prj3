from __future__ import annotations

import json
import os
import shutil
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from . import reporting as reporting_module
from .analyses import plot_threshold_scan
from .common import read_csv
from .latex_tables import make_latex_tables
from .model_output_reporting import make_model_output_reports
from .plot_style import LABELS
from .reporting import make_plots, plot_model_study_windows

HISTOGRAM_BINS_ENV = "TOFPET_HISTOGRAM_BINS"
DEFAULT_HISTOGRAM_BINS = 22


def _reset(directory: Path) -> None:
    if directory.is_dir():
        shutil.rmtree(directory)


def _resolve_histogram_bins(histogram_bins: int | None) -> int:
    value = histogram_bins
    if value is None:
        raw = os.environ.get(HISTOGRAM_BINS_ENV)
        value = DEFAULT_HISTOGRAM_BINS if raw in (None, "") else int(raw)
    value = int(value)
    if value <= 0:
        raise ValueError("histogram_bins must be a positive integer")
    return value


@contextmanager
def _reporting_plot_options(histogram_bins: int):
    """Apply reporting-only histogram and legend options without changing CTR values."""
    from matplotlib.axes import Axes

    original_edges = reporting_module._median_centered_display_edges
    original_legend = Axes.legend

    def configured_edges(values, xlim, n_bins=DEFAULT_HISTOGRAM_BINS):
        return original_edges(values, xlim, histogram_bins)

    def configured_legend(self, *args, **kwargs):
        if self.get_xlabel() == "Residual [ps]" and self.get_ylabel() == "Events [count]":
            kwargs["loc"] = "lower center"
            kwargs["bbox_to_anchor"] = (0.5, 1.02)
            kwargs.setdefault("ncol", 2)
            kwargs.setdefault("borderaxespad", 0.0)
        return original_legend(self, *args, **kwargs)

    reporting_module._median_centered_display_edges = configured_edges
    Axes.legend = configured_legend
    try:
        yield
    finally:
        reporting_module._median_centered_display_edges = original_edges
        Axes.legend = original_legend


def rebuild_study_plots(
    run_dir: str | Path,
    output_dir: str | Path | None = None,
    *,
    latex_tables: bool = False,
    histogram_bins: int | None = None,
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
    bins = _resolve_histogram_bins(histogram_bins)

    paths: list[Path] = []
    with _reporting_plot_options(bins):
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
    local = root / window
    if local.is_dir():
        return local.resolve()
    configured = (manifest.get("subruns") or {}).get(window)
    if configured:
        candidate = Path(str(configured)).expanduser()
        if candidate.is_dir():
            return candidate.resolve()
    return local.resolve()


def rebuild_experiment_plots(
    run_dir: str | Path,
    output_dir: str | Path | None = None,
    *,
    latex_tables: bool = False,
    histogram_bins: int | None = None,
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
                    histogram_bins=histogram_bins,
                )
            )
        plot_model_study_windows(
            run,
            manifest,
            paths,
            output_dir=destination / "plots",
        )
        return paths

    if experiment_type == "threshold_scan":
        destination = run if output_dir is None else Path(output_dir).expanduser().resolve()
        _reset(destination / "plots")
        rows = read_csv(run / "csv" / "threshold_scan.csv")
        if not rows:
            raise FileNotFoundError(
                f"Threshold-scan results not found: {run / 'csv' / 'threshold_scan.csv'}"
            )
        paths = plot_threshold_scan(destination, rows)
        if latex_tables:
            table_root = destination / "latex_tables"
            _reset(table_root)
            paths.extend(make_latex_tables(run, table_root))
        return paths

    return rebuild_study_plots(
        run,
        output_dir,
        latex_tables=latex_tables,
        histogram_bins=histogram_bins,
    )
