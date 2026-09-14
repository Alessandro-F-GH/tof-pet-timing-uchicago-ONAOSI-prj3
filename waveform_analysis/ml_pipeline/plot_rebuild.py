from __future__ import annotations

import shutil
from pathlib import Path

from .analyses import make_analysis_plots
from .model_output_reporting import make_model_output_reports
from .plot_style import LABELS
from .reporting import make_plots


def _reset(directory: Path) -> None:
    if directory.is_dir():
        shutil.rmtree(directory)


def rebuild_study_plots(
    run_dir: str | Path,
    output_dir: str | Path | None = None,
) -> list[Path]:
    """Recreate every study/report plot from persisted run artifacts only."""
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
    paths.extend(make_analysis_plots(run, plot_root / "analyses"))
    return paths
