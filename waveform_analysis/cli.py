from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from utils_fit import (
    CTR_DEFINITIONS,
    DEFAULT_CTR_DEFINITION,
    DEFAULT_HISTOGRAM_BIN_WIDTH_PS,
)

from .ml_pipeline.concatenate import concatenate_prepared_datasets
from .ml_pipeline.config import discover_root_files, load_config, public_config
from .ml_pipeline.data import preprocess_selected
from .ml_pipeline.event_selection import select_events
from .ml_pipeline.prepared_data import prepare_ml_dataset
from .ml_pipeline.preflight import confirm_overwrite, inspect_preprocessing, study_overwrite_path
from .ml_pipeline.plot_rebuild import rebuild_experiment_plots
from .ml_pipeline.model_run_reporting import compare_model_runs
from .ml_pipeline.latex_tables import make_dataset_latex_table
from .ml_pipeline.selection_outputs import ensure_selection_outputs
from .ml_pipeline.study import run_study

PROJECT_ROOT = Path(__file__).resolve().parent


def _config_path(path: str | Path) -> Path:
    """Resolve CLI config paths from either the working directory or waveform_analysis/."""
    candidate = Path(path).expanduser()
    if candidate.is_file():
        return candidate
    project_candidate = PROJECT_ROOT / candidate
    return project_candidate if project_candidate.is_file() else candidate


def _positive_float(value: str) -> float:
    parsed = float(value)
    if not parsed > 0.0:
        raise argparse.ArgumentTypeError("value must be a positive number")
    return parsed


def _add_reporting_fit_options(command: argparse.ArgumentParser) -> None:
    command.add_argument(
        "--histogram-bin-width-ps",
        type=_positive_float,
        default=DEFAULT_HISTOGRAM_BIN_WIDTH_PS,
        help=(
            "fixed residual-histogram bin width in ps used for reporting and "
            "histogram-based CTR definitions; 0 ps is always a bin center "
            f"(default: {DEFAULT_HISTOGRAM_BIN_WIDTH_PS:g} ps)"
        ),
    )
    command.add_argument(
        "--ctr-definition",
        choices=CTR_DEFINITIONS,
        default=DEFAULT_CTR_DEFINITION,
        help=(
            "CTR definition used only while generating reports; experiment selection "
            f"always uses {DEFAULT_CTR_DEFINITION!r} (default: {DEFAULT_CTR_DEFINITION})"
        ),
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m waveform_analysis.cli", description="TOF-PET waveform pipeline: single-mode selection, native-time preprocessing, holdout ML")
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("check", "prepare", "run"):
        command = commands.add_parser(name)
        command.add_argument("--config", type=Path, required=True)
    commands.choices["prepare"].add_argument("--rebuild", action="store_true")
    run = commands.choices["run"]
    mode = run.add_mutually_exclusive_group()
    mode.add_argument("--overwrite", action="store_true")
    mode.add_argument("--resume", action="store_true")
    mode.add_argument(
        "--remake-plots",
        action="store_true",
        help="recreate plots from existing run artifacts without rerunning preprocessing or training",
    )
    run.add_argument("--rebuild-preprocessing", action="store_true")
    _add_reporting_fit_options(run)
    compare_runs = commands.add_parser("compare-runs")
    compare_runs.add_argument(
        "--runs",
        type=Path,
        nargs="+",
        required=True,
        help="completed model-study directories copied from any machines",
    )
    compare_runs.add_argument("--output-dir", type=Path, required=True)
    dataset_table = commands.add_parser(
        "dataset-table",
        help="export a LaTeX dataset table from existing selection caches only",
    )
    dataset_table.add_argument("--config", type=Path, required=True)
    dataset_table.add_argument("--output-file", type=Path, required=True)
    dataset_table.add_argument("--caption", required=True)
    dataset_table.add_argument("--label", required=True)
    report = commands.add_parser("report")
    report.add_argument("--run-dir", type=Path, required=True)
    report.add_argument("--output-dir", type=Path)
    _add_reporting_fit_options(report)
    report.add_argument(
        "--latex-tables",
        action="store_true",
        help="also export LaTeX tables from persisted numerical results",
    )
    return parser


def _prepare(config, rebuild: bool) -> int:
    roots = discover_root_files(config)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    logger = logging.getLogger("waveform-prepare")
    prepared = []
    for root in roots:
        selection = select_events(root, config, rebuild=rebuild, logger=logger)
        ensure_selection_outputs(root, selection, config, logger)
        preprocessed = preprocess_selected(root, selection, config, rebuild=rebuild, logger=logger)
        prepared.append(prepare_ml_dataset(preprocessed, config, rebuild=rebuild, logger=logger))

    if bool(config["experiment"].get("concatenate_datasets", False)):
        name = str(config["experiment"].get("concatenated_dataset_name", "concatenated"))
        concatenate_prepared_datasets(
            prepared,
            Path(config["preprocessing"]["prepared_dir"]) / name,
            config,
            name=name,
            rebuild=rebuild,
            logger=logger,
        )
    return len(roots)


def main() -> None:
    args = _parser().parse_args()
    if args.command == "compare-runs":
        for path in compare_model_runs(args.runs, args.output_dir):
            print(path)
        return
    if args.command == "report":
        for path in rebuild_experiment_plots(
            args.run_dir,
            args.output_dir,
            latex_tables=args.latex_tables,
            histogram_bin_width_ps=args.histogram_bin_width_ps,
            ctr_definition=args.ctr_definition,
        ):
            print(path)
        return
    if args.command == "dataset-table":
        config = load_config(_config_path(args.config), PROJECT_ROOT)
        print(
            make_dataset_latex_table(
                config,
                args.output_file,
                caption=args.caption,
                label=args.label,
            )
        )
        return
    config = load_config(_config_path(args.config), PROJECT_ROOT)
    if args.command == "run" and args.remake_plots:
        run_dir = Path(config["experiment"]["output_dir"])
        for path in rebuild_experiment_plots(
            run_dir,
            histogram_bin_width_ps=args.histogram_bin_width_ps,
            ctr_definition=args.ctr_definition,
        ):
            print(path)
        return
    if args.command == "check":
        print(json.dumps({"config": "OK", "root_files": [str(path) for path in discover_root_files(config)], "resolved": public_config(config)}, indent=2))
        return
    if args.command == "prepare":
        preflight = inspect_preprocessing(config, rebuild=args.rebuild)
        if not confirm_overwrite(list(preflight.overwrite_paths)):
            print("No files were changed.")
            return
        print(f"Prepared {_prepare(config, args.rebuild)} source dataset(s)")
        return
    experiment_type = str(config["experiment"].get("type", "standard"))
    include_prepared = experiment_type == "standard"
    preflight = inspect_preprocessing(
        config,
        rebuild=args.rebuild_preprocessing,
        include_prepared=include_prepared,
    )
    run_overwrite = study_overwrite_path(
        config,
        overwrite=args.overwrite,
        resume=args.resume,
    )
    overwrite_paths = list(preflight.overwrite_paths)
    if run_overwrite is not None:
        overwrite_paths.append(run_overwrite)
    if not confirm_overwrite(overwrite_paths):
        print("No files were changed.")
        return
    run_dir = run_study(
        config,
        overwrite=args.overwrite,
        resume=args.resume,
        rebuild_preprocessing=args.rebuild_preprocessing,
    )
    if (
        args.histogram_bin_width_ps != DEFAULT_HISTOGRAM_BIN_WIDTH_PS
        or args.ctr_definition != DEFAULT_CTR_DEFINITION
    ):
        rebuild_experiment_plots(
            run_dir,
            histogram_bin_width_ps=args.histogram_bin_width_ps,
            ctr_definition=args.ctr_definition,
        )
    print(run_dir)


if __name__ == "__main__":
    main()
