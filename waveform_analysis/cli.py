from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from .ml_pipeline.concatenate import concatenate_prepared_datasets
from .ml_pipeline.config import discover_root_files, load_config, public_config
from .ml_pipeline.data import preprocess_selected
from .ml_pipeline.event_selection import select_events
from .ml_pipeline.prepared_data import prepare_ml_dataset
from .ml_pipeline.preflight import confirm_overwrite, inspect_preprocessing, study_overwrite_path
from .ml_pipeline.reporting import make_plots
from .ml_pipeline.selection_outputs import ensure_selection_outputs
from .ml_pipeline.study import run_study

PROJECT_ROOT = Path(__file__).resolve().parent


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m waveform_analysis.cli", description="TOF-PET waveform pipeline: single-mode selection, native-time preprocessing, holdout ML")
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("check", "prepare", "run"):
        command = commands.add_parser(name)
        command.add_argument("--config", type=Path, required=True)
    commands.choices["prepare"].add_argument("--rebuild", action="store_true")
    run = commands.choices["run"]
    run.add_argument("--overwrite", action="store_true")
    run.add_argument("--rebuild-preprocessing", action="store_true")
    report = commands.add_parser("report")
    report.add_argument("--run-dir", type=Path, required=True)
    report.add_argument("--output-dir", type=Path)
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
            logger=logger,
        )
    return len(roots)


def main() -> None:
    args = _parser().parse_args()
    if args.command == "report":
        for path in make_plots(args.run_dir, args.output_dir):
            print(path)
        return
    config = load_config(args.config, PROJECT_ROOT)
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
    preflight = inspect_preprocessing(config, rebuild=args.rebuild_preprocessing)
    run_overwrite = study_overwrite_path(config, overwrite=args.overwrite)
    overwrite_paths = list(preflight.overwrite_paths)
    if run_overwrite is not None:
        overwrite_paths.append(run_overwrite)
    if not confirm_overwrite(overwrite_paths):
        print("No files were changed.")
        return
    print(run_study(config, overwrite=args.overwrite, rebuild_preprocessing=args.rebuild_preprocessing))


if __name__ == "__main__":
    main()
