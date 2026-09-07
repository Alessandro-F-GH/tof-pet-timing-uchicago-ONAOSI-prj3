from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from .ml_pipeline.config import discover_root_files, load_config, public_config
from .ml_pipeline.prepared_data import plot_prepared_signal_examples, prepare_file_dataset
from .ml_pipeline.reporting import make_plots
from .ml_pipeline.study import run_study

PROJECT_ROOT = Path(__file__).resolve().parent


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m waveform_analysis.cli",
        description="TOF-PET waveform timing pipeline: one deterministic holdout + untouched blind set",
    )
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("check", "prepare", "run"):
        command = commands.add_parser(name)
        command.add_argument("--config", type=Path, required=True)
    prepare = commands.choices["prepare"]
    prepare.add_argument("--rebuild", action="store_true")
    run = commands.choices["run"]
    run.add_argument("--overwrite", action="store_true")
    run.add_argument("--rebuild-preprocessing", action="store_true")
    report = commands.add_parser("report")
    report.add_argument("--run-dir", type=Path, required=True)
    report.add_argument("--output-dir", type=Path)
    return parser


def _prepare(config, rebuild: bool) -> int:
    roots = discover_root_files(config)
    if not roots:
        raise FileNotFoundError("No ROOT files matched the configured data source")
    logger = logging.getLogger("waveform-prepare")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    examples = Path(config["preprocessing"]["prepared_dir"]) / "examples"
    for root in roots:
        dataset = prepare_file_dataset(config, root, rebuild=rebuild, logger=logger)
        plot_prepared_signal_examples(dataset, examples / f"{root.stem}.png", dpi=int(config.get("reporting", {}).get("dpi", 180)))
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
        print(f"Prepared {_prepare(config, args.rebuild)} dataset(s)")
        return
    print(run_study(config, overwrite=args.overwrite, rebuild_preprocessing=args.rebuild_preprocessing))


if __name__ == "__main__":
    main()
