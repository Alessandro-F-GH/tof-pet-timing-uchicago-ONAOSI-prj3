from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from .ml_pipeline.batch import run_batch
from .ml_pipeline.binning_scan import run_ctr_binning_scan
from .ml_pipeline.config import load_batch_config, load_config, public_batch_config, public_config
from .ml_pipeline.postprocess import remake_batch_plots, remake_study_plots
from .ml_pipeline.report import batch_result_dirs, generate_report
from .ml_pipeline.report_naming import compact_report_filenames
from .ml_pipeline.study import run_study

PROJECT_ROOT = Path(__file__).resolve().parent


def _config_path(path):
    path = Path(path).expanduser()
    if path.is_file():
        return path
    candidate = PROJECT_ROOT / path
    return candidate if candidate.is_file() else path


def _parser():
    parser = argparse.ArgumentParser(
        prog="python -m waveform_analysis.cli",
        description="TOF-PET fixed-control ML studies with fixed hyperparameter validation and blind replicas",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    for name in ("check", "run", "remake-plots"):
        command = sub.add_parser(name)
        command.add_argument("--config", type=Path, required=True)

    check_batch = sub.add_parser("check-batch")
    check_batch.add_argument("--config", type=Path, required=True)

    run = sub.choices["run"]
    run_mode = run.add_mutually_exclusive_group()
    run_mode.add_argument("--overwrite", action="store_true")
    run_mode.add_argument("--resume", action="store_true")
    run.add_argument("--rebuild-preprocessing", action="store_true")
    run.add_argument("--remake-plots", action="store_true")

    batch = sub.add_parser("batch")
    batch.add_argument("--config", type=Path, required=True)
    batch.add_argument("--remake-plots", action="store_true")

    remake_batch = sub.add_parser("remake-batch-plots")
    remake_batch.add_argument("--config", type=Path, required=True)

    report = sub.add_parser(
        "report",
        help="Compare completed replica results by model, formulation, mode, and window",
    )
    source = report.add_mutually_exclusive_group(required=True)
    source.add_argument("--batch-config", type=Path)
    source.add_argument("--studies", type=Path, nargs="+")
    report.add_argument("--output-dir", type=Path)
    report.add_argument(
        "--report-config",
        type=Path,
        help="Optional partial JSON override for waveform_analysis/config/reporting.json",
    )

    scan = sub.add_parser(
        "ctr-binning-scan",
        help="Recompute blind-replica CTR for multiple histogram bin widths and correlate with RMSE",
    )
    scan.add_argument("--config", type=Path, required=True)
    scan.add_argument(
        "--bin-widths",
        type=float,
        nargs="+",
        default=[5.0, 7.5, 10.0, 12.5, 15.0, 20.0],
    )
    return parser


def _validate_remake_args(args):
    if not getattr(args, "remake_plots", False):
        return
    if (
        getattr(args, "overwrite", False)
        or getattr(args, "resume", False)
        or getattr(args, "rebuild_preprocessing", False)
    ):
        raise SystemExit(
            "--remake-plots cannot be combined with --overwrite, --resume, or --rebuild-preprocessing"
        )


def main():
    args = _parser().parse_args()

    if args.command == "check":
        config = load_config(_config_path(args.config), PROJECT_ROOT)
        print(json.dumps(public_config(config), indent=2))
        return

    if args.command == "check-batch":
        batch = load_batch_config(_config_path(args.config), PROJECT_ROOT)
        print(json.dumps(public_batch_config(batch), indent=2))
        return

    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    logger = logging.getLogger("waveform-batch")

    if args.command == "report":
        if args.batch_config is not None:
            batch = load_batch_config(_config_path(args.batch_config), PROJECT_ROOT)
            paths = batch_result_dirs(batch)
            output_dir = args.output_dir or (Path(batch.output_dir) / "report")
        else:
            paths = [Path(path).expanduser().resolve() for path in args.studies]
            output_dir = args.output_dir or (PROJECT_ROOT / "results" / "reports" / "comparison")
        report_config = None if args.report_config is None else _config_path(args.report_config)
        report_root = generate_report(
            paths,
            output_dir,
            logger=logger,
            report_config=report_config,
        )
        compact_report_filenames(report_root)
        print(report_root)
        return

    if args.command == "ctr-binning-scan":
        config = load_config(_config_path(args.config), PROJECT_ROOT)
        outputs = run_ctr_binning_scan(config, args.bin_widths, logger=logger)
        for path in outputs.values():
            print(Path(path).resolve())
        return

    if args.command == "remake-plots":
        config = load_config(_config_path(args.config), PROJECT_ROOT)
        remake_study_plots(config, logger=logger)
        print(Path(config["output_dir"]).resolve())
        return

    if args.command == "remake-batch-plots":
        batch = load_batch_config(_config_path(args.config), PROJECT_ROOT)
        for run_dir, _ in remake_batch_plots(batch, logger=logger):
            print(run_dir)
        return

    _validate_remake_args(args)

    if args.command == "batch":
        batch = load_batch_config(_config_path(args.config), PROJECT_ROOT)
        if args.remake_plots:
            for run_dir, _ in remake_batch_plots(batch, logger=logger):
                print(run_dir)
            return
        for output in run_batch(
            batch,
            logger=logger,
        ):
            print(output)
        return

    config = load_config(_config_path(args.config), PROJECT_ROOT)
    if args.remake_plots:
        remake_study_plots(config, logger=logger)
        print(Path(config["output_dir"]).resolve())
        return
    print(
        run_study(
            config,
            overwrite=args.overwrite,
            resume=args.resume,
            rebuild_preprocessing=args.rebuild_preprocessing,
        )
    )


if __name__ == "__main__":
    main()
