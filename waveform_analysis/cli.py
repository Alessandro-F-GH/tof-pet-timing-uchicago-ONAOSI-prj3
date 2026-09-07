from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from .ml_pipeline.config import discover_root_files, load_config, public_config
from .ml_pipeline.data import preprocess_selected
from .ml_pipeline.event_selection import select_events
from .ml_pipeline.prepared_data import prepare_ml_dataset
from .ml_pipeline.reporting import make_plots
from .ml_pipeline.selection_outputs import ensure_selection_outputs
from .ml_pipeline.study import run_study

PROJECT_ROOT = Path(__file__).resolve().parent


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m waveform_analysis.cli", description="TOF-PET waveform pipeline: selection-first, native-time preprocessing, holdout ML")
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
    if not roots:
        raise FileNotFoundError("No ROOT files matched the configured source")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    logger = logging.getLogger("waveform-prepare")
    for root in roots:
        selection = select_events(root, config, rebuild=rebuild, logger=logger)
        ensure_selection_outputs(root, selection, config, logger)
        preprocessed = preprocess_selected(root, selection, config, rebuild=rebuild, logger=logger)
        prepare_ml_dataset(preprocessed, config, rebuild=rebuild, logger=logger)
    return len(roots)


def _confirm_same_config_rerun(config, *, overwrite: bool, rebuild_preprocessing: bool) -> bool:
    if not (overwrite or rebuild_preprocessing):
        return True
    run_dir = Path(config["experiment"]["output_dir"]).resolve()
    manifest_path = run_dir / "manifest.json"
    if not manifest_path.is_file():
        return True
    try:
        previous = json.loads(manifest_path.read_text(encoding="utf-8")).get("config")
    except (OSError, json.JSONDecodeError):
        return True
    if previous != public_config(config):
        return True
    action = []
    if overwrite:
        action.append("overwrite the existing run")
    if rebuild_preprocessing:
        action.append("rebuild preprocessing")
    prompt = f"Configuration is unchanged from the previous run. Do you still want to {' and '.join(action)}? [y/N]: "
    try:
        answer = input(prompt).strip().lower()
    except EOFError:
        answer = ""
    if answer in {"y", "yes"}:
        return True
    print(f"Keeping existing result: {run_dir}")
    return False


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
    if not _confirm_same_config_rerun(config, overwrite=args.overwrite, rebuild_preprocessing=args.rebuild_preprocessing):
        return
    print(run_study(config, overwrite=args.overwrite, rebuild_preprocessing=args.rebuild_preprocessing))


if __name__ == "__main__":
    main()
