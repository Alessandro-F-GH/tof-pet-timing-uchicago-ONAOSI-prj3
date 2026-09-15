from __future__ import annotations

import argparse
from pathlib import Path

from waveform_analysis.ml_pipeline.model_run_reporting import compare_model_runs


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Combine completed single-model studies produced on the same or "
            "different machines after verifying blind-event compatibility."
        )
    )
    parser.add_argument(
        "--runs",
        type=Path,
        nargs="+",
        required=True,
        help="completed model_study directories",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="destination for combined CSV files and publication plots",
    )
    return parser


def main() -> None:
    args = _parser().parse_args()
    for path in compare_model_runs(args.runs, args.output_dir):
        print(path)


if __name__ == "__main__":
    main()
