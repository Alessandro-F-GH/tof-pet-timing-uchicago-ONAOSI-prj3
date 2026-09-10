from __future__ import annotations

import argparse
from pathlib import Path

from ..ml_pipeline.model_output_reporting import make_model_output_reports
from ..ml_pipeline.reporting import LABELS, make_plots


def main() -> None:
    parser = argparse.ArgumentParser(description="Create presentation plots from stored holdout/blind results")
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("presentation/plots"))
    args = parser.parse_args()

    paths = make_plots(args.run_dir, args.output_dir)
    paths.extend(
        make_model_output_reports(
            args.run_dir,
            args.output_dir / "model_output",
            labels=LABELS,
        )
    )
    for path in paths:
        print(path)


if __name__ == "__main__":
    main()
