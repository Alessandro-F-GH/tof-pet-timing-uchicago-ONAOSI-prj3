from __future__ import annotations

import argparse
from pathlib import Path

from ..ml_pipeline.reporting import make_plots


def main() -> None:
    parser = argparse.ArgumentParser(description="Create presentation plots from stored holdout/blind results")
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("presentation/plots"))
    args = parser.parse_args()
    for path in make_plots(args.run_dir, args.output_dir):
        print(path)


if __name__ == "__main__":
    main()
