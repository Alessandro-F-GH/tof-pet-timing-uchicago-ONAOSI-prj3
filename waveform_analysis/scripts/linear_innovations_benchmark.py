"""Build/run UC and FBK energy/timing comparisons without modifying original batches.

Run from the repository root:
  python -m waveform_analysis.scripts.linear_innovations_benchmark --output-root /tmp/innovations
  python -m waveform_analysis.scripts.linear_innovations_benchmark --output-root /tmp/innovations --execute

Generated batches run the standard preprocessing, development CV, blind CTR
and saved-model workflow. TSHAP can be applied afterwards to saved models,
occluding ORIGINAL waveforms before applying the frozen linear transformation.
"""
from __future__ import annotations

import argparse
import copy
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = {
    "UC": ROOT / "config/batches/benchmark_UC_test.json",
    "FBK": ROOT / "config/batches/benchmark_FBK_test.json",
}
MODELS = (
    "direct_linear_ridge", "direct_linear_innovations",
    "shared_linear_ridge", "shared_linear_innovations",
)


def generate(output_root: Path, boards: tuple[str, ...], models: tuple[str, ...], order: str):
    output_root = output_root.expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    configs = []
    for board in boards:
        original_path = SOURCE[board]
        original = json.loads(original_path.read_text(encoding="utf-8"))
        batch = copy.deepcopy(original)
        batch["name"] = f"{board}_linear_innovations_{order}"
        # Preserve real source paths: the relative paths in the original config
        # resolve relative to waveform_analysis/, not to the generated file.
        for role in ("control_dataset", "development_dataset", "blind_dataset"):
            entry = batch[role]
            if role == "control_dataset" and isinstance(entry, str):
                batch[role] = str((original_path.parent / entry).resolve())
            elif isinstance(entry, dict) and "root_file" in entry:
                path = Path(entry["root_file"])
                if not path.is_absolute():
                    entry["root_file"] = str((ROOT / path).resolve())
        batch["protocol"]["preprocessing_config"] = str(
            (original_path.parent / original["protocol"]["preprocessing_config"]).resolve()
        )
        batch["results"] = {
            "root": str(output_root), "folder": f"{board}/linear_innovations_{order}"
        }
        batch["sweep"]["models"] = [
            {"name": model, "space": _space(model, order)}
            for model in models
        ]
        path = output_root / f"{board}_{order}.json"
        path.write_text(json.dumps(batch, indent=2) + "\n", encoding="utf-8")
        configs.append(path)
    return configs


def _space(model, order):
    path = ROOT / "config/model_spaces" / f"{model}.json"
    space = json.loads(path.read_text(encoding="utf-8"))
    if "transform" in space:
        space["transform"]["order"] = order
    return space


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--boards", nargs="+", choices=tuple(SOURCE), default=list(SOURCE))
    parser.add_argument("--models", nargs="+", choices=MODELS, default=list(MODELS))
    parser.add_argument("--order", choices=("forward", "reverse"), default="forward")
    parser.add_argument("--execute", action="store_true", help="Run standard batch CLI after generating configs")
    args = parser.parse_args(argv)
    paths = generate(args.output_root, tuple(args.boards), tuple(args.models), args.order)
    for path in paths:
        print(path, flush=True)
        if args.execute:
            subprocess.run(
                [sys.executable, "-m", "waveform_analysis.cli", "batch", "--config", str(path)],
                check=True,
            )


if __name__ == "__main__":
    main()
