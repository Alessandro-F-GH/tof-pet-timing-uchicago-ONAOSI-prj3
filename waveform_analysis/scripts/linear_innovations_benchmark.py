"""Build/run UC and FBK linear-innovation comparisons using existing cache bundles.

By default, reuses $HOME/UC_cache and $HOME/FBK_cache. Override with
--uc-cache-root /path/to/UC_cache and --fbk-cache-root /path/to/FBK_cache.
Run from the repository root:
  python -m waveform_analysis.scripts.linear_innovations_benchmark --output-root "$HOME/linear_innovations"
  python -m waveform_analysis.scripts.linear_innovations_benchmark --output-root "$HOME/linear_innovations" --execute
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
DEFAULT_CACHES = {
    "UC": Path.home() / "UC_cache",
    "FBK": Path.home() / "FBK_cache",
}
MODELS = (
    "direct_linear_ridge", "direct_linear_innovations",
    "shared_linear_ridge", "shared_linear_innovations",
)


def _space(model):
    path = ROOT / "config/model_spaces" / f"{model}.json"
    return json.loads(path.read_text(encoding="utf-8"))


def _cache_paths(board, cache_root):
    root = Path(cache_root).expanduser().resolve()
    preprocessing = root / "preprocessing"
    artifact_store = root / "artifact_store"
    missing = [str(path) for path in (preprocessing, artifact_store) if not path.is_dir()]
    if missing:
        raise FileNotFoundError(
            f"{board} cache bundle is incomplete at {root}; missing: "
            + ", ".join(missing)
            + ". Point the corresponding --*-cache-root option to the existing "
              "cache bundle (the directory containing preprocessing/ and artifact_store/)."
        )
    return root, preprocessing, artifact_store


def generate(
    output_root: Path,
    boards: tuple[str, ...],
    models: tuple[str, ...],
    cache_roots: dict[str, Path],
):
    output_root = output_root.expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    configs = []
    for board in boards:
        original_path = SOURCE[board]
        original = json.loads(original_path.read_text(encoding="utf-8"))
        batch = copy.deepcopy(original)
        batch["name"] = f"{board}_linear_innovations"

        # Resolve source paths against the original configuration location.
        for role in ("control_dataset", "development_dataset", "blind_dataset"):
            entry = batch[role]
            if role == "control_dataset" and isinstance(entry, str):
                batch[role] = str((original_path.parent / entry).resolve())
            elif isinstance(entry, dict) and "root_file" in entry:
                path = Path(entry["root_file"])
                if not path.is_absolute():
                    entry["root_file"] = str((ROOT / path).resolve())

        cache_root, preprocessing_root, artifact_root = _cache_paths(
            board, cache_roots[board]
        )
        original_preprocessing_path = (
            original_path.parent / original["protocol"]["preprocessing_config"]
        ).resolve()
        preprocessing = json.loads(
            original_preprocessing_path.read_text(encoding="utf-8")
        )
        # Match the cache bundle layout created by cache preparation. This is
        # essential: otherwise the pipeline silently falls back to repository
        # local processed_data/ and artifact_store directories and rebuilds.
        preprocessing["cache_dir"] = str(preprocessing_root)
        preprocessing["artifact_root"] = str(artifact_root)
        preprocessing_path = output_root / f"preprocessing_{board}.json"
        preprocessing_path.write_text(
            json.dumps(preprocessing, indent=2) + "\\n", encoding="utf-8"
        )
        batch["protocol"]["preprocessing_config"] = str(preprocessing_path)

        batch["results"] = {
            "root": str(output_root), "folder": f"{board}/linear_innovations"
        }
        batch["sweep"]["models"] = [
            {"name": model, "space": _space(model)} for model in models
        ]
        path = output_root / f"{board}.json"
        path.write_text(json.dumps(batch, indent=2) + "\\n", encoding="utf-8")
        print(
            f"{board} cache: {cache_root} "
            f"(preprocessing={preprocessing_root}, artifact_store={artifact_root})",
            flush=True,
        )
        configs.append(path)
    return configs


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--boards", nargs="+", choices=tuple(SOURCE), default=list(SOURCE))
    parser.add_argument("--models", nargs="+", choices=MODELS, default=list(MODELS))
    parser.add_argument("--uc-cache-root", type=Path, default=DEFAULT_CACHES["UC"])
    parser.add_argument("--fbk-cache-root", type=Path, default=DEFAULT_CACHES["FBK"])
    parser.add_argument(
        "--execute", action="store_true",
        help="Run standard batch CLI after generating configs",
    )
    args = parser.parse_args(argv)
    cache_roots = {"UC": args.uc_cache_root, "FBK": args.fbk_cache_root}
    paths = generate(args.output_root, tuple(args.boards), tuple(args.models), cache_roots)
    for path in paths:
        print(path, flush=True)
        if args.execute:
            subprocess.run(
                [sys.executable, "-m", "waveform_analysis.cli", "batch", "--config", str(path)],
                check=True,
            )


if __name__ == "__main__":
    main()
