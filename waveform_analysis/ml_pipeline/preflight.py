from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .config import discover_root_files
from .data import load_preprocessed
from .dataset import load_prepared_dataset
from .event_selection import load_selection
from .prepared_data import dataset_fingerprint


@dataclass(frozen=True)
class Preflight:
    roots: tuple[Path, ...]
    overwrite_paths: tuple[Path, ...]


def _cache_path(config, key: str, root: Path) -> Path:
    return Path(config["preprocessing"][key]).resolve() / root.stem


def inspect_preprocessing(config, *, rebuild: bool) -> Preflight:
    """Validate every input cache before a multi-file run starts."""
    roots = tuple(discover_root_files(config))
    if not roots:
        raise FileNotFoundError("No ROOT files matched the configured source")
    overwrite: list[Path] = []
    problems: list[str] = []
    for root in roots:
        paths = {key: _cache_path(config, key, root) for key in ("selection_store_dir", "preprocessed_dir", "prepared_dir")}
        if rebuild:
            overwrite.extend(path for path in paths.values() if path.exists())
            continue
        selection = None
        if paths["selection_store_dir"].exists():
            try:
                selection = load_selection(paths["selection_store_dir"], root, config)
            except Exception as exc:
                problems.append(f"{root.name}: stale selection cache ({exc})")
        elif paths["preprocessed_dir"].exists() or paths["prepared_dir"].exists():
            problems.append(f"{root.name}: preprocessing cache exists without its selection cache")
        preprocessed = None
        if selection is not None and paths["preprocessed_dir"].exists():
            try:
                preprocessed = load_preprocessed(paths["preprocessed_dir"], root, selection, config)
            except Exception as exc:
                problems.append(f"{root.name}: stale native preprocessing cache ({exc})")
        elif paths["prepared_dir"].exists() and not paths["preprocessed_dir"].exists():
            problems.append(f"{root.name}: prepared ML cache exists without native preprocessing")
        if preprocessed is not None and paths["prepared_dir"].exists():
            try:
                dataset = load_prepared_dataset(paths["prepared_dir"])
                if dataset.manifest.get("fingerprint") != dataset_fingerprint(preprocessed, config):
                    raise ValueError("fingerprint changed")
            except Exception as exc:
                problems.append(f"{root.name}: stale prepared ML cache ({exc})")
    if problems:
        details = "\n  - ".join(problems)
        raise RuntimeError(f"Preflight found stale caches. Re-run with --rebuild-preprocessing after reviewing them:\n  - {details}")
    return Preflight(roots, tuple(dict.fromkeys(path.resolve() for path in overwrite)))


def study_overwrite_path(config, *, overwrite: bool) -> Path | None:
    run_dir = Path(config["experiment"]["output_dir"]).resolve()
    nonempty = run_dir.is_dir() and any(run_dir.iterdir())
    if nonempty and not overwrite:
        raise FileExistsError(f"Run directory is not empty: {run_dir}. Use --overwrite to replace it.")
    return run_dir if nonempty and overwrite else None


def confirm_overwrite(paths: list[Path] | tuple[Path, ...]) -> bool:
    unique = tuple(dict.fromkeys(Path(path).resolve() for path in paths))
    if not unique:
        return True
    print("The following existing results will be overwritten:")
    for path in unique:
        print(f"  - {path}")
    try:
        answer = input("Continue? [y/N]: ").strip().lower()
    except EOFError:
        answer = ""
    return answer in {"y", "yes"}
