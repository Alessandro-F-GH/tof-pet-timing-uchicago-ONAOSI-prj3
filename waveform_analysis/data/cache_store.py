"""Atomic, human-named cache publication without recipe hashes or symlink aliases."""
from __future__ import annotations

import os
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

COMPLETE_FILE = ".complete"


@dataclass(frozen=True)
class CacheEntry:
    path: Path
    identifier: str

    def payload_path(self, name: str) -> Path:
        if not name or Path(name).is_absolute() or ".." in Path(name).parts:
            raise ValueError("A relative cache payload name is required")
        return self.path / name


def cache_complete(path) -> bool:
    directory = Path(path)
    return directory.is_dir() and (directory / COMPLETE_FILE).is_file()


def get_or_build(root, path, builder: Callable[[Path], None], *, rebuild=False) -> CacheEntry:
    """Reuse a complete directory or build it off-path and publish after success.

    The selected semantic name is the authority. Changes to scientific settings
    require a new cache name or an explicit rebuild. Interrupted builds never
    publish a partial directory. Concurrent writers to one name are unsupported.
    """
    root = Path(root).expanduser().resolve()
    path = Path(path).expanduser().resolve()
    if not path.is_relative_to(root):
        raise ValueError(f"Cache path {path} must be inside cache root {root}")
    entry = CacheEntry(path, path.relative_to(root).as_posix())
    if cache_complete(path) and not rebuild:
        return entry
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{path.name}.building-", dir=path.parent))
    try:
        builder(temporary)
        (temporary / COMPLETE_FILE).write_text("complete\n", encoding="utf-8")
        if path.exists():
            shutil.rmtree(path)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
    return entry
