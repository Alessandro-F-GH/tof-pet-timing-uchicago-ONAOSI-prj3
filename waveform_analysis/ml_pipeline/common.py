from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
from pathlib import Path
from typing import Any

import numpy as np


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def canonical_json(value: Any) -> str:
    return json.dumps(json_safe(value), sort_keys=True, separators=(",", ":"))


def canonical_hash(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def source_signature(path: Path) -> dict[str, Any]:
    stat = path.stat()
    return {"path": str(path.resolve()), "size_bytes": int(stat.st_size), "mtime_ns": int(stat.st_mtime_ns)}


def voltage_from_name(value: str | Path) -> float:
    """Extract detector bias from names such as ``49V-490mV.root``."""
    name = Path(value).stem
    match = re.search(r"(?:^|[^0-9.])(\d+(?:\.\d+)?)V(?:-|_|$)", name, flags=re.IGNORECASE)
    return float(match.group(1)) if match else float("nan")


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(json_safe(value), indent=2, sort_keys=True, allow_nan=False) + "\n"
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as stream:
        value = json.load(stream)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return value
