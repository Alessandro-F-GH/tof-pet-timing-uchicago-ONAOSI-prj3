"""Verified prepared cache loading for TSHAP and waveform imputation."""
from __future__ import annotations
import json
from pathlib import Path, PureWindowsPath
import re
from typing import Any
from waveform_analysis.data.dataset import PreparedDataset, load_prepared_dataset

def _read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _cache_root(config: dict[str, Any], run: Path, override: Path | None, *, roles=("control", "blind")) -> Path:
    if override is not None:
        return override.expanduser().resolve()
    configured = Path(config["preprocessing"]["cache_dir"]).expanduser()
    candidates = [configured]
    parts = PureWindowsPath(str(configured)).parts
    anchors = [i for i, part in enumerate(parts) if part.lower() == "waveform_analysis"]
    if anchors:
        suffix = Path(*parts[anchors[-1] + 1:])
        roots = [parent for parent in run.parents if parent.name.lower() == "waveform_analysis"]
        roots.append(Path(__file__).resolve().parents[1])
        candidates.extend(root / suffix for root in roots)
    for candidate in candidates:
        if all((candidate / f"{role}_ml" / "prepared").is_dir() for role in roles):
            return candidate.resolve()
    return configured.resolve()


class PreparedCacheLookup:
    """Index metadata once and reuse memory-mapped datasets across saved models."""

    def __init__(self):
        self.index: dict[tuple[Path, str], list[tuple[Path, dict[str, Any]]]] = {}
        self.datasets: dict[Path, PreparedDataset] = {}

    def find(self, root: Path, role: str, config: dict[str, Any], manifest: dict[str, Any]) -> PreparedDataset:
        raise ValueError(
            f"No prepared artifact reference for role={role!r}; "
            "new runs must publish prepared_artifact_refs in their manifest."
        )
