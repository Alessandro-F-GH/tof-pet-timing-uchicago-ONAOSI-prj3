from __future__ import annotations

from pathlib import Path


def replica_tag(replica_index: int, seed: int) -> str:
    """Compact, stable filesystem tag for one replica."""
    return f"r{int(replica_index):03d}_s{int(seed)}"


def seed_tag(seed: int) -> str:
    """Compact, stable filesystem tag for one seed."""
    return f"s{int(seed)}"


def prefer_existing(primary: str | Path, legacy: str | Path) -> Path:
    """Prefer compact naming, but keep old results readable/resumable."""
    primary = Path(primary)
    legacy = Path(legacy)
    if primary.exists() or not legacy.exists():
        return primary
    return legacy
