"""Portable, human-readable cache settings (no hash or absolute-path checks).

Named cache directories select artifacts. Explicit settings guard against
accidental reuse when an artifact was created with different parameters.
Legacy caches lacking a settings record remain the user's responsibility.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any


_PATH_KEYS = {"root_file", "source", "dataset_source", "control_source", "reference_source"}
_LOCAL_KEYS = {"cache_dir", "artifact_root", "output_dir", "batch_output_dir"}


def portable_settings(value: Any) -> Any:
    """Normalize settings for comparison across Windows and POSIX machines."""
    if isinstance(value, dict):
        return {
            str(key): (str(raw).replace("\\", "/").split("/")[-1]
                       if key in _PATH_KEYS else portable_settings(raw))
            for key, raw in value.items()
            if key not in _LOCAL_KEYS
        }
    if isinstance(value, (list, tuple)):
        return [portable_settings(item) for item in value]
    if isinstance(value, Path):
        return value.name
    return value


def verify_settings(manifest: dict, expected: dict, *, label: str) -> None:
    """Validate explicit cache settings, without comparing SHA fingerprints.

    Existing cache manifests predate the explicit 'cache_settings' field.
    They can still be read: callers should separately check the available
    legacy fields such as mode and window.
    """
    recorded = manifest.get("cache_settings")
    if recorded is None:
        return
    actual = portable_settings(recorded)
    wanted = portable_settings(expected)
    if actual != wanted:
        keys = sorted(set(actual) | set(wanted))
        changed = [key for key in keys if actual.get(key) != wanted.get(key)]
        raise ValueError(
            f"{label} cache settings differ | fields={changed} | "
            f"cached={{{', '.join(f'{key}: {actual.get(key)!r}' for key in changed)}}} | "
            f"expected={{{', '.join(f'{key}: {wanted.get(key)!r}' for key in changed)}}}"
        )
