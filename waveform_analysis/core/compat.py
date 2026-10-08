"""Preserve legacy module identity for imports, monkeypatching and pickles."""

from __future__ import annotations

import sys


def preserve_legacy_identity(module_name: str, legacy_name: str) -> None:
    """Alias a migrated module and retain the import paths of its own definitions.

    Existing artifacts may pickle dataclasses, sklearn wrappers or feature
    transforms by module path. Imported definitions keep their original identity.
    Aliasing the module itself also preserves monkeypatch behavior in callers.
    """
    module = sys.modules[module_name]
    for value in vars(module).values():
        if isinstance(value, type) or callable(value):
            if getattr(value, "__module__", None) == module_name:
                value.__module__ = legacy_name
    sys.modules[legacy_name] = module
