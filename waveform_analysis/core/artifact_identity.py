"""Canonical, path-independent identities for reusable scientific products."""
from __future__ import annotations

import dataclasses
import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any, Mapping


IDENTITY_SCHEMA_VERSION = 1
_PRODUCT_TYPE_PART = re.compile(r"^[A-Za-z0-9_.-]+$")


def canonical_value(value: Any) -> Any:
    """Return a deterministic JSON value without silently coercing unsupported data."""
    if dataclasses.is_dataclass(value):
        value = dataclasses.asdict(value)
    if isinstance(value, Path):
        raise TypeError("Filesystem paths cannot be recipe identity values; use a stable source ID")
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("Recipe identity cannot contain NaN or infinity")
        return value
    if isinstance(value, Mapping):
        if not all(isinstance(key, str) for key in value):
            raise TypeError("Recipe mappings must have string keys")
        return {key: canonical_value(value[key]) for key in sorted(value)}
    if isinstance(value, (list, tuple)):
        return [canonical_value(item) for item in value]
    if isinstance(value, (set, frozenset)):
        normalized = [canonical_value(item) for item in value]
        return sorted(normalized, key=lambda item: json.dumps(item, sort_keys=True, separators=(",", ":")))
    if type(value).__module__.startswith("numpy") and hasattr(value, "item"):
        return canonical_value(value.item())
    raise TypeError(f"Unsupported recipe identity value: {type(value).__name__}")


def canonical_json(value: Any) -> str:
    return json.dumps(
        canonical_value(value), sort_keys=True, separators=(",", ":"), ensure_ascii=False,
        allow_nan=False,
    )


def canonical_digest(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


@dataclasses.dataclass(frozen=True)
class ProductRecipe:
    """Complete scientific recipe for one immutable artifact.

    Parent order is significant. Machine-local output roots and report presentation
    settings belong in views, not in scientific recipes.
    """

    product_type: str
    parameters: Mapping[str, Any]
    parents: tuple[str, ...] = ()
    implementation: str = "unspecified"
    compatibility: Mapping[str, Any] = dataclasses.field(default_factory=dict)
    schema_version: int = IDENTITY_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if not self.product_type or not self.product_type.strip():
            raise ValueError("product_type must be non-empty")
        parts = self.product_type.split("/")
        if any(part in {"", ".", ".."} or not _PRODUCT_TYPE_PART.fullmatch(part) for part in parts):
            raise ValueError("product_type must be a safe slash-separated identifier")
        if not self.implementation or not self.implementation.strip():
            raise ValueError("implementation must be non-empty")
        if self.schema_version < 1:
            raise ValueError("schema_version must be positive")
        if any(not isinstance(parent, str) or not parent for parent in self.parents):
            raise ValueError("parents must contain non-empty artifact IDs")
        canonical_json(self.identity_payload)

    @property
    def identity_payload(self) -> dict[str, Any]:
        return {
            "identity_schema_version": IDENTITY_SCHEMA_VERSION,
            "product_type": self.product_type,
            "schema_version": self.schema_version,
            "parameters": self.parameters,
            "parents": list(self.parents),
            "implementation": self.implementation,
            "compatibility": self.compatibility,
        }

    @property
    def product_id(self) -> str:
        return canonical_digest(self.identity_payload)
