"""Rebuildable artifact index and explicit source bindings."""
from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any

from waveform_analysis.core.artifact_identity import canonical_digest
from waveform_analysis.data.artifact_store import ArtifactStore, MANIFEST_NAME, _process_lock


class ArtifactRegistry:
    """A convenience index; product manifests remain the source of truth."""

    def __init__(self, store: ArtifactStore):
        self.store = store
        self.path = store.root / "registry.json"
        self.bindings_path = store.root / "bindings.json"

    @staticmethod
    def _atomic_json(path: Path, value: Any) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
        try:
            with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as stream:
                json.dump(value, stream, sort_keys=True, indent=2, ensure_ascii=False)
                stream.write("\n")
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, path)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)

    def rebuild(self) -> dict[str, Any]:
        products: dict[str, dict[str, Any]] = {}
        for manifest_path in self.store.products_root.rglob(MANIFEST_NAME):
            try:
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                product_type = str(manifest["product_type"])
                product_id = str(manifest["product_id"])
                if (
                    manifest.get("status") != "complete"
                    or manifest_path.parent.name != product_id
                    or canonical_digest(manifest.get("recipe")) != product_id
                ):
                    continue
                key = f"{product_type}/{product_id}"
                products[key] = {
                    "product_type": product_type,
                    "product_id": product_id,
                    "manifest_sha256": canonical_digest(manifest),
                }
            except (OSError, ValueError, KeyError, TypeError):
                continue
        index = {"schema_version": 1, "products": dict(sorted(products.items()))}
        with _process_lock(self.store.locks_root / "registry.lock"):
            self._atomic_json(self.path, index)
        return index

    def bind_source(self, source_key: str, product_type: str, product_id: str) -> None:
        if not source_key.strip():
            raise ValueError("source_key must be non-empty")
        self.store.resolve(product_type, product_id)
        with _process_lock(self.store.locks_root / "bindings.lock"):
            try:
                bindings = json.loads(self.bindings_path.read_text(encoding="utf-8"))
            except FileNotFoundError:
                bindings = {"schema_version": 1, "bindings": {}}
            bindings.setdefault("bindings", {})[source_key] = {
                "product_type": product_type,
                "product_id": product_id,
            }
            self._atomic_json(self.bindings_path, bindings)

    def resolve_source(self, source_key: str):
        try:
            bindings = json.loads(self.bindings_path.read_text(encoding="utf-8"))
        except FileNotFoundError as exc:
            raise KeyError(f"No source bindings registered for {source_key!r}") from exc
        binding = bindings.get("bindings", {}).get(source_key)
        if binding is None:
            raise KeyError(f"No source binding registered for {source_key!r}")
        return self.store.resolve(binding["product_type"], binding["product_id"])
