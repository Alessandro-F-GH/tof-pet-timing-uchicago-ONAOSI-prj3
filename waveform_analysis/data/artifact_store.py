"""Immutable, content-verified artifact storage for scientific pipeline products."""
from __future__ import annotations

import contextlib
import hashlib
import json
import os
import re
import shutil
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Any

import numpy as np

from waveform_analysis.core.artifact_identity import ProductRecipe, canonical_digest, canonical_json


MANIFEST_NAME = ".artifact.json"
STORE_SCHEMA_VERSION = 1
_PRODUCT_ID = re.compile(r"^[0-9a-f]{64}$")


class ArtifactError(RuntimeError):
    """Base class for artifact storage failures."""


class ArtifactMissing(ArtifactError):
    pass


class ArtifactIncompatible(ArtifactError):
    pass


class ArtifactCorrupt(ArtifactError):
    pass


@dataclass(frozen=True)
class ArtifactReference:
    product_type: str
    product_id: str
    path: Path
    manifest: dict[str, Any]

    def payload_path(self, name: str) -> Path:
        if not name or name == MANIFEST_NAME:
            raise ValueError("A payload name, not the manifest, is required")
        candidate = (self.path / name).resolve()
        if not candidate.is_relative_to(self.path.resolve()):
            raise ValueError("Artifact payload path escapes its product directory")
        return candidate


@dataclass(frozen=True)
class ArtifactLookup:
    status: str
    reference: ArtifactReference | None = None
    reason: str | None = None


@contextlib.contextmanager
def _process_lock(path: Path):
    """Cross-process advisory lock implemented with the Python standard library."""
    path.parent.mkdir(parents=True, exist_ok=True)
    stream = path.open("a+b")
    try:
        if os.name == "nt":
            import msvcrt
            stream.seek(0, os.SEEK_END)
            if stream.tell() == 0:
                stream.write(b"0")
                stream.flush()
            stream.seek(0)
            msvcrt.locking(stream.fileno(), msvcrt.LK_LOCK, 1)
        else:
            import fcntl
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX)
        yield
    finally:
        try:
            if os.name == "nt":
                import msvcrt
                stream.seek(0)
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl
                fcntl.flock(stream.fileno(), fcntl.LOCK_UN)
        finally:
            stream.close()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _fsync_directory(path: Path) -> None:
    if os.name == "nt":
        return
    fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _fsync_tree(root: Path) -> None:
    """Flush payload bytes and directory entries before publishing the product."""
    for path in root.rglob("*"):
        if path.is_file() and not path.is_symlink():
            with path.open("rb") as stream:
                os.fsync(stream.fileno())
    directories = [root, *(path for path in root.rglob("*") if path.is_dir() and not path.is_symlink())]
    for directory in sorted(directories, key=lambda item: len(item.parts), reverse=True):
        _fsync_directory(directory)


def _payload_inventory(root: Path) -> dict[str, dict[str, Any]]:
    inventory: dict[str, dict[str, Any]] = {}
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise ArtifactError(f"Symlinks are not allowed inside artifacts: {path}")
        if not path.is_file() or path.name == MANIFEST_NAME:
            continue
        relative = path.relative_to(root).as_posix()
        record: dict[str, Any] = {"sha256": _sha256(path), "size_bytes": path.stat().st_size}
        if path.suffix == ".npy":
            array = np.load(path, mmap_mode="r", allow_pickle=False)
            record["shape"] = list(array.shape)
            record["dtype"] = array.dtype.str
            del array
        inventory[relative] = record
    if not inventory:
        raise ArtifactError("An artifact must contain at least one payload file")
    return inventory


class ArtifactStore:
    """Store immutable artifacts by recipe identity, independent of report folders."""

    def __init__(self, root: str | Path):
        self.root = Path(root).expanduser().resolve()
        self.products_root = self.root / "products"
        self.locks_root = self.root / ".locks"
        self.products_root.mkdir(parents=True, exist_ok=True)
        self.locks_root.mkdir(parents=True, exist_ok=True)

    def _product_path(self, recipe: ProductRecipe) -> Path:
        return self.products_root / recipe.product_type / recipe.product_id

    def lookup(self, recipe: ProductRecipe, *, verify_payloads: bool = True) -> ArtifactLookup:
        path = self._product_path(recipe)
        if not path.exists():
            return ArtifactLookup("missing", reason=f"Product {recipe.product_id} does not exist")
        if not path.is_dir() or path.is_symlink():
            return ArtifactLookup("corrupt", reason="Product path is not a regular directory")
        manifest_path = path / MANIFEST_NAME
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            return ArtifactLookup("corrupt", reason=f"Cannot read completion manifest: {exc}")
        if manifest.get("product_id") != recipe.product_id or manifest.get("recipe") != json.loads(canonical_json(recipe.identity_payload)):
            return ArtifactLookup("incompatible", reason="Manifest recipe does not match requested recipe")
        if manifest.get("status") != "complete" or manifest.get("store_schema_version") != STORE_SCHEMA_VERSION:
            return ArtifactLookup("incomplete", reason="Artifact is not complete or uses an unsupported store schema")
        expected = manifest.get("payloads")
        if not isinstance(expected, dict) or not expected:
            return ArtifactLookup("corrupt", reason="Manifest has no payload inventory")
        if verify_payloads:
            try:
                actual_names = {
                    p.relative_to(path).as_posix() for p in path.rglob("*")
                    if p.is_file() and p.name != MANIFEST_NAME
                }
                if actual_names != set(expected):
                    return ArtifactLookup("corrupt", reason="Payload inventory differs from the manifest")
                for name, record in expected.items():
                    payload = path / name
                    if payload.is_symlink() or not payload.is_file() or _sha256(payload) != record.get("sha256"):
                        return ArtifactLookup("corrupt", reason=f"Payload checksum mismatch: {name}")
                    if payload.suffix == ".npy":
                        array = np.load(payload, mmap_mode="r", allow_pickle=False)
                        if list(array.shape) != record.get("shape") or array.dtype.str != record.get("dtype"):
                            return ArtifactLookup("corrupt", reason=f"Array metadata mismatch: {name}")
                        del array
            except (OSError, ValueError, KeyError) as exc:
                return ArtifactLookup("corrupt", reason=f"Cannot validate payloads: {exc}")
        return ArtifactLookup("hit", ArtifactReference(recipe.product_type, recipe.product_id, path, manifest))

    def resolve(self, product_type: str, product_id: str, *, verify_payloads: bool = True) -> ArtifactReference:
        parts = product_type.split("/")
        if any(part in {"", ".", ".."} for part in parts) or not _PRODUCT_ID.fullmatch(product_id):
            raise ArtifactMissing(f"Invalid artifact reference: {product_type}/{product_id}")
        path = self.products_root.joinpath(*parts, product_id)
        if not path.is_dir() or path.is_symlink():
            raise ArtifactMissing(f"Artifact {product_type}/{product_id} is missing")
        try:
            manifest = json.loads((path / MANIFEST_NAME).read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise ArtifactCorrupt(f"Cannot read manifest for {product_type}/{product_id}: {exc}") from exc
        if manifest.get("product_type") != product_type or manifest.get("product_id") != product_id:
            raise ArtifactCorrupt(f"Manifest identity mismatch for {product_type}/{product_id}")
        if manifest.get("status") != "complete" or manifest.get("store_schema_version") != STORE_SCHEMA_VERSION:
            raise ArtifactCorrupt(f"Artifact {product_type}/{product_id} is incomplete or incompatible")
        recipe_payload = manifest.get("recipe")
        if not isinstance(recipe_payload, dict) or canonical_digest(recipe_payload) != product_id:
            raise ArtifactCorrupt(f"Recipe identity checksum mismatch for {product_type}/{product_id}")
        expected = manifest.get("payloads")
        if not isinstance(expected, dict) or not expected:
            raise ArtifactCorrupt(f"Missing payload inventory for {product_type}/{product_id}")
        if verify_payloads:
            actual_names = {
                p.relative_to(path).as_posix() for p in path.rglob("*")
                if p.is_file() and p.name != MANIFEST_NAME
            }
            if actual_names != set(expected):
                raise ArtifactCorrupt(f"Payload inventory mismatch for {product_type}/{product_id}")
            for name, record in expected.items():
                payload = (path / name).resolve()
                if not payload.is_relative_to(path.resolve()) or payload.is_symlink() or not payload.is_file():
                    raise ArtifactCorrupt(f"Invalid payload path: {path / name}")
                if _sha256(payload) != record.get("sha256"):
                    raise ArtifactCorrupt(f"Payload checksum mismatch: {payload}")
                if payload.suffix == ".npy":
                    array = np.load(payload, mmap_mode="r", allow_pickle=False)
                    if list(array.shape) != record.get("shape") or array.dtype.str != record.get("dtype"):
                        raise ArtifactCorrupt(f"Array metadata mismatch: {payload}")
                    del array
        return ArtifactReference(product_type, product_id, path, manifest)

    def get_or_build(
        self,
        recipe: ProductRecipe,
        builder: Callable[[Path], Any],
        *,
        metadata: dict[str, Any] | None = None,
    ) -> ArtifactReference:
        final_path = self._product_path(recipe)
        lock_path = self.locks_root / f"{recipe.product_id}.lock"
        with _process_lock(lock_path):
            found = self.lookup(recipe)
            if found.status == "hit":
                return found.reference  # type: ignore[return-value]
            if found.status != "missing":
                raise ArtifactCorrupt(
                    f"Refusing to overwrite {found.status} artifact {recipe.product_id}: {found.reason}. "
                    "Repair or quarantine it explicitly."
                )
            final_path.parent.mkdir(parents=True, exist_ok=True)
            temporary = Path(tempfile.mkdtemp(prefix=f".{recipe.product_id}.", suffix=".building", dir=final_path.parent))
            try:
                builder(temporary)
                payloads = _payload_inventory(temporary)
                manifest = {
                    "store_schema_version": STORE_SCHEMA_VERSION,
                    "product_type": recipe.product_type,
                    "product_id": recipe.product_id,
                    "recipe": json.loads(canonical_json(recipe.identity_payload)),
                    "parents": list(recipe.parents),
                    "implementation": recipe.implementation,
                    "compatibility": json.loads(canonical_json(recipe.compatibility)),
                    "payloads": payloads,
                    "metadata": metadata or {},
                    "status": "complete",
                    "created_unix_time": time.time(),
                }
                manifest_path = temporary / MANIFEST_NAME
                with manifest_path.open("w", encoding="utf-8", newline="\n") as stream:
                    stream.write(json.dumps(manifest, sort_keys=True, indent=2, ensure_ascii=False) + "\n")
                    stream.flush()
                    os.fsync(stream.fileno())
                _fsync_tree(temporary)
                if final_path.exists():
                    raise ArtifactError(f"Refusing to replace existing artifact path: {final_path}")
                os.replace(temporary, final_path)
                _fsync_directory(final_path.parent)
            except Exception:
                if temporary.exists():
                    shutil.rmtree(temporary, ignore_errors=True)
                raise
            return self.resolve(recipe.product_type, recipe.product_id)

    def open_array(self, reference: ArtifactReference, name: str) -> np.ndarray:
        """Open a .npy payload as a read-only memory map."""
        payload = reference.payload_path(name)
        if payload.suffix != ".npy":
            raise ValueError("open_array accepts .npy payloads only")
        array = np.load(payload, mmap_mode="r", allow_pickle=False)
        if array.flags.writeable:
            raise ArtifactCorrupt(f"Expected a read-only array: {payload}")
        return array
