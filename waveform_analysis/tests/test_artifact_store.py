from __future__ import annotations

import multiprocessing
import os
from pathlib import Path

import numpy as np
import pytest

from waveform_analysis.core.artifact_identity import ProductRecipe, canonical_digest
from waveform_analysis.data.artifact_store import ArtifactCorrupt, ArtifactStore
from waveform_analysis.data.artifact_registry import ArtifactRegistry


def _build_array(root: Path, value: float = 1.0) -> None:
    np.save(root / "values.npy", np.asarray([value, value + 1], dtype=np.float32), allow_pickle=False)


def _concurrent_build(root: str, counter: str) -> None:
    store = ArtifactStore(root)
    recipe = ProductRecipe("test/array", {"seed": 12}, implementation="test-v1")
    def builder(directory):
        with open(counter, "a", encoding="utf-8") as stream:
            stream.write("build\n")
        _build_array(directory)
    store.get_or_build(recipe, builder)


def test_recipe_identity_is_canonical_but_parent_order_matters():
    first = ProductRecipe("features", {"b": 2, "a": 1}, parents=("raw-a", "mask-b"))
    same = ProductRecipe("features", {"a": 1, "b": 2}, parents=("raw-a", "mask-b"))
    changed = ProductRecipe("features", {"a": 1, "b": 2}, parents=("mask-b", "raw-a"))
    assert first.product_id == same.product_id
    assert first.product_id != changed.product_id
    assert canonical_digest({"x": np.int64(3)}) == canonical_digest({"x": 3})


def test_recipe_rejects_paths_nonfinite_values_and_unsafe_product_types(tmp_path):
    with pytest.raises(TypeError):
        ProductRecipe("x", {"root": tmp_path})
    with pytest.raises(ValueError):
        ProductRecipe("x", {"alpha": float("nan")})
    with pytest.raises(ValueError):
        ProductRecipe("../outside", {"seed": 1})


def test_store_builds_once_and_opens_arrays_read_only(tmp_path):
    store = ArtifactStore(tmp_path / "store")
    recipe = ProductRecipe("features", {"seed": 7}, implementation="test-v1")
    calls = []
    ref = store.get_or_build(recipe, lambda directory: (calls.append(1), _build_array(directory)))
    second = store.get_or_build(recipe, lambda directory: (_ for _ in ()).throw(AssertionError("unexpected rebuild")))
    assert calls == [1]
    assert ref.product_id == second.product_id
    array = store.open_array(ref, "values.npy")
    assert np.array_equal(array, np.asarray([1, 2], dtype=np.float32))
    assert not array.flags.writeable


def test_store_refuses_corrupt_payload_rebuild(tmp_path):
    store = ArtifactStore(tmp_path / "store")
    recipe = ProductRecipe("features", {"seed": 8})
    ref = store.get_or_build(recipe, lambda directory: _build_array(directory))
    ref.payload_path("values.npy").write_bytes(b"corrupt")
    with pytest.raises(ArtifactCorrupt):
        store.get_or_build(recipe, lambda directory: _build_array(directory, 10.0))


def test_registry_rebuild_and_source_bindings(tmp_path):
    store = ArtifactStore(tmp_path / "store")
    recipe = ProductRecipe("native", {"source_id": "acq-001"})
    ref = store.get_or_build(recipe, lambda directory: _build_array(directory))
    registry = ArtifactRegistry(store)
    index = registry.rebuild()
    assert f"native/{ref.product_id}" in index["products"]
    registry.bind_source("acquisition:001", ref.product_type, ref.product_id)
    assert registry.resolve_source("acquisition:001").product_id == ref.product_id


@pytest.mark.skipif(os.name == "nt", reason="Multiprocessing spawn test is exercised on POSIX CI hosts")
def test_parallel_builders_serialize_and_compute_once(tmp_path):
    store_root = str(tmp_path / "store")
    counter = str(tmp_path / "calls.txt")
    context = multiprocessing.get_context("fork")
    processes = [context.Process(target=_concurrent_build, args=(store_root, counter)) for _ in range(4)]
    for process in processes:
        process.start()
    for process in processes:
        process.join(timeout=20)
        assert process.exitcode == 0
    assert Path(counter).read_text(encoding="utf-8").splitlines() == ["build"]


def test_domain_manifest_is_payload_not_store_completion_record(tmp_path):
    store = ArtifactStore(tmp_path / "store")
    recipe = ProductRecipe("prepared", {"source": "control-001"})
    def builder(directory):
        np.save(directory / "values.npy", np.asarray([1, 2], dtype=np.int16), allow_pickle=False)
        (directory / "manifest.json").write_text('{"scientific": "metadata"}', encoding="utf-8")
    reference = store.get_or_build(recipe, builder)
    assert reference.payload_path("manifest.json").is_file()
    assert (reference.path / ".artifact.json").is_file()
    assert store.lookup(recipe).status == "hit"
