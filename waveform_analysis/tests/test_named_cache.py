from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from waveform_analysis.core.artifact_identity import ProductRecipe
from waveform_analysis.data.artifact_store import ArtifactStore
from waveform_analysis.data.named_cache import (
    get_or_build_named, load_named_reference, publish_alias, resolve_alias,
)


def _builder(value):
    def build(directory, recipe):
        np.save(directory / "values.npy", np.asarray([value], dtype=np.int64))
    return build


def test_named_cache_reuses_by_name_even_when_recipe_changes(tmp_path):
    store = ArtifactStore(tmp_path / "artifact_store")
    alias = tmp_path / "datasets" / "control" / "blind" / "wide"
    first_recipe = ProductRecipe("prepared-datasets", {"format": 1}, implementation="v1")
    first = get_or_build_named(store, first_recipe, alias, _builder(7))
    second_recipe = ProductRecipe("prepared-datasets", {"format": 99}, implementation="v99")
    calls = []
    second = get_or_build_named(
        store, second_recipe, alias,
        lambda directory, recipe: calls.append(recipe),
    )
    assert second.product_id == first.product_id
    assert not calls
    np.testing.assert_array_equal(
        np.load(second.payload_path("values.npy")), np.asarray([7])
    )


def test_named_cache_adopts_existing_content_addressed_product(tmp_path):
    store = ArtifactStore(tmp_path / "artifact_store")
    recipe = ProductRecipe("feature-transforms", {"model": "minirocket"})
    original = store.get_or_build(recipe, _builder(11))
    alias = tmp_path / "transforms" / "minirocket" / "control" / "wide"
    calls = []
    adopted = get_or_build_named(
        store, recipe, alias, lambda directory, effective: calls.append(effective)
    )
    assert adopted.product_id == original.product_id
    assert not calls
    assert resolve_alias(alias) == original.path


def test_named_cache_explicit_rebuild_retargets_same_alias(tmp_path):
    store = ArtifactStore(tmp_path / "artifact_store")
    alias = tmp_path / "features" / "minirocket" / "blind"
    recipe = ProductRecipe("feature-matrices", {"dataset": "blind"})
    first = get_or_build_named(store, recipe, alias, _builder(1))
    second = get_or_build_named(store, recipe, alias, _builder(2), rebuild=True)
    assert first.product_id != second.product_id
    np.testing.assert_array_equal(
        np.load(resolve_alias(alias) / "values.npy"), np.asarray([2])
    )


def test_named_cache_rejects_wrong_product_type(tmp_path):
    store = ArtifactStore(tmp_path / "artifact_store")
    wrong = ProductRecipe("native-waveforms", {"dataset": "blind"})
    reference = store.get_or_build(wrong, _builder(1))
    alias = tmp_path / "prepared"
    publish_alias(alias, reference.path)
    expected = ProductRecipe("prepared-datasets", {"dataset": "blind"})
    import pytest
    from waveform_analysis.data.artifact_store import ArtifactCorrupt
    with pytest.raises(ArtifactCorrupt, match="expected prepared-datasets"):
        get_or_build_named(store, expected, alias, _builder(2))
