from pathlib import Path
import numpy as np
import pytest

from waveform_analysis.data.cache_store import cache_complete, get_or_build
from waveform_analysis.data.named_cache import dataset_name, prepared_dataset_alias


def _builder(value):
    def build(directory):
        np.save(directory / "values.npy", np.asarray([value], dtype=np.int64))
    return build


def test_named_cache_reuses_completed_directory(tmp_path):
    path = tmp_path / "datasets" / "control" / "blind" / "wide"
    first = get_or_build(tmp_path, path, _builder(7))
    reused = get_or_build(tmp_path, path, lambda directory: pytest.fail("unexpected rebuild"))
    assert first.identifier == reused.identifier
    assert cache_complete(path)
    np.testing.assert_array_equal(np.load(path / "values.npy"), [7])


def test_named_cache_rebuild_replaces_existing_output(tmp_path):
    path = tmp_path / "features" / "ridge" / "blind"
    get_or_build(tmp_path, path, _builder(1))
    get_or_build(tmp_path, path, _builder(2), rebuild=True)
    np.testing.assert_array_equal(np.load(path / "values.npy"), [2])


def test_incomplete_cache_is_rebuilt_and_complete_cache_is_reused(tmp_path):
    path = tmp_path / "native" / "energy"
    path.mkdir(parents=True)
    (path / "values.npy").write_bytes(b"incomplete")
    assert not cache_complete(path)
    result = get_or_build(tmp_path, path, _builder(9))
    assert result.path == path
    assert cache_complete(path)
    np.testing.assert_array_equal(np.load(path / "values.npy"), [9])


def test_failed_build_does_not_publish_partial_cache(tmp_path):
    path = tmp_path / "prepared" / "wide"
    def fail(directory):
        (directory / "values.npy").write_bytes(b"partial")
        raise RuntimeError("interrupted")
    with pytest.raises(RuntimeError, match="interrupted"):
        get_or_build(tmp_path, path, fail)
    assert not cache_complete(path)


def test_cache_path_cannot_escape_root(tmp_path):
    with pytest.raises(ValueError, match="inside cache root"):
        get_or_build(tmp_path, tmp_path.parent / "outside", _builder(2))


def test_dataset_cache_names_are_portable(tmp_path):
    windows = {"root_file": r"C:\Users\aless\Desktop\data\48V-R0.root"}
    posix = {"root_file": "/home/aless/data/48V-R0.root"}
    assert dataset_name(windows) == dataset_name(posix) == "48V-R0"
    w = {"control": windows, "blind": windows, "mode": "energy_to_energy", "window_name": "wide"}
    u = {"control": posix, "blind": posix, "mode": "energy_to_energy", "window_name": "wide"}
    assert prepared_dataset_alias(tmp_path, w, "blind") == prepared_dataset_alias(tmp_path, u, "blind")


def test_failed_rebuild_preserves_last_complete_cache(tmp_path):
    path = tmp_path / "features" / "complete"
    get_or_build(tmp_path, path, _builder(7))

    def fail(directory):
        np.save(directory / "values.npy", np.asarray([99]))
        raise RuntimeError("build failed")

    with pytest.raises(RuntimeError, match="build failed"):
        get_or_build(tmp_path, path, fail, rebuild=True)
    assert cache_complete(path)
    np.testing.assert_array_equal(np.load(path / "values.npy"), [7])


def test_changed_named_cache_rebuild_publishes_new_content(tmp_path):
    path = tmp_path / "features" / "shared"
    get_or_build(tmp_path, path, _builder(2))
    get_or_build(tmp_path, path, _builder(8), rebuild=True)
    assert cache_complete(path)
    np.testing.assert_array_equal(np.load(path / "values.npy"), [8])
