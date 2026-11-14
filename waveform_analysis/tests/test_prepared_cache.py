from types import SimpleNamespace
import pytest
from waveform_analysis.data import prepared_cache


def _config(tmp_path):
    return {
        "mode": "energy_to_energy",
        "preprocessing": {"cache_dir": str(tmp_path / "preprocessing")},
    }


def test_prepared_cache_uses_selected_named_directory(tmp_path, monkeypatch):
    path = tmp_path / "preprocessing" / "datasets" / "control"
    path.mkdir(parents=True)
    (path / ".complete").write_text("complete")
    dataset = SimpleNamespace(directory=path, manifest={"dataset_role": "control"})
    monkeypatch.setattr(prepared_cache, "prepared_dataset_alias", lambda *args: path)
    monkeypatch.setattr(prepared_cache, "load_prepared_dataset", lambda _: dataset)
    monkeypatch.setattr(prepared_cache, "attach_control_sample_mask", lambda value, control, **kwargs: value)
    actual = prepared_cache.PreparedCacheLookup().find("control", _config(tmp_path), {})
    assert actual is dataset


def test_prepared_cache_rejects_incomplete_directory(tmp_path, monkeypatch):
    path = tmp_path / "preprocessing" / "datasets" / "blind"
    path.mkdir(parents=True)
    monkeypatch.setattr(prepared_cache, "prepared_dataset_alias", lambda *args: path)
    with pytest.raises(FileNotFoundError, match="incomplete"):
        prepared_cache.PreparedCacheLookup().find("blind", _config(tmp_path), {})


def test_prepared_cache_rejects_wrong_role(tmp_path, monkeypatch):
    path = tmp_path / "preprocessing" / "datasets" / "blind"
    path.mkdir(parents=True)
    (path / ".complete").write_text("complete")
    dataset = SimpleNamespace(directory=path, manifest={"dataset_role": "blind"})
    monkeypatch.setattr(prepared_cache, "prepared_dataset_alias", lambda *args: path)
    monkeypatch.setattr(prepared_cache, "load_prepared_dataset", lambda _: dataset)
    with pytest.raises(ValueError, match="role mismatch"):
        prepared_cache.PreparedCacheLookup().find("control", _config(tmp_path), {})
