from types import SimpleNamespace

import pytest

from waveform_analysis.data import prepared_cache


def _config(tmp_path):
    return {
        "mode": "energy_to_energy",
        "preprocessing": {
            "named_cache_root": str(tmp_path),
            "cache_dir": str(tmp_path / "preprocessing"),
            "artifact_root": str(tmp_path / "artifact_store"),
        },
    }


def test_named_prepared_cache_is_not_rejected_for_old_protocol_fingerprint(
    tmp_path, monkeypatch
):
    dataset_path = tmp_path / "prepared" / "control"
    dataset = SimpleNamespace(
        directory=dataset_path,
        manifest={
            "dataset_role": "control",
            "analysis_protocol_identity": "new-cache-protocol",
        },
    )
    monkeypatch.setattr(
        prepared_cache, "prepared_dataset_alias", lambda *args: tmp_path / "alias"
    )
    monkeypatch.setattr(prepared_cache, "alias_exists", lambda _: True)
    monkeypatch.setattr(prepared_cache, "resolve_alias", lambda _: dataset_path)
    monkeypatch.setattr(prepared_cache, "load_prepared_dataset", lambda _: dataset)
    monkeypatch.setattr(
        prepared_cache, "attach_control_sample_mask",
        lambda selected, control, **kwargs: selected,
        raising=False,
    )
    # The run manifest comes from an older results folder. The user-selected
    # named cache remains authoritative despite a different protocol hash.
    old_manifest = {
        "dataset_populations": {
            "control": {"analysis_protocol_identity": "old-run-protocol"}
        }
    }

    result = prepared_cache.PreparedCacheLookup().find(
        "control", _config(tmp_path), old_manifest
    )

    assert result is dataset


def test_prepared_cache_still_rejects_wrong_dataset_role(tmp_path, monkeypatch):
    dataset_path = tmp_path / "prepared" / "blind"
    dataset = SimpleNamespace(
        directory=dataset_path,
        manifest={
            "dataset_role": "blind",
            "analysis_protocol_identity": "some-protocol",
        },
    )
    monkeypatch.setattr(
        prepared_cache, "prepared_dataset_alias", lambda *args: tmp_path / "alias"
    )
    monkeypatch.setattr(prepared_cache, "alias_exists", lambda _: True)
    monkeypatch.setattr(prepared_cache, "resolve_alias", lambda _: dataset_path)
    monkeypatch.setattr(prepared_cache, "load_prepared_dataset", lambda _: dataset)

    with pytest.raises(ValueError, match="role mismatch"):
        prepared_cache.PreparedCacheLookup().find(
            "control", _config(tmp_path), {}
        )
