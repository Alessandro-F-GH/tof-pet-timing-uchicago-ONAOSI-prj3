from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from waveform_analysis.core.config import BatchConfig
from waveform_analysis.engine import cache_preparation


def _batch(tmp_path):
    run = {
        "run_id": "energy__wide__direct_ridge",
        "mode": "energy_to_energy",
        "window_name": "wide",
        "window_ns": {"start": -2.0, "end": 50.0},
        "seed": 17,
        "fit": {"histogram_bin_width_ps": 5},
        "model": {"name": "direct_ridge", "space": {"parameters": {}}},
        "ml_input": {"subsampling": 1},
        "preprocessing": {
            "cache_dir": str(tmp_path / "old-cache"),
            "artifact_root": str(tmp_path / "old-artifacts"),
        },
        "control": {"root_file": "control.root", "channels": {"energy": ["a"]}},
        "development": {"root_file": "development.root", "channels": {"energy": ["a"]}},
        "blind": {"root_file": "blind.root", "channels": {"energy": ["a"]}},
    }
    return BatchConfig(
        name="test-batch",
        output_dir=str(tmp_path / "results"),
        source_path=str(tmp_path / "batch.json"),
        results={"folder": str(tmp_path / "results")},
        datasets={},
        protocol={},
        axes={},
        plot_config={},
        runs=(run,),
    )


def test_prepare_batch_cache_records_complete_without_training(tmp_path, monkeypatch):
    batch = _batch(tmp_path)
    dataset = SimpleNamespace(
        manifest={"analysis_protocol_identity": "dataset-protocol-identity"}
    )
    calls = []

    monkeypatch.setattr(
        cache_preparation, "fit_control",
        lambda config, **kwargs: (calls.append(("control", config["run_id"])) or {}, None),
    )
    monkeypatch.setattr(
        cache_preparation, "prepare_role_dataset",
        lambda config, role, control, **kwargs: (
            None, None, calls.append((role, config["run_id"])) or dataset
        ),
    )
    monkeypatch.setattr(
        cache_preparation, "get_model",
        lambda name: SimpleNamespace(feature_transform=None),
    )

    manifest_path = cache_preparation.prepare_batch_cache(
        batch, cache_root=tmp_path / "portable-cache"
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    assert manifest["status"] == "complete"
    assert manifest["prepared_runs"][0]["run_id"] == "energy__wide__direct_ridge"
    assert (tmp_path / "portable-cache" / "preprocessing").is_dir()
    assert (tmp_path / "portable-cache" / "artifact_store").is_dir()
    assert [call[0] for call in calls] == ["control", "control", "development", "blind"]
    assert not any("fit" in str(call[0]).lower() for call in calls if call[0] not in {"control", "development", "blind"})


def test_prepare_batch_cache_records_failure_and_reraises(tmp_path, monkeypatch):
    batch = _batch(tmp_path)
    monkeypatch.setattr(
        cache_preparation, "fit_control",
        lambda config, **kwargs: (_ for _ in ()).throw(RuntimeError("deliberate test failure")),
    )

    import pytest
    with pytest.raises(RuntimeError, match="deliberate test failure"):
        cache_preparation.prepare_batch_cache(batch, cache_root=tmp_path / "portable-cache")

    manifest = json.loads(
        (tmp_path / "portable-cache" / "cache_manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["status"] == "failed"
    assert "deliberate test failure" in manifest["error"]
