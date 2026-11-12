from __future__ import annotations

import pickle
from types import SimpleNamespace

import numpy as np

from waveform_analysis.data.artifact_store import ArtifactStore
from waveform_analysis.data.named_cache import transform_alias
from waveform_analysis.engine import transform_migration


def test_import_fitted_transform_publishes_shared_alias(tmp_path, monkeypatch):
    run = tmp_path / "historical" / "energy" / "wide" / "direct_minirocket"
    model = run / "model"
    transform_dir = model / "feature_transform"
    transform_dir.mkdir(parents=True)
    np.save(model / "sample_mask.npy", np.asarray([True, False, True, True]))
    source_artifact = SimpleNamespace(
        metadata={"training_seed": 123, "feature_count": 17}
    )
    with (transform_dir / "transform.pkl").open("wb") as stream:
        pickle.dump(source_artifact, stream)

    transform_spec = SimpleNamespace(
        name="direct_minirocket_multivariate",
        fit_role="control",
        save=lambda *_: None,
        parameters=lambda candidate, space: {"num_kernels": 10000, "n_jobs": -1},
    )
    spec = SimpleNamespace(
        name="direct_minirocket",
        feature_transform=transform_spec,
        requires_frozen_features=False,
    )
    config = {
        "run_id": "energy__wide__direct_minirocket",
        "seed": 42,
        "mode": "energy_to_energy",
        "window_name": "wide",
        "window_ns": {"start": -2.0, "end": 50.0},
        "model": {"name": "direct_minirocket", "space": {"parameters": {}}},
        "control": {"root_file": "control.root"},
    }
    run_record = {
        "directory": run,
        "manifest": {
            "model": "direct_minirocket",
            "dataset_populations": {
                "control": {"analysis_protocol_identity": "control-protocol-v1"}
            },
        },
    }
    monkeypatch.setattr(transform_migration, "discover_runs", lambda *a, **k: [run_record])
    monkeypatch.setattr(transform_migration, "resolve_run_config", lambda *_: config)
    monkeypatch.setattr(transform_migration, "get_model", lambda _: spec)

    cache_root = tmp_path / "portable-cache"
    records = transform_migration.import_fitted_transforms(
        tmp_path / "historical", cache_root=cache_root
    )

    assert records[0]["status"] == "imported"
    alias = transform_alias(
        cache_root, config, transform_spec.name, transform_spec.fit_role
    )
    assert alias.is_symlink() or (alias / ".cache_link.json").is_file()
    reference = transform_migration.load_named_reference(
        ArtifactStore(cache_root / "artifact_store"),
        alias,
        expected_type="feature-transforms",
    )
    with reference.payload_path("transform.pkl").open("rb") as stream:
        restored = pickle.load(stream)
    assert restored.metadata["training_seed"] == 123
    assert restored.metadata["control_analysis_protocol_identity"] == "control-protocol-v1"
    np.testing.assert_array_equal(
        np.load(model / "sample_mask.npy"),
        np.asarray([True, False, True, True]),
    )
