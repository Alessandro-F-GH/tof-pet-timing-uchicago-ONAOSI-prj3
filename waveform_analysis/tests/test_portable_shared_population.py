"""Shared batch population metadata must survive Linux-to-Windows resume."""
import json

import pytest

from waveform_analysis.data import results


def _population(identity, count=70490):
    return {
        "development": {
            "n_selected": count,
            "bias_voltage_V": [48.0],
            "population_identity": identity,
            "analysis_protocol_identity": identity,
            "n_before_fixed_led": 71000,
            "n_after_fixed_led": 70500,
            "n_dropped_window": 10,
        }
    }


def test_cross_machine_shared_population_reuses_existing_context(tmp_path, monkeypatch):
    root = tmp_path / "batch"
    run = root / "energy" / "onishi" / "direct_mlp"
    run.mkdir(parents=True)
    monkeypatch.setattr(results, "batch_root", lambda path: root)
    monkeypatch.setattr(results, "resolve_run_config", lambda path: {})
    context = run.parent / "population.json"
    context.write_text(json.dumps({
        "dataset_populations": _population("linux-sha"),
        "control_artifact": "/home/alessandro/cache/control",
        "sample_mask_identity": "linux-mask",
    }), encoding="utf-8")
    results.write_run_manifest(run, {
        "status": "complete",
        "dataset_populations": _population("windows-sha"),
        "control_artifact": r"C:\Users\aless\cache\control",
        "sample_mask_identity": "windows-mask",
    })
    actual = results.read_json(context)
    assert actual["dataset_populations"]["development"]["population_identity"] == "linux-sha"
    assert actual["control_artifact"] == "/home/alessandro/cache/control"
    assert results.read_json(run / "metadata" / "manifest.json")["status"] == "complete"


def test_shared_population_count_mismatch_is_rejected(tmp_path, monkeypatch):
    root = tmp_path / "batch"
    run = root / "energy" / "onishi" / "direct_mlp"
    run.mkdir(parents=True)
    monkeypatch.setattr(results, "batch_root", lambda path: root)
    monkeypatch.setattr(results, "resolve_run_config", lambda path: {})
    context = run.parent / "population.json"
    context.write_text(json.dumps({
        "dataset_populations": _population("linux-sha")
    }), encoding="utf-8")
    with pytest.raises(ValueError, match="development.n_selected"):
        results.write_run_manifest(run, {
            "status": "complete",
            "dataset_populations": _population("windows-sha", count=70491),
        })
