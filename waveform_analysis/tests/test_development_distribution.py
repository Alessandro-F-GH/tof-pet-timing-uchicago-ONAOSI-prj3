"""Training-population plots never replace validation or modify blind artifacts."""

import json
import random
from pathlib import Path
from types import SimpleNamespace

import matplotlib.pyplot as plt
import numpy as np
from waveform_analysis.data.storage import RunStore
from waveform_analysis.core.io import canonical_hash
from waveform_analysis.engine.diagnostics import diagnostic_random_state, save_development_diagnostic
from waveform_analysis.engine.train import FittedModel
from waveform_analysis.reporting import development, plotting


def test_development_predictions_use_all_events_and_output_limit(tmp_path):
    store = RunStore(tmp_path)
    store.save_predictions(event_id=[1], prediction_ps=[1], corrected_ps=[2], led_residual_ps=[3])
    blind_before = store.predictions_path.read_bytes()
    dataset = SimpleNamespace(
        n_events=4, event_index=np.array([9, 8, 7, 6]),
        energy_windows=np.arange(16, dtype=np.float32).reshape(4, 2, 2),
        energy_time_ps=np.array([0., 1.]), energy_target_ps=np.array([1., 3., 9., 20.]),
        manifest={"analysis_protocol_identity": "prepared"},
    )
    spec = SimpleNamespace(predict=lambda artifact, pair: pair[:, 0, 0])
    fitted = FittedModel(None, {}, output_max_abs_ps=5.)
    config = {"mode": "energy_to_energy", "runtime": {"prediction_chunk_size": 2}}
    save_development_diagnostic(store, spec, fitted, dataset, config, "final")
    with np.load(store.development_predictions_path) as diagnostic:
        np.testing.assert_array_equal(diagnostic["event_id"], [9, 8, 7, 6])
        np.testing.assert_array_equal(diagnostic["prediction_ps"], [0, 4, 5, 5])
        np.testing.assert_array_equal(diagnostic["corrected_ps"], [1, -1, 4, 15])
    assert store.predictions_path.read_bytes() == blind_before
    store.invalidate_from("bootstrap")
    assert store.development_predictions_path.exists()
    store.invalidate_from("final_fit")
    assert not store.development_predictions_path.exists()


def test_diagnostic_restores_random_generators():
    import torch
    python_state = random.getstate()
    numpy_state = np.random.get_state()
    torch_state = torch.random.get_rng_state().clone()
    with diagnostic_random_state():
        random.random()
        np.random.random(7)
        torch.rand(7)
    assert random.getstate() == python_state
    current = np.random.get_state()
    assert current[0] == numpy_state[0] and current[2:] == numpy_state[2:]
    np.testing.assert_array_equal(current[1], numpy_state[1])
    assert torch.equal(torch.random.get_rng_state(), torch_state)


def test_development_and_blind_use_identical_histogram_style(tmp_path, monkeypatch):
    store = RunStore(tmp_path)
    cfg = json.loads((Path(__file__).parents[1] / "config/plots/default.json").read_text())
    led, corrected = np.arange(12.) + 10., np.arange(12.) - 4.
    for role in ("blind", "development"):
        store.save_predictions(event_id=np.arange(12), prediction_ps=led-corrected,
                               corrected_ps=corrected, led_residual_ps=led, dataset_role=role)
    captured = []

    def save(fig, path):
        ax = fig.axes[0]
        captured.append((ax.get_xlabel(), [p.get_xy().copy() for p in ax.patches]))
        plt.close(fig)
        return path

    monkeypatch.setattr(plotting, "_save", save)
    assert plotting.plot_run_blind(tmp_path, cfg).name == "blind.png"
    assert plotting.plot_run_development(tmp_path, cfg).name == "development.png"
    assert captured[0][0] == "Blind residual [ps]"
    assert captured[1][0] == "Development residual [ps] (final model)"
    for blind, train in zip(captured[0][1], captured[1][1]):
        np.testing.assert_array_equal(blind, train)
    with np.load(store.development_predictions_path) as data:
        np.testing.assert_array_equal(data["led_residual_ps"], led)  # centering is display-only
    # A diagnostic from a different final fit must never be presented as current.
    store.write_manifest({"stage_fingerprints": {"final_fit": "new-fit"}})
    stale_plot = plotting.output_path(store.plots_dir, "development", cfg)
    stale_plot.parent.mkdir(parents=True, exist_ok=True)
    stale_plot.write_text("obsolete")
    assert plotting.plot_run_development(tmp_path, cfg) is None
    assert not stale_plot.exists()


def test_reporting_backfills_only_unique_matching_cache(tmp_path, monkeypatch):
    store = RunStore(tmp_path / "run")
    control = tmp_path / "control"
    control.mkdir()
    (control / "manifest.json").write_text(json.dumps({"fingerprint": "control"}))
    config = {"mode": "timing_to_timing", "window_ns": [-1.5, 2.],
              "ml_input": {"subsampling": 1}, "preprocessing": {"cache_dir": str(tmp_path)},
              "development": {"root_file": "development.root", "true_tof_ps": 0.}}
    store.write_resolved_config(config)
    cache = tmp_path / "development_ml/prepared/one"
    cache.mkdir(parents=True)
    cache_metadata = {"dataset_role": "development", "dataset_source": "development.root",
                      "mode": "timing_to_timing", "window_ns": [-1.5, 2.], "subsampling": 1,
                      "true_tof_ps": 0., "control_fingerprint": "control",
                      "event_population_identity": "events"}
    (cache / "manifest.json").write_text(json.dumps(cache_metadata))
    run = {"directory": store.root, "best": {"parameters": {}},
           "manifest": {"model": "direct_linear_ridge", "control_artifact": str(control),
                        "stage_fingerprints": {"final_fit": "final"},
                        "dataset_populations": {"development": {"population_identity": "events"}}}}
    calls = []
    monkeypatch.setattr(development, "saved_model_complete", lambda *args: True)
    monkeypatch.setattr(development, "load_prepared_dataset",
                        lambda path: SimpleNamespace(n_events=12))
    monkeypatch.setattr(development, "load_fitted_model", lambda *args: calls.append("load"))
    monkeypatch.setattr(development, "save_development_diagnostic",
                        lambda *args: calls.append("predict") or store.development_predictions_path)
    monkeypatch.setattr(development, "release_training_memory", lambda: None)
    assert development.ensure_development_predictions(run) == store.development_predictions_path
    assert calls == ["load", "predict"]
    calls.clear()
    (cache / "manifest.json").write_text(json.dumps({**cache_metadata, "control_fingerprint": "wrong"}))
    assert development.ensure_development_predictions(run) is None
    assert not calls
    (cache / "manifest.json").write_text(json.dumps(cache_metadata))
    other = cache.parent / "two"
    other.mkdir()
    (other / "manifest.json").write_text(json.dumps(cache_metadata))
    assert development.ensure_development_predictions(run) is None
    assert not calls


def test_missing_control_json_uses_saved_protocol_identity(tmp_path, monkeypatch):
    store = RunStore(tmp_path / "run")
    config = {"mode": "timing_to_timing", "window_ns": {"start": -1.5, "end": 2.},
              "ml_input": {"subsampling": 1}, "preprocessing": {"cache_dir": str(tmp_path)},
              "development": {"root_file": "train.root", "true_tof_ps": 0.}}
    store.write_resolved_config(config)
    cache = tmp_path / "development_ml/prepared/one"
    cache.mkdir(parents=True)
    metadata = {"dataset_role": "development", "dataset_source": "train.root",
                "mode": config["mode"], "window_ns": config["window_ns"], "subsampling": 1,
                "true_tof_ps": 0., "analysis_protocol_identity": "trained-protocol",
                "event_population_identity": "events"}
    (cache / "manifest.json").write_text(json.dumps(metadata))
    run = {"directory": store.root, "best": {"parameters": {}},
           "manifest": {"model": "independent_cnn1d", "control_artifact": "missing/control",
                        "stage_fingerprints": {"final_fit": "final"},
                        "dataset_populations": {"development": {
                            "population_identity": "events", "analysis_protocol_identity": "trained-protocol"}}}}
    calls = []
    monkeypatch.setattr(development, "saved_model_complete", lambda *args: True)
    monkeypatch.setattr(development, "load_prepared_dataset", lambda _: SimpleNamespace(n_events=12))
    monkeypatch.setattr(development, "load_fitted_model", lambda *args: calls.append("load"))
    monkeypatch.setattr(development, "save_development_diagnostic",
                        lambda *args: calls.append("predict") or store.development_predictions_path)
    monkeypatch.setattr(development, "release_training_memory", lambda: None)
    assert development.ensure_development_predictions(run) == store.development_predictions_path
    assert calls == ["load", "predict"]
    calls.clear()
    (cache / "manifest.json").write_text(json.dumps({**metadata, "analysis_protocol_identity": "wrong"}))
    assert development.ensure_development_predictions(run) is None
    assert not calls


def test_old_neural_run_matches_exact_final_fit_without_control_json():
    from waveform_analysis.models import get_model
    spec = get_model("independent_cnn1d")
    config = {"model": {"name": spec.name}, "seed": 1001}
    best = {"candidate_id": "selected", "parameters": {"architecture": [8]}}
    payload = {"development": "protocol", "model": config["model"],
               "candidate_id": best["candidate_id"], "parameters": best["parameters"],
               "seed": 1001, "frozen_transform": None}
    fingerprint = canonical_hash({"schema_version": 51, "stage": "final_fit", "payload": payload})
    run = {"best": best, "manifest": {"schema_version": 51,
           "stage_fingerprints": {"final_fit": fingerprint}, "control_artifact": "missing/control"}}
    assert development._matches_training_identity(
        {"analysis_protocol_identity": "protocol"}, run, config, spec, None)
    assert not development._matches_training_identity(
        {"analysis_protocol_identity": "other"}, run, config, spec, None)


def test_missing_control_json_uses_directory_fingerprint():
    from waveform_analysis.models import get_model
    spec = get_model("shared_linear_ridge")
    run = {"manifest": {"control_artifact": r"C:\old\control\1234567890abcdef"}}
    assert development._matches_training_identity(
        {"control_fingerprint": "1234567890abcdef" + "a" * 48}, run, {}, spec, None)
    assert not development._matches_training_identity(
        {"control_fingerprint": "wrong"}, run, {}, spec, None)


def test_cache_path_resolves_after_moving_windows_checkout(tmp_path):
    root = tmp_path / "waveform_analysis"
    cache = root / "processed_data/ml_protocol_v3"
    (cache / "development_ml/prepared").mkdir(parents=True)
    run = root / "results/FBK/batch/timing/short/independent_cnn1d"
    config = {"preprocessing": {"cache_dir": r"C:\old\waveform_analysis\processed_data\ml_protocol_v3"}}
    assert development._development_cache_root(config, run) == cache
