"""Training-population plots never replace validation or modify blind artifacts."""

import json
import random
from pathlib import Path
from types import SimpleNamespace

import matplotlib.pyplot as plt
import numpy as np
from waveform_analysis.data.storage import RunStore
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
