"""Control calibration is frozen across roles, windows and resampled events."""

import json
from pathlib import Path
from types import SimpleNamespace

import matplotlib.pyplot as plt
import numpy as np
import pytest

from waveform_analysis.data import preparation
from waveform_analysis.data.storage import RunStore
from waveform_analysis.data.view import model_target, standard_delta
from waveform_analysis.engine import control_preprocessing as control
from waveform_analysis.reporting import plotting, report_engine, stats
from waveform_analysis.signal.metrics import rmse_ps


def test_control_mean_uses_selected_threshold_finite_coincidences(monkeypatch):
    # Threshold selection is unchanged; calibration excludes nonfinite/outside-window events.
    grid = np.zeros((5, 2, 2))
    grid[:, 0, 0] = [20., 40., 80., 5000., np.nan]
    grid[:, 0, 1] = [120., 140., 180., 5000., np.nan]
    monkeypatch.setattr(control, "led_grid", lambda *a, **k: grid)
    monkeypatch.setattr(control, "ctr_estimate",
                        lambda values, *a, **k: SimpleNamespace(ctr_ps=float(np.mean(values))))
    threshold, rows = control._select_led(
        SimpleNamespace(n_events=5), "energy_to_energy", {"true_tof_ps": 10.},
        {"selection": {"baseline_window_ns": [-2., -1.]},
         "led_selection": {"thresholds_mV": [5., 15.], "coincidence_window_ns": 1.,
                           "minimum_crossing_efficiency": .5}}, {},
    )
    assert threshold == 5.
    assert rows[0]["control_mean_ps"] == np.mean([10., 30., 70.])
    assert rows[0]["n"] == 3


@pytest.mark.parametrize("role", ["control", "development", "blind"])
@pytest.mark.parametrize("control_mean", [50., 4000.])
def test_prepared_targets_and_selection_use_frozen_control_mean(
    tmp_path, monkeypatch, role, control_mean,
):
    native = SimpleNamespace(
        n_events=4, manifest={"fingerprint": "native", "source": "acquisition"},
        energy_windows_mV=np.zeros((4, 2, 6)),
        energy_sample_interval_s=np.full((4, 2), 1e-9),
        event_index=np.arange(4), bias_voltage_V=np.full(4, 48.),
    )
    grid = np.zeros((4, 2, 1))
    grid[:, 0, 0] = [110., 130., 5000., np.nan]
    monkeypatch.setattr(preparation, "led_grid", lambda *a, **k: grid)
    monkeypatch.setattr(preparation, "anchor_grid", lambda *a, **k: np.full((4, 2), 2))
    mode = "energy_to_energy"
    dataset_cfg = {"root_file": str(tmp_path / 'data.root'), "true_tof_ps": 10.}
    cfg = {"mode": mode, "window_ns": {"start": -1, "end": 1},
           "ml_input": {"subsampling": 1}, role: dataset_cfg,
           "preprocessing": {"selection": {"baseline_window_ns": [-2, -1]},
                             "led_selection": {"coincidence_window_ns": 1},
                             "energy": {"vertical_scale_limit_mV": [[-1, 1], [-1, 1]]}}}
    artifact = {"fingerprint": "control", "selected_led_threshold_mV": {mode: 5.},
                "led_control_mean_ps": {mode: control_mean}}
    prepared = preparation.prepare_ml_dataset(
        native, artifact, cfg, dataset_role=role, cache_dir=tmp_path,
    )
    np.testing.assert_array_equal(prepared.event_index, [0, 1])
    target = model_target(prepared, mode)
    np.testing.assert_array_equal(target, np.array([100., 120.]) - control_mean)
    np.testing.assert_array_equal(standard_delta(prepared, mode) - 10., target)
    assert prepared.manifest['led_control_mean_ps'] == control_mean
    # A dataset-local centering would remove this transfer bias, which must be retained.
    assert rmse_ps(target) == np.sqrt(np.mean(target ** 2))
    if control_mean == 50.:
        assert target.mean() == 60.
        assert rmse_ps(target) > np.std(target)


def test_histogram_does_not_center_blind_again(tmp_path, monkeypatch):
    led = np.array([40., 50., 60., 70.])  # Already subtracts control mean; blind mean != 0.
    store = RunStore(tmp_path)
    store.save_predictions(event_id=np.arange(4), prediction_ps=np.zeros(4),
                           corrected_ps=led, led_residual_ps=led)
    cfg = json.loads((Path(__file__).parents[1] / 'config/plots/default.json').read_text())
    captured = []
    original = plt.Axes.hist
    def hist(ax, values, *a, **k):
        captured.append(np.asarray(values).copy())
        return original(ax, values, *a, **k)
    monkeypatch.setattr(plt.Axes, 'hist', hist)
    monkeypatch.setattr(plotting, '_save', lambda fig, path: plt.close(fig) or path)
    plotting.plot_run_blind(tmp_path, cfg)
    np.testing.assert_array_equal(captured[0], led)


def test_bootstrap_keeps_control_centering_and_transfer_bias(monkeypatch):
    monkeypatch.setattr(stats, 'ctr_estimate',
                        lambda *a, **k: SimpleNamespace(ctr_ps=1.))
    led = np.array([40., 50., 60., 70., 80.])
    central = stats.paired_central_metrics(led / 2., led, {}, seed=42)
    assert central['led_rmse_ps'] == rmse_ps(led)
    assert central['rmse_improvement_ps'] == rmse_ps(led) - rmse_ps(led / 2.)
    _, draws = stats.blind_event_bootstrap(led / 2., led, {}, n_resamples=12, seed=42)
    # Even the smallest draw has RMSE >= 40; per-draw recentering would be <= 15.
    assert np.min(draws['led_rmse_ps']) >= 40.


def test_reporting_rejects_uncentered_legacy_results(tmp_path):
    root = tmp_path / 'batch'
    store = RunStore(root / 'run')
    store.write_manifest({'schema_version': 51, 'status': 'complete'})
    tables = root / 'tables'
    tables.mkdir()
    (tables / 'runs.csv').write_text('path\nrun\n')
    with pytest.raises(ValueError, match='rerun the batch'):
        report_engine.collect_runs(root)
    with pytest.raises(ValueError, match='rerun the batch'):
        plotting.render_run_plots(store.root, {})


def test_control_artifact_persists_separate_mode_calibrations(tmp_path, monkeypatch):
    source = tmp_path / 'control.root'
    source.write_bytes(b'synthetic control')
    monkeypatch.setattr(control, 'fit_selection_rules', lambda *a, **k: {})
    monkeypatch.setattr(control, 'apply_selection_rules', lambda *a, **k: None)
    monkeypatch.setattr(control, 'preprocess_selected', lambda *a, **k: None)
    monkeypatch.setattr(control, 'plot_led_selection', lambda *a, **k: None)
    def select(prepared, mode, *a):
        mean = 100. if mode == 'energy_to_energy' else -20.
        return 5., [{'threshold_mV': 5., 'ctr_ps': 200., 'n': 10,
                     'efficiency': 1., 'control_mean_ps': mean}]
    monkeypatch.setattr(control, '_select_led', select)
    dataset = {'root_file': str(source), 'true_tof_ps': 0., 'channels': {'timing': [3, 4]}}
    manifest, directory = control.fit_control_artifact(
        source, dataset, {}, {}, cache_root=tmp_path,
    )
    means = {'energy_to_energy': 100., 'timing_to_timing': -20.}
    assert manifest['led_control_mean_ps'] == means
    assert json.loads((directory / 'manifest.json').read_text())['led_control_mean_ps'] == means
    loaded = control.load_control_artifact(directory, source, dataset, {}, {})
    assert loaded['led_control_mean_ps'] == means


def test_batch_planner_rebuilds_old_uncentered_runs(tmp_path):
    from waveform_analysis.engine.batch import _planner_fingerprints, _run_state
    from waveform_analysis.tests.test_batch_planner import config
    cfg = config(tmp_path)
    store = RunStore(cfg['output_dir'])
    store.write_manifest({'schema_version': 51, 'status': 'complete'})
    assert _run_state(cfg, {cfg['run_id']: _planner_fingerprints(cfg)})[:2] == (
        'rebuild', 'incompatible_schema',
    )
