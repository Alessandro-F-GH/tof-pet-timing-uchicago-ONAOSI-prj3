"""Regression checks for fixed prepared multithreshold energy references."""
from pathlib import Path

import numpy as np
import pytest

from waveform_analysis.engine.multithreshold_reference import (
    normalize_reference, reference_directory, load_reference,
)
from waveform_analysis.core.exceptions import ConfigError
from waveform_analysis.reporting import plotting


def test_fixed_reference_defaults_and_config_validation():
    config = normalize_reference({"enabled": True})
    assert config["candidate_levels_mV"] == [10., 20., 30., 50., 100.]
    assert not normalize_reference({})["enabled"]
    for invalid in (
        {"candidate_levels_mV": [10, 20]},
        {"candidate_levels_mV": [10, 20, 30, 50, 101]},
        {"candidate_levels_mV": [20, 10, 30, 50, 100]},
        {"minimum_coverage": 1.1},
        {"enabled": "yes"},
        {"unknown_key": 2},
    ):
        with pytest.raises(ConfigError):
            normalize_reference(invalid)


def test_missing_reference_is_optional(tmp_path):
    assert load_reference(tmp_path, "wide") is None
    assert reference_directory(tmp_path, "wide") == tmp_path / "references/energy/wide"


def test_mt_reference_is_line_not_model_bar(tmp_path, monkeypatch):
    import json
    import matplotlib.pyplot as plt
    cfg = json.loads((Path(__file__).parents[1] /
                      "config/plots/default.json").read_text())
    observed = []

    def capture(fig, path):
        observed.append(fig.axes[0])
        plt.close(fig)
        return Path(path)

    monkeypatch.setattr(plotting, "_save", capture)
    row = {"code": "D-LIN", "display_name": "Ridge", "formulation": "direct",
           "ctr_ps": 110., "ctr_std_ps": 2.}
    plotting.blind_metric_bar([row], tmp_path / "ctr.pdf", cfg,
                             metric="ctr", led_value=150., mt_value=125.)
    ax = observed[0]
    assert len(ax.patches) == 1
    reference = {line.get_label(): line for line in ax.lines}
    assert "LED reference" in reference
    assert "Multi-threshold reference" in reference
    np.testing.assert_allclose(reference["Multi-threshold reference"].get_ydata(),
                               [125., 125.])


def test_crossing_diagnostics_classify_missing_and_valid():
    from waveform_analysis.reporting.multithreshold_diagnostics import _failure_reasons
    crossings = np.array([
        [[1., 2., 3., 4.], [1., 2., 3., 4.]],
        [[1., 2., np.nan, 4.], [1., 2., 3., 4.]],
    ])
    diag = {
        "crossing_times_ps": crossings,
        "led_crossing_ps": np.zeros((2, 2)),
        "intercept_ps": np.zeros((2, 2, 1)),
    }
    reason = _failure_reasons(np.array([1., np.nan]),
                              np.array([0., 0.]), diag)
    assert reason.tolist() == ["valid", "missing_threshold_crossing"]


def test_thresholds_to_100_mv_are_batch_opt_in():
    from waveform_analysis.engine.multithreshold_led import validate_threshold_sets
    assert validate_threshold_sets([(20, 30, 50, 100)], maximum_mV=100.) == (
        (20., 30., 50., 100.),)
    with pytest.raises(ValueError):
        validate_threshold_sets([(20, 30, 50, 100)])


def test_led_crossings_need_not_be_close_to_zero():
    """Regression for an otherwise valid pair with LED offsets >2.5 samples."""
    from waveform_analysis.engine.multithreshold_led import multithreshold_corrections
    grid = np.arange(-40, 81, dtype=float) * 10.  # ps
    signals = np.zeros((1, 2, grid.size), dtype=float)
    # LED=30 mV at t=+100 ps for detector 1 and t=-100 ps for detector 2.
    # All 30/50/100 mV crossings are inside the allowed rising-edge window.
    signals[0, 0] = np.maximum(0., grid + 50.) / 500.
    signals[0, 1] = np.maximum(0., grid + 250.) / 500.
    corrected, diag = multithreshold_corrections(
        signals, grid, threshold_sets=[(30., 50., 100.)],
        baseline_indices=[0, 1], led_threshold_mV=30.,
        physical_scale_mV=[500., 500.],
        crossing_window_ns=(-.3, .7), maximum_threshold_mV=100.,
    )
    assert np.isfinite(corrected[0, 0])
    assert np.isfinite(diag["led_crossing_ps"]).all()
    assert np.max(np.abs(diag["led_crossing_ps"])) > 25.


def test_native_control_candidates_and_event_order():
    from waveform_analysis.engine.multithreshold_native import (
        threshold_combinations, native_positions)
    from types import SimpleNamespace
    groups = threshold_combinations()
    assert len(groups) == 10
    assert groups[0] == (10., 20., 30.)
    assert groups[-1] == (30., 50., 100.)
    native = SimpleNamespace(event_index=np.array([13, 7, 10], np.int64))
    np.testing.assert_array_equal(native_positions(native, [10, 13]), [2, 0])
    with pytest.raises(ValueError, match="misses"):
        native_positions(native, [10, 11])


def test_native_pickoff_control_freezes_best_candidate(monkeypatch):
    from waveform_analysis.engine import multithreshold_native as native
    from types import SimpleNamespace
    waves = SimpleNamespace(n_events=20)
    groups = native.threshold_combinations()
    targets = np.linspace(-30., 30., 20)
    saved = np.repeat(targets[:, None], 10, axis=1)
    saved[:, 1] = 0.
    saved[0, 1] = np.nan
    monkeypatch.setattr(native, "candidate_pickoffs",
                        lambda *args, **kwargs: (saved, None))
    monkeypatch.setattr(native, "metric_values",
                        lambda values, fit, seed: {"ctr_ps": float(np.std(values))})
    choice, scan = native.select_control_thresholds(
        waves, {"histogram_bin_width_ps": 10.},
        levels=[10, 20, 30, 50, 100], minimum_coverage=.90,
        seed=3, baseline_window_ns=(-2., -1.), true_tof_ps=0.)
    assert choice["thresholds_mV"] == list(groups[1])
    assert choice["coverage"] == pytest.approx(.95)
    assert len(scan) == 10


def test_native_predictions_keep_original_led_target(tmp_path):
    from waveform_analysis.engine.multithreshold_reference import _save_predictions
    from types import SimpleNamespace
    prepared = SimpleNamespace(
        event_index=np.array([7, 9]),
        energy_target_ps=np.array([14., -3.]),
        manifest={"event_population_identity": "pop",
                  "analysis_protocol_identity": "protocol"})
    valid, residual, target = _save_predictions(
        tmp_path / "native.npz", prepared, np.array([111., np.nan]),
        control_offset_ps=10., true_tof_ps=1.,
        thresholds_mV=(10., 20., 30.), role="blind")
    np.testing.assert_array_equal(valid, [True, False])
    assert residual[0] == pytest.approx(100.)
    with np.load(tmp_path / "native.npz") as saved:
        assert saved["prediction_ps"][0] == pytest.approx(14. - 100.)
        assert saved["corrected_ps"][0] == pytest.approx(100.)
        assert np.isnan(saved["corrected_ps"][1])
