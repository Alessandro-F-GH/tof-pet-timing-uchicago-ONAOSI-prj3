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
    assert config["thresholds_mV"] == [30., 50., 100.]
    assert config["baseline_window_ns"] == [-1.4, -1.0]
    assert not normalize_reference({})["enabled"]
    for invalid in (
        {"thresholds_mV": [10, 20]},
        {"thresholds_mV": [10, 20, 101]},
        {"thresholds_mV": [20, 10, 30]},
        {"minimum_coverage": 1.1},
        {"baseline_window_ns": [0, 0]},
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
