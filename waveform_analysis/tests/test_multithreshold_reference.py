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
    assert config["thresholds_mV"] == [10., 20., 30., 40., 50.]
    assert config["baseline_window_ns"] == [-1.4, -1.0]
    assert not normalize_reference({})["enabled"]
    for invalid in (
        {"thresholds_mV": [10, 20]},
        {"thresholds_mV": [10, 20, 55]},
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
