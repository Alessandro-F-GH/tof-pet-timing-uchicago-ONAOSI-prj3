"""Boxplots retain original blind residual values and the complete finite tails."""

import json
from pathlib import Path

import matplotlib.axes
import matplotlib.pyplot as plt
import numpy as np
import pytest

from waveform_analysis.data.storage import RunStore
from waveform_analysis.reporting import plotting


def config():
    return json.loads((Path(__file__).parents[1] / "config/plots/default.json").read_text())


def test_blind_boxplot_preserves_bias_outliers_and_saved_predictions(tmp_path, monkeypatch):
    store = RunStore(tmp_path)
    led = np.array([10., 11., 12., 13., 14., 100., np.nan])
    corrected = np.array([-10., -5., 0., 5., 10., 50., -10000.])
    store.save_predictions(event_id=np.arange(7), prediction_ps=led-corrected,
                           corrected_ps=corrected, led_residual_ps=led)
    before = store.predictions_path.read_bytes()
    original_boxplot = matplotlib.axes.Axes.boxplot
    captured = {}

    def boxplot(ax, data, **kwargs):
        captured["data"] = [values.copy() for values in data]
        captured["kwargs"] = kwargs
        captured["artists"] = original_boxplot(ax, data, **kwargs)
        return captured["artists"]

    def save(fig, path):
        captured["ylim"] = fig.axes[0].get_ylim()
        plt.close(fig)
        return path

    monkeypatch.setattr(matplotlib.axes.Axes, "boxplot", boxplot)
    monkeypatch.setattr(plotting, "_save", save)
    output = plotting.plot_run_blind_boxplot(tmp_path, config())
    assert output.name == "blind_boxplot.png"
    np.testing.assert_array_equal(captured["data"][0], led[:6])
    np.testing.assert_array_equal(captured["data"][1], corrected[:6])
    assert captured["kwargs"]["showfliers"] is True
    assert captured["kwargs"]["whis"] == 1.5
    assert captured["artists"]["medians"][0].get_ydata()[0] == 12.5
    np.testing.assert_array_equal(captured["artists"]["fliers"][0].get_ydata(), [100.])
    np.testing.assert_array_equal(captured["artists"]["fliers"][1].get_ydata(), [50.])
    assert captured["ylim"][0] < -10 and captured["ylim"][1] > 100
    assert store.predictions_path.read_bytes() == before


def test_boxplot_whisker_configuration_and_missing_data(tmp_path, monkeypatch):
    cfg = config()
    cfg["boxplot"]["whisker_iqr"] = 3.0
    captured = {}
    original = matplotlib.axes.Axes.boxplot

    def boxplot(ax, data, **kwargs):
        captured.update(kwargs)
        return original(ax, data, **kwargs)

    monkeypatch.setattr(matplotlib.axes.Axes, "boxplot", boxplot)
    plotting.blind_residual_boxplot(
        [{"label": "D-MLP", "led": [1, 2, 3], "corrected": [-1, 0, 1]}],
        tmp_path / "comparison.png", cfg, title="Blind comparison",
    )
    assert captured["whis"] == 3.0
    assert (tmp_path / "comparison.png").is_file()
    assert plotting.plot_run_blind_boxplot(tmp_path, cfg) is None
    with pytest.raises(ValueError, match="must be paired"):
        plotting.blind_residual_boxplot(
            [{"label": "bad", "led": [1, 2], "corrected": [1]}],
            tmp_path / "bad.png", cfg, title="Invalid",
        )
