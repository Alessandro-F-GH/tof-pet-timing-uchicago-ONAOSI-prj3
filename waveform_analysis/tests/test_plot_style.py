"""Presentation configuration applies locally and preserves plotted values."""

import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

from waveform_analysis.reporting import plotting, preprocessing_plots


def config():
    return json.loads(
        (Path(__file__).parents[1] / "config/plots/default.json").read_text()
    )


def test_style_overrides_and_global_state_restore():
    cfg = config()
    cfg["font"]["family"] = "DejaVu Sans"
    cfg["font"]["label_size"] = 17
    cfg["legend"]["frame"] = True
    cfg["axes"]["spines"]["top"] = True
    cfg["palette"] = ["#123456"]
    original = {
        key: mpl.rcParams[key]
        for key in (
            "font.family",
            "axes.labelsize",
            "legend.frameon",
            "axes.spines.top",
            "axes.prop_cycle",
        )
    }
    with plotting.plot_context(cfg):
        fig, ax = plt.subplots()
        (line,) = ax.plot([1, 2], [3, 4], label="Measurement")
        ax.set_xlabel("Time [ps]")
        legend = ax.legend()
        assert line.get_color() == "#123456"
        assert ax.xaxis.label.get_fontsize() == 17
        assert legend.get_frame_on()
        assert ax.spines["top"].get_visible()
        plt.close(fig)
    assert {key: mpl.rcParams[key] for key in original} == original


def test_preprocessing_uses_same_config_and_preserves_scan(tmp_path, monkeypatch):
    cfg = config()
    cfg["font"]["label_size"] = 15
    captured = []

    def inspect(fig, path):
        ax = fig.axes[0]
        np.testing.assert_array_equal(ax.lines[0].get_xdata(), [1, 2])
        np.testing.assert_array_equal(ax.lines[0].get_ydata(), [100, 90])
        assert ax.xaxis.label.get_fontsize() == 15
        assert not ax.spines["right"].get_visible()
        captured.append(True)
        plt.close(fig)

    monkeypatch.setattr(preprocessing_plots, "_save", inspect)
    token = preprocessing_plots._CONFIG.set(cfg)
    try:
        preprocessing_plots.plot_led_selection(
            [{"threshold_mV": 1, "ctr_ps": 100}, {"threshold_mV": 2, "ctr_ps": 90}],
            2,
            tmp_path / "led.png",
            "Control",
        )
    finally:
        preprocessing_plots._CONFIG.reset(token)
    assert captured


def test_vector_export_and_configured_bar_style(tmp_path, monkeypatch):
    cfg = config()
    cfg["output"]["format"] = "pdf"
    cfg["bar"]["show_values"] = False
    cfg["formulations"]["shared"]["color"] = "#123456"
    original_save = plotting._save

    def inspect(fig, path):
        ax = fig.axes[0]
        np.testing.assert_array_equal([p.get_height() for p in ax.patches], [100, 90])
        assert ax.patches[0].get_facecolor() == mpl.colors.to_rgba("#123456")
        assert not ax.texts
        assert not ax.get_legend().get_frame_on()
        return original_save(fig, path)

    monkeypatch.setattr(plotting, "_save", inspect)
    output = plotting.grouped_bar(
        ["A", "B"],
        [
            {
                "label": cfg["formulations"]["shared"]["label"],
                "values": [100, 90],
                "errors": [2, 3],
            }
        ],
        plotting.output_path(tmp_path, "comparison", cfg),
        cfg,
        ylabel="CTR [ps]",
        title="Blind timing resolution",
    )
    assert output.read_bytes().startswith(b"%PDF")
