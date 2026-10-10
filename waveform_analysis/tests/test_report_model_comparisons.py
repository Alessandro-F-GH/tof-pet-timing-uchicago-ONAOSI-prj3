"""Reporting order and filters preserve the persisted scientific comparisons."""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pytest

from waveform_analysis.cli import _parser
from waveform_analysis.reporting import plotting, report_engine as engine


def plot_config():
    return json.loads((Path(__file__).parents[1] / "config/plots/default.json").read_text())


@pytest.fixture
def report_inputs(tmp_path, monkeypatch):
    runs = []
    for name, form, ctr in [
        ("shared_cnn1d", "shared", 110.),
        ("independent_cnn1d", "direct", 95.),
        ("antisymmetric_mlp", "shared", 105.),
        ("direct_mlp", "direct", 90.),
    ]:
        directory = tmp_path / name
        (directory / "artifacts").mkdir(parents=True)
        np.savez(directory / "artifacts/pred.npz", event_id=np.arange(32),
                 prediction_ps=np.arange(32) * ctr, corrected_ps=np.arange(32))
        runs.append({
            "directory": directory,
            "manifest": {"model": name, "estimator_formulation": form,
                         "mode": "timing_to_timing", "window_name": "short"},
            "best": {"candidate_id": "stored", "selection_metric": "ctr",
                     "validation_ctr_mean_ps": ctr + 1, "validation_ctr_std_ps": 2.,
                     "validation_rmse_mean_ps": 55., "validation_rmse_std_ps": 1., "folds": 3},
            "blind": {"ctr_ps": ctr, "rmse_ps": 50., "n_events": 32,
                      "led_ctr_ps": 130., "led_rmse_ps": 70.,
                      "ctr_improvement_ps": 130. - ctr, "rmse_improvement_ps": 20.},
            "bootstrap": {"ctr_bootstrap_std_ps": 2., "rmse_bootstrap_std_ps": 1.,
                          "ctr_improvement_bootstrap_std_ps": 3.,
                          "rmse_improvement_bootstrap_std_ps": 1.},
        })
    (tmp_path / "config.json").write_text(json.dumps({"protocol": {"seed": 1001}}))
    monkeypatch.setattr(engine, "load_plot_config", lambda _: plot_config())
    monkeypatch.setattr(engine, "collect_runs", lambda _: runs)
    monkeypatch.setattr(engine, "render_run_plots", lambda *args: None)
    monkeypatch.setattr(engine, "export_dataset_tables", lambda *args: [])
    monkeypatch.setattr(engine, "heatmap", lambda *args, **kwargs: None)
    monkeypatch.setattr(engine, "blind_ctr_bar", lambda *args, **kwargs: None)
    monkeypatch.setattr(engine, "_scatters", lambda *args: None)
    calls = []

    def paired(group, payloads, metric, config, seed):
        calls.append(metric)
        values = np.array([r["blind"]["ctr_ps"] for r in group])
        difference = values[:, None] - values[None, :]
        return ([r["manifest"]["model"] for r in group], difference,
                np.ones_like(difference), np.full(difference.shape, 32))

    monkeypatch.setattr(engine, "_paired", paired)
    return tmp_path, runs, calls


def test_sorted_matrices_cached_subset_and_removed_std(report_inputs):
    root, runs, calls = report_inputs
    output = engine.generate_report(root)
    tables = output / "tables/timing/short"
    expected = ["direct_mlp", "independent_cnn1d", "antisymmetric_mlp", "shared_cnn1d"]
    labels, before = engine._read_matrix(tables / "paired_ctr.csv")
    assert labels == expected
    np.testing.assert_array_equal(before, np.array([90, 95, 105, 110])[:, None]
                                  - np.array([90, 95, 105, 110])[None, :])
    assert engine._read_matrix(tables / "output_correlation.csv")[0] == expected
    assert calls == ["ctr", "rmse"]
    # Simulate outputs from an older reporting run, then filter the cached matrices.
    (tables / "paired_ctr_std.csv").write_text("obsolete")
    plots = output / "plots/timing/short"
    (plots / "paired_ctr_std.png").write_text("obsolete")
    engine.generate_report(root, reuse_numeric=True, exclude_models=["independent_cnn1d"])
    labels, subset = engine._read_matrix(tables / "paired_ctr.csv")
    assert labels == [expected[i] for i in [0, 2, 3]]
    np.testing.assert_array_equal(subset, before[np.ix_([0, 2, 3], [0, 2, 3])])
    assert calls == ["ctr", "rmse"]  # no new bootstrap for a cached subset
    assert not list(tables.glob("*_std.csv"))
    assert not list(plots.glob("*_std.*"))
    ranked = engine._csv(tables / "blind_ctr.csv")
    assert [r["code"] for r in ranked] == ["D-MLP", "S-MLP", "S-CNN"]
    assert "independent_cnn1d" not in (output / "tables/blind.csv").read_text()
    tex = (tables / "blind_ctr.tex").read_text()
    assert "Code & Model" in tex and "D-MLP" in tex and "130.00" in tex
    assert "D-CNN" not in tex
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["excluded_models"] == ["independent_cnn1d"]
    # Restoring a missing model requires recalculation rather than stale subsets.
    engine.generate_report(root, reuse_numeric=True)
    assert calls == ["ctr", "rmse", "ctr", "rmse"]


def test_invalid_and_empty_exclusions_fail_before_output(report_inputs):
    root, runs, _ = report_inputs
    with pytest.raises(ValueError, match="Unknown excluded models"):
        engine.generate_report(root, exclude_models=["typo"])
    with pytest.raises(ValueError, match="No complete models remain"):
        engine.generate_report(root, exclude_models=[r["manifest"]["model"] for r in runs])
    assert not (root / "report").exists()


@pytest.mark.parametrize("command", [["report", "results"], ["plots", "--results", "results"]])
def test_cli_exclusions(command):
    args = _parser().parse_args(command + ["--exclude-models", "direct_mlp", "shared_cnn1d"])
    assert args.exclude_models == ["direct_mlp", "shared_cnn1d"]


def test_ranked_ctr_bar_and_unannotated_scatter(tmp_path, monkeypatch):
    cfg = plot_config()
    rows = [
        {"code": "S-MLP", "display_name": "Shared MLP", "formulation": "shared",
         "ctr_ps": 90., "ctr_std_ps": 2.},
        {"code": "D-CNN", "display_name": "Independent CNN", "formulation": "direct",
         "ctr_ps": 100., "ctr_std_ps": 3.},
    ]
    captured = []

    def inspect(fig, path):
        captured.append(fig.axes[0])
        plt.close(fig)

    monkeypatch.setattr(plotting, "_save", inspect)
    plotting.blind_ctr_bar(rows, tmp_path / "ctr.png", cfg, led_ctr=130.)
    ax = captured.pop()
    assert [p.get_height() for p in ax.patches] == [90., 100.]
    assert [t.get_text() for t in ax.get_xticklabels()] == ["S-MLP", "D-CNN"]
    reference = next(line for line in ax.lines if line.get_label() == "LED reference")
    assert reference.get_linestyle() == "--"
    np.testing.assert_array_equal(reference.get_ydata(), [130., 130.])
    assert [t.get_text() for t in ax.get_legend().get_texts()][0] == "S-MLP: Shared MLP"
    plotting.scatter_with_labels([10, 20], [30, 40], ["A", "B"], tmp_path / "scatter.png",
                                 cfg, xlabel="x", ylabel="y", title="comparison",
                                 annotation="Pearson r = 1.000")
    ax = captured.pop()
    assert not ax.texts
    assert [t.get_text() for t in ax.get_legend().get_texts()] == ["A", "B"]
    assert "Pearson r" in ax.get_title()
