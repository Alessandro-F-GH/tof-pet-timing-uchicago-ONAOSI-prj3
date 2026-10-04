from __future__ import annotations

import numpy as np

from waveform_analysis.ml_pipeline.report import (
    _prepare_report_layout,
    best_by_formulation,
    paired_model_comparisons,
    pareto_frontier,
    plot_best_models_by_mode,
    plot_ctr_vs_time,
    plot_led_improvements,
    plot_rmse_comparisons,
    plot_rmse_ctr_correlation,
    plot_window_model_comparisons,
    replica_wall_times,
    study_summary,
)
from waveform_analysis.ml_pipeline.reporting_config import load_reporting_config


def _row(
    *,
    study,
    dataset,
    population,
    sampling,
    model,
    formulation,
    mode,
    window,
    replica,
    ctr,
    led,
    wall_time=1.0,
):
    start, end = window
    return {
        "study": study,
        "source_run": study,
        "dataset_key": dataset,
        "dataset": dataset,
        "population_identity": population,
        "sampling_identity": sampling,
        "model": model,
        "estimator_formulation": formulation,
        "mode": mode,
        "window": f"{start:g}_{end:g}ns",
        "window_start_ns": float(start),
        "window_end_ns": float(end),
        "replica_index": int(replica),
        "seed": 1000 + int(replica),
        "candidate_id": "candidate",
        "shared_replica": "",
        "ctr_ps": float(ctr),
        "led_ctr_ps": float(led),
        "improvement_ps": float(led - ctr),
        "improvement_percent": float(100.0 * (led - ctr) / led),
        "rmse_ps": float(ctr + 10.0),
        "led_rmse_ps": float(led + 10.0),
        "rmse_improvement_ps": float(led - ctr),
        "rmse_improvement_percent": float(100.0 * (led - ctr) / (led + 10.0)),
        "replica_wall_time_s": float(wall_time),
    }


def test_study_summary_never_pools_modes_or_formulations():
    rows = [
        _row(study="s", dataset="d", population="pe", sampling="se", model="m1", formulation="shared", mode="energy_to_energy", window=(-1.5, 2.0), replica=1, ctr=60, led=90),
        _row(study="s", dataset="d", population="pt", sampling="st", model="m1", formulation="shared", mode="timing_to_timing", window=(-1.5, 2.0), replica=1, ctr=70, led=100),
        _row(study="s", dataset="d", population="pe", sampling="se", model="m2", formulation="direct", mode="energy_to_energy", window=(-1.5, 2.0), replica=1, ctr=62, led=90),
    ]
    summary = study_summary(rows)
    assert len(summary) == 3
    assert {row["estimator_formulation"] for row in summary} == {"shared", "direct"}


def test_paired_model_comparison_requires_same_context_sampling_and_replica():
    base = [
        _row(study="s", dataset="d", population="p", sampling="s", model="A", formulation="shared", mode="energy_to_energy", window=(-1.5, 2.0), replica=1, ctr=60, led=90),
        _row(study="s", dataset="d", population="p", sampling="s", model="A", formulation="shared", mode="energy_to_energy", window=(-1.5, 2.0), replica=2, ctr=61, led=91),
        _row(study="s", dataset="d", population="p", sampling="s", model="B", formulation="direct", mode="energy_to_energy", window=(-1.5, 2.0), replica=1, ctr=55, led=90),
        _row(study="s", dataset="d", population="p", sampling="s", model="B", formulation="direct", mode="energy_to_energy", window=(-1.5, 2.0), replica=2, ctr=56, led=91),
    ]
    comparison = paired_model_comparisons(base)
    assert len(comparison) == 1
    assert comparison[0]["n_paired_replicas"] == 2
    assert np.isclose(comparison[0]["ctr_difference_reference_minus_candidate_ps"], 5.0)
    assert comparison[0]["reference_formulation"] == "shared"
    assert comparison[0]["candidate_formulation"] == "direct"

    changed = [dict(row) for row in base]
    for row in changed[2:]:
        row["sampling_identity"] = "other"
    assert paired_model_comparisons(changed) == []


def test_reporting_config_supports_partial_override():
    config = load_reporting_config({"global": {"font_size": 12}})
    assert config["global"]["font_size"] == 12
    assert "shared" in config["formulations"]
    assert "correlation_heatmap" in config["plots"]


def test_replica_wall_time_parser(tmp_path):
    (tmp_path / "study.log").write_text(
        "2026-10-02 10:00:00,000 | INFO | Replica 1/2 | seed=1\n"
        "2026-10-02 10:00:03,250 | INFO | Replica result | replica=1 | CTR=60.0 ps\n"
        "2026-10-02 10:00:04,000 | INFO | Replica 2/2 | seed=2\n"
        "2026-10-02 10:00:09,500 | INFO | Replica result | replica=2 | CTR=61.0 ps\n",
        encoding="utf-8",
    )
    assert replica_wall_times(tmp_path) == {1: 3.25, 2: 5.5}


def test_pareto_frontier_minimizes_ctr_and_time():
    points = [
        {"time": 1.0, "ctr": 65.0},
        {"time": 2.0, "ctr": 60.0},
        {"time": 3.0, "ctr": 66.0},
        {"time": 4.0, "ctr": 58.0},
    ]
    assert pareto_frontier(points, x_key="time", y_key="ctr") == [0, 1, 3]


def test_best_by_formulation_uses_lowest_mean_ctr():
    rows = []
    for model, formulation, base_ctr in (
        ("shared_a", "shared", 60.0),
        ("shared_b", "shared", 58.0),
        ("direct_a", "direct", 57.0),
        ("direct_b", "direct", 59.0),
    ):
        for replica in (1, 2):
            rows.append(
                _row(
                    study="s",
                    dataset="d",
                    population="p",
                    sampling="q",
                    model=model,
                    formulation=formulation,
                    mode="energy_to_energy",
                    window=(-1.5, 2.0),
                    replica=replica,
                    ctr=base_ctr + replica * 0.1,
                    led=90,
                    wall_time=replica + base_ctr,
                )
            )
    best = best_by_formulation(study_summary(rows))
    assert len(best) == 1
    assert best[0]["best_shared_model"] == "shared_b"
    assert best[0]["best_direct_model"] == "direct_a"
    assert best[0]["direct_minus_shared_ctr_ps"] < 0


def test_report_layout_uses_compact_hierarchy(tmp_path):
    layout = _prepare_report_layout(tmp_path / "report")
    root = layout["root"]
    assert layout["rmse_ctr_plots"] == root / "plots" / "rmse_vs_ctr"
    assert layout["ctr_time_plots"] == root / "plots" / "ctr_vs_time"
    assert layout["window_plots"] == root / "plots" / "window_comparison"
    assert layout["best_model_plots"] == root / "plots" / "best_model"
    assert not (root / "plots" / "tradeoffs").exists()
    assert not (root / "plots" / "architecture").exists()


def test_cross_window_plots_group_window_specific_identities(tmp_path):
    rows = []
    models = (("shared_model", "shared"), ("direct_model", "direct"))
    for mode, base_population, base_sampling in (
        ("energy_to_energy", "pe", "se"),
        ("timing_to_timing", "pt", "st"),
    ):
        for window_index, window in enumerate(((-1.5, 2.0), (-2.0, 30.0))):
            population = f"{base_population}_window_{window_index}"
            sampling = f"{base_sampling}_window_{window_index}"
            for model_index, (model, formulation) in enumerate(models):
                for replica in (1, 2):
                    rows.append(
                        _row(
                            study="s",
                            dataset="d",
                            population=population,
                            sampling=sampling,
                            model=model,
                            formulation=formulation,
                            mode=mode,
                            window=window,
                            replica=replica,
                            ctr=60 + replica + model_index - 2 * window_index,
                            led=90 + replica,
                            wall_time=5 + 10 * model_index + replica,
                        )
                    )

    summary = study_summary(rows)
    assert all("led_rmse_mean_ps" in row for row in summary)
    assert all("replica_wall_time_mean_s" in row for row in summary)

    led_outputs = plot_led_improvements(summary, tmp_path / "led")
    rmse_outputs = plot_rmse_comparisons(summary, tmp_path / "rmse")
    scatter_outputs = plot_rmse_ctr_correlation(summary, tmp_path / "rmse_ctr")
    time_outputs = plot_ctr_vs_time(summary, tmp_path / "ctr_time")
    window_outputs = plot_window_model_comparisons(summary, tmp_path / "window", metric="ctr")
    window_rmse_outputs = plot_window_model_comparisons(summary, tmp_path / "window", metric="rmse")
    best_outputs = plot_best_models_by_mode(summary, tmp_path / "best")

    assert len(led_outputs) == 4
    assert len(rmse_outputs) == 4
    assert len(scatter_outputs) == 4
    assert len(time_outputs) == 4
    assert len(window_outputs) == 2
    assert len(window_rmse_outputs) == 2
    assert len(best_outputs) == 2
    assert {path.name for path in best_outputs} == {"energy_to_energy.png", "timing_to_timing.png"}
    assert {path.name for path in window_outputs} == {"ctr.png"}
    assert {path.name for path in window_rmse_outputs} == {"rmse.png"}
    assert all(
        path.is_file()
        for path in (
            led_outputs
            + rmse_outputs
            + scatter_outputs
            + time_outputs
            + window_outputs
            + window_rmse_outputs
            + best_outputs
        )
    )
