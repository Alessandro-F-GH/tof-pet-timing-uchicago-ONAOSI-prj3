from __future__ import annotations

import numpy as np

from waveform_analysis.ml_pipeline.report import paired_model_comparisons, plot_led_improvements, study_summary


def _row(*, study, dataset, population, model, mode, window, seed, ctr, led):
    start, end = window
    return {
        "study": study, "source_run": study, "dataset_key": dataset, "dataset": dataset,
        "population_identity": population, "model": model, "mode": mode,
        "window": f"{start:g}_{end:g}ns", "window_start_ns": float(start), "window_end_ns": float(end),
        "seed": int(seed), "ctr_ps": float(ctr), "led_ctr_ps": float(led),
        "improvement_ps": float(led - ctr), "improvement_percent": float(100.0 * (led - ctr) / led),
        "rmse_ps": float(ctr), "led_rmse_ps": float(led), "rmse_improvement_ps": float(led - ctr),
    }


def test_study_summary_never_pools_modes():
    rows = [
        _row(study="s", dataset="d", population="pe", model="m", mode="energy_to_energy", window=(-1.5, 2.0), seed=1, ctr=60, led=90),
        _row(study="s", dataset="d", population="pt", model="m", mode="timing_to_timing", window=(-1.5, 2.0), seed=1, ctr=70, led=100),
    ]
    summary = study_summary(rows)
    assert len(summary) == 2
    assert {r["mode"] for r in summary} == {"energy_to_energy", "timing_to_timing"}


def test_paired_model_comparison_requires_same_dataset_mode_window_population_and_seed():
    base = [
        _row(study="a", dataset="d", population="p", model="A", mode="energy_to_energy", window=(-1.5, 2.0), seed=1, ctr=60, led=90),
        _row(study="a", dataset="d", population="p", model="A", mode="energy_to_energy", window=(-1.5, 2.0), seed=2, ctr=61, led=91),
        _row(study="b", dataset="d", population="p", model="B", mode="energy_to_energy", window=(-1.5, 2.0), seed=1, ctr=55, led=90),
        _row(study="b", dataset="d", population="p", model="B", mode="energy_to_energy", window=(-1.5, 2.0), seed=2, ctr=56, led=91),
    ]
    comparison = paired_model_comparisons(base)
    assert len(comparison) == 1
    assert comparison[0]["n_paired_replicas"] == 2
    assert np.isclose(comparison[0]["ctr_difference_reference_minus_candidate_ps"], 5.0)

    changed_mode = [dict(r) for r in base]
    for row in changed_mode[2:]: row["mode"] = "timing_to_timing"
    assert paired_model_comparisons(changed_mode) == []

    changed_window = [dict(r) for r in base]
    for row in changed_window[2:]:
        row["window"] = "-2_30ns"; row["window_start_ns"] = -2.0; row["window_end_ns"] = 30.0
    assert paired_model_comparisons(changed_window) == []

    changed_dataset = [dict(r) for r in base]
    for row in changed_dataset[2:]: row["dataset_key"] = "other"; row["dataset"] = "other"
    assert paired_model_comparisons(changed_dataset) == []

    changed_population = [dict(r) for r in base]
    for row in changed_population[2:]: row["population_identity"] = "other"
    assert paired_model_comparisons(changed_population) == []


def test_led_improvement_plot_is_separate_per_mode_and_window(tmp_path):
    rows = []
    for mode, population in (("energy_to_energy", "pe"), ("timing_to_timing", "pt")):
        for window in ((-1.5, 2.0), (-2.0, 30.0)):
            for seed in (1, 2):
                rows.append(_row(study="s", dataset="d", population=population, model="m", mode=mode, window=window, seed=seed, ctr=60 + seed, led=90 + seed))
    outputs = plot_led_improvements(study_summary(rows), tmp_path)
    assert len(outputs) == 4
    assert all(path.is_file() for path in outputs)
