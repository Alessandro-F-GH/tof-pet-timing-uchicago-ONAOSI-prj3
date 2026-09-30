from __future__ import annotations

import numpy as np

from waveform_analysis.ml_pipeline.report import (
    paired_model_comparisons,
    plot_led_improvements,
    study_summary,
)


def _row(*, study, dataset, population, sampling, model, mode, window, replica, ctr, led):
    start, end = window
    return {
        "study": study,
        "source_run": study,
        "dataset_key": dataset,
        "dataset": dataset,
        "population_identity": population,
        "sampling_identity": sampling,
        "model": model,
        "mode": mode,
        "window": f"{start:g}_{end:g}ns",
        "window_start_ns": float(start),
        "window_end_ns": float(end),
        "replica_index": int(replica),
        "seed": 1000 + int(replica),
        "ctr_ps": float(ctr),
        "led_ctr_ps": float(led),
        "improvement_ps": float(led - ctr),
        "improvement_percent": float(100.0 * (led - ctr) / led),
        "rmse_ps": float(ctr),
        "led_rmse_ps": float(led),
        "rmse_improvement_ps": float(led - ctr),
    }


def test_study_summary_never_pools_modes():
    rows = [
        _row(study="s", dataset="d", population="pe", sampling="se", model="m", mode="energy_to_energy", window=(-1.5, 2.0), replica=1, ctr=60, led=90),
        _row(study="s", dataset="d", population="pt", sampling="st", model="m", mode="timing_to_timing", window=(-1.5, 2.0), replica=1, ctr=70, led=100),
    ]
    summary = study_summary(rows)
    assert len(summary) == 2


def test_paired_model_comparison_requires_same_sampling_identity_and_replica():
    base = [
        _row(study="a", dataset="d", population="p", sampling="s", model="A", mode="energy_to_energy", window=(-1.5, 2.0), replica=1, ctr=60, led=90),
        _row(study="a", dataset="d", population="p", sampling="s", model="A", mode="energy_to_energy", window=(-1.5, 2.0), replica=2, ctr=61, led=91),
        _row(study="b", dataset="d", population="p", sampling="s", model="B", mode="energy_to_energy", window=(-1.5, 2.0), replica=1, ctr=55, led=90),
        _row(study="b", dataset="d", population="p", sampling="s", model="B", mode="energy_to_energy", window=(-1.5, 2.0), replica=2, ctr=56, led=91),
    ]
    comparison = paired_model_comparisons(base)
    assert len(comparison) == 1
    assert comparison[0]["n_paired_replicas"] == 2
    assert np.isclose(comparison[0]["ctr_difference_reference_minus_candidate_ps"], 5.0)

    changed = [dict(row) for row in base]
    for row in changed[2:]:
        row["sampling_identity"] = "other"
    assert paired_model_comparisons(changed) == []


def test_led_improvement_plot_is_separate_per_mode_and_window(tmp_path):
    rows = []
    for mode, population, sampling in (
        ("energy_to_energy", "pe", "se"),
        ("timing_to_timing", "pt", "st"),
    ):
        for window in ((-1.5, 2.0), (-2.0, 30.0)):
            for replica in (1, 2):
                rows.append(
                    _row(
                        study="s",
                        dataset="d",
                        population=population,
                        sampling=sampling,
                        model="m",
                        mode=mode,
                        window=window,
                        replica=replica,
                        ctr=60 + replica,
                        led=90 + replica,
                    )
                )
    outputs = plot_led_improvements(study_summary(rows), tmp_path)
    assert len(outputs) == 4
    assert all(path.is_file() for path in outputs)
