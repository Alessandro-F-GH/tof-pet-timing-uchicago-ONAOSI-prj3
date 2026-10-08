"""CV axes show evaluation order while keeping metrics attached to candidates."""

import csv
import json
from pathlib import Path

import numpy as np
import pytest

from waveform_analysis.reporting import plotting
from waveform_analysis.engine import study
from waveform_analysis.engine.search import candidate_manifest, grid_candidates
from waveform_analysis.data.storage import RunStore


@pytest.mark.parametrize(
    "registry,expected",
    [
        (
            {
                "hash-a": {"trial_number": 9},
                "hash-b": {"trial_number": 1},
                "hash-c": {"trial_number": 4},
            },
            [(2, 20.0), (5, 30.0), (10, 10.0)],
        ),
        (
            {
                "hash-a": {"candidate_order": 3},
                "hash-b": {"candidate_order": 1},
                "hash-c": {"candidate_order": 2},
            },
            [(1, 20.0), (2, 30.0), (3, 10.0)],
        ),
        (None, [(1, 10.0), (2, 20.0), (3, 30.0)]),
        ({"hash-a": {"candidate_order": 5}}, [(5, 10.0), (6, 20.0), (7, 30.0)]),
    ],
)
def test_cv_plot_uses_sorted_candidate_numbers(
    tmp_path, monkeypatch, registry, expected
):
    (tmp_path / "tables").mkdir()
    (tmp_path / "metadata").mkdir()
    rows = [
        {
            "candidate_id": "hash-a",
            "ctr_mean_ps": 10.0,
            "ctr_std_ps": 1.0,
            "pruned": False,
        },
        {
            "candidate_id": "hash-b",
            "ctr_mean_ps": 20.0,
            "ctr_std_ps": 2.0,
            "pruned": False,
        },
        {
            "candidate_id": "hash-c",
            "ctr_mean_ps": 30.0,
            "ctr_std_ps": 3.0,
            "pruned": True,
        },
    ]
    with (tmp_path / "tables/cv.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows)
    (tmp_path / "metadata/manifest.json").write_text(
        json.dumps({"cv": {"metric": "ctr"}})
    )
    if registry is not None:
        (tmp_path / "metadata/candidates.json").write_text(json.dumps(registry))
    config = json.loads(
        (Path(__file__).parents[1] / "config/plots/default.json").read_text()
    )

    def inspect_and_save(figure, path):
        axis = figure.axes[0]
        assert axis.get_xlabel() == "Candidate order"
        assert [label.get_text() for label in axis.get_xticklabels()] == [
            str(n) for n, _ in expected
        ]
        np.testing.assert_array_equal(axis.get_xticks(), [n for n, _ in expected])
        line = axis.containers[0].lines[0]
        complete = [(n, value) for n, value in expected if value != 30.0]
        np.testing.assert_array_equal(line.get_xdata(), [n for n, _ in complete])
        np.testing.assert_array_equal(
            line.get_ydata(), [value for _, value in complete]
        )
        np.testing.assert_array_equal(
            axis.collections[-1].get_offsets(),
            [(n, value) for n, value in expected if value == 30.0],
        )
        return original_save(figure, path)

    original_save = plotting._save
    monkeypatch.setattr(plotting, "_save", inspect_and_save)
    before = (tmp_path / "tables/cv.csv").read_bytes()
    output = plotting.plot_run_cv(tmp_path, config)
    assert output.is_file()
    assert (tmp_path / "tables/cv.csv").read_bytes() == before


def test_grid_registry_retains_evaluation_order(tmp_path, monkeypatch):
    store = RunStore(tmp_path)
    space = {
        "optimization": {"strategy": "grid"},
        "parameters": {
            "ridge_alpha": {"type": "categorical", "choices": [10.0, 0.1, 1.0]}
        },
    }
    evaluated = []
    monkeypatch.setattr(
        study, "_evaluate_candidate", lambda *args, **kwargs: evaluated.append(args[6])
    )
    study._fixed_grid(store, None, space, {}, None, None, None)
    registry = json.loads(store.candidates_path.read_text())
    expected = list(candidate_manifest(grid_candidates(space)))
    assert evaluated == expected
    assert [registry[identifier]["candidate_order"] for identifier in expected] == [
        1,
        2,
        3,
    ]
