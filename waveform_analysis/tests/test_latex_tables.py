"""Population exports must reflect prepared events, without refitting or guessing."""

from copy import deepcopy
import csv
import json
from types import SimpleNamespace

import numpy as np
import pytest

from waveform_analysis.data import preparation
from waveform_analysis.reporting.latex_tables import (
    dataset_rows,
    escape_latex,
    export_dataset_tables,
    population_metadata,
)


def run(tmp_path, *, mode="timing_to_timing", window="short", counts=(10, 20, 30)):
    manifest = {
        "mode": mode,
        "window_name": window,
        "window_ns": {"start": -1.0, "end": 2.0},
        "control_artifact": str(tmp_path / "control"),
        "dataset_populations": {},
    }
    for role, count in zip(("control", "development", "blind"), counts):
        source = str(tmp_path / f"{role}_48V.root")
        manifest[f"{role}_dataset"] = {"root_file": source}
        manifest["dataset_populations"][role] = {
            "source": source,
            "n_selected": count,
            "bias_voltage_V": [48.0],
            "population_identity": f"{role}-{mode}-{window}",
        }
    return {"directory": tmp_path / "run", "manifest": manifest}


@pytest.mark.parametrize("board", ["UC", "FBK"])
def test_board_table_deduplicates_models_and_keeps_modes_windows(board, tmp_path):
    first = run(tmp_path)
    runs = [first, deepcopy(first), run(tmp_path, mode="energy_to_energy")]
    wide = run(tmp_path, window="wide", counts=(0, 17, 25))
    wide["manifest"]["window_ns"] = {"start": -2.0, "end": 30.0}
    runs.append(wide)
    saved = deepcopy(runs)
    paths = export_dataset_tables(
        tmp_path, runs, {"results": {"folder": f"{board}/benchmark"}}
    )
    assert runs == saved
    assert len(paths) == 1
    text = paths[0].read_text()
    assert f"{board} selected waveform populations" in text
    assert text.count("Train (development) &") == 3
    assert text.count("Control &") == 3
    assert r"\multirow[t]{6}{*}{Timing}" in text
    assert r"\multirow[t]{3}{*}{wide (-2 to 30)}" in text
    assert r"Control & control\_48V.root & \multirow{3}{*}{48} & 0" in text
    assert text.count(r"\hdashline") == 1
    assert text.count(r"\cdashline{2-6}") == 1
    assert not (
        paths[0].parent / f"{'FBK' if board == 'UC' else 'UC'}_selected_events.tex"
    ).exists()
    with (paths[0].parent / "selected_events.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 9
    assert {row["board"] for row in rows} == {board}
    assert sum(int(row["n_selected"]) for row in rows if row["role"] == "blind") == 85


def test_conflicting_populations_cannot_be_silently_collapsed(tmp_path):
    first = run(tmp_path)
    second = deepcopy(first)
    second["manifest"]["dataset_populations"]["blind"]["population_identity"] = (
        "different-events"
    )
    with pytest.raises(ValueError, match="Inconsistent prepared populations"):
        dataset_rows([first, second], "UC")


def test_missing_metadata_is_a_dash_not_zero_and_unknown_board_stays_csv(tmp_path):
    item = run(tmp_path)
    item["manifest"].pop("dataset_populations")
    path = export_dataset_tables(tmp_path, [item], {"results": {"folder": "UC/study"}})[
        0
    ]
    assert (
        r"Control & control\_48V.root & \multirow{3}{*}{---} & ---" in path.read_text()
    )
    other = tmp_path / "undeclared"
    assert export_dataset_tables(other, [item], {"results": {"folder": "study"}}) == []
    assert (other / "report/tables/datasets/selected_events.csv").exists()
    assert not list(other.rglob("*.tex"))


def test_old_run_backfills_only_matching_prepared_cache_without_mutation(tmp_path):
    item = run(tmp_path)
    item["manifest"].pop("dataset_populations")
    config = {
        "mode": "timing_to_timing",
        "window_ns": {"start": -1.0, "end": 2.0},
        "ml_input": {"subsampling": 1},
        "preprocessing": {"cache_dir": str(tmp_path / "cache")},
        **{
            role: item["manifest"][f"{role}_dataset"]
            for role in ("control", "development", "blind")
        },
    }
    directory = item["directory"] / "metadata"
    directory.mkdir(parents=True)
    (directory / "config.json").write_text(json.dumps(config))
    control = tmp_path / "control"
    control.mkdir()
    (control / "manifest.json").write_text(
        json.dumps({"fingerprint": "frozen-control"})
    )
    for role in ("control", "development", "blind"):
        path = tmp_path / "cache" / f"{role}_ml/prepared/candidate"
        path.mkdir(parents=True)
        (path / "manifest.json").write_text(
            json.dumps(
                {
                    "dataset_role": role,
                    "mode": config["mode"],
                    "window_ns": config["window_ns"],
                    "subsampling": 1,
                    "dataset_source": config[role]["root_file"],
                    "control_fingerprint": "frozen-control",
                    "n_final": 7,
                    "event_population_identity": role,
                }
            )
        )
        np.save(path / "bias_voltage_V.npy", np.full(7, 48.0))
    original = {
        path: path.read_bytes() for path in tmp_path.rglob("*") if path.is_file()
    }
    rows = dataset_rows([item], "UC")
    assert [row["n_selected"] for row in rows] == [7, 7, 7]
    assert all(path.read_bytes() == content for path, content in original.items())
    config["window_ns"]["end"] = 30
    (directory / "config.json").write_text(json.dumps(config))
    assert all(row["n_selected"] is None for row in dataset_rows([item], "UC"))


def test_plain_text_is_escaped_and_population_snapshot_retains_zero_count():
    assert escape_latex("run_a&b%{c}\\") == r"run\_a\&b\%\{c\}\textbackslash{}"
    dataset = SimpleNamespace(manifest={}, n_events=0, bias_voltage_V=np.array([]))
    assert population_metadata(dataset, "control.root")["n_selected"] == 0


def test_empty_control_bookkeeping_does_not_relax_training_requirements(
    tmp_path, monkeypatch
):
    native = SimpleNamespace(
        n_events=2,
        manifest={"fingerprint": "native", "source": "control.root"},
        energy_windows_mV=np.zeros((2, 2, 6)),
        energy_sample_interval_s=np.full((2, 2), 1e-9),
        event_index=np.arange(2),
        bias_voltage_V=np.full(2, 48.0),
    )
    config = {
        "mode": "energy_to_energy",
        "window_ns": {"start": -1, "end": 1},
        "ml_input": {"subsampling": 1},
        "control": {"root_file": str(tmp_path / "control.root"), "true_tof_ps": 0},
        "preprocessing": {
            "selection": {"baseline_window_ns": [-2, -1]},
            "led_selection": {"coincidence_window_ns": 2},
            "energy": {"vertical_scale_limit_mV": [[-1, 1], [-1, 1]]},
        },
    }
    artifact = {
        "fingerprint": "control",
        "selected_led_threshold_mV": {config["mode"]: 5},
    }
    monkeypatch.setattr(preparation, "led_grid", lambda *a, **kw: np.zeros((2, 2, 1)))
    # Every pulse lacks the requested pre-crossing part of the window.
    monkeypatch.setattr(
        preparation, "anchor_grid", lambda *a, **kw: np.zeros((2, 2), dtype=int)
    )
    dataset = preparation.prepare_ml_dataset(
        native,
        artifact,
        config,
        dataset_role="control",
        cache_dir=tmp_path,
        allow_empty=True,
    )
    assert dataset.n_events == 0
    assert dataset.manifest["n_dropped_window"] == 2
    with pytest.raises(RuntimeError, match="No events remain"):
        preparation.prepare_ml_dataset(
            native, artifact, config, dataset_role="control", cache_dir=tmp_path
        )
