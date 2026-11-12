"""Tests for using the original, cached wide ML input."""
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from waveform_analysis.scripts.test_tail_phase_ridge import _align, _cached_wide, _features


def test_join_uses_event_identity_not_position():
    rows = _align([101, 103, 102, 105], [105, 101, 102], role="development")
    np.testing.assert_array_equal(rows, [3, 0, 2])


def test_join_rejects_missing_event():
    with pytest.raises(ValueError, match="missing"):
        _align([1, 3], [1, 2], role="blind")


def test_join_rejects_duplicate_event():
    with pytest.raises(ValueError, match="duplicate"):
        _align([1, 1], [1], role="blind")


def test_cached_wide_phase_with_nonmatching_event_order(tmp_path):
    base = tmp_path / "artifact_store/products/prepared-datasets/identity"
    base.mkdir(parents=True)
    (base / "manifest.json").write_text(json.dumps({
        "dataset_role": "development",
        "dataset_source": "/original/48V-R1.root",
        "mode": "energy_to_energy",
        "window_ns": {"start": -2.0, "end": 50.0},
        "subsampling": 1,
    }))
    dt = .0125
    time = np.arange(-2., 50. + .0001, dt)
    phase = .65
    x = np.zeros((3, 2, len(time)), np.float32)
    for row in range(3):
        x[row, 0] = .5 + .08 * np.cos(2 * np.pi * 2.5 * time)
        x[row, 1] = .5 + .08 * np.cos(2 * np.pi * 2.5 * time + phase + .2 * row)
    np.save(base / "energy_windows.npy", x)
    np.save(base / "event_index.npy", np.array([18, 23, 19]))
    np.save(base / "energy_time_ps.npy", time * 1000.)
    np.savez(base / "energy_transform.npz",
             minimum=np.zeros((2, 1)), maximum=np.ones((2, 1)))
    args = SimpleNamespace(frequency_ghz=2.5, tail_start_ns=5.,
                           tail_end_ns=30., batch_size=2)
    coeff, path = _cached_wide(
        tmp_path, "development", np.array([19, 18]), "/same/48V-R1.root", args
    )
    assert Path(path) == base
    assert coeff.shape == (2, 2)
    features = _features(coeff, "relative")
    np.testing.assert_allclose(
        features[:, 0], np.cos([phase + .4, phase]), atol=.01
    )
    np.testing.assert_allclose(
        features[:, 1], -np.sin([phase + .4, phase]), atol=.01
    )
