"""Tests for representative-event explanation selection and overwrite protection."""
import numpy as np

from waveform_analysis.engine.representative_explainer import _confirm_overwrite
from waveform_analysis.engine.stratified_explainer import INTERVALS, select_quantile_groups


def test_representatives_reuse_existing_prediction_rank_bands():
    predictions = np.arange(1000, dtype=float)
    event_ids = np.array([f"event-{i}" for i in range(len(predictions))])
    _, groups = select_quantile_groups(event_ids, predictions, INTERVALS)
    assert set(groups) == {"low", "central", "high"}
    for name, (lo, hi) in INTERVALS.items():
        idx = groups[name]["representative_index"]
        assert lo <= idx / len(predictions) < hi


def test_existing_output_requires_explicit_confirmation(tmp_path, monkeypatch):
    destination = tmp_path / "representatives"
    destination.mkdir()
    monkeypatch.setattr("builtins.input", lambda prompt: "n")
    assert not _confirm_overwrite(destination, yes=False)
    monkeypatch.setattr("builtins.input", lambda prompt: "yes")
    assert _confirm_overwrite(destination, yes=False)
    assert _confirm_overwrite(destination, yes=True)


def test_missing_output_does_not_prompt(tmp_path, monkeypatch):
    monkeypatch.setattr("builtins.input", lambda prompt: (_ for _ in ()).throw(AssertionError()))
    assert _confirm_overwrite(tmp_path / "missing", yes=False)
