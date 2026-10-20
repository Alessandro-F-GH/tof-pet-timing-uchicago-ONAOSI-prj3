"""Tests for representative-event explanation selection and overwrite protection."""
import numpy as np

from waveform_analysis.engine.representative_explainer import (
    _confirm_overwrite, _importance_scale, _importance_zoom_limits,
)
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



def test_importance_display_scaling_preserves_sign_and_limits_outliers():
    relevance = np.array([-1., -2., 0., 1., 100.])
    scaled = _importance_scale(relevance)
    assert scaled[0] < 0
    assert scaled[2] == 0
    assert scaled[3] > 0
    assert np.max(np.abs(scaled)) <= 1.
    # Extreme samples are clipped for display, not mutated in saved arrays.
    assert scaled[-1] == 1.


def test_importance_zoom_finds_localized_region():
    time = np.arange(1000, dtype=float) * 0.0125
    relevance = np.zeros((1, 1000), dtype=float)
    relevance[0, 450:510] = 1.
    limits = _importance_zoom_limits(time, relevance)
    assert limits is not None
    assert limits[0] < time[450]
    assert limits[1] > time[509]
    assert limits[1] - limits[0] < 0.3 * (time[-1] - time[0])


def test_importance_zoom_skips_when_relevance_is_not_localized():
    time = np.arange(1000, dtype=float) * 0.0125
    relevance = np.ones((1, 1000), dtype=float)
    assert _importance_zoom_limits(time, relevance) is None
