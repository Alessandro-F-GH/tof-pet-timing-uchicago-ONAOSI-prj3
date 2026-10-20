"""Tests for representative-event explanation selection and overwrite protection."""
import numpy as np
import pytest

from waveform_analysis.engine.representative_explainer import (
    _confirm_overwrite, _importance_scale, _importance_zoom_limits, _aggregate_signed_importance,
    _legend_location, _cached_representatives_are_reusable,
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



def test_signed_importance_is_summed_in_eight_sample_bins():
    time_ns = np.arange(19, dtype=float) * 0.0125
    relevance = np.zeros((2, 19), dtype=float)
    relevance[0, :8] = 1.0
    relevance[0, 8:16] = -0.5
    relevance[1, 16:] = 2.0
    centers, binned, widths = _aggregate_signed_importance(time_ns, relevance, bin_samples=8)
    assert binned.shape == (2, 3)
    np.testing.assert_allclose(binned[0], [8.0, -4.0, 0.0])
    np.testing.assert_allclose(binned[1], [0.0, 0.0, 6.0])
    assert widths[0] == pytest.approx(0.1)
    assert widths[-1] == pytest.approx(0.0375)
    assert len(centers) == 3


def test_aggregated_signed_importance_preserves_cancellation():
    time_ns = np.arange(8, dtype=float) * 0.0125
    relevance = np.array([[1., 1., 1., 1., -1., -1., -1., -1.]])
    _, binned, _ = _aggregate_signed_importance(time_ns, relevance, bin_samples=8)
    assert binned[0, 0] == pytest.approx(0.0)



def test_legend_location_depends_on_channel_and_onishi_window():
    assert _legend_location("energy_to_energy", False) == "upper left"
    assert _legend_location("timing_to_timing", False) == "upper right"
    assert _legend_location("timing_to_timing", True) == "upper left"


def test_existing_attributions_are_detected_for_plot_only_reuse(tmp_path):
    destination = tmp_path / "representatives"
    destination.mkdir()
    (destination / "representatives.csv").write_text(
        "group,population_index\nlow,1\ncentral,2\nhigh,3\n", encoding="utf-8"
    )
    for group in ("low", "central", "high"):
        np.savez_compressed(
            destination / f"{group}_attribution.npz",
            time_ps=np.arange(16, dtype=float),
            signed_contribution_ps=np.zeros((1, 16)),
            absolute_contribution_ps=np.zeros((1, 16)),
            event_id=np.asarray(group),
            prediction_ps=np.asarray(0.),
            target_ps=np.asarray(0.),
        )
    assert _cached_representatives_are_reusable(destination)
    (destination / "high_attribution.npz").unlink()
    assert not _cached_representatives_are_reusable(destination)
