"""Tests for saved-prediction quantile-based waveform examples."""
import numpy as np
import pytest

from waveform_analysis.scripts.correction_examples import (
    align_prediction_events, select_examples, validate_quantiles, read_predictions,
    plot_example_waveforms, plot_prediction_distribution,
)


def test_quantiles_select_actual_events_and_never_generate_waveforms():
    predictions = np.array([-100., -60., -20., 0., 15., 40., 90.])
    ids = np.array([21, 22, 23, 24, 25, 26, 27])
    rows = select_examples(predictions, ids, [.2, .5, .8])
    assert [row["event_id"] for row in rows] == [22, 24, 26]
    assert [row["prediction_ps"] for row in rows] == [-60., 0., 40.]
    assert all(row["prediction_ps"] in predictions for row in rows)


def test_tie_break_independent_of_row_order():
    preds = np.array([-2., 2., 0.])
    ids = np.array([20, 10, 30])
    chosen = select_examples(preds, ids, [.5])
    assert chosen[0]["event_id"] == 30
    ids2 = np.array([30, 20, 10])
    preds2 = np.array([0., -2., 2.])
    assert select_examples(preds2, ids2, [.5])[0]["event_id"] == 30


def test_duplicate_selection_and_invalid_quantiles():
    with pytest.raises(ValueError, match="same event"):
        select_examples([1., 1., 1.], [10, 11, 12], [.1, .9])
    for q in ([0], [1], [.4, .2], [.1, .1], [float("nan")]):
        with pytest.raises(ValueError):
            validate_quantiles(q)


def test_event_alignment_checks_identifiers_not_row_positions():
    prepared = np.array([4, 7, 9])
    predictions = np.array([9, 4, 7])
    assert align_prediction_events(prepared, predictions, [10., 20., 30.]).tolist() == [2, 0, 1]
    with pytest.raises(ValueError, match="differ"):
        align_prediction_events(prepared, [4, 5, 9], [1., 2., 3.])
    with pytest.raises(ValueError, match="unique"):
        align_prediction_events(prepared, [4, 4, 9], [1., 2., 3.])


def test_saved_prediction_loading_and_metadata(tmp_path):
    path = tmp_path / "pred.npz"
    np.savez(path, event_id=np.array([1, 2]), prediction_ps=np.array([-4., 5.]),
             analysis_protocol_identity=np.array("protocol-a"),
             dataset_role=np.array("blind"), final_fit_fingerprint=np.array("fit-a"))
    ids, values = read_predictions(path, role="blind", protocol_identity="protocol-a",
                                   final_fit_fingerprint="fit-a")
    np.testing.assert_array_equal(ids, [1, 2])
    np.testing.assert_array_equal(values, [-4., 5.])
    with pytest.raises(ValueError, match="mismatch"):
        read_predictions(path, role="development", protocol_identity="protocol-a")


def test_waveform_and_histogram_figures(tmp_path):
    matplotlib = pytest.importorskip("matplotlib")
    rows = select_examples([-8., -2., 1., 4., 10.], [11, 12, 13, 14, 15], [.2, .5, .8])
    waves = np.arange(3 * 2 * 12, dtype=np.float32).reshape(3, 2, 12) / 50
    plot_example_waveforms(waves, np.arange(12) * 50, rows, tmp_path / "waveforms.png")
    plot_prediction_distribution(np.array([-8., -2., 1., 4., 10.]), rows, tmp_path / "distribution.png")
    assert (tmp_path / "waveforms.png").stat().st_size > 0
    assert (tmp_path / "distribution.png").stat().st_size > 0
