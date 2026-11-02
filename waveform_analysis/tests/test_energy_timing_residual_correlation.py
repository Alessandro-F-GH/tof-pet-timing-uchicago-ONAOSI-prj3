"""Tests for matched-event energy/timing residual comparison."""
import numpy as np
import pytest

from waveform_analysis.scripts.compare_energy_timing_residuals import (
    read_residuals, align_residuals, correlations,
)


def test_align_by_event_id_not_by_array_position():
    ids, energy, timing, excluded = align_residuals(
        np.array([30, 10, 20, 50]), np.array([3., 1., 2., 5.]),
        np.array([20, 30, 40, 10]), np.array([20., 30., 40., 10.]),
    )
    np.testing.assert_array_equal(ids, [10, 20, 30])
    np.testing.assert_array_equal(energy, [1., 2., 3.])
    np.testing.assert_array_equal(timing, [10., 20., 30.])
    assert excluded == 0
    assert correlations(energy, timing)["pearson_r"] == pytest.approx(1.)


def test_no_duplicate_ids(tmp_path):
    file = tmp_path / "pred.npz"
    np.savez(file, event_id=np.array([1, 1, 2]),
             corrected_ps=np.array([1., 2., 3.]))
    with pytest.raises(ValueError, match="duplicate"):
        read_residuals(file)


def test_intersection_too_small_is_rejected():
    with pytest.raises(ValueError, match="Only 1 matched"):
        align_residuals(
            np.array([1, 2, 3]), np.array([1., 2., 3.]),
            np.array([3, 4, 5]), np.array([3., 4., 5.]),
        )


def test_difference_statistics_use_paired_residuals():
    from waveform_analysis.scripts.compare_energy_timing_residuals import difference_statistics
    timing = np.array([-2., -1., 0., 1., 2.])
    energy = timing + np.array([3., -3., 3., -3., 0.])
    result = difference_statistics(energy, timing)
    assert result["difference_mean_ps"] == pytest.approx(0.)
    assert result["difference_std_ps"] == pytest.approx(np.std(energy-timing, ddof=1))
    assert result["difference_rmse_ps"] == pytest.approx(np.sqrt(np.mean((energy-timing)**2)))
