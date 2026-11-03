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


def test_unit_slope_exact_common_component():
    from waveform_analysis.scripts.compare_energy_timing_residuals import test_unit_slope
    timing = np.linspace(-5, 5, 101)
    energy = timing + 3.0
    result = test_unit_slope(energy, timing)
    assert result["beta_energy_on_timing"] == pytest.approx(1.0)
    assert result["beta_test_p_value_two_sided"] == pytest.approx(1.0)
    assert result["beta_reject_h0_at_0_05"] is False


def test_unit_slope_rejects_nonunit_slope():
    from waveform_analysis.scripts.compare_energy_timing_residuals import test_unit_slope
    timing = np.linspace(-5, 5, 101)
    energy = 2.0 * timing + 1.0 + 0.01 * np.sin(np.arange(101))
    result = test_unit_slope(energy, timing)
    assert result["beta_energy_on_timing"] == pytest.approx(2.0, abs=0.01)
    assert result["beta_test_p_value_two_sided"] < 0.05
    assert result["beta_reject_h0_at_0_05"] is True


def test_trimmed_slope_quantile_intersection_and_counts():
    from waveform_analysis.scripts.compare_energy_timing_residuals import trimmed_slope_statistics
    timing = np.arange(1001, dtype=float)
    energy = 2.0 * timing + 10.0
    stats = trimmed_slope_statistics(energy, timing)
    assert [row["central_percent_per_channel"] for row in stats] == [99.9, 99.0, 95.0]
    assert [row["n_retained"] for row in stats] == [999, 991, 951]
    for row in stats:
        assert row["beta_energy_on_timing"] == pytest.approx(2.0)
        assert row["retained_fraction"] == pytest.approx(row["n_retained"] / 1001)


def test_trimmed_slope_requires_both_channels_to_pass():
    from waveform_analysis.scripts.compare_energy_timing_residuals import trimmed_slope_statistics
    timing = np.arange(1001, dtype=float)
    energy = timing.copy()
    energy[500], energy[0] = energy[0], energy[500]
    trimmed = trimmed_slope_statistics(energy, timing, retained_percentages=(95.0,))[0]
    # One point in the center of timing is excluded because energy is in its tail.
    assert trimmed["n_retained"] < 951
