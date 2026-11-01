"""Synthetic regression tests for deterministic mean-conditioned AR imputation."""
import numpy as np

from waveform_analysis.scripts.evaluate_mean_ar import (
    active_interval, choose_indices, dense_cutoffs, distributed_cutoffs,
    local_forecast_end, valid_intervals, fit_moments,
    fit_ridge_orders, forecast_suffix, damped_ar_forecast, metric_row,
)


def test_forecast_only_uses_prefix_and_clips():
    mean = np.full(25, .5)
    std = np.ones(25)
    prefix = np.full((2, 10), .9)
    predicted = forecast_suffix(prefix, mean, std, np.array([2.0]), 10, 25)
    assert predicted.shape == (2, 15)
    assert np.isfinite(predicted).all()
    assert np.all((predicted >= 0) & (predicted <= 1))
    assert predicted[0, 0] == 1.0


def test_ar_improves_short_horizon_over_mean_for_persistent_residuals():
    rng = np.random.default_rng(15)
    time = np.arange(220)
    mean = .1 + .65 * np.exp(-.5 * ((time - 85) / 17) ** 2)
    scale = .03 + .08 * np.exp(-.5 * ((time - 85) / 25) ** 2)
    residual = np.empty((120, len(time)))
    residual[:, 0] = rng.normal(size=120)
    for t in range(1, len(time)):
        residual[:, t] = .96 * residual[:, t-1] + rng.normal(scale=.1, size=120)
    signal = np.clip(mean + scale * residual, 0, 1)
    training, testing = signal[:100], signal[100:]
    mu, sigma, _ = fit_moments(training)
    fitted = fit_ridge_orders(training, mu, sigma, (0, 220), [4, 8],
                              stride=2, batch_events=12)
    cutoff = 105
    truth = testing[:, cutoff:]
    prediction = forecast_suffix(testing[:, :cutoff], mu, sigma,
        fitted["standardized_ar"][8]["coefficients"], cutoff, 220, True)
    baseline = np.broadcast_to(mu[cutoff:], truth.shape)
    assert np.mean((prediction[:, :20] - truth[:, :20])**2) < np.mean(
        (baseline[:, :20] - truth[:, :20])**2)


def test_dense_cutoffs_concentrate_near_training_mean_rising_edge():
    time = np.arange(400)
    mean = .1 + .7/(1+np.exp(-(time-145)/4)) - .7/(1+np.exp(-(time-210)/20))
    cutoffs = dense_cutoffs(mean, (35, 390), 32, 10)
    assert len(cutoffs) == 10
    assert np.all(np.diff(cutoffs) > 0)
    assert cutoffs.min() >= 67 and cutoffs.max() < 390
    assert sum(abs(cutoffs-145) < 75) >= 6


def test_cropped_intervals_are_never_crossed():
    mask = np.zeros(270, dtype=bool)
    mask[4:92], mask[111:255] = True, True
    mean = np.zeros(270)
    mean[148:156] = np.arange(8) * .1
    assert active_interval(mask, mean, 16) == (111, 255)


def test_physical_unit_error_scaling():
    metrics = metric_row(np.zeros((2, 10)), np.ones((2, 10)) * .5, 160.)
    assert metrics["mse_normalized"] == .25
    assert metrics["rmse_mV"] == 80.


def test_event_selection_is_reproducible():
    assert np.array_equal(choose_indices(100, 15, 123),
                          choose_indices(100, 15, 123))
    assert len(choose_indices(100, 0, 123)) == 100


def test_cropped_regions_use_mean_after_gap_and_local_history_after_gap():
    mask = np.zeros(430, dtype=bool)
    mask[0:175] = True
    mask[200:420] = True
    mean = .05 + .7/(1+np.exp(-(np.arange(430)-147)/7))
    intervals = valid_intervals(mask, 32)
    assert intervals == [(0, 175), (200, 420)]
    assert local_forecast_end(mask, 100) == 175
    assert local_forecast_end(mask, 240) == 420
    cutoffs = distributed_cutoffs(mean, mask, 32, 10)
    assert len(cutoffs) == 10
    assert np.any(cutoffs < 175)
    assert np.any(cutoffs >= 232)
    assert np.all(mask[cutoffs])


def test_decay_is_postprocessing_and_tends_to_mean():
    mean = np.full(31, .2)
    time_ps = np.arange(31) * 100.
    cutoff = 5
    raw = np.full((3, 20), .8)
    same = damped_ar_forecast(raw, mean, time_ps, cutoff, None)
    np.testing.assert_allclose(same, raw)
    faded = damped_ar_forecast(raw, mean, time_ps, cutoff, .5)
    np.testing.assert_allclose(faded[:, 0], raw[:, 0])
    # 0.5 ns = five 100-ps samples, thus one half-life at index five.
    np.testing.assert_allclose(faded[:, 5], .5)
    assert np.all(np.diff(faded[0]) < 0)
    # The input AR result cannot be modified by fade-in-place feedback.
    np.testing.assert_allclose(raw, .8)


def test_decay_rejects_invalid_half_life_and_time_grid():
    import pytest
    mean = np.full(12, .4)
    raw = np.full((2, 4), .7)
    with pytest.raises(ValueError, match="positive"):
        damped_ar_forecast(raw, mean, np.arange(12)*100., 3, -1)
    with pytest.raises(ValueError, match="increasing"):
        damped_ar_forecast(raw, mean, np.zeros(12), 3, 1.)


def test_waveform_example_suffix_coordinates(tmp_path):
    from waveform_analysis.scripts.evaluate_mean_ar import plot_waveform_example
    n, cutoff = 200, 35
    mean = np.full(n, .25)
    prediction_suffix = np.full(n-cutoff, .5)
    baseline_suffix = np.full(n-cutoff, .25)
    path = tmp_path / "example.png"
    plot_waveform_example(
        path, time_ps=np.arange(n)*100., cutoff=cutoff,
        original=np.full(n, .35), prediction_suffix=prediction_suffix,
        baseline_suffix=baseline_suffix, mean=mean, mask=np.ones(n, bool),
        event_id=4, title="synthetic", decay_half_lives_ns=[1.],
    )
    assert path.is_file() and path.stat().st_size > 100
    import pytest
    with pytest.raises(ValueError, match="suffix-only"):
        plot_waveform_example(
            path, time_ps=np.arange(n)*100., cutoff=cutoff,
            original=np.full(n, .35),
            prediction_suffix=prediction_suffix[cutoff:],
            baseline_suffix=baseline_suffix, mean=mean,
            mask=np.ones(n, bool), event_id=4, title="synthetic",
            decay_half_lives_ns=[1.],
        )
