"""Exact comparisons against independent, frozen pre-migration kernels."""

from __future__ import annotations

from types import SimpleNamespace
from dataclasses import astuple

import numpy as np
import pytest

from waveform_analysis.data.timing_adapter import anchor_grid, led_grid, cfd_grid
from waveform_analysis.signal.baseline import (
    baseline_level_mV,
    baseline_quality_metrics,
)
from waveform_analysis.signal.timing import (
    crossing_ps,
    led_times_ps,
    cfd_times_ps,
    pair_delta,
)
from waveform_analysis.signal.metrics import rmse_ps
from waveform_analysis.signal.pulses import (
    decode_oriented,
    robust_center_scale,
    pulse_hits,
    photopeak_mask,
)
from waveform_analysis.reporting.stats import metric_values
from .fixtures import legacy_signal_reference as original


def assert_exact(actual, expected) -> None:
    """Check shapes, dtypes, NaN masks and finite bytes, including signed zero."""
    actual = np.asarray(actual)
    expected = np.asarray(expected)
    assert actual.shape == expected.shape
    assert actual.dtype == expected.dtype
    np.testing.assert_array_equal(np.isnan(actual), np.isnan(expected))
    np.testing.assert_array_equal(actual, expected)
    mask = ~np.isnan(expected)
    assert actual[mask].tobytes() == expected[mask].tobytes()


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize(
    "signal,a,b,level",
    [
        ([0, 1, 2, 3], 0, 3, 1.5),
        ([0, 2, 0, 2], 0, 3, 1),  # last strict crossing
        ([0, 2, 1, 2], 0, 3, 1),  # strict beats later equality fallback
        ([1, 1, 2, 3], 0, 3, 1),  # equality fallback
        ([0, 1, 2, 3], 0, 3, 3),  # exact peak
        ([0, 1, 2, 3], 0, 3, 4),  # missing crossing
        ([np.nan, 0, 2, np.inf], 0, 3, 1),
        ([0, 1, 2, 3], -1, 3, 1),
        ([0, 1, 2, 3], 0, 4, 1),
        ([0, 1, 2, 3], 2, 2, 1),
        ([0, 1, 2, 3], 0, 3, np.nan),
    ],
)
def test_crossing_is_bitwise_identical(dtype, signal, a, b, level):
    values = np.asarray(signal, dtype=dtype)
    args = (values, -3.713e-9, 2.5e-12, a, b, level)
    assert_exact(crossing_ps(*args), original._crossing_ps(*args))


def synthetic_data(dtype=np.float32) -> SimpleNamespace:
    """Jittered double-detector pulses, including invalid baselines/crossings."""
    rng = np.random.default_rng(917)
    x = np.arange(160, dtype=np.float64)
    center = rng.normal(86, 2, size=(512, 2, 1))
    amplitude = rng.uniform(40, 100, size=(512, 2, 1))
    waves = (
        amplitude * np.exp(-0.5 * ((x - center) / 10) ** 2)
        + rng.normal(0, 0.2, size=(512, 2, 160))
    ).astype(dtype)
    waves[0, 0, :] = np.nan
    waves[1, 1, :] = 0
    waves[2, 0, 20:25] = np.inf
    starts = rng.normal(-4e-9, 10e-12, size=(512, 2))
    intervals = np.full((512, 2), 50e-12)
    a = np.full((512, 2), 30, dtype=np.int32)
    b = np.broadcast_to(center[:, :, 0].astype(np.int32), (512, 2)).copy()
    return SimpleNamespace(
        energy_windows_mV=waves,
        energy_window_start_time_s=starts,
        energy_sample_interval_s=intervals,
        energy_rising_start=a,
        energy_rising_stop=b,
        n_events=512,
        manifest={"materialized_window_ns": {"before": 3.0}},
    )


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("window", [None, (-2.0, -1.0), (10.0, 11.0)])
def test_synthetic_led_and_cfd_grids_are_identical(dtype, window):
    data = synthetic_data(dtype)
    # Include repeated and reordered event IDs to check population ordering.
    indices = np.r_[np.arange(512), [2, 0, 511]]
    thresholds = np.array([5, 15, 25, 35, 45, 55, np.nan], dtype=float)
    arrays = original.family_arrays(data, "energy")
    expected_led = original.led_grid(
        data, "energy", indices, thresholds, baseline_window_ns=window
    )
    assert_exact(
        led_times_ps(
            *arrays,
            indices,
            thresholds,
            baseline_window_ns=window,
            materialized_before_ns=3.0,
        ),
        expected_led,
    )
    assert_exact(
        led_grid(data, "energy", indices, thresholds, baseline_window_ns=window),
        expected_led,
    )
    fractions = np.array([0.05, 0.2, 0.5, 1.0, np.nan])
    with np.errstate(invalid="ignore"):
        expected_cfd = original.cfd_grid(data, "energy", indices, fractions)
        assert_exact(cfd_times_ps(*arrays, indices, fractions), expected_cfd)
        assert_exact(cfd_grid(data, "energy", indices, fractions), expected_cfd)
    assert_exact(
        anchor_grid(data, "energy", 15, baseline_window_ns=window),
        original.anchor_grid(data, "energy", 15, baseline_window_ns=window),
    )


@pytest.mark.parametrize("window", [(-2, -1), (-100, 100), (10, 11), (0, 0)])
def test_baseline_mean_rms_and_clipping_are_identical(window):
    signal = np.random.default_rng(31).normal(size=160)
    signal[2] = np.nan
    signal[40] = np.inf
    assert_exact(
        baseline_level_mV(signal, 50e-12, 3.0, window),
        original._baseline_level_mV(signal, 50e-12, 3.0, window),
    )
    args = (signal, 60, 50e-12, window, (-5, 5), 1.0)
    assert_exact(
        baseline_quality_metrics(*args), original._baseline_quality_metrics(*args)
    )


def test_led_and_cfd_residual_metrics_are_identical():
    data = synthetic_data(np.float64)
    indices = np.arange(data.n_events)
    for actual, expected in [
        (
            led_grid(data, "energy", indices, np.array([15.0])),
            original.led_grid(data, "energy", indices, np.array([15.0])),
        ),
        (
            cfd_grid(data, "energy", indices, np.array([0.5])),
            original.cfd_grid(data, "energy", indices, np.array([0.5])),
        ),
    ]:
        delta = pair_delta(actual[:, :, 0])
        before = original.pair_delta(expected[:, :, 0])
        assert_exact(delta, before)
        assert_exact(rmse_ps(delta), original.rmse_ps(before))
        fit = {"histogram_bin_width_ps": 10.0}
        assert metric_values(delta, fit, seed=918) == original.metric_values(
            before, fit, seed=918
        )


@pytest.mark.parametrize("fractions", [[0], [-0.1], [1.1]])
def test_invalid_cfd_fractions_preserve_errors(fractions):
    data = synthetic_data()
    for function in (cfd_grid, original.cfd_grid):
        with pytest.raises(ValueError, match=r"CFD fractions must lie in"):
            function(data, "energy", np.array([0]), np.asarray(fractions))


@pytest.mark.parametrize("polarity", [-1, 1])
def test_adc_calibration_and_pulse_selection_are_identical(polarity):
    raw = np.random.default_rng(88).integers(-100, 500, size=160, dtype=np.int16)
    args = (raw, 0.00025, -0.012, polarity)
    signal = decode_oriented(*args)
    assert_exact(signal, original.decode_oriented(*args))
    assert_exact(robust_center_scale(signal), original.robust_center_scale(signal))
    for threshold in (5.0, 15.0, 40.0):
        before = original.pulse_hits(signal, threshold, 2.5e-12)
        after = pulse_hits(signal, threshold, 2.5e-12)
        assert [astuple(hit) for hit in after] == [astuple(hit) for hit in before]
    amplitudes = signal[:80].reshape(-1, 2)
    rules = {"photopeak_intervals_mV": [[-5.0, 50.0], [0.0, 100.0]]}
    assert_exact(
        photopeak_mask(amplitudes, rules), original._photo_mask(amplitudes, rules)
    )
