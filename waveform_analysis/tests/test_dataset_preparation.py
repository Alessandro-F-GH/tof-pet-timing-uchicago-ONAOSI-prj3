import unittest
from pathlib import Path

import numpy as np

from waveform_analysis.ml_pipeline.data import PreprocessedData
from waveform_analysis.ml_pipeline.prepared_data import (
    _learn_dead_time_mask,
    _materialize_family,
    _normalize_family,
)
from waveform_analysis.ml_pipeline.timing import led_grid


class DatasetPreparationTests(unittest.TestCase):
    def _data(self):
        n, length = 6, 32
        waves = np.zeros((n, 2, length), dtype=np.float32)
        for event in range(n):
            waves[event, 0] = np.arange(length, dtype=np.float32) + event * 0.1
            waves[event, 1] = 2.0 * np.arange(length, dtype=np.float32) + event * 0.2
        starts = np.zeros((n, 2), dtype=np.float64)
        intervals = np.full((n, 2), 1e-9, dtype=np.float64)
        rising_start = np.full((n, 2), 4, dtype=np.int32)
        rising_stop = np.full((n, 2), 20, dtype=np.int32)
        return PreprocessedData(
            Path("."),
            {"source": "synthetic.root", "fingerprint": "x"},
            np.arange(n),
            np.array([0, 0, 0, 0, 1, 1], dtype=np.int8),
            np.ones(n) * 45,
            waves,
            None,
            starts,
            None,
            intervals,
            None,
            rising_start,
            None,
            rising_stop,
            None,
        )

    def _config(self):
        return {
            "ml_input": {
                "window_ns": {"start": -2.0, "end": 4.0},
                "subsampling": 2,
            },
            "preprocessing": {
                "energy": {
                    "vertical_scale_limit_mV": [[-10.0, 40.0], [-20.0, 80.0]]
                }
            },
        }

    def test_continuous_alignment_places_led_crossing_exactly_at_zero(self):
        data = self._data()
        threshold = 10.5
        led = led_grid(
            data,
            "energy",
            np.arange(data.n_events),
            np.asarray([threshold]),
        )[:, :, 0]
        raw, time_ps = _materialize_family(
            data,
            "energy",
            led,
            np.arange(data.n_events, dtype=np.int64),
            self._config(),
            threshold,
        )
        zero = int(np.flatnonzero(np.isclose(time_ps, 0.0))[0])
        np.testing.assert_allclose(raw[:, :, zero], threshold, rtol=0.0, atol=0.0)

        plus_two_ns = int(np.flatnonzero(np.isclose(time_ps, 2000.0))[0])
        np.testing.assert_allclose(raw[:, 0, plus_two_ns], threshold + 2.0, atol=1e-6)
        np.testing.assert_allclose(raw[:, 1, plus_two_ns], threshold + 4.0, atol=1e-6)

    def test_dead_region_mask_is_learned_from_development_and_removes_crossing(self):
        time_ps = np.asarray([-1000.0, 0.0, 1000.0, 2000.0, 3000.0])
        raw = np.zeros((100, 2, 5), dtype=np.float32)
        raw[:, :, 0] = -5.0
        raw[:, :, 1] = 10.5
        raw[:, 0, 2] = np.arange(100, dtype=np.float32)
        raw[:, 1, 2] = np.arange(100, dtype=np.float32) * 2.0
        raw[:99, :, 3] = 40.0
        raw[99, :, 3] = 39.0
        raw[:, 0, 4] = np.arange(100, dtype=np.float32) + 0.5
        raw[:, 1, 4] = np.arange(100, dtype=np.float32) * 3.0 + 0.5

        keep, fractions = _learn_dead_time_mask(
            raw,
            np.arange(100, dtype=np.int64),
            time_ps,
            threshold=0.99,
        )
        np.testing.assert_array_equal(keep, [False, False, True, False, True])
        np.testing.assert_allclose(fractions[:, 3], 0.99)

    def test_normalization_uses_fixed_detector_limits_after_interpolation(self):
        data = self._data()
        threshold = 10.5
        led = led_grid(
            data,
            "energy",
            np.arange(data.n_events),
            np.asarray([threshold]),
        )[:, :, 0]
        raw, _time = _materialize_family(
            data,
            "energy",
            led,
            np.arange(data.n_events, dtype=np.int64),
            self._config(),
            threshold,
        )
        normalized, minimum, maximum = _normalize_family(raw, "energy", self._config())
        self.assertEqual(minimum.shape, (2, 1))
        self.assertEqual(maximum.shape, (2, 1))
        np.testing.assert_allclose(minimum[:, 0], [-10.0, -20.0])
        np.testing.assert_allclose(maximum[:, 0], [40.0, 80.0])
        reconstructed = normalized * (maximum - minimum)[None, :, :] + minimum[None, :, :]
        np.testing.assert_allclose(reconstructed, raw, rtol=1e-6, atol=1e-6)


if __name__ == "__main__":
    unittest.main()
