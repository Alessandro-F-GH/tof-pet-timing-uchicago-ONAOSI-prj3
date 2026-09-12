import unittest
from pathlib import Path

import numpy as np

from waveform_analysis.scripts.sweep_led_thresholds import (
    _common_validation_events,
    _threshold_config,
)


class LedThresholdSweepTests(unittest.TestCase):
    def test_threshold_config_isolates_prepared_cache_and_run_output(self):
        base = {
            "standard_methods": {"led_thresholds_mV": [5, 15, 25]},
            "models": {"cnn": {"model": "cnn"}},
            "experiment": {"name": "timing", "output_dir": "/tmp/base_run"},
            "preprocessing": {
                "selection_store_dir": "/tmp/selection",
                "preprocessed_dir": "/tmp/preprocessed",
                "prepared_dir": "/tmp/prepared",
            },
        }
        output = Path("/tmp/sweep")
        config = _threshold_config(
            base,
            threshold_mV=25.0,
            output_root=output,
            requested_models=None,
        )
        self.assertEqual(config["standard_methods"]["led_thresholds_mV"], [25.0])
        self.assertEqual(
            config["preprocessing"]["selection_store_dir"],
            base["preprocessing"]["selection_store_dir"],
        )
        self.assertEqual(
            config["preprocessing"]["preprocessed_dir"],
            base["preprocessing"]["preprocessed_dir"],
        )
        self.assertNotEqual(
            config["preprocessing"]["prepared_dir"],
            base["preprocessing"]["prepared_dir"],
        )
        self.assertIn("25mV", config["preprocessing"]["prepared_dir"])
        self.assertIn("25mV", config["experiment"]["output_dir"])

    def test_common_validation_events_intersects_threshold_populations(self):
        payloads = {
            5.0: ({}, {"dataset": {"validation_event_ids": np.asarray([1, 2, 3, 4])}}),
            15.0: ({}, {"dataset": {"validation_event_ids": np.asarray([2, 3, 4, 5])}}),
            25.0: ({}, {"dataset": {"validation_event_ids": np.asarray([3, 4, 5, 6])}}),
        }
        np.testing.assert_array_equal(
            _common_validation_events(payloads, "dataset"),
            np.asarray([3, 4], dtype=np.int64),
        )


if __name__ == "__main__":
    unittest.main()
