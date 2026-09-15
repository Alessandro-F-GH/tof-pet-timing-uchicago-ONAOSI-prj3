import unittest
from pathlib import Path

import numpy as np

from waveform_analysis.ml_pipeline.analyses import (
    _paired_blind_improvement,
    _threshold_config,
)


class ThresholdExperimentTests(unittest.TestCase):
    def test_threshold_config_isolates_prepared_cache(self):
        base = {
            "standard_methods": {"led_thresholds_mV": [5, 15, 25]},
            "preprocessing": {"prepared_dir": "/tmp/prepared"},
            "experiment": {"output_dir": "/tmp/run"},
        }
        config = _threshold_config(base, 25.0, Path("/tmp/run"))
        self.assertEqual(config["standard_methods"]["led_thresholds_mV"], [25.0])
        self.assertIn("_led_threshold_scan", config["preprocessing"]["prepared_dir"])
        self.assertIn("25mV", config["preprocessing"]["prepared_dir"])

    def test_paired_improvement_uses_aligned_events(self):
        led = np.asarray([-2.0, -1.0, 1.0, 2.0, 3.0])
        corrected = 0.5 * led
        fit = {"coverage_fraction": 0.8, "bootstrap_samples": 10}
        improvement, uncertainty, successful = _paired_blind_improvement(
            led,
            corrected,
            fit,
            samples=10,
            seed=1,
        )
        self.assertGreater(improvement, 0.0)
        self.assertGreater(successful, 0)
        self.assertTrue(np.isfinite(uncertainty))


if __name__ == "__main__":
    unittest.main()
