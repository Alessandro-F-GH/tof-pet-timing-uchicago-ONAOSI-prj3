import unittest
from pathlib import Path

from waveform_analysis.ml_pipeline.analyses import (
    _common_event_keys,
    _threshold_config,
    _ThresholdPoint,
)


class IntegratedAnalysisTests(unittest.TestCase):
    def test_threshold_config_isolates_prepared_cache(self):
        base = {
            "standard_methods": {"led_thresholds_mV": [5, 15, 25]},
            "preprocessing": {"prepared_dir": "/tmp/prepared"},
            "experiment": {"output_dir": "/tmp/run", "concatenate_datasets": False},
        }
        config = _threshold_config(base, 25.0, Path("/tmp/run/analyses/led_threshold"))
        self.assertEqual(config["standard_methods"]["led_thresholds_mV"], [25.0])
        self.assertIn("_led_threshold_scan", config["preprocessing"]["prepared_dir"])
        self.assertIn("25mV", config["preprocessing"]["prepared_dir"])

    def test_common_validation_population_is_intersection(self):
        def point(threshold, keys):
            return _ThresholdPoint(
                threshold_mV=threshold,
                dataset_name="dataset",
                prepared_dir=Path("/tmp"),
                event_keys=tuple(keys),
                model_residual_ps=None,
                led_residual_ps=None,
                validation_ctr_ps=1.0,
                selected_parameters_json="{}",
                retained_events=10,
                validation_events=len(keys),
            )

        common = _common_event_keys(
            [
                point(5.0, [1, 2, 3, 4]),
                point(15.0, [2, 3, 4, 5]),
                point(25.0, [3, 4, 5, 6]),
            ]
        )
        self.assertEqual(common, (3, 4))


if __name__ == "__main__":
    unittest.main()
