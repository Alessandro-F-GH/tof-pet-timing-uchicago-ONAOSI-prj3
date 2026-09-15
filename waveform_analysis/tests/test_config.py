import copy
import unittest
from pathlib import Path

from waveform_analysis.ml_pipeline.config import ConfigError, load_config, validate_config


class ConfigTests(unittest.TestCase):
    def setUp(self):
        root = Path(__file__).resolve().parents[1]
        self.config = load_config(root / "config" / "experiments" / "timing.json")

    def test_default_experiment_has_no_training_target_filter(self):
        self.assertNotIn("ml_training", self.config)

    def test_integrated_analyses_are_removed(self):
        self.assertNotIn("analyses", self.config)

    def test_obsolete_ml_training_is_rejected(self):
        stale = copy.deepcopy(self.config)
        stale["ml_training"] = {"target_abs_max_ps": [100, 200]}
        with self.assertRaisesRegex(ConfigError, "Obsolete configuration"):
            validate_config(stale)

    def test_removed_min_events_is_rejected(self):
        bad = copy.deepcopy(self.config)
        bad["fit"]["min_events"] = 100
        with self.assertRaisesRegex(ConfigError, "Unknown/obsolete fit option"):
            validate_config(bad)

    def test_removed_bin_width_is_rejected(self):
        bad = copy.deepcopy(self.config)
        bad["fit"]["bin_width_ps"] = 5.0
        with self.assertRaisesRegex(ConfigError, "Unknown/obsolete fit option"):
            validate_config(bad)

    def test_validation_fraction_above_half_is_valid(self):
        valid = copy.deepcopy(self.config)
        valid["validation"]["test_fraction"] = 0.6
        valid["validation"]["validation_fraction"] = 0.7
        validate_config(valid)

    def test_coverage_fraction_below_half_is_valid(self):
        valid = copy.deepcopy(self.config)
        valid["fit"]["coverage_fraction"] = 0.4
        validate_config(valid)


    def test_model_comparison_requires_final_model_pair(self):
        bad = copy.deepcopy(self.config)
        bad["experiment"]["type"] = "model_comparison"
        bad["experiment"]["fixed_led_threshold_mV"] = 15.0
        bad["experiment"]["windows"] = {
            "onishi": {"start": -1.5, "end": 2.0},
            "wide": {"start": -2.0, "end": 30.0},
        }
        bad["models"] = {"mlp": self.config["models"]["mlp"]}
        with self.assertRaisesRegex(ConfigError, "mlp.*onishi_cnn"):
            validate_config(bad)

    def test_threshold_scan_requires_voltage(self):
        bad = copy.deepcopy(self.config)
        bad["experiment"]["type"] = "threshold_scan"
        bad["models"] = {"mlp": self.config["models"]["mlp"]}
        with self.assertRaisesRegex(ConfigError, "voltage_V"):
            validate_config(bad)

    def test_model_comparison_accepts_arbitrary_window_names(self):
        valid = copy.deepcopy(self.config)
        valid["experiment"].update(
            {
                "type": "model_comparison",
                "fixed_led_threshold_mV": 15.0,
                "windows": {
                    "onishi_window": {"start": -1.5, "end": 2.0},
                    "wide_window": {"start": -2.0, "end": 30.0},
                },
            }
        )
        validate_config(valid)

    def test_model_comparison_requires_onishi_reference_bounds(self):
        bad = copy.deepcopy(self.config)
        bad["experiment"].update(
            {
                "type": "model_comparison",
                "fixed_led_threshold_mV": 15.0,
                "windows": {
                    "short_window": {"start": -1.0, "end": 2.0},
                    "wide_window": {"start": -2.0, "end": 30.0},
                },
            }
        )
        with self.assertRaisesRegex(ConfigError, "reference window"):
            validate_config(bad)

    def test_obsolete_analyses_section_is_rejected(self):
        bad = copy.deepcopy(self.config)
        bad["analyses"] = {"led_threshold_scan": {"enabled": True}}
        with self.assertRaisesRegex(ConfigError, "Obsolete configuration"):
            validate_config(bad)


if __name__ == "__main__":
    unittest.main()
