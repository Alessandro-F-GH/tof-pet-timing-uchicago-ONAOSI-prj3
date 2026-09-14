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

    def test_optional_analyses_default_to_disabled(self):
        self.assertFalse(self.config["analyses"]["led_threshold_scan"]["enabled"])
        self.assertFalse(self.config["analyses"]["window_scan"]["enabled"])

    def test_obsolete_ml_training_is_rejected(self):
        stale = copy.deepcopy(self.config)
        stale["ml_training"] = {"target_abs_max_ps": [100, 200]}
        with self.assertRaisesRegex(ConfigError, "ml_training is obsolete"):
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

    def test_threshold_selection_model_must_be_configured(self):
        bad = copy.deepcopy(self.config)
        bad["analyses"] = {
            "led_threshold_scan": {
                "enabled": True,
                "selection_model": "not_a_model",
            }
        }
        with self.assertRaisesRegex(ConfigError, "selection_model"):
            validate_config(bad)

    def test_window_scan_must_fit_inside_base_window(self):
        bad = copy.deepcopy(self.config)
        bad["analyses"] = {
            "window_scan": {
                "enabled": True,
                "right_limits_ns": [1, 1000],
                "models": ["cnn"],
            }
        }
        with self.assertRaisesRegex(ConfigError, "right_limits_ns"):
            validate_config(bad)


if __name__ == "__main__":
    unittest.main()
