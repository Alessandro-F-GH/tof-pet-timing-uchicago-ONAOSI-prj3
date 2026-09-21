import copy
import unittest
from pathlib import Path

from waveform_analysis.ml_pipeline.config import ConfigError, load_config, validate_config


class ConfigTests(unittest.TestCase):
    def setUp(self):
        root = Path(__file__).resolve().parents[1]
        self.config = load_config(root / "config" / "experiments" / "timing.json")

    def test_removed_min_events_is_rejected(self):
        bad = copy.deepcopy(self.config)
        bad["fit"]["min_events"] = 100
        with self.assertRaisesRegex(ConfigError, "Unknown fit option"):
            validate_config(bad)

    def test_removed_bin_width_is_rejected(self):
        bad = copy.deepcopy(self.config)
        bad["fit"]["bin_width_ps"] = 5.0
        with self.assertRaisesRegex(ConfigError, "Unknown fit option"):
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


    def test_mlp_training_has_no_random_restart_configuration(self):
        self.assertNotIn(
            "random_restarts",
            self.config["models"]["mlp"]["training"],
        )

    def test_model_study_requires_single_model(self):
        bad = copy.deepcopy(self.config)
        bad["experiment"]["type"] = "model_study"
        bad["experiment"]["fixed_led_threshold_mV"] = 15.0
        with self.assertRaisesRegex(ConfigError, "exactly one"):
            validate_config(bad)

    def test_model_study_uses_profile_windows(self):
        valid = copy.deepcopy(self.config)
        valid["experiment"]["type"] = "model_study"
        valid["experiment"]["fixed_led_threshold_mV"] = 15.0
        valid["models"] = {"mlp": self.config["models"]["mlp"]}
        validate_config(valid)

    def test_threshold_scan_requires_voltage(self):
        bad = copy.deepcopy(self.config)
        bad["experiment"]["type"] = "threshold_scan"
        bad["models"] = {"mlp": self.config["models"]["mlp"]}
        with self.assertRaisesRegex(ConfigError, "voltage_V"):
            validate_config(bad)

    def test_model_study_requires_profile_windows(self):
        bad = copy.deepcopy(self.config)
        bad["experiment"]["type"] = "model_study"
        bad["experiment"]["fixed_led_threshold_mV"] = 15.0
        bad["models"] = {"mlp": self.config["models"]["mlp"]}
        bad["ml_input"].pop("windows", None)
        with self.assertRaisesRegex(ConfigError, "ml_input.windows"):
            validate_config(bad)

    def test_model_study_config_files_load_shared_windows(self):
        root = Path(__file__).resolve().parents[1]
        for name, model in (
            ("model_study_mlp.json", "mlp"),
            ("model_study_onishi.json", "onishi_cnn"),
        ):
            config = load_config(root / "config" / "experiments" / name)
            self.assertEqual(config["experiment"]["type"], "model_study")
            self.assertEqual(set(config["models"]), {model})
            self.assertEqual(
                set(config["ml_input"]["windows"]),
                {"onishi_window", "wide_window"},
            )
            self.assertEqual(
                config["ml_input"]["window_ns"],
                config["ml_input"]["windows"]["wide_window"],
            )
    def test_unknown_top_level_section_is_rejected(self):
        bad = copy.deepcopy(self.config)
        bad["extra_section"] = {}
        with self.assertRaisesRegex(ConfigError, "Unknown configuration section"):
            validate_config(bad)


if __name__ == "__main__":
    unittest.main()
