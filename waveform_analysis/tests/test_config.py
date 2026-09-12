import copy
import unittest
from pathlib import Path

from waveform_analysis.ml_pipeline.config import ConfigError, load_config, validate_config


class ConfigTests(unittest.TestCase):
    def test_default_experiment_has_no_training_target_filter(self):
        root = Path(__file__).resolve().parents[1]
        config = load_config(root / "config" / "experiments" / "timing.json")
        self.assertNotIn("ml_training", config)

    def test_obsolete_ml_training_is_rejected(self):
        root = Path(__file__).resolve().parents[1]
        config = load_config(root / "config" / "experiments" / "timing.json")
        stale = copy.deepcopy(config)
        stale["ml_training"] = {"target_abs_max_ps": [100, 200]}
        with self.assertRaisesRegex(ConfigError, "ml_training is obsolete"):
            validate_config(stale)


if __name__ == "__main__":
    unittest.main()
