import json
import tempfile
import unittest
from pathlib import Path

from waveform_analysis.ml_pipeline.common import canonical_hash
from waveform_analysis.ml_pipeline.study import _check_experiment_resume_config


class ExperimentResumeTests(unittest.TestCase):
    def test_resume_accepts_matching_config(self):
        config = {"experiment": {"type": "threshold_scan"}, "mode": "timing_to_timing"}
        config["_config_fingerprint"] = canonical_hash(config)
        public = {key: value for key, value in config.items() if not key.startswith("_")}
        config["_config_fingerprint"] = canonical_hash(public)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "manifest.json").write_text(
                json.dumps({"config": public}),
                encoding="utf-8",
            )
            _check_experiment_resume_config(config, root, resume=True)

    def test_resume_rejects_changed_config(self):
        config = {"experiment": {"type": "threshold_scan"}, "mode": "timing_to_timing"}
        public = dict(config)
        config["_config_fingerprint"] = canonical_hash(public)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "manifest.json").write_text(
                json.dumps({"config": {**public, "mode": "energy_to_energy"}}),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(RuntimeError, "different configuration"):
                _check_experiment_resume_config(config, root, resume=True)


if __name__ == "__main__":
    unittest.main()
