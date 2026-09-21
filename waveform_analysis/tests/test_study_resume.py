import json
import tempfile
import unittest
from pathlib import Path

from waveform_analysis.ml_pipeline.common import canonical_hash
from unittest.mock import patch

from waveform_analysis.ml_pipeline.study import (
    _check_experiment_resume_config,
    _run_model_study_experiment,
)


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


    def test_model_study_rebuilds_prepared_cache_for_every_window(self):
        config = {
            "experiment": {
                "type": "model_study",
                "output_dir": "unused",
                "name": "test",
                "fixed_led_threshold_mV": 15.0,
            },
            "models": {"mlp": {}},
            "mode": "timing_to_timing",
            "ml_input": {
                "windows": {
                    "first": {"start": -1.5, "end": 2.0},
                    "second": {"start": -2.0, "end": 30.0},
                },
                "window_ns": {"start": -1.5, "end": 2.0},
                "subsampling": 1,
            },
            "preprocessing": {"prepared_dir": "prepared"},
            "standard_methods": {"led_thresholds_mV": [15.0]},
            "ml_output": {"max_abs_ps": 500.0},
            "data": {},
            "validation": {},
            "fit": {},
            "cfd": False,
            "_config_fingerprint": "x",
        }

        calls = []

        def fake_run_standard(sub, **kwargs):
            calls.append(kwargs)
            return Path(sub["experiment"]["output_dir"])

        with tempfile.TemporaryDirectory() as tmp:
            config["experiment"]["output_dir"] = str(Path(tmp) / "study")
            config["preprocessing"]["prepared_dir"] = str(Path(tmp) / "prepared")
            with (
                patch(
                    "waveform_analysis.ml_pipeline.study._run_standard_study",
                    side_effect=fake_run_standard,
                ),
                patch(
                    "waveform_analysis.ml_pipeline.study._model_study_compatibility",
                    return_value={"signature": "x", "payload": {}},
                ),
                patch(
                    "waveform_analysis.ml_pipeline.study.plot_model_study_windows"
                ),
            ):
                _run_model_study_experiment(
                    config,
                    overwrite=False,
                    resume=False,
                    rebuild_preprocessing=True,
                )

        self.assertEqual(len(calls), 2)
        self.assertTrue(calls[0]["rebuild_preprocessing"])
        self.assertFalse(calls[1]["rebuild_preprocessing"])
        self.assertTrue(calls[0]["rebuild_prepared"])
        self.assertTrue(calls[1]["rebuild_prepared"])


if __name__ == "__main__":
    unittest.main()
