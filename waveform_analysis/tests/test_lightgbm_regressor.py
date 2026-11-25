"""Direct LightGBM pipeline registration, predictions and model persistence."""
import json
import pickle
import tempfile
import unittest
from pathlib import Path

import numpy as np

from waveform_analysis.models import get_model
from waveform_analysis.engine.search import grid_candidates


ROOT = Path(__file__).resolve().parents[1]


class LightGBMTests(unittest.TestCase):
    def setUp(self):
        try:
            import lightgbm  # noqa: F401
        except ImportError:
            self.skipTest("lightgbm is not installed")
        rng = np.random.default_rng(33)
        self.pair = rng.normal(size=(90, 2, 8)).astype(np.float32)
        self.target = 14.0 * self.pair[:, 0, 3] - 9.0 * self.pair[:, 1, 2]
        self.params = {
            "n_estimators": 20, "learning_rate": 0.1,
            "num_leaves": 7, "max_depth": -1,
            "min_child_samples": 5, "reg_lambda": 0.0,
        }

    def test_registration_and_parameter_space(self):
        spec = get_model("direct_lightgbm_regressor")
        self.assertEqual(spec.estimator_formulation, "direct")
        self.assertEqual(spec.artifact_format, "pickle")
        self.assertFalse(spec.swap_antisymmetry_enforced)
        space = json.loads((ROOT / "config/model_spaces/direct_lightgbm_regressor.json").read_text())
        self.assertEqual(len(grid_candidates(space)), 8)

    def test_fit_predict_explain_and_pickle(self):
        spec = get_model("direct_lightgbm_regressor")
        model = spec.fit(self.params, self.pair, self.target, seed=42, config={})
        self.assertEqual(model.metadata["feature_count"], 16)
        prediction = spec.predict(model, self.pair)
        self.assertEqual(prediction.shape, (90,))
        self.assertTrue(np.all(np.isfinite(prediction)))
        self.assertEqual(spec.explain(model, self.pair).shape, (16,))
        with tempfile.TemporaryDirectory() as tmp:
            spec.save(model, Path(tmp))
            with (Path(tmp) / "model.pkl").open("rb") as stream:
                restored = pickle.load(stream)
            np.testing.assert_allclose(prediction, spec.predict(restored, self.pair))

    def test_determinism_and_invalid_inputs(self):
        spec = get_model("direct_lightgbm_regressor")
        a = spec.fit(self.params, self.pair, self.target, seed=42, config={})
        b = spec.fit(self.params, self.pair, self.target, seed=42, config={})
        np.testing.assert_allclose(spec.predict(a, self.pair), spec.predict(b, self.pair))
        with self.assertRaises(ValueError):
            spec.fit(self.params, self.pair[:, 0, :], self.target, seed=42, config={})
        invalid = self.pair.copy()
        invalid[0, 0, 0] = np.nan
        with self.assertRaises(ValueError):
            spec.predict(a, invalid)

    def test_lowpass_batches(self):
        for board in ("UC", "FBK"):
            path = ROOT / "config/batches" / (board + "_lowpass_final.json")
            batch = json.loads(path.read_text())
            self.assertIn("direct_lightgbm_regressor", batch["sweep"]["models"])
            self.assertNotIn("shared_lightgbm_regressor", batch["sweep"]["models"])


if __name__ == "__main__":
    unittest.main()
