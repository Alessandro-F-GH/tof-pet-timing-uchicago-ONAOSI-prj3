"""Smoke tests for RuleCard model registration, formulations and persistence."""
import json
import pickle
import tempfile
import unittest
from pathlib import Path
import numpy as np

from waveform_analysis.models import get_model
from waveform_analysis.engine.search import grid_candidates

ROOT = Path(__file__).resolve().parents[1]


class RuleCardTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(19)
        self.pair = rng.normal(size=(90, 2, 9)).astype(np.float32)
        self.target = 8 * (self.pair[:, 0, 2] - self.pair[:, 1, 2])
        self.params = dict(learning_rate=0.3, max_n_iter=6, patience=3,
                           val_size=0.2, max_depth=2, min_samples_leaf=5,
                           top_k_features=4, include_pairs=True)

    def test_models(self):
        for form, count in (("direct", 18), ("shared", 9)):
            with self.subTest(form=form):
                name = form + "_rulecard_regressor"
                spec = get_model(name)
                self.assertEqual(spec.estimator_formulation, form)
                self.assertFalse(spec.swap_antisymmetry_enforced)
                self.assertEqual(spec.artifact_format, "pickle")
                space = json.loads((ROOT / "config/model_spaces" / (name + ".json")).read_text())
                self.assertEqual(len(grid_candidates(space)), 4)
                artifact = spec.fit(self.params, self.pair, self.target, seed=11, config={})
                self.assertEqual(artifact.metadata["feature_count"], count)
                p = spec.predict(artifact, self.pair)
                self.assertEqual(p.shape, (90,))
                self.assertTrue(np.all(np.isfinite(p)))
                self.assertEqual(spec.explain(artifact, self.pair).shape, (count,))
                with tempfile.TemporaryDirectory() as directory:
                    spec.save(artifact, Path(directory))
                    with open(Path(directory) / "model.pkl", "rb") as stream:
                        loaded = pickle.load(stream)
                    np.testing.assert_allclose(p, spec.predict(loaded, self.pair))

    def test_seed_and_invalid_input(self):
        spec = get_model("direct_rulecard_regressor")
        a = spec.fit(self.params, self.pair, self.target, seed=4, config={})
        b = spec.fit(self.params, self.pair, self.target, seed=4, config={})
        np.testing.assert_array_equal(spec.predict(a, self.pair), spec.predict(b, self.pair))
        with self.assertRaises(ValueError):
            spec.fit(self.params, self.pair[:, 0], self.target, seed=4, config={})
        bad = self.pair.copy()
        bad[0, 0, 0] = np.nan
        with self.assertRaises(ValueError):
            spec.predict(a, bad)

    def test_lowpass_batches(self):
        for board in ("UC", "FBK"):
            path = ROOT / "config/batches" / (board + "_lowpass_final.json")
            batch = json.loads(path.read_text())
            self.assertIn("direct_rulecard_regressor", batch["sweep"]["models"])
            self.assertIn("shared_rulecard_regressor", batch["sweep"]["models"])


if __name__ == "__main__":
    unittest.main()
