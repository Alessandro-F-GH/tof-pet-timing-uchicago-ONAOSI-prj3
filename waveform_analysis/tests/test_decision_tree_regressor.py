"""Decision tree registration, input formulations, persistence and batch coverage."""
from __future__ import annotations

import json
import pickle
import tempfile
import unittest
from pathlib import Path

import numpy as np

from waveform_analysis.models import get_model
from waveform_analysis.engine.search import grid_candidates


ROOT = Path(__file__).resolve().parents[1]


class DecisionTreeRegressorTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(51)
        self.pair = rng.normal(size=(100, 2, 12)).astype(np.float32)
        self.target = 18 * (self.pair[:, 0, 3] - self.pair[:, 1, 3]) + 2
        self.params = {
            "criterion": "squared_error", "max_depth": 5,
            "min_samples_leaf": 10, "ccp_alpha": 0.0,
        }

    def test_registration_and_candidate_spaces(self):
        for form in ("direct", "shared"):
            name = f"{form}_decision_tree_regressor"
            spec = get_model(name)
            self.assertEqual(spec.name, name)
            self.assertEqual(spec.estimator_formulation, form)
            self.assertEqual(spec.artifact_format, "pickle")
            self.assertIsNone(spec.feature_transform)
            self.assertFalse(spec.swap_antisymmetry_enforced)
            space = json.loads((ROOT / "config/model_spaces" / f"{name}.json").read_text())
            self.assertEqual(len(grid_candidates(space)), 12)

    def test_formulations_prediction_and_pickle_roundtrip(self):
        for form, n_features in (("direct", 24), ("shared", 12)):
            with self.subTest(form=form):
                spec = get_model(f"{form}_decision_tree_regressor")
                model = spec.fit(self.params, self.pair, self.target, seed=12, config={})
                self.assertEqual(model.metadata["feature_count"], n_features)
                self.assertEqual(model.regressor.n_features_in_, n_features)
                original = spec.predict(model, self.pair)
                self.assertEqual(original.shape, (100,))
                self.assertTrue(np.all(np.isfinite(original)))
                self.assertEqual(spec.explain(model, self.pair).shape, (n_features,))
                with tempfile.TemporaryDirectory() as tmp:
                    spec.save(model, Path(tmp))
                    with (Path(tmp) / "model.pkl").open("rb") as stream:
                        restored = pickle.load(stream)
                    np.testing.assert_array_equal(original, spec.predict(restored, self.pair))

    def test_shared_uses_difference_not_independent_scorers(self):
        spec = get_model("shared_decision_tree_regressor")
        model = spec.fit(self.params, self.pair, self.target, seed=12, config={})
        shifted = self.pair + 0.5
        np.testing.assert_array_equal(spec.predict(model, self.pair), spec.predict(model, shifted))
        self.assertFalse(model.metadata["detector_swap_antisymmetry_enforced"])

    def test_invalid_inputs(self):
        spec = get_model("direct_decision_tree_regressor")
        with self.assertRaises(ValueError):
            spec.fit(self.params, self.pair[:, 0], self.target, seed=12, config={})
        bad = self.pair.copy()
        bad[0, 0, 0] = np.nan
        with self.assertRaises(ValueError):
            spec.fit(self.params, bad, self.target, seed=12, config={})

    def test_both_filtered_batches_include_requested_models(self):
        required = {"direct_mlp", "direct_decision_tree_regressor", "shared_decision_tree_regressor"}
        for board in ("UC", "FBK"):
            filename = ROOT / "config/batches" / f"{board}_lowpass_final.json"
            batch = json.loads(filename.read_text())
            self.assertTrue(required <= set(batch["sweep"]["models"]))


if __name__ == "__main__":
    unittest.main()
