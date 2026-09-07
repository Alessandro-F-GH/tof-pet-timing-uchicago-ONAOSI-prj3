import unittest
import numpy as np

from waveform_analysis.ml_pipeline.splits import split_development_test, split_training_validation


class SplitTests(unittest.TestCase):
    def test_raw_test_split_and_development_holdout_are_disjoint_and_deterministic(self):
        indices = np.arange(1000)
        first = split_development_test(indices, test_fraction=0.2, seed=7)
        second = split_development_test(indices, test_fraction=0.2, seed=7)
        np.testing.assert_array_equal(first.development, second.development)
        np.testing.assert_array_equal(first.test, second.test)
        self.assertFalse(set(first.development) & set(first.test))
        holdout = split_training_validation(first.development, validation_fraction=0.2, seed=11)
        self.assertFalse(set(holdout.training) & set(holdout.validation))
        self.assertEqual(set(holdout.training) | set(holdout.validation), set(first.development))
