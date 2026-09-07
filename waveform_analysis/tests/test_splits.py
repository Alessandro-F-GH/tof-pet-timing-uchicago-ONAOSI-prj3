import unittest
import numpy as np

from waveform_analysis.ml_pipeline.splits import make_split


class SplitTests(unittest.TestCase):
    def test_isolation_and_determinism(self):
        a = make_split(1000, blind_fraction=0.2, validation_fraction=0.2, seed=17)
        b = make_split(1000, blind_fraction=0.2, validation_fraction=0.2, seed=17)
        for name in ("development", "blind", "training", "validation"):
            np.testing.assert_array_equal(getattr(a, name), getattr(b, name))
        self.assertFalse(set(a.development) & set(a.blind))
        self.assertFalse(set(a.training) & set(a.validation))
        self.assertTrue(set(a.training) <= set(a.development))
        self.assertTrue(set(a.validation) <= set(a.development))
