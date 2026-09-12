import unittest

import numpy as np

from waveform_analysis.scripts.analyze_instance_hardness import (
    _difference_pair,
    _instance_hardness,
)


class InstanceHardnessTests(unittest.TestCase):
    def test_difference_pair_exposes_only_signal_difference(self):
        pair = np.asarray(
            [
                [[1.0, 3.0, 5.0], [0.5, 1.0, 2.0]],
                [[2.0, 4.0, 8.0], [1.0, 1.5, 3.0]],
            ],
            dtype=np.float32,
        )
        transformed = _difference_pair(pair)
        np.testing.assert_allclose(transformed[:, 0, :], pair[:, 0, :] - pair[:, 1, :])
        np.testing.assert_array_equal(transformed[:, 1, :], 0.0)

    def test_instance_hardness_is_zero_for_exact_predictions(self):
        target = np.asarray([-2.0, 1.0, 3.0])
        predictions = np.stack([target, target], axis=0)
        hardness, gamma = _instance_hardness(target, predictions)
        np.testing.assert_allclose(hardness, 0.0)
        self.assertGreater(gamma, 0.0)

    def test_instance_hardness_increases_with_ensemble_error(self):
        target = np.asarray([1.0, 1.0])
        predictions = np.asarray([[1.0, 2.0], [1.0, 3.0]])
        hardness, _ = _instance_hardness(target, predictions)
        self.assertEqual(hardness[0], 0.0)
        self.assertGreater(hardness[1], hardness[0])
        self.assertLessEqual(hardness[1], 1.0)


if __name__ == "__main__":
    unittest.main()
