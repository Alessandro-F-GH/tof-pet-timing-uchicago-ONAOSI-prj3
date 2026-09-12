import unittest

import numpy as np

from waveform_analysis.scripts.analyze_instance_hardness import (
    MODEL_SETTINGS,
    _difference_pair,
    _ensemble_disagreement,
)


class EnsembleDisagreementTests(unittest.TestCase):
    def test_fixed_model_pool_has_dual_knn_and_minirocket(self):
        self.assertEqual(
            set(MODEL_SETTINGS),
            {"linear_svr", "difference_knn_k2", "difference_knn_k50", "minirocket"},
        )
        self.assertEqual(MODEL_SETTINGS["difference_knn_k2"]["parameters"]["n_neighbors"], 2)
        self.assertEqual(MODEL_SETTINGS["difference_knn_k50"]["parameters"]["n_neighbors"], 50)
        self.assertEqual(MODEL_SETTINGS["minirocket"]["parameters"]["n_kernels"], 10000)

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

    def test_disagreement_is_zero_when_models_agree(self):
        predictions = np.asarray(
            [
                [1.0, 2.0, 3.0],
                [1.0, 2.0, 3.0],
                [1.0, 2.0, 3.0],
            ]
        )
        disagreement, prediction_range = _ensemble_disagreement(predictions)
        np.testing.assert_allclose(disagreement, 0.0)
        np.testing.assert_allclose(prediction_range, 0.0)

    def test_disagreement_increases_with_prediction_spread(self):
        predictions = np.asarray(
            [
                [0.0, 0.0],
                [0.0, 2.0],
                [0.0, 4.0],
            ]
        )
        disagreement, prediction_range = _ensemble_disagreement(predictions)
        self.assertEqual(disagreement[0], 0.0)
        self.assertGreater(disagreement[1], disagreement[0])
        np.testing.assert_allclose(prediction_range, [0.0, 4.0])


if __name__ == "__main__":
    unittest.main()
