import unittest

import numpy as np

from waveform_analysis.scripts.analyze_instance_hardness import (
    MODEL_SETTINGS,
    _difference_pair,
    _oof_null_prediction,
    _prediction_gain,
)


class PredictionGainTests(unittest.TestCase):
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

    def test_gain_is_positive_when_models_beat_null(self):
        target = np.asarray([-10.0, 10.0])
        null = np.asarray([0.0, 0.0])
        predictions = np.asarray([[-9.0, 9.0], [-8.0, 8.0]])
        gain, null_error, model_error = _prediction_gain(target, predictions, null)
        np.testing.assert_allclose(null_error, [10.0, 10.0])
        np.testing.assert_allclose(model_error, [1.5, 1.5])
        np.testing.assert_allclose(gain, [8.5, 8.5])

    def test_gain_is_negative_when_models_are_worse_than_null(self):
        target = np.asarray([1.0])
        null = np.asarray([0.0])
        predictions = np.asarray([[4.0], [-3.0]])
        gain, _, _ = _prediction_gain(target, predictions, null)
        self.assertLess(gain[0], 0.0)

    def test_null_prediction_is_out_of_fold(self):
        target = np.asarray([0.0, 2.0, 4.0, 6.0])
        prediction = _oof_null_prediction(target, folds=2, split_seed=7)
        self.assertEqual(prediction.shape, target.shape)
        self.assertTrue(np.all(np.isfinite(prediction)))
        self.assertFalse(np.allclose(prediction, np.mean(target)))


if __name__ == "__main__":
    unittest.main()
