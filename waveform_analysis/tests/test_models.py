import unittest
import numpy as np

from waveform_analysis.ml_pipeline.models.cnn import CNNArtifact, SharedScorerCNN, predict as predict_cnn
from waveform_analysis.ml_pipeline.models.cnn_2d import (
    CNN2DArtifact,
    JointPairCNN2D,
    explain as explain_cnn_2d,
    predict as predict_cnn_2d,
)
from waveform_analysis.ml_pipeline.models.linear_svr import fit as fit_svr, predict as predict_svr
from waveform_analysis.ml_pipeline.view import corrected_timing_residual


class AntisymmetryTests(unittest.TestCase):
    def test_linear_svr_pair_antisymmetry(self):
        rng = np.random.default_rng(12)
        pair = rng.normal(size=(160, 2, 24))
        target = rng.normal(size=160)
        artifact = fit_svr(
            {"C": 1.0, "epsilon_ps": 0.0}, pair, target,
            seed=1, config={"max_iterations": 5000},
        )
        forward = predict_svr(artifact, pair)
        reverse = predict_svr(artifact, pair[:, ::-1, :])
        np.testing.assert_allclose(forward, -reverse, rtol=1e-9, atol=1e-9)

    def test_cnn_pair_antisymmetry(self):
        rng = np.random.default_rng(13)
        pair = rng.normal(size=(8, 2, 64)).astype(np.float32)
        artifact = CNNArtifact(SharedScorerCNN({"channels": [4], "kernels": [5], "strides": [1], "dilations": [1], "dense_units": [4]}), "cpu", {})
        forward = predict_cnn(artifact, pair)
        reverse = predict_cnn(artifact, pair[:, ::-1, :])
        np.testing.assert_allclose(forward, -reverse, rtol=1e-6, atol=1e-6)


    def test_cnn_2d_delays_detector_fusion(self):
        model = JointPairCNN2D(
            {
                "channels": [4, 8, 12],
                "kernels": [5, 3, 3],
                "strides": [1, 1, 1],
                "dilations": [1, 1, 1],
                "detector_fusion_layer": 1,
                "adaptive_pool_length": 8,
                "dense_units": [4],
            }
        )
        conv_layers = [layer for layer in model.features if hasattr(layer, "kernel_size")]
        self.assertEqual([layer.kernel_size[0] for layer in conv_layers], [1, 2, 1])
        self.assertEqual(model.detector_fusion_layer, 1)

    def test_cnn_2d_joint_pair_forward_and_xai_shape(self):
        rng = np.random.default_rng(14)
        pair = rng.normal(size=(6, 2, 64)).astype(np.float32)
        model = JointPairCNN2D(
            {
                "channels": [4],
                "kernels": [5],
                "strides": [1],
                "dilations": [1],
                "adaptive_pool_length": 8,
                "dense_units": [4],
            }
        )
        artifact = CNN2DArtifact(model, "cpu", {})
        prediction = predict_cnn_2d(artifact, pair)
        importance = explain_cnn_2d(artifact, pair)
        self.assertEqual(prediction.shape, (6,))
        self.assertEqual(importance.shape, (64,))
        self.assertTrue(np.all(np.isfinite(prediction)))
        self.assertTrue(np.all(np.isfinite(importance)))

    def test_corrected_timing_is_slide_target_minus_prediction(self):
        target = np.asarray([35.0, -20.0, 5.0])
        prediction = np.asarray([10.0, -5.0, 8.0])
        np.testing.assert_allclose(
            corrected_timing_residual(target, prediction),
            [25.0, -15.0, -3.0],
        )
