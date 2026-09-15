import unittest

import numpy as np

from waveform_analysis.ml_pipeline.models.mlp import (
    MLPArtifact,
    SharedScorerMLP,
    candidates as mlp_candidates,
    predict as predict_mlp,
)
from waveform_analysis.ml_pipeline.models.onishi_cnn import (
    OnishiCNNArtifact,
    OnishiPairedCNN,
    explain as explain_onishi_cnn,
    predict as predict_onishi_cnn,
)
from waveform_analysis.ml_pipeline.view import corrected_timing_residual


class ActiveModelTests(unittest.TestCase):
    def test_mlp_pair_antisymmetry(self):
        rng = np.random.default_rng(16)
        pair = rng.normal(size=(8, 2, 24)).astype(np.float32)
        artifact = MLPArtifact(
            SharedScorerMLP(24, [8, 4], "silu"),
            "cpu",
            {},
        )
        forward = predict_mlp(artifact, pair)
        reverse = predict_mlp(artifact, pair[:, ::-1, :])
        np.testing.assert_allclose(forward, -reverse, rtol=1e-6, atol=1e-6)

    def test_mlp_candidate_grid_uses_requested_hyperparameters(self):
        config = {
            "parameters": {
                "architecture": [[32], [64, 32]],
                "activation": ["relu", "silu"],
                "learning_rate": [1e-3, 5e-4],
                "batch_size": [32, 64],
                "weight_decay": [0.0, 1e-5],
            }
        }
        rows = mlp_candidates(config)
        self.assertEqual(len(rows), 32)
        self.assertEqual(
            set(rows[0]),
            {
                "architecture",
                "activation",
                "learning_rate",
                "batch_size",
                "weight_decay",
            },
        )

    def test_onishi_cnn_matches_reference_architecture(self):
        import torch

        model = OnishiPairedCNN(
            {
                "channels": [32, 32, 64],
                "kernels": [5, 3, 3],
                "dense_units": 256,
            }
        )
        conv_layers = [
            layer for layer in model.features
            if isinstance(layer, torch.nn.Conv2d)
        ]
        self.assertEqual(
            [layer.kernel_size for layer in conv_layers],
            [(2, 5), (1, 3), (1, 3)],
        )
        self.assertEqual(
            [layer.out_channels for layer in conv_layers],
            [32, 32, 64],
        )
        self.assertFalse(
            any(isinstance(layer, torch.nn.MaxPool2d) for layer in model.features)
        )

    def test_onishi_cnn_forward_and_xai_shape(self):
        rng = np.random.default_rng(14)
        pair = rng.normal(size=(6, 2, 64)).astype(np.float32)
        model = OnishiPairedCNN(
            {
                "channels": [32, 32, 64],
                "kernels": [5, 3, 3],
                "dense_units": 256,
            }
        )
        artifact = OnishiCNNArtifact(model, "cpu", {})
        prediction = predict_onishi_cnn(artifact, pair)
        importance = explain_onishi_cnn(artifact, pair)
        self.assertEqual(prediction.shape, (6,))
        self.assertEqual(importance.shape, (64,))
        self.assertTrue(np.all(np.isfinite(prediction)))
        self.assertTrue(np.all(np.isfinite(importance)))

    def test_corrected_timing_is_target_minus_prediction(self):
        target = np.asarray([35.0, -20.0, 5.0])
        prediction = np.asarray([10.0, -5.0, 8.0])
        np.testing.assert_allclose(
            corrected_timing_residual(target, prediction),
            [25.0, -15.0, -3.0],
        )


if __name__ == "__main__":
    unittest.main()
