import unittest
import numpy as np

from waveform_analysis.ml_pipeline.models.cnn import (
    CNNArtifact,
    SharedScorerCNN,
    fit as fit_cnn,
    predict as predict_cnn,
)
from waveform_analysis.ml_pipeline.models.onishi_cnn import (
    OnishiCNNArtifact,
    OnishiPairedCNN,
    explain as explain_onishi_cnn,
    predict as predict_onishi_cnn,
)
from waveform_analysis.ml_pipeline.models.linear_svr import fit as fit_svr, predict as predict_svr
from waveform_analysis.ml_pipeline.models.mlp import (
    MLPArtifact,
    SharedScorerMLP,
    candidates as mlp_candidates,
    predict as predict_mlp,
)
from waveform_analysis.ml_pipeline.models.mlp_2d import (
    JointPairMLP,
    predict as predict_mlp_2d,
)
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


    def test_cnn_training_is_repeatable_for_same_seed(self):
        rng = np.random.default_rng(15)
        train_x = rng.normal(size=(32, 2, 32)).astype(np.float32)
        train_y = rng.normal(scale=20.0, size=32)
        validation_x = rng.normal(size=(12, 2, 32)).astype(np.float32)
        validation_y = rng.normal(scale=20.0, size=12)
        params = {"learning_rate": 1e-3, "weight_decay": 0.0, "batch_size": 8}
        config = {
            "architecture": {
                "channels": [4],
                "kernels": [5],
                "strides": [1],
                "dilations": [1],                "dense_units": [4],
            },
            "training": {
                "device": "cpu",
                "epochs": 4,
                "patience": 4,
                "min_delta": 0.0,
                "gradient_clip_norm": 10.0,
            },
        }
        first = fit_cnn(
            params,
            train_x,
            train_y,
            seed=12345,
            config=config,
            validation_x=validation_x,
            validation_target=validation_y,
        )
        second = fit_cnn(
            params,
            train_x,
            train_y,
            seed=12345,
            config=config,
            validation_x=validation_x,
            validation_target=validation_y,
        )
        np.testing.assert_array_equal(
            predict_cnn(first, validation_x),
            predict_cnn(second, validation_x),
        )
        self.assertEqual(first.metadata["training_seed"], 12345)
        self.assertEqual(first.metadata["best_epoch"], second.metadata["best_epoch"])
        self.assertEqual(
            first.metadata["best_early_stopping_rmse_ps"],
            second.metadata["best_early_stopping_rmse_ps"],
        )
        self.assertFalse(first.metadata["refit_on_full_training_split"])
        self.assertFalse(first.metadata["external_validation_used_for_early_stopping"])


    def test_onishi_cnn_matches_reference_architecture(self):
        import torch

        model = OnishiPairedCNN(
            {
                "channels": [32, 64, 64],
                "kernels": [5, 3, 3],
                "pool_size": 3,
                "dense_units": 256,
            }
        )
        conv_layers = [
            layer for layer in model.features
            if isinstance(layer, torch.nn.Conv2d)
        ]
        pool_layers = [
            layer for layer in model.features
            if isinstance(layer, torch.nn.MaxPool2d)
        ]
        self.assertEqual(
            [layer.kernel_size for layer in conv_layers],
            [(2, 5), (1, 3), (1, 3)],
        )
        self.assertEqual(
            [layer.out_channels for layer in conv_layers],
            [32, 64, 64],
        )
        self.assertEqual(len(pool_layers), 3)

    def test_onishi_cnn_forward_and_xai_shape(self):
        rng = np.random.default_rng(14)
        pair = rng.normal(size=(6, 2, 64)).astype(np.float32)
        model = OnishiPairedCNN(
            {
                "channels": [32, 64, 64],
                "kernels": [5, 3, 3],
                "pool_size": 3,
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


    def test_mlp_pair_antisymmetry(self):
        rng = np.random.default_rng(16)
        pair = rng.normal(size=(8, 2, 24)).astype(np.float32)
        artifact = MLPArtifact(SharedScorerMLP(24, [8, 4], "silu"), "cpu", {})
        forward = predict_mlp(artifact, pair)
        reverse = predict_mlp(artifact, pair[:, ::-1, :])
        np.testing.assert_allclose(forward, -reverse, rtol=1e-6, atol=1e-6)

    def test_mlp_2d_joint_pair_forward_shape(self):
        rng = np.random.default_rng(17)
        pair = rng.normal(size=(7, 2, 24)).astype(np.float32)
        artifact = MLPArtifact(JointPairMLP(24, [8, 4], "relu"), "cpu", {})
        prediction = predict_mlp_2d(artifact, pair)
        self.assertEqual(prediction.shape, (7,))
        self.assertTrue(np.all(np.isfinite(prediction)))

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
            {"architecture", "activation", "learning_rate", "batch_size", "weight_decay"},
        )


    def test_corrected_timing_is_slide_target_minus_prediction(self):
        target = np.asarray([35.0, -20.0, 5.0])
        prediction = np.asarray([10.0, -5.0, 8.0])
        np.testing.assert_allclose(
            corrected_timing_residual(target, prediction),
            [25.0, -15.0, -3.0],
        )
