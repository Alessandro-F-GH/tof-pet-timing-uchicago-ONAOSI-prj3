import unittest
import numpy as np

from waveform_analysis.ml_pipeline.models.cnn import CNNArtifact, SharedScorerCNN, predict as predict_cnn
from waveform_analysis.ml_pipeline.models.linear_svr import fit as fit_svr, predict as predict_svr


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
