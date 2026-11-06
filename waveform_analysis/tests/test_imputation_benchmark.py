"""Synthetic consistency checks for the joint-PPCA imputation benchmark."""
import numpy as np

from waveform_analysis.models.ppca_imputer import ProbabilisticPCA
from waveform_analysis.scripts.evaluate_imputation_benchmark import (
    conditional_mean, bootstrap_rmse, METHODS,
)


def test_benchmark_methods_no_knn_hermite_or_blend():
    assert METHODS == ("ppca_joint", "median", "original_tshap")


def test_joint_ppca_conditions_on_both_detectors_and_both_mask_types():
    rng = np.random.default_rng(13)
    n_events, length = 120, 16
    z = rng.normal(size=(n_events, 1))
    # Detectors are strongly correlated through their common latent pulse.
    a = z * np.linspace(1.0, 2.0, length) + rng.normal(0, .04, (n_events, length))
    b = z * np.linspace(1.8, .6, length) + rng.normal(0, .04, (n_events, length))
    waves = np.concatenate((a, b), axis=1)
    model = ProbabilisticPCA(2).fit(waves[:100])
    test = waves[100:]
    for keep_only in (False, True):
        segment = np.zeros(length, bool)
        segment[4:9] = True
        observed = np.tile(segment if keep_only else ~segment, 2)
        missing = ~observed
        prediction = conditional_mean(model, test, observed, missing)
        assert prediction.shape == (len(test), int(np.sum(missing)))
        assert np.isfinite(prediction).all()
        # The joint conditional estimator should beat an unconditional median.
        baseline = np.median(waves[:100], axis=0)[missing]
        assert np.mean((prediction - test[:, missing]) ** 2) < np.mean(
            (baseline - test[:, missing]) ** 2
        )


def test_joint_ppca_uses_cross_detector_information():
    rng = np.random.default_rng(21)
    latent = rng.normal(size=160)
    one = latent[:, None] * np.ones((160, 8)) + rng.normal(0, .02, (160, 8))
    two = 2 * latent[:, None] * np.ones((160, 8)) + rng.normal(0, .02, (160, 8))
    waves = np.concatenate([one, two], axis=1)
    model = ProbabilisticPCA(1).fit(waves[:130])
    observed = np.zeros(16, bool)
    observed[:8] = True
    missing = ~observed
    prediction = conditional_mean(model, waves[130:], observed, missing)
    assert np.mean((prediction - waves[130:, missing]) ** 2) < 0.1


def test_bootstrap_rmse_finite_interval():
    low, high = bootstrap_rmse(np.arange(1, 11, dtype=float), 5, 42, 40)
    assert 0 < low <= high < 2
