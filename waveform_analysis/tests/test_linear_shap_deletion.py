"""Numerical checks for analytical Ridge Linear SHAP deletion."""
import numpy as np
import pytest

from waveform_analysis.engine.linear_shap import linear_attributions
from waveform_analysis.engine.linear_shap_deletion import deletion_predictions


@pytest.mark.parametrize("shared", [False, True])
def test_deletion_reproduces_explicit_mean_replacement(shared):
    rng = np.random.default_rng(17)
    n_events, n_samples = 13, 11
    pair = rng.normal(size=(n_events, 2, n_samples))
    mean = rng.normal(size=(2, n_samples))
    coefficients = rng.normal(size=n_samples if shared else 2 * n_samples)
    intercept = 3.0
    phi, baseline, prediction = linear_attributions(
        pair, mean, coefficients, shared=shared, intercept=intercept
    )
    starts = np.array([0, 4, 8])
    grouped = np.add.reduceat(phi, starts, axis=2)
    grouped = grouped.sum(axis=1) if shared else grouped.reshape(n_events, -1)
    importance = rng.random(grouped.shape[1])
    order = np.argsort(-importance)
    result = deletion_predictions(prediction, grouped, order)
    assert np.allclose(result[:, 0], prediction)
    assert np.allclose(result[:, -1], baseline)

    for step in range(1, len(order) + 1):
        replaced = pair.copy()
        for index in order[:step]:
            group = index % len(starts)
            lo = starts[group]
            hi = starts[group + 1] if group + 1 < len(starts) else n_samples
            if shared:
                replaced[:, :, lo:hi] = mean[:, lo:hi]
            else:
                detector = index // len(starts)
                replaced[:, detector, lo:hi] = mean[detector, lo:hi]
        _, _, brute = linear_attributions(
            replaced, mean, coefficients, shared=shared, intercept=intercept
        )
        assert np.allclose(result[:, step], brute, atol=1e-10)


def test_clipping_is_applied_after_subtracting_attributions():
    contribution = np.array([[8.0, -13.0], [-15.0, 9.0]])
    original = np.array([12.0, -9.0])
    computed = deletion_predictions(
        original, contribution, [1, 0], limit=10.0,
    )
    assert np.allclose(computed[:, 0], [10.0, -9.0])
    assert np.allclose(computed[:, 1], [10.0, -10.0])
    assert np.allclose(computed[:, -1], [10.0, -3.0])


def test_invalid_order_is_rejected():
    with pytest.raises(ValueError, match="ordering"):
        deletion_predictions(np.zeros(2), np.ones((2, 3)), [0, 0, 2])
