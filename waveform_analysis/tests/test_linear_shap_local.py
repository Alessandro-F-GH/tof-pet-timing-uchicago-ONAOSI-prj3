"""Exact local Linear SHAP and additive grouped-attribution tests."""
import numpy as np
import pytest

from waveform_analysis.engine.linear_shap import (
    development_mean,
    grouped_contributions,
    linear_attributions,
)


@pytest.mark.parametrize("shared", [False, True])
def test_exact_linear_attributions_reconstruct_prediction(shared):
    rng = np.random.default_rng(3)
    pair = rng.normal(size=(5, 2, 7))
    mean_pair = rng.normal(size=(2, 7))
    weights = rng.normal(size=7 if shared else 14)
    phi, base, predicted = linear_attributions(
        pair, mean_pair, weights, shared=shared, intercept=1.4
    )
    if shared:
        expected = (pair[:, 0] - pair[:, 1]) @ weights + 1.4
        np.testing.assert_allclose(phi[:, 0] + phi[:, 1],
                                   (pair[:, 0] - pair[:, 1] - mean_pair[0] + mean_pair[1]) * weights)
    else:
        expected = pair.reshape(5, -1) @ weights + 1.4
    np.testing.assert_allclose(predicted, expected, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(base + phi.sum(axis=(1, 2)), predicted, atol=1e-12)


def test_grouped_signed_contributions_keep_partial_tail():
    phi = np.arange(30, dtype=float).reshape(1, 2, 15)
    starts, grouped = grouped_contributions(phi, 8)
    np.testing.assert_array_equal(starts, [0, 8])
    assert grouped.shape == (1, 2, 2)
    np.testing.assert_allclose(grouped.sum(axis=-1), phi.sum(axis=-1))


def test_invalid_feature_shapes_are_rejected():
    with pytest.raises(ValueError, match="coefficient"):
        linear_attributions(np.ones((1, 2, 7)), np.ones((2, 7)),
                            np.ones(9), shared=True)


def test_masked_development_mean_matches_full_population():
    class Dummy:
        n_events = 5

    # Only the reusable array-view function is replaced: all 5 events must
    # contribute, not a sample of the blind population.
    from unittest.mock import patch
    data = np.arange(5 * 2 * 6, dtype=float).reshape(5, 2, 6)
    class View:
        def __init__(self, indices):
            self.indices = indices
        def materialize(self):
            return data[self.indices]
    with patch("waveform_analysis.engine.linear_shap.waveform_view",
               side_effect=lambda _dataset, _mode, indices: View(indices)):
        result = development_mean(Dummy(), "energy_to_energy",
                                  np.array([1, 0, 1, 0, 1, 0], dtype=bool),
                                  chunk_size=2)
    np.testing.assert_allclose(result, data[:, :, [0, 2, 4]].mean(axis=0))
