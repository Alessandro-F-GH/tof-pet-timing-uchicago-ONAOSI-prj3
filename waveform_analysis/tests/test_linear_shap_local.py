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
    phi = np.array([[[2., -5., 4., -1., 7.],
                     [-3., 6., -2., 8., -9.]]])
    starts, positive, negative = grouped_contributions(phi, 3)
    np.testing.assert_array_equal(starts, [0, 3])
    assert positive.shape == negative.shape == (1, 2, 2)
    np.testing.assert_allclose(positive[0], [[6., 7.], [6., 8.]])
    np.testing.assert_allclose(negative[0], [[-5., -1.], [-5., -9.]])
    assert np.all(positive >= 0)
    assert np.all(negative <= 0)
    np.testing.assert_allclose((positive + negative).sum(axis=-1), phi.sum(axis=-1))


def test_grouping_does_not_cancel_opposite_signs():
    phi = np.array([[[4.0, -4.0], [-7.0, 7.0]]])
    starts, positive, negative = grouped_contributions(phi, 2)
    np.testing.assert_array_equal(starts, [0])
    np.testing.assert_allclose(positive[0, :, 0], [4., 7.])
    np.testing.assert_allclose(negative[0, :, 0], [-4., -7.])


def test_local_linear_plot_shows_separate_detector_panels_without_difference(tmp_path):
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib.axes import Axes
    from unittest.mock import patch
    from waveform_analysis.engine.tshap_explainer import _plot_local_event

    seen = []
    old_title = Axes.set_title

    def record_title(self, label, *args, **kwargs):
        seen.append(label)
        return old_title(self, label, *args, **kwargs)

    with patch.object(Axes, "set_title", record_title):
        _plot_local_event(
            tmp_path / "direct.pdf",
            time_ps=np.arange(10, dtype=float) * 12.5,
            waveform_mv=np.ones((2, 10)),
            dataset_mean_difference_mv=np.zeros(10),
            led_baseline_mv=None,
            attribution=np.zeros((2, 10)),
            group="typical", event_id=10,
            target_ps=2.0, prediction_ps=1.0,
            improvement_ps=1.0, led_threshold_mv=5.0,
            window_starts=np.array([0, 5]),
            window_samples=5,
            positive_window_scores=np.ones((2, 2)),
            negative_window_scores=-np.ones((2, 2)),
            attribution_method="Linear SHAP",
            separate_detector_importance=True,
        )
    assert (tmp_path / "direct.pdf").is_file()
    assert "Detector 1 Linear SHAP" in seen
    assert "Detector 2 Linear SHAP" in seen
    assert not any("difference" in title.lower() for title in seen)


def test_shared_linear_and_original_tshap_keep_difference_panel(tmp_path):
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib.axes import Axes
    from unittest.mock import patch
    from waveform_analysis.engine.tshap_explainer import _plot_local_event

    for method in ("shared", "tshap"):
        titles = []
        original = Axes.set_title

        def record(self, value, *args, **kwargs):
            titles.append(value)
            return original(self, value, *args, **kwargs)

        kwargs = dict(
            path=tmp_path / f"{method}.pdf",
            time_ps=np.arange(10, dtype=float) * 12.5,
            waveform_mv=np.ones((2, 10)),
            dataset_mean_difference_mv=np.zeros(10),
            led_baseline_mv=None,
            attribution=np.zeros((2, 10)),
            group="typical", event_id=22,
            target_ps=3.0, prediction_ps=2.0, improvement_ps=1.0,
            led_threshold_mv=5.0,
            window_starts=np.array([0, 5]), window_samples=5,
        )
        if method == "shared":
            kwargs["positive_window_scores"] = np.array([[1., 2.]])
            kwargs["negative_window_scores"] = np.array([[-2., -1.]])
            kwargs["attribution_method"] = "Linear SHAP"
        else:
            kwargs["window_scores"] = np.array([[2., -1.]])
        with patch.object(Axes, "set_title", record):
            _plot_local_event(**kwargs)
        assert (tmp_path / f"{method}.pdf").is_file()
        assert "Detector difference relative to dataset mean" in titles


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
