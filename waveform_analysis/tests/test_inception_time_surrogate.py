"""Tests for full-signal InceptionTime black-box distillation backbone."""
import numpy as np
import pytest

torch = pytest.importorskip("torch")
from waveform_analysis.engine.inception_time_surrogate import InceptionTimeSurrogate
from waveform_analysis.scripts.evaluate_full_signal_distillation import (
    full_features, full_metrics,
)


def test_inception_time_forward_full_signal():
    torch.manual_seed(17)
    model = InceptionTimeSurrogate(n_samples=64, filters=4, bottleneck=8,
                                    depth=6, kernel_size=15)
    samples = torch.randn(3, 2, 64)
    features = full_features(samples, None, None)
    output = model(features)
    assert output.shape == (3,)
    assert torch.isfinite(output).all()
    assert torch.all(features[:, 128:] == 1)


def test_inception_time_backprop_and_small_batch():
    torch.manual_seed(11)
    model = InceptionTimeSurrogate(n_samples=48, filters=4, bottleneck=8,
                                    depth=3, kernel_size=15)
    data = torch.randn(2, 2, 48)
    predictions = model(full_features(data, None, None))
    loss = predictions.square().mean()
    loss.backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all()
               for p in model.parameters())
    with pytest.raises(ValueError, match="4\\*T"):
        model(torch.randn(2, 49))


def test_full_metrics_reference():
    reference = np.array([-2., 0., 3.])
    reconstructed = np.array([-1., 0., 2.])
    metrics = full_metrics(reference, reconstructed)
    assert metrics["rmse_ps"] == pytest.approx(np.sqrt(2. / 3.))
    assert metrics["mae_ps"] == pytest.approx(2. / 3.)


def test_position_aware_pooling_retains_temporal_order():
    # Confirm the ordered head distinguishes same features in different bins.
    from torch.nn import functional as F
    activations = torch.zeros(1, 4, 64)
    activations[:, 0, 4] = 2.0
    shifted = torch.zeros_like(activations)
    shifted[:, 0, 52] = 2.0
    left = torch.cat((F.adaptive_avg_pool1d(activations, 8),
                      F.adaptive_max_pool1d(activations, 8)), dim=1)
    right = torch.cat((F.adaptive_avg_pool1d(shifted, 8),
                       F.adaptive_max_pool1d(shifted, 8)), dim=1)
    assert not torch.equal(left, right)
    assert left.shape == (1, 8, 8)


def test_position_aware_regression_head_shape():
    model = InceptionTimeSurrogate(
        n_samples=64, filters=4, bottleneck=8, depth=3,
        kernel_size=15, bins=8
    )
    # Four filters per inception branch, then avg+max over eight time bins.
    assert model.output[0].in_features == 4 * 4 * 2 * 8
