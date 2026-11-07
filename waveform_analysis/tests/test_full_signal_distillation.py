"""Full-signal distillation: complete masks, train-only preprocessing, metrics."""
import numpy as np
import pytest

torch = pytest.importorskip("torch")
from waveform_analysis.engine.sample_normalization import fit_time_sample_stats, tensor_stats
from waveform_analysis.engine.temporal_conditional_cnn import TemporalConditionalCNN
from waveform_analysis.scripts.evaluate_full_signal_distillation import (
    full_features, full_metrics, evaluate,
)


def test_all_samples_are_observed_and_centered():
    rng = np.random.default_rng(4)
    data = rng.normal(size=(8, 2, 64)).astype(np.float32)
    mean, scale, _ = fit_time_sample_stats(data, np.arange(6), chunk_size=2)
    tm, ts = tensor_stats(mean, scale, device="cpu")
    waves = torch.from_numpy(data[:3])
    features = full_features(waves, tm, ts)
    assert features.shape == (3, 256)
    torch.testing.assert_close(features[:, :128].reshape(3, 2, 64),
                               (waves - tm.unsqueeze(0)) / ts.unsqueeze(0))
    torch.testing.assert_close(features[:, 128:], torch.ones(3, 128))


def test_full_metrics_and_inference():
    rng = np.random.default_rng(2)
    data = rng.normal(size=(9, 2, 64)).astype(np.float32)
    labels = rng.normal(size=9)
    model = TemporalConditionalCNN(64, width=16, depth=1, bins=4)
    prediction, metrics = evaluate(
        model, data, labels, np.array([6, 7, 8]), "cpu", 2, 0., 1., None, None
    )
    assert prediction.shape == (3,)
    assert np.isfinite(prediction).all()
    assert metrics["n_events"] == 3
    assert metrics["rmse_ps"] >= 0
    assert np.isfinite(metrics["mae_ps"])
    perfect = full_metrics(labels[:3], labels[:3])
    assert perfect["rmse_ps"] == 0.0
    assert perfect["correlation"] == pytest.approx(1.0)


def test_first_difference_full_signal_channels_and_gradients():
    from waveform_analysis.engine.sample_normalization import fit_first_difference_stats
    torch.manual_seed(14)
    x = torch.randn(6, 2, 64)
    raw = x.numpy()
    dmean, dscale, _ = fit_first_difference_stats(raw, np.arange(5), chunk_size=2)
    mean = torch.from_numpy(dmean)
    scale = torch.from_numpy(dscale)
    features = full_features(x[:3], None, None,
                             derivative_mean=mean, derivative_scale=scale)
    assert features.shape == (3, 6 * 64)
    view = features.reshape(3, 6, 64)
    torch.testing.assert_close(view[:, :2], x[:3])
    torch.testing.assert_close(view[:, 2:4], torch.ones(3, 2, 64))
    raw_diff = torch.zeros_like(x[:3])
    raw_diff[..., 1:] = x[:3, :, 1:] - x[:3, :, :-1]
    torch.testing.assert_close(view[:, 4:],
                               (raw_diff - mean) / scale, rtol=1e-5, atol=1e-5)
    network = TemporalConditionalCNN(64, width=16, depth=1, bins=4,
                                      use_derivative=True)
    assert network.stem[0].in_channels == 7
    result = network(features)
    assert result.shape == (3,)
    result.square().mean().backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all()
               for p in network.parameters())


def test_first_difference_is_not_derivative_of_z_scored_waveform():
    # The intended order is difference(raw x), THEN separately z-score dx.
    x = torch.tensor([[[1., 2., 5., 6., 11., 12., 14., 20.],
                       [2., 5., 7., 11., 12., 14., 15., 17.]]])
    mean = torch.full((2, 8), 3.)
    scale = torch.full((2, 8), 2.)
    dmean = torch.zeros(2, 8)
    dscale = torch.ones(2, 8)
    result = full_features(x, mean, scale, derivative_mean=dmean,
                           derivative_scale=dscale).reshape(1, 6, 8)
    torch.testing.assert_close(result[0, 4, 1:], x[0, 0, 1:] - x[0, 0, :-1])
    torch.testing.assert_close(result[0, 0], (x[0, 0] - 3) / 2)


def test_full_signal_wide_olsen_with_derivatives():
    from waveform_analysis.engine.nn_olsen import NNOlsenSurrogate
    torch.manual_seed(23)
    waves = torch.randn(4, 2, 64)
    mean = torch.zeros(2, 64)
    scale = torch.ones(2, 64)
    features = full_features(waves, mean, scale,
                             derivative_mean=mean, derivative_scale=scale)
    model = NNOlsenSurrogate(64, width=256, depth=3, use_derivative=True)
    assert model.input.in_features == 6 * 64
    predicted = model(features)
    assert predicted.shape == (4,)
    predicted.square().mean().backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all()
               for p in model.parameters())


def test_full_signal_tuning_scheduler_decay():
    # Scheduler reacts to the same eval-mode tuning RMSE used for checkpoint selection.
    parameter = torch.nn.Parameter(torch.zeros(()))
    optimizer = torch.optim.Adam([parameter], lr=0.001)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=2,
        threshold=0.05, threshold_mode="abs", cooldown=0, min_lr=1e-6)
    for rmse in [10.0, 9.9, 9.92, 9.91, 9.93]:
        scheduler.step(rmse)
    assert optimizer.param_groups[0]["lr"] == pytest.approx(0.0005)
    scheduler.step(8.0)
    assert optimizer.param_groups[0]["lr"] == pytest.approx(0.0005)
