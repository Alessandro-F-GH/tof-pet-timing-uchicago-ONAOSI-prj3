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
