"""Capacity experiment checks: deterministic disjoint subset and model contract."""
import numpy as np
import pytest

torch = pytest.importorskip("torch")

from waveform_analysis.scripts.test_temporal_cnn_capacity import training_rows
from waveform_analysis.scripts.evaluate_full_signal_distillation import full_features, evaluate
from waveform_analysis.engine.temporal_conditional_cnn import TemporalConditionalCNN


def test_fixed_memorization_cohort_excludes_validation_and_holdout():
    n = 1200
    train, selected = training_rows(n, seed=42, tuning_queries=50,
                                     holdout_queries=50, fit_events=256)
    train_again, selected_again = training_rows(
        n, seed=42, tuning_queries=50, holdout_queries=50, fit_events=256)
    np.testing.assert_array_equal(selected, selected_again)
    np.testing.assert_array_equal(train, train_again)
    assert len(train) == 1100
    assert len(selected) == 256
    assert len(np.unique(selected)) == len(selected)
    assert set(selected).issubset(set(train))


def test_derivative_capacity_model_prediction_is_same_event_eval():
    rng = np.random.default_rng(11)
    waves = rng.normal(size=(7, 2, 64)).astype(np.float32)
    labels = rng.normal(size=7)
    network = TemporalConditionalCNN(64, width=16, depth=1, bins=4,
                                      use_derivative=True)
    mean = torch.zeros(2, 64)
    scale = torch.ones(2, 64)
    features = full_features(torch.from_numpy(waves[:3]), mean, scale,
                             derivative_mean=mean, derivative_scale=scale)
    assert features.shape == (3, 6 * 64)
    with torch.no_grad():
        baseline = network(features).numpy()
    reconstructed, metrics = evaluate(
        network, waves, labels, np.arange(3), "cpu", 2, 0., 1.,
        mean, scale, mean, scale)
    np.testing.assert_allclose(reconstructed, baseline, atol=1e-5, rtol=1e-5)
    assert metrics["n_events"] == 3


def test_insufficient_train_events_rejected():
    with pytest.raises(ValueError, match="Insufficient"):
        training_rows(120, seed=42, tuning_queries=50,
                      holdout_queries=50, fit_events=30)


def test_plateau_scheduler_reduces_lr_after_nonimprovements():
    optimizer = torch.optim.Adam([torch.nn.Parameter(torch.ones(()))], lr=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=2,
        threshold=0.05, threshold_mode="abs", min_lr=1e-6)
    for rmse in (4., 3.9, 3.95, 3.91, 3.93):
        scheduler.step(rmse)
    assert optimizer.param_groups[0]["lr"] == pytest.approx(5e-5)
    scheduler.step(2.0)
    assert optimizer.param_groups[0]["lr"] == pytest.approx(5e-5)
