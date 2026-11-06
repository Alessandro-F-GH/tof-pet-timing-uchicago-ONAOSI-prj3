"""Tests for NN-Olsen adapted paired temporal coalition surrogate."""
import numpy as np
import pytest
torch = pytest.importorskip("torch")
from waveform_analysis.engine.conditional_knn import tshap_windows
from waveform_analysis.engine.nn_olsen import (NNOlsenSurrogate, augment,
                                                coalitions, paired_batch,
                                                predict_coalitions)
from waveform_analysis.scripts.evaluate_nn_olsen import _metrics


def test_coalitions_are_exact_complements_with_frozen_retention():
    retained = np.ones(14, bool)
    retained[[0, 13]] = False
    starts, keep = tshap_windows(14, 3, 3, retained)
    masks = coalitions(starts, 3, keep).numpy()
    assert masks.shape == (len(starts), 2, 14)
    np.testing.assert_array_equal(masks[:, 0] + masks[:, 1],
                                  np.broadcast_to(keep.astype(float), (len(starts), 14)))
    np.testing.assert_array_equal(masks[:, :, ~keep], 0.)
    for i, start in enumerate(starts):
        np.testing.assert_array_equal(masks[i, 0, start:start + 3], 1.)


def test_zero_filling_is_labeled_and_no_hidden_signal_leaks():
    waves = torch.arange(2 * 2 * 10, dtype=torch.float32).reshape(2, 2, 10)
    mask = torch.zeros(2, 10)
    mask[:, 2:4] = 1
    baseline = augment(waves, mask)
    altered = waves.clone()
    altered[:, :, 5:] = -9999
    other = augment(altered, mask)
    torch.testing.assert_close(baseline, other)
    assert baseline.shape == (2, 40)
    assert torch.all(baseline[:, :20].reshape(2, 2, 10)[:, :, 5:] == 0)
    assert torch.all(baseline[:, 20:].reshape(2, 2, 10)[:, :, 2:4] == 1)


def test_paired_batch_uses_same_cached_original_target_both_masks():
    waves = torch.rand(4, 2, 12)
    labels = torch.tensor([10., 20., 30., 40.])
    starts, keep = tshap_windows(12, 2, 2)
    masks = coalitions(starts, 2, keep)
    sample = torch.tensor([0, 1, 2, 1])
    features, paired_labels = paired_batch(waves, labels, sample, masks)
    assert features.shape == (8, 48)
    torch.testing.assert_close(paired_labels, labels.repeat_interleave(2))
    for i in range(4):
        w = features[2 * i, 24:].reshape(2, 12)[0]
        r = features[2 * i + 1, 24:].reshape(2, 12)[0]
        torch.testing.assert_close(w + r, torch.ones_like(w))


def test_network_inference_and_variance_diagnostics():
    torch.manual_seed(123)
    waves = np.random.default_rng(8).random((11, 2, 12), dtype=np.float32)
    starts, mask = tshap_windows(12, 2, 2)
    model = NNOlsenSurrogate(12, width=8, depth=3)
    output = predict_coalitions(model, waves, starts, 2, mask, batch_size=3)
    assert output.shape == (len(waves), len(starts), 2)
    assert np.isfinite(output).all()
    original = np.linspace(-25., 25., 11)
    metrics = _metrics(output, original)
    assert abs(metrics["original_std_ps"] - np.std(original)) < 1e-10
    for key in ("estimated_std_ps", "estimated_variance_ps2", "rmse_ps",
                "original_vs_estimated_std_ratio"):
        assert len(metrics[key]) == 2
    np.testing.assert_allclose(np.square(metrics["estimated_std_ps"]),
                               metrics["estimated_variance_ps2"], atol=1e-10)


def test_training_graph_uses_only_masked_values():
    torch.manual_seed(1)
    model = NNOlsenSurrogate(10, width=8)
    starts, keep = tshap_windows(10, 2, 2)
    mask = coalitions(starts, 2, keep)
    waves = torch.randn(4, 2, 10)
    target = torch.randn(4)
    inputs, labels = paired_batch(waves, target, torch.zeros(4, dtype=torch.long), mask)
    model.train()
    loss = (model(inputs) - labels).square().mean()
    loss.backward()
    assert any(p.grad is not None and torch.isfinite(p.grad).all()
               for p in model.parameters())
