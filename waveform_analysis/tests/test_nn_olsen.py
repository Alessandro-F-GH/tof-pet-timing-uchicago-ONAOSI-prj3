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


def test_wide_derivative_mlp_masks_hidden_neighbours():
    torch.manual_seed(31)
    t = 24
    network = NNOlsenSurrogate(t, width=256, depth=3, use_derivative=True)
    network.eval()
    assert network.input.in_features == 6 * t
    wave = torch.randn(3, 2, t)
    mask = torch.zeros(3, t)
    mask[:, 8:14] = 1.0
    derivative_mean = torch.zeros(2, t)
    derivative_scale = torch.ones(2, t)
    encoded = augment(wave, mask, derivative_mean=derivative_mean,
                      derivative_scale=derivative_scale)
    channels = encoded.reshape(3, 6, t)
    assert torch.all(channels[:, 4:, :9] == 0)
    assert torch.all(channels[:, 4:, 14:] == 0)
    torch.testing.assert_close(
        channels[:, 4:, 9:14], wave[:, :, 9:14] - wave[:, :, 8:13])
    changed = wave.clone()
    changed[:, :, :8] += 500
    changed[:, :, 14:] -= 500
    other = augment(changed, mask, derivative_mean=derivative_mean,
                    derivative_scale=derivative_scale)
    torch.testing.assert_close(encoded, other, atol=0, rtol=0)
    with torch.no_grad():
        torch.testing.assert_close(network(encoded), network(other),
                                   atol=0, rtol=0)


def test_derivative_paired_masks_keep_W_R_target_contract():
    torch.manual_seed(41)
    starts, keep = tshap_windows(24, 6, 6)
    masks = coalitions(starts, 6, keep)
    wave = torch.randn(4, 2, 24)
    target = torch.randn(4)
    dmean, dscale = torch.zeros(2, 24), torch.ones(2, 24)
    values, labels = paired_batch(
        wave, target, torch.tensor([0, 1, 2, 3]), masks,
        derivative_mean=dmean, derivative_scale=dscale)
    assert values.shape == (8, 6 * 24)
    torch.testing.assert_close(labels, target.repeat_interleave(2))
    model = NNOlsenSurrogate(24, width=32, use_derivative=True)
    outputs = model(values)
    outputs.square().mean().backward()
    assert outputs.shape == (8,)
    assert all(p.grad is not None and torch.isfinite(p.grad).all()
               for p in model.parameters())
