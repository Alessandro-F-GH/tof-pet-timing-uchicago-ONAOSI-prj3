"""Temporal CNN conditional surrogate: masking, gradients, paired output and layout."""
import numpy as np
import pytest

torch = pytest.importorskip("torch")
from waveform_analysis.engine.conditional_knn import tshap_windows
from waveform_analysis.engine.nn_olsen import augment, coalitions, paired_batch, predict_coalitions
from waveform_analysis.engine.temporal_conditional_cnn import TemporalConditionalCNN


def test_temporal_cnn_input_and_output():
    torch.manual_seed(4)
    net = TemporalConditionalCNN(n_samples=64, width=16, depth=2, bins=4)
    net.eval()
    waves = torch.randn(5, 2, 64)
    mask = torch.randint(0, 2, (5, 64)).float()
    with torch.no_grad():
        result = net(augment(waves, mask))
    assert result.shape == (5,)
    assert torch.isfinite(result).all()
    with pytest.raises(ValueError, match="4\\*T"):
        net(torch.randn(5, 255))


def test_hidden_signal_never_changes_output():
    torch.manual_seed(8)
    net = TemporalConditionalCNN(n_samples=64, width=16, depth=1, bins=4)
    net.eval()
    waves = torch.randn(4, 2, 64)
    mask = torch.zeros(4, 64)
    mask[:, 10:25] = 1
    altered = waves.clone()
    altered[:, :, 25:] += 1000
    with torch.no_grad():
        before = net(augment(waves, mask))
        after = net(augment(altered, mask))
    torch.testing.assert_close(before, after, rtol=0, atol=0)


def test_temporal_cnn_uses_identical_tshap_coalitions_and_targets():
    torch.manual_seed(22)
    starts, keep = tshap_windows(64, 8, 8)
    mask = coalitions(starts, 8, keep)
    wave = torch.randn(3, 2, 64)
    original_targets = torch.randn(3)
    encoded, paired_targets = paired_batch(
        wave, original_targets, torch.tensor([0, 4, 7]), mask
    )
    net = TemporalConditionalCNN(n_samples=64, width=16, depth=1, bins=4)
    net.train()
    output = net(encoded)
    assert output.shape == (6,)
    torch.testing.assert_close(paired_targets, original_targets.repeat_interleave(2))
    loss = (output - paired_targets).square().mean()
    loss.backward()
    assert torch.isfinite(loss)
    assert all(p.grad is not None and torch.isfinite(p.grad).all()
               for p in net.parameters())


def test_predict_coalitions_same_array_contract_as_nn_olsen():
    torch.manual_seed(1)
    starts, keep = tshap_windows(64, 8, 8)
    waves = np.random.default_rng(10).normal(size=(3, 2, 64)).astype("float32")
    net = TemporalConditionalCNN(n_samples=64, width=16, depth=1, bins=4)
    result = predict_coalitions(net, waves, starts, 8, keep,
                                batch_size=2, device="cpu")
    assert result.shape == (3, len(starts), 2)
    assert np.isfinite(result).all()
