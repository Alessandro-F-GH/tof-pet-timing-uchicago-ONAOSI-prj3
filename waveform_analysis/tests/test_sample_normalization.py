"""Streaming time-sample standardization without dataset-wide materialization."""
import numpy as np
import pytest

torch = pytest.importorskip("torch")
from waveform_analysis.engine.sample_normalization import fit_time_sample_stats, tensor_stats
from waveform_analysis.engine.nn_olsen import augment, coalitions, paired_batch, predict_coalitions
from waveform_analysis.engine.temporal_conditional_cnn import TemporalConditionalCNN
from waveform_analysis.engine.conditional_knn import tshap_windows


def test_training_only_statistics_and_std_floor():
    rng = np.random.default_rng(9)
    waves = rng.normal(size=(15, 2, 32)).astype("float32")
    waves[:, :, 1] = 5.0  # zero variability: must remain finite
    waves[12:] += 1000.0  # validation/holdout must never enter fitting
    idx = np.arange(12)
    mean, scale, diag = fit_time_sample_stats(waves, idx, chunk_size=5,
                                              scale_floor_ratio=0.1)
    np.testing.assert_allclose(mean, waves[idx].mean(axis=0), atol=2e-6)
    train_std = waves[idx].astype("float64").std(axis=0)
    np.testing.assert_allclose(scale, np.maximum(train_std,
        np.sqrt((train_std ** 2).mean(axis=1))[:, None] * .1), rtol=1e-5, atol=1e-6)
    assert np.all(scale > 0)
    assert diag["fraction_floored"][0] > 0


def test_standardize_before_masking_and_do_not_leak_hidden_values():
    rng = np.random.default_rng(5)
    data = rng.normal(size=(18, 2, 64)).astype("float32")
    mean, scale, _ = fit_time_sample_stats(data, np.arange(16), chunk_size=4)
    tm, ts = tensor_stats(mean, scale, device="cpu")
    waves = torch.from_numpy(data[:3].copy())
    mask = torch.zeros(3, 64)
    mask[:, 20:40] = 1
    encoded = augment(waves, mask, sample_mean=tm, sample_scale=ts)
    expected = (waves - tm.unsqueeze(0)) / ts.unsqueeze(0)
    torch.testing.assert_close(encoded[:, :128].reshape(3, 2, 64),
                               expected * mask[:, None, :])
    modified = waves.clone()
    modified[:, :, :20] = 1e6
    modified[:, :, 40:] = -1e6
    torch.testing.assert_close(encoded, augment(modified, mask,
        sample_mean=tm, sample_scale=ts), rtol=0, atol=0)


def test_train_and_inference_share_normalization():
    rng = np.random.default_rng(2)
    waves = rng.normal(size=(9, 2, 64)).astype("float32")
    mean, scale, _ = fit_time_sample_stats(waves, np.arange(6))
    tm, ts = tensor_stats(mean, scale, device="cpu")
    starts, keep = tshap_windows(64, 8, 8)
    masks = coalitions(starts, 8, keep)
    selected = torch.tensor([0, 2, 4])
    original = torch.tensor([1., 2., 3.])
    encoded, targets = paired_batch(torch.from_numpy(waves[:3]), original,
        selected, masks, sample_mean=tm, sample_scale=ts)
    assert encoded.shape == (6, 256)
    torch.testing.assert_close(targets, original.repeat_interleave(2))
    model = TemporalConditionalCNN(64, width=16, depth=1, bins=4)
    model.eval()
    with torch.no_grad():
        batch_predictions = model(encoded).reshape(3, 2).numpy()
    # Compare direct paired-batch evaluation with the generic inference helper.
    by_helper = predict_coalitions(model, waves[:3], starts, 8, keep,
        batch_size=2, device="cpu", sample_mean=tm, sample_scale=ts)
    np.testing.assert_allclose(by_helper[np.arange(3), selected.numpy()],
                               batch_predictions, rtol=1e-5, atol=1e-5)
