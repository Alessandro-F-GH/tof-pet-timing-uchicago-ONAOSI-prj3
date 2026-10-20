"""Tests for ROCKET-LRP PPV pooling weights and frozen feature layout."""
import numpy as np
import pytest
from types import SimpleNamespace

from waveform_analysis.engine.rocket_ppv_lrp import _trace_rep, _ppv_columns


def test_ppv_uniform_relevance_and_zero_positive_mass():
    # One 9-tap kernel with 3 selected positive taps; zero-valued input
    # yields C(t) = 0 so bias -1 activates every response.
    x = np.zeros((1, 21), dtype=np.float32)
    indices = np.array([[0, 1, 2]], dtype=np.int32)
    offsets = np.array([0, 1], dtype=np.int32)
    dilations = np.array([1], dtype=np.int32)
    features_per_dilation = np.array([2], dtype=np.int32)
    biases = np.array([-1., 1.], dtype=np.float32)
    relevance = np.array([4., -7.])
    scales = np.ones(2)
    ppv, signed, absolute, count, missing_abs, missing_signed = _trace_rep(
        x, np.array([1], dtype=np.int32), np.array([0], dtype=np.int32),
        offsets, dilations, features_per_dilation, biases,
        indices, relevance, scales, 0)
    assert ppv.tolist() == pytest.approx([1., 0.])
    assert signed.sum() == pytest.approx(4.)
    assert absolute.sum() == pytest.approx(4.)
    assert count.sum() == 21
    assert missing_abs == pytest.approx(7.)
    assert missing_signed == pytest.approx(-7.)
    np.testing.assert_allclose(signed, np.full(21, 4./21))


def test_ppv_output_blocks_match_aeon_order():
    raw = (np.array([1], dtype=np.int32), np.array([0], dtype=np.int32),
           np.array([2], dtype=np.int32), np.array([1], dtype=np.int32),
           np.array([0.], dtype=np.float32))
    diff = raw
    transform = SimpleNamespace(parameter=raw, parameter1=diff,
                                _indices=np.array([[0, 1, 2]], dtype=np.int32))
    artifact = SimpleNamespace(transformer=transform,
                               metadata={"feature_count": 8})
    col_raw, col_diff, *_ = _ppv_columns(artifact)
    assert col_raw.tolist() == [0]
    assert col_diff.tolist() == [4]
