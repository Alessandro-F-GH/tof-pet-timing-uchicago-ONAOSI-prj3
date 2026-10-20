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


@pytest.mark.parametrize("n_channels", [1, 2])
def test_exact_ppv_reconstruction_against_aeon_160(n_channels):
    """Compare against actual fitted aeon, including scaler and all PPV blocks."""
    aeon = pytest.importorskip("aeon")
    from aeon.transformations.collection.convolution_based import MultiRocket
    from sklearn.preprocessing import StandardScaler
    from waveform_analysis.engine.rocket_ppv_lrp import _params, _ppv_columns
    rng = np.random.default_rng(47)
    x = rng.standard_normal((8, n_channels, 72)).astype(np.float32)
    transformer = MultiRocket(n_kernels=84, max_dilations_per_kernel=4,
                               n_features_per_kernel=4, random_state=7, n_jobs=1)
    transformed = np.asarray(transformer.fit_transform(x), dtype=np.float32)
    scaler = StandardScaler(with_mean=False).fit(transformed)
    scaled = scaler.transform(transformed)
    artifact = SimpleNamespace(transformer=transformer, scaler=scaler,
                               metadata={"feature_count": transformed.shape[1]})
    raw_cols, diff_cols, raw, diff = _ppv_columns(artifact)
    for row in (0, 3, 7):
        for rep, columns, params, values, trim in (
                ("raw", raw_cols, raw, x[row], 0),
                ("difference", diff_cols, diff, np.diff(x[row], axis=-1), 1)):
            recovered, *_ = _trace_rep(
                np.asarray(values, dtype=np.float32),
                *params, np.zeros(len(columns)), scaler.scale_[columns], trim)
            # In a valid tracer, every PPV output matches the true frozen
            # transformer. Any discrepancy blocks model explanations.
            np.testing.assert_allclose(
                recovered, scaled[row, columns], atol=2e-4, rtol=2e-4,
                err_msg=f"{n_channels} channels, {rep}, event {row}")
