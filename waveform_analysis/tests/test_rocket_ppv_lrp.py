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


def test_ppv_subset_skips_inactive_biases_and_conserves_selected_mass():
    x = np.zeros((1, 21), dtype=np.float32)
    indices = np.array([[0, 1, 2]], dtype=np.int32)
    args = (x, np.array([1], dtype=np.int32), np.array([0], dtype=np.int32),
            np.array([0, 1], dtype=np.int32), np.array([1], dtype=np.int32),
            np.array([2], dtype=np.int32), np.array([-1., 1.], dtype=np.float32),
            indices, np.array([4., -7.]), np.ones(2), 0)
    full, signed, absolute, _, missing_abs, _ = _trace_rep(*args)
    selected, subset_signed, subset_absolute, _, subset_missing, _ = _trace_rep(
        *args, selected_mask=np.array([True, False]))
    assert np.isnan(selected[1])
    assert selected[0] == pytest.approx(full[0])
    np.testing.assert_allclose(subset_signed, signed)
    np.testing.assert_allclose(subset_absolute, absolute)
    assert subset_missing == pytest.approx(0.)
    assert missing_abs == pytest.approx(7.)


def test_ppv_subset_matches_aeon_selected_columns_only():
    pytest.importorskip("aeon")
    from aeon.transformations.collection.convolution_based import MultiRocket
    from sklearn.preprocessing import StandardScaler
    from waveform_analysis.engine.rocket_ppv_lrp import _ppv_columns
    rng = np.random.default_rng(51)
    x = rng.standard_normal((8, 2, 72)).astype(np.float32)
    t = MultiRocket(n_kernels=84, max_dilations_per_kernel=4,
                    n_features_per_kernel=4, random_state=7, n_jobs=1)
    raw = np.asarray(t.fit_transform(x), dtype=np.float32)
    scaler = StandardScaler(with_mean=False).fit(raw)
    scaled = scaler.transform(raw)
    artifact = SimpleNamespace(transformer=t, scaler=scaler,
                               metadata={"feature_count": raw.shape[1]})
    raw_columns, diff_columns, params_raw, params_diff = _ppv_columns(artifact)
    for rep, columns, params, signal, trim in (
            ("raw", raw_columns, params_raw, x[0], 0),
            ("diff", diff_columns, params_diff, np.diff(x[0], axis=-1), 1)):
        selected = np.zeros(len(columns), dtype=np.bool_)
        selected[[1, 11, 25, 51, 70]] = True
        restored, *_ = _trace_rep(
            np.asarray(signal, dtype=np.float32), *params,
            np.ones(len(columns)), scaler.scale_[columns], trim,
            selected_mask=selected)
        assert np.isnan(restored[~selected]).all()
        np.testing.assert_allclose(restored[selected], scaled[0, columns[selected]],
                                   atol=2e-4, rtol=2e-4, err_msg=rep)
