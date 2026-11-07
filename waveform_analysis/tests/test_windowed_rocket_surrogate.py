"""Windowed Rocket/RDST surrogate structural and disk-backed Ridge tests."""
import numpy as np
import pytest

from waveform_analysis.scripts.evaluate_windowed_rocket_surrogate import (
    BlockRidgeOperator, disjoint_windows, select_columns, split_budgets,
)


def test_windows_cover_signal_with_final_short_tail_merged():
    assert disjoint_windows(4159, 256)[0] == (0, 256)
    windows = disjoint_windows(4159, 256)
    assert len(windows) == 17
    assert windows[-1] == (4096, 4159)
    assert disjoint_windows(513, 256) == [(0, 256), (256, 513)]
    assert disjoint_windows(520, 256) == [(0, 256), (256, 520)]
    assert disjoint_windows(516, 256) == [(0, 256), (256, 516)]


def test_feature_allocation_and_columns():
    assert split_budgets(256, "both") == (128, 128)
    assert split_budgets(255, "both") == (127, 128)
    assert split_budgets(256, "rdst") == (0, 256)
    assert split_budgets(256, "multirocket") == (256, 0)
    np.testing.assert_array_equal(select_columns(5, 10), np.arange(5))
    np.testing.assert_array_equal(select_columns(7, 3), [0, 3, 6])


def test_disk_style_ridge_operator_matches_centered_matrix():
    rng = np.random.default_rng(10)
    blocks = [rng.normal(size=(30, 4)).astype(np.float32),
              rng.normal(size=(30, 3)).astype(np.float32)]
    train = np.arange(5, 25, 2)
    op = BlockRidgeOperator(blocks, train, chunk_size=3)
    x = np.concatenate(blocks, axis=1).astype(np.float64)
    centered = x[train] - x[train].mean(axis=0)
    weight = rng.normal(size=7)
    residual = rng.normal(size=len(train))
    np.testing.assert_allclose(op.matvec(weight), centered @ weight, atol=1e-12)
    np.testing.assert_allclose(op.rmatvec(residual), centered.T @ residual, atol=1e-12)
    np.testing.assert_allclose(op.predict(weight, np.array([2, 5, 9])),
                               x[[2, 5, 9]] @ weight, atol=1e-12)


def test_disk_style_lsqr_agrees_with_sklearn_ridge():
    from scipy.sparse.linalg import lsqr
    from sklearn.linear_model import Ridge

    rng = np.random.default_rng(11)
    blocks = [rng.normal(size=(28, 5)).astype(np.float32),
              rng.normal(size=(28, 3)).astype(np.float32)]
    y = rng.normal(size=28)
    train = np.arange(22)
    op = BlockRidgeOperator(blocks, train, chunk_size=6)
    coefficients = lsqr(op.scipy_operator(), y[train] - y[train].mean(),
                        damp=np.sqrt(3.0), atol=1e-12, btol=1e-12,
                        iter_lim=150)[0]
    intercept = y[train].mean() - op.col_mean @ coefficients
    x = np.concatenate(blocks, axis=1)
    expected = Ridge(alpha=3.0).fit(x[train], y[train])
    np.testing.assert_allclose(intercept + op.predict(coefficients, np.arange(22, 28)),
                               expected.predict(x[22:]), rtol=1e-5, atol=1e-5)
