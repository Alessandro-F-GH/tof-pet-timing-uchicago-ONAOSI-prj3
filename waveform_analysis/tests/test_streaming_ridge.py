"""Regression tests for bounded-RAM final composite Ridge fitting."""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import pytest
from sklearn.linear_model import Ridge

from waveform_analysis.engine.streaming_ridge import fit_streamed_composite_ridge


@dataclass
class _Frozen:
    feature_blocks: tuple
    block_names: tuple = ("multirocket", "rdst")


@pytest.mark.parametrize("shared", [False, True])
def test_streamed_lsqr_matches_dense_penalized_composite(shared):
    rng = np.random.default_rng(104)
    n, p, q = 109, 13, 7
    a = rng.normal(size=(n, p)).astype(np.float32)
    b = rng.normal(size=(n, q)).astype(np.float32)
    y = rng.normal(size=n)
    params = {"multirocket_alpha": 4., "rdst_alpha": 17.}
    scale = np.sqrt(params["multirocket_alpha"] / params["rdst_alpha"])
    dense = np.concatenate([a, b * scale], axis=1)
    expected = Ridge(alpha=params["multirocket_alpha"],
                     fit_intercept=not shared, solver="lsqr", tol=1e-4).fit(dense, y)
    expected_prediction = expected.predict(dense)
    result = fit_streamed_composite_ridge(
        _Frozen((a, b)), y, params, shared=shared, rows_per_chunk=11,
    )
    actual = result.regressor.predict(np.concatenate([a, b], axis=1))
    np.testing.assert_allclose(actual, expected_prediction, atol=4e-3, rtol=4e-3)
    assert result.metadata["streamed_rows_per_chunk"] == 11
    assert result.metadata["feature_count"] == p + q
    assert result.metadata["solver"] == "streamed_lsqr"


def test_streamed_lsqr_rejects_invalid_penalty():
    frozen = _Frozen((np.ones((5, 2), dtype=np.float32),
                      np.ones((5, 3), dtype=np.float32)))
    with pytest.raises(ValueError, match="positive"):
        fit_streamed_composite_ridge(
            frozen, np.ones(5), {"multirocket_alpha": 0., "rdst_alpha": 3.}
        )
