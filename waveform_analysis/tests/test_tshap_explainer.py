"""Numerical and persistence checks for the bounded upstream TSHAP adapter."""
import json

import numpy as np
import pytest

from waveform_analysis.engine.tshap_explainer import (
    _choose_indices, _complete, _new_explainer, _publish_directory,
)


@pytest.mark.parametrize("roi", [False, True])
def test_tshap_matches_upstream_reference(roi):
    from tshap.tshap import TSHAPExplainer

    rng = np.random.default_rng(47)
    waveforms = rng.normal(size=(2, 2, 32)).astype(np.float32)
    baselines = rng.normal(size=(3, 2, 32)).astype(np.float32)
    def predict(x):
        x = np.asarray(x)
        return (x[:, 0, 3:13].sum(axis=1)
                - 0.4 * x[:, 1, 9:22].sum(axis=1)
                + 0.07 * x[:, 0, 4] * x[:, 1, 6])
    reference = TSHAPExplainer(window_length=8, stride=4, interpolation=True, roi=roi)
    bounded = _new_explainer(8, 4, roi, predict_batch_size=5)
    expected = reference.explain(waveforms, baselines, predict)
    actual = bounded.explain(waveforms, baselines, predict)
    for observed, original in zip(actual, expected):
        np.testing.assert_allclose(observed, original, rtol=1e-5, atol=1e-5)


def test_tshap_constant_prediction_is_zero():
    model = _new_explainer(6, 3, True, 4)
    x = np.ones((1, 2, 24), dtype=np.float32)
    refs = np.zeros((2, 2, 24), dtype=np.float32)
    maps = model.explain(x, refs, lambda v: np.full(len(v), 7.0))
    for value in maps:
        np.testing.assert_allclose(value, 0)


def test_event_selection_is_reproducible():
    np.testing.assert_array_equal(_choose_indices(75, 10, 14), _choose_indices(75, 10, 14))
    assert len(_choose_indices(75, 10, 14)) == 10
    np.testing.assert_array_equal(_choose_indices(3, 10, 14), [0, 1, 2])


def test_complete_identity_and_replace(tmp_path):
    target = tmp_path / "explanation"
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    (scratch / "summary.json").write_text(json.dumps({"identity_sha256": "123"}))
    for file in ("attributions.npz", "global_importance.csv", "global_importance.png"):
        (scratch / file).touch()
    _publish_directory(scratch, target, overwrite=False)
    assert _complete(target, "123")
    assert not _complete(target, "456")
    newer = tmp_path / "replacement"
    newer.mkdir()
    (newer / "summary.json").write_text(json.dumps({"identity_sha256": "456"}))
    for file in ("attributions.npz", "global_importance.csv", "global_importance.png"):
        (newer / file).touch()
    _publish_directory(newer, target, overwrite=True)
    assert _complete(target, "456")
