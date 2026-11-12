"""Development-only cumulative-prefix transform study checks."""
import numpy as np
from waveform_analysis.scripts.evaluate_progressive_windows import (
    temporal_cutoffs, fingerprint, alpha_grid, effective_alphas,
    original_feature_counts, regularized_feature_matrix,
)


def test_cumulative_prefixes_refit_on_full_window():
    time = np.arange(-2000, 4000, 12.5)
    windows = temporal_cutoffs(time, np.ones(len(time), bool), -2, 4, 1)
    assert len(windows) == 6
    assert [round(end) for _, end, _ in windows] == [-1, 0, 1, 2, 3, 4]
    for start, end, selected in windows:
        assert start == -2
        np.testing.assert_array_equal(
            selected, np.flatnonzero(time < end*1000 - 1e-5))
    assert len(windows[0][2]) < len(windows[-1][2])


def test_cumulative_prefix_respects_frozen_mask():
    time = np.arange(-2000, 4000, 12.5)
    mask = np.ones(len(time), bool)
    mask[(time >= 1000) & (time < 2000)] = False
    windows = temporal_cutoffs(time, mask, -2, 4, 1)
    assert all(mask[ids].all() for _, _, ids in windows)
    assert len(windows[-1][2]) == int(mask.sum())


def test_fingerprint_order_independent():
    assert fingerprint({"a": 1, "b": 2}) == fingerprint({"b": 2, "a": 1})


def test_alpha_grid_reuses_batch_model_spaces():
    candidates = list(alpha_grid("both").values())
    assert len(candidates) == 25
    assert all(set(c) == {"multirocket_alpha", "rdst_alpha"}
               for c in candidates)


def test_effective_alphas_calculated_once_from_full_window():
    features = np.zeros((3, 18), np.float32)
    base = {"multirocket_alpha": 1000., "rdst_alpha": 100.}
    reference = {"multirocket": 120, "rdst": 60}
    fixed = effective_alphas(features, [(12, 18)], "both", base, reference)
    assert fixed == {"multirocket_alpha": 100., "rdst_alpha": 10.}
    # A later shorter window MUST NOT rescale these selected values.
    shorter = np.zeros((3, 9), np.float32)
    transformed, alpha = regularized_feature_matrix(
        shorter, [(6,9)], "both", fixed)
    assert alpha == 100.
    np.testing.assert_allclose(transformed[:,6:9], 0.)


def test_two_penalties_ridge_equivalence():
    from sklearn.linear_model import Ridge
    rng = np.random.default_rng(15)
    x = rng.normal(size=(70,5)).astype(np.float32)
    copy = x.copy()
    y = rng.normal(size=70)
    penalty = {"multirocket_alpha":3., "rdst_alpha":17.}
    scaled, alpha = regularized_feature_matrix(x, [(3,5)], "both", penalty)
    model = Ridge(alpha=alpha, solver="lsqr", tol=1e-9).fit(scaled,y)
    centered = x - x.mean(axis=0)
    beta = np.linalg.solve(
        centered.T @ centered + np.diag([3.,3.,3.,17.,17.]),
        centered.T @ (y-y.mean()))
    np.testing.assert_allclose(model.predict(scaled),
                               y.mean()+centered@beta, atol=1e-4)
    np.testing.assert_array_equal(x,copy)


def test_reference_dimensions_from_existing_batch_model_space():
    assert original_feature_counts("both") == {
        "multirocket":50000, "rdst":3072}


def test_pruning_still_uses_batch_study():
    import inspect
    from waveform_analysis.engine.validation import evaluate_candidate
    from waveform_analysis.scripts.evaluate_progressive_windows import evaluate_prefix
    assert callable(evaluate_candidate)
    assert "evaluate_candidate(" in inspect.getsource(evaluate_prefix)


def test_batch_transform_parameters_are_not_reduced():
    from waveform_analysis.scripts.evaluate_progressive_windows import batch_model_space
    params = batch_model_space("both")["parameters"]
    assert params["n_kernels"]["value"] == 6250
    assert params["max_dilations_per_kernel"]["value"] == 32
    assert params["n_features_per_kernel"]["value"] == 4
    assert params["max_shapelets"]["value"] == 1024
    assert params["shapelet_lengths"]["value"] == [11]
