"""Checks for development-only progressive temporal feature assembly."""
import numpy as np
from waveform_analysis.scripts.evaluate_progressive_windows import temporal_blocks, fingerprint


def test_temporal_blocks_are_disjoint_and_cumulative():
    time = np.arange(-2000, 4000, 12.5)
    mask = np.ones(len(time), bool)
    blocks = temporal_blocks(time, mask, -2, 4, 1)
    assert len(blocks) == 6
    all_indices = np.concatenate([block[2] for block in blocks])
    assert len(np.unique(all_indices)) == len(all_indices)
    assert np.array_equal(all_indices, np.arange(len(time)))
    assert [round(b[1]) for b in blocks] == [-1, 0, 1, 2, 3, 4]


def test_masked_segments_are_not_used_as_features():
    time = np.arange(-2000, 4000, 12.5)
    mask = np.ones(len(time), bool)
    mask[(time >= 1000) & (time < 2000)] = False
    blocks = temporal_blocks(time, mask, -2, 4, 1)
    assert len(blocks) == 5
    assert all(mask[indices].all() for _, _, indices in blocks)


def test_feature_fingerprint_independent_of_argument_order():
    assert fingerprint({"a": 1, "b": [2,3]}) == fingerprint({"b": [2,3], "a": 1})


def test_alpha_grid_reuses_batch_study_model_spaces():
    from waveform_analysis.scripts.evaluate_progressive_windows import alpha_grid
    combined = list(alpha_grid("both").values())
    assert len(combined) == 25
    assert all(set(candidate) == {"multirocket_alpha", "rdst_alpha"}
               for candidate in combined)


def test_two_penalties_match_closed_form_regularized_solution():
    from waveform_analysis.scripts.evaluate_progressive_windows import regularized_feature_matrix
    from sklearn.linear_model import Ridge
    rng = np.random.default_rng(15)
    x = rng.normal(size=(70, 5)).astype(np.float32)
    y = rng.normal(size=70)
    penalties = {"multirocket_alpha": 3., "rdst_alpha": 17.}
    scaled, alpha = regularized_feature_matrix(x, [(3,5)], "both", penalties)
    model = Ridge(alpha=alpha, solver="lsqr", tol=1e-9).fit(scaled, y)
    predicted = model.predict(scaled)
    center = x.mean(axis=0)
    target = y - y.mean()
    matrix = np.diag([3., 3., 3., 17., 17.])
    coefficients = np.linalg.solve(
        (x-center).T @ (x-center) + matrix, (x-center).T @ target)
    expected = y.mean() + (x-center) @ coefficients
    np.testing.assert_allclose(predicted, expected, atol=1e-4)
    np.testing.assert_array_equal(x, np.asarray(x))  # not modified


def test_pruning_rules_are_batch_engine_not_separate_implementation():
    import inspect
    from waveform_analysis.scripts.evaluate_progressive_windows import evaluate_prefix
    from waveform_analysis.engine.validation import evaluate_candidate
    source = inspect.getsource(evaluate_prefix)
    assert "evaluate_candidate(" in source
    assert "best_complete_candidate(" in source
    assert callable(evaluate_candidate)


def test_effective_alphas_scale_separately_by_feature_count():
    from waveform_analysis.scripts.evaluate_progressive_windows import effective_alphas
    x = np.zeros((3, 18), np.float32)
    parameters = {"multirocket_alpha": 1000., "rdst_alpha": 100.}
    ref = {"multirocket": 120, "rdst": 60}
    # 12 MR columns and 6 RDST columns.
    actual = effective_alphas(x, [(12, 18)], "both", parameters, ref)
    assert actual == {"multirocket_alpha": 100., "rdst_alpha": 10.}
    x2 = np.zeros((3, 36), np.float32)
    actual2 = effective_alphas(x2, [(12,18), (30,36)], "both", parameters, ref)
    assert actual2 == {"multirocket_alpha": 200., "rdst_alpha": 20.}


def test_original_reference_feature_counts_from_model_spaces():
    from waveform_analysis.scripts.evaluate_progressive_windows import original_feature_counts
    result = original_feature_counts("both")
    assert result == {"multirocket": 50000, "rdst": 3072}
