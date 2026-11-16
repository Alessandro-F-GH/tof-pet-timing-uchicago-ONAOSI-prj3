"""Regression checks for the frozen benchmark and its TSHAP adapter."""
import json
from types import SimpleNamespace

import numpy as np
import pytest

from waveform_analysis.engine.imputer_selection import (
    _masks, _score_candidates, _winner_per_method, FrozenImputers,
)
from waveform_analysis.engine.conditional_knn_imputer import PairedWindowKNN
from waveform_analysis.engine.frozen_imputer_tshap import _configure_frozen_explainer
from waveform_analysis.engine.tshap_explainer import _new_explainer


def test_two_coalition_masks_consistent_with_knn():
    starts = np.asarray([3, 6])
    retained = np.ones(16, dtype=bool)
    masks = _masks(16, starts, 3, retained)
    assert [item[0] for item in masks] == [
        "remove_window", "remove_window", "keep_only", "keep_only",
    ]
    for task, start, observed, hidden in masks:
        assert not np.any(observed & hidden)
        np.testing.assert_array_equal(observed | hidden, retained)
        assert observed[start] == (task == "keep_only")


def test_global_knn_candidate_scoring_no_per_window_selection():
    rng = np.random.default_rng(14)
    reference = rng.normal(size=(15, 2, 16)).astype(np.float32)
    validation = rng.normal(size=(3, 2, 16)).astype(np.float32)
    starts = np.asarray([3, 6])
    retained = np.ones(16, dtype=bool)
    masks = _masks(16, starts, 3, retained)
    knn = PairedWindowKNN(reference, starts, 3, retained, [1, 2],
                          scale_mv=np.ones((2, 16)), reference_block=5)
    scores = _score_candidates(
        validation, masks, minimum=np.zeros((2, 16)),
        span=np.ones((2, 16)), models={},
        median=np.median(reference.reshape(15, -1), axis=0),
        control=reference[:2], knn=knn, knn_k=(1, 2),
    )
    for task in ("remove_window", "keep_only"):
        chosen = _winner_per_method(scores, task)
        assert "conditional_knn" in chosen
        assert chosen["conditional_knn"]["setting"] in {
            "1:uniform", "1:distance", "2:uniform", "2:distance",
        }
        assert all(np.isfinite(item["rmse_mv"]) for item in chosen.values())


def test_saved_frozen_reference_and_median_no_refit(tmp_path):
    folder = tmp_path / "frozen"
    folder.mkdir()
    (folder / ".complete").write_text("complete")
    metadata = {
        "development_identity": "population-123",
        "window_samples": 3, "stride_samples": 3,
        "selected": {
            "remove_window": {"method": "median", "setting": ""},
            "keep_only": {"method": "reference", "setting": ""},
        },
    }
    (folder / "selection.json").write_text(json.dumps(metadata))
    base = np.zeros((2, 16), np.float32)
    baselines = np.stack([base + 1, base + 2]).astype(np.float32)
    np.savez(
        folder / "fitted_imputers.npz",
        fit_reference=np.stack((base, base + 3)),
        control_baselines=baselines,
        median_physical=np.ones(32),
        minimum=np.zeros((2, 16)), span=np.ones((2, 16)),
        retained=np.ones(16, dtype=bool), starts=np.asarray([3, 6]),
    )
    frozen = FrozenImputers(
        folder, development_identity="population-123",
        window_samples=3, stride_samples=3,
    )
    sample = np.zeros((2, 16), np.float32)
    removed = frozen.reconstruct(sample, "remove_window", 0)
    np.testing.assert_array_equal(removed[0, :, 3:6], 1)
    kept = frozen.reconstruct(sample, "keep_only", 0)
    assert kept.shape == (2, 2, 16)
    np.testing.assert_array_equal(kept[:, :, 3:6], 0)
    with pytest.raises(ValueError, match="development_identity"):
        FrozenImputers(folder, development_identity="wrong",
                       window_samples=3, stride_samples=3)


def test_frozen_shapley_zero_for_constant_model(tmp_path):
    folder = tmp_path / "frozen"
    folder.mkdir()
    (folder / ".complete").touch()
    (folder / "selection.json").write_text(json.dumps({
        "development_identity": "p", "window_samples": 4,
        "stride_samples": 4,
        "selected": {
            "remove_window": {"method": "median", "setting": ""},
            "keep_only": {"method": "median", "setting": ""},
        },
    }))
    np.savez(folder / "fitted_imputers.npz",
             fit_reference=np.zeros((2, 2, 24), np.float32),
             control_baselines=np.zeros((1, 2, 24), np.float32),
             median_physical=np.zeros(48),
             minimum=np.zeros((2, 24)), span=np.ones((2, 24)),
             retained=np.ones(24, dtype=bool),
             starts=np.asarray([4, 8, 12, 16]))
    frozen = FrozenImputers(folder, development_identity="p",
                            window_samples=4, stride_samples=4)
    # Avoid fitting an ML model; the scorer only requires a frozen predict contract.
    predict = lambda batch: np.full(len(batch), 12., np.float64)
    spec = SimpleNamespace()
    fit = SimpleNamespace(sample_mask=None, feature_transform=None)
    # _configure calls _predict_bounded for the empty coalition; monkeypatch it.
    from waveform_analysis.engine import frozen_imputer_tshap as module
    original = module.base._predict_bounded
    module.base._predict_bounded = lambda *a: predict(a[2])
    try:
        explainer = _new_explainer(4, 4, False, 8)
        explainer = _configure_frozen_explainer(explainer, frozen, (spec, fit, 8))
        result = explainer._calculate_window_attrib_scores(
            predict, np.ones((2, 24), np.float32),
            np.zeros((1, 2, 24), np.float32), [(-1, 4, 4)],
        )
        np.testing.assert_allclose(result, 0.)
    finally:
        module.base._predict_bounded = original


def test_position_rmse_matches_pooled_rmse_and_writes_plots(tmp_path):
    from waveform_analysis.engine.imputer_diagnostics import (
        position_rows, save_diagnostics, diagnostics_complete,
    )
    rng = np.random.default_rng(211)
    reference = rng.normal(size=(9, 2, 16)).astype(np.float32)
    validation = rng.normal(size=(4, 2, 16)).astype(np.float32)
    starts = np.asarray([3, 6])
    masks = _masks(16, starts, 3, np.ones(16, dtype=bool))
    engine = PairedWindowKNN(
        reference, starts, 3, np.ones(16, bool), [1, 2],
        scale_mv=np.ones((2, 16)), reference_block=4,
    )
    keywords = dict(
        minimum=np.zeros((2, 16)), span=np.ones((2, 16)),
        models={}, median=np.median(reference.reshape(9, -1), axis=0),
        control=reference[:2], knn=engine, knn_k=(1, 2),
    )
    global_scores, per_position = _score_candidates(
        validation, masks, return_positions=True, **keywords,
    )
    chosen = {task: _winner_per_method(global_scores, task)
              for task in ("remove_window", "keep_only")}
    records = position_rows(
        chosen, per_position, per_position, starts,
        np.arange(16, dtype=float) * 1000, 3,
    )
    assert len(records) == sum(len(chosen[task]) for task in chosen) * len(starts)
    for row in records:
        assert np.isfinite(row["evaluation_rmse_mv"])
        assert row["evaluation_rmse_mv"] >= 0
    # Remove-window positions have equal numbers of hidden samples; pooled
    # squared RMSE must equal the mean of the per-window squared RMSE.
    for method, item in chosen["remove_window"].items():
        values = [r["evaluation_rmse_mv"] for r in records
                  if r["task"] == "remove_window" and r["method"] == method]
        assert np.sqrt(np.mean(np.square(values))) == pytest.approx(
            global_scores[("remove_window", method, item["setting"])], rel=1e-6,
        )
    summary = [
        dict(task=task, method=method, setting=item["setting"],
             tuning_rmse_mv=item["rmse_mv"],
             evaluation_rmse_mv=item["rmse_mv"])
        for task, winners in chosen.items()
        for method, item in winners.items()
    ]
    selected = {task: min(
        (row for row in summary if row["task"] == task),
        key=lambda row: row["tuning_rmse_mv"],
    ) for task in chosen}
    save_diagnostics(tmp_path, records, summary, selected, "UC", "energy")
    assert diagnostics_complete(tmp_path)
    with np.load(tmp_path / "position_rmse.npz") as artifact:
        assert artifact["evaluation_rmse_mv"].shape[1] == len(starts)
    assert (tmp_path / "best_methods_summary.pdf").stat().st_size > 0
