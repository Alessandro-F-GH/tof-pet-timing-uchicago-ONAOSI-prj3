"""Tests for standalone UC/FBK IQR classification and comparison."""
import json

import numpy as np
import pytest

from waveform_analysis.scripts.compare_energy_timing_residuals import (
    align_prediction_pairs, analyze_distribution, compare_boards,
    empirical_statistics, load_predictions, main, resolve_board_files, tukey_fences,
)


def test_tukey_3iqr_either_channel():
    # IQR is 2 for both coordinates: fences [-7, 9].
    points = np.array([
        [-1., -1.], [0., 0.], [1., 1.], [2., 2.], [3., 3.],
        [4., 4.], [11., 0.], [0., -20.],
    ])
    fences, outlier = tukey_fences(points, multiplier=3.)
    assert outlier[-2:].all()
    assert not outlier[:-2].any()
    assert fences["multiplier"] == 3.


def test_no_gaussian_assumption():
    rng = np.random.default_rng(12)
    points = rng.laplace(size=(5000, 2))
    result, mask = analyze_distribution(points)
    assert result["method"].startswith("Extreme Tukey IQR")
    assert mask.shape == (len(points),)
    assert result["empirical_outliers"] is not None
    assert "core_gaussian_fit" not in result
    assert result["outlier_count"] == int(mask.sum())


def test_npz_validations(tmp_path):
    p = tmp_path / "x.npz"
    np.savez(p, event_id=np.arange(25), corrected_ps=np.zeros(25))
    with pytest.raises(ValueError, match="led_residual_ps"):
        load_predictions(p)
    np.savez(p, event_id=np.zeros(25, dtype=int),
             led_residual_ps=np.zeros(25), corrected_ps=np.zeros(25))
    with pytest.raises(ValueError, match="unique"):
        load_predictions(p)


def test_join_is_by_event_id():
    ids = np.arange(30)
    energy = (ids[::-1], 100. + ids[::-1], 200. + ids[::-1])
    timing = (ids, 300. + ids, 400. + ids)
    joined, led, corrected = align_prediction_pairs(energy, timing)
    np.testing.assert_array_equal(joined, ids)
    np.testing.assert_array_equal(led[:, 0], ids + 100.)
    np.testing.assert_array_equal(led[:, 1], ids + 300.)
    np.testing.assert_array_equal(corrected[:, 0], ids + 200.)


def test_end_to_end_uc_fbk(tmp_path):
    rng = np.random.default_rng(13)
    root = tmp_path / "predictions"
    for board, scale in (("UC", 1.), ("FBK", 1.1)):
        directory = root / board
        directory.mkdir(parents=True)
        n = 2500
        ids = rng.permutation(n)
        common = rng.normal(0, 20*scale, size=n)
        extreme = rng.random(n) < .03
        common[extreme] += rng.normal(0, 400, size=extreme.sum())
        energy = common + rng.normal(0, 40, size=n)
        timing = common + rng.normal(0, 15, size=n)
        np.savez(directory / "energy_predictions.npz",
                 event_id=ids, led_residual_ps=energy, corrected_ps=.8*energy)
        np.savez(directory / "timing_predictions.npz", event_id=ids[::-1],
                 led_residual_ps=timing[::-1], corrected_ps=(.85*timing)[::-1])
        assert len(resolve_board_files(root, board)) == 2
    output = tmp_path / "result"
    main(["--predictions-root", str(root), "--output", str(output),
          "--bootstrap-resamples", "50"])
    comparison = json.loads((output / "uc_fbk_iqr_comparison.json").read_text())
    assert comparison["bootstrap_draws"] == 50
    assert "outliers" in comparison["states"]["ml"]["populations"]
    for board in ("UC", "FBK"):
        assert (output / board / "iqr_summary.json").is_file()
        assert (output / board / "led_vs_ml_distributions.png").is_file()
        assert (output / board / "led_vs_ml_iqr_scatter.pdf").is_file()
    assert (output / "uc_vs_fbk_iqr_scatter.png").is_file()
    assert (output / "uc_vs_fbk_distributions.pdf").is_file()


def test_resolve_ambiguous_files(tmp_path):
    folder = tmp_path / "UC"
    folder.mkdir()
    for name in ("energy_predictions.npz", "energy.npz", "timing_predictions.npz"):
        (folder / name).write_bytes(b"unused")
    with pytest.raises(ValueError, match="exactly one"):
        resolve_board_files(tmp_path, "UC")
