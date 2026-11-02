import numpy as np

from waveform_analysis.scripts.evaluate_imputation_benchmark import (
    format_rmse_label,
    physical_unit_scale,
)


def test_benchmark_physical_unit_scale_covers_millivolts_to_picovolts():
    assert physical_unit_scale([2.0, 1.0]) == (1.0, "mV")
    assert physical_unit_scale([2e-2, 1e-3]) == (1e3, "µV")
    assert physical_unit_scale([2e-6, 1e-6]) == (1e6, "nV")
    assert physical_unit_scale([2e-9, 1e-9]) == (1e9, "pV")


def test_benchmark_physical_unit_scale_ignores_nonfinite_and_nonpositive():
    assert physical_unit_scale([np.nan, np.inf, 0.0, -1.0, 2e-6]) == (1e6, "nV")


def test_benchmark_rmse_labels_use_adaptive_units():
    assert format_rmse_label(2.0) == "2 mV"
    assert format_rmse_label(2e-3) == "2 µV"
    assert format_rmse_label(2e-6) == "2 nV"
    assert format_rmse_label(2e-9) == "2 pV"


def test_keep_only_benchmark_knn_single_neighbor_and_ppca():
    from types import SimpleNamespace
    from waveform_analysis.scripts.evaluate_imputation_benchmark import (
        keep_only_benchmark,
    )
    rng = np.random.default_rng(7)
    waves = rng.normal(size=(42, 2, 30)).astype(np.float64)
    mask = np.ones(30, dtype=bool)
    mask[20:24] = False
    args = SimpleNamespace(components=2, batch_size=4, seed=8, bootstrap=20)
    rows, overall, config = keep_only_benchmark(
        "UC", "energy", args, waves, np.arange(30) * 25., mask,
        np.arange(30), np.arange(30, 34), np.arange(34, 42),
        [5, 10], 4, 25., "synthetic", {},
    )
    assert config["occlusion_mode"] == "keep_only"
    assert config["knn_k"] == 1
    assert len(rows) == 2 * 2 * 3 * 2  # positions, detectors, methods
    assert {r["method"] for r in overall} == {"median", "ppca", "knn"}
    assert all(np.isfinite(r["rmse"]) for r in overall)
    assert all(r["missing_valid_samples"] == 22 for r in rows)


def test_benchmark_knn_method_in_remove_window_set():
    from waveform_analysis.scripts.evaluate_imputation_benchmark import METHODS
    assert "knn" in METHODS
