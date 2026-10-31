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
