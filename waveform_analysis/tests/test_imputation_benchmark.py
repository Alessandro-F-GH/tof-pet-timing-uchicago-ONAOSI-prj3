import numpy as np

from waveform_analysis.scripts.evaluate_imputation_benchmark import physical_unit_scale


def test_benchmark_physical_unit_scale_covers_millivolts_to_picovolts():
    assert physical_unit_scale([2.0, 1.0]) == (1.0, "mV")
    assert physical_unit_scale([2e-2, 1e-3]) == (1e3, "µV")
    assert physical_unit_scale([2e-6, 1e-6]) == (1e6, "nV")
    assert physical_unit_scale([2e-9, 1e-9]) == (1e9, "pV")


def test_benchmark_physical_unit_scale_ignores_nonfinite_and_nonpositive():
    assert physical_unit_scale([np.nan, np.inf, 0.0, -1.0, 2e-6]) == (1e6, "nV")
