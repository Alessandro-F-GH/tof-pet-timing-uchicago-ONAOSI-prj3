import unittest

import numpy as np

from waveform_analysis.ml_pipeline.model_run_reporting import _paired_difference


class ModelRunReportingTests(unittest.TestCase):
    def test_paired_difference_is_zero_for_identical_residuals(self):
        residual = np.linspace(-100.0, 100.0, 501)
        result = _paired_difference(
            residual,
            residual.copy(),
            {"coverage_fraction": 0.9, "bootstrap_samples": 20},
            samples=20,
            seed=17,
        )
        delta, delta_unc, relative, relative_unc, successful = result
        self.assertAlmostEqual(delta, 0.0, places=12)
        self.assertAlmostEqual(relative, 0.0, places=12)
        self.assertAlmostEqual(delta_unc, 0.0, places=12)
        self.assertAlmostEqual(relative_unc, 0.0, places=12)
        self.assertGreater(successful, 1)


if __name__ == "__main__":
    unittest.main()
