import unittest
import numpy as np

from waveform_analysis.ml_pipeline.stats import bootstrap_ctr_uncertainty, gaussian_ctr


class GaussianCTRTests(unittest.TestCase):
    def test_gaussian_width(self):
        rng = np.random.default_rng(1); sigma = 42.0; values = rng.normal(0.0, sigma, 40000); result = gaussian_ctr(values); expected = 2.354820045 * sigma; self.assertLess(abs(result.ctr_ps - expected) / expected, 0.05)

    def test_result_comes_from_gaussian_fit(self):
        rng = np.random.default_rng(2); result = gaussian_ctr(rng.normal(12.0, 30.0, 10000)); self.assertTrue(result.success); self.assertAlmostEqual(result.ctr_ps, 2.354820045 * result.sigma_ps, places=6); self.assertTrue(np.isfinite(result.chi2_ndof))

    def test_bootstrap_uses_gaussian_estimator(self):
        rng = np.random.default_rng(4); uncertainty = bootstrap_ctr_uncertainty(rng.normal(0.0, 30.0, 4000), 25, 7); self.assertTrue(np.isfinite(uncertainty)); self.assertGreater(uncertainty, 0.0)
