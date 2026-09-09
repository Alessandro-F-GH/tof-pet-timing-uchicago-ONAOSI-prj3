import unittest
import numpy as np

from utils_fit import fit_ctr_ps


class GaussianCTRTests(unittest.TestCase):
    def test_gaussian_width(self):
        rng = np.random.default_rng(1); sigma = 42.0; values = rng.normal(0.0, sigma, 40000); result = fit_ctr_ps(values); expected = 2.354820045 * sigma; self.assertLess(abs(result.ctr_ps - expected) / expected, 0.05)

    def test_result_comes_from_gaussian_fit(self):
        rng = np.random.default_rng(2); result = fit_ctr_ps(rng.normal(12.0, 30.0, 10000)); self.assertTrue(result.success); self.assertAlmostEqual(result.ctr_ps, 2.354820045 * result.sigma_ps, places=6); self.assertTrue(np.isfinite(result.chi2_ndof))

    def test_fit_reports_ctr_uncertainty(self):
        rng = np.random.default_rng(4); result = fit_ctr_ps(rng.normal(0.0, 30.0, 4000)); self.assertTrue(np.isfinite(result.ctr_error_ps)); self.assertGreater(result.ctr_error_ps, 0.0)
