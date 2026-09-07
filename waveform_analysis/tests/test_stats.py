import unittest
import numpy as np

from waveform_analysis.ml_pipeline.stats import bootstrap_ctr, ctr_fwhm


class FWHMTests(unittest.TestCase):
    def test_gaussian_width(self):
        rng = np.random.default_rng(1)
        sigma = 42.0
        values = rng.normal(0.0, sigma, 40000)
        result = ctr_fwhm(values)
        expected = 2.354820045 * sigma
        self.assertLess(abs(result.ctr_ps - expected) / expected, 0.12)

    def test_non_gaussian_distribution(self):
        rng = np.random.default_rng(2)
        values = np.concatenate([rng.laplace(0.0, 25.0, 30000), rng.normal(120.0, 15.0, 2500)])
        result = ctr_fwhm(values)
        self.assertTrue(np.isfinite(result.ctr_ps))
        self.assertGreater(result.ctr_ps, 0.0)

    def test_outliers_do_not_dominate(self):
        rng = np.random.default_rng(3)
        core = rng.normal(0.0, 35.0, 30000)
        baseline = ctr_fwhm(core).ctr_ps
        contaminated = np.concatenate([core, np.array([-10000.0, 15000.0, 25000.0])])
        result = ctr_fwhm(contaminated).ctr_ps
        self.assertLess(abs(result - baseline) / baseline, 0.08)

    def test_bootstrap_uses_same_estimator(self):
        rng = np.random.default_rng(4)
        result = bootstrap_ctr(rng.normal(0.0, 30.0, 4000), 25, 7)
        self.assertTrue(np.isfinite(result.uncertainty_ps))
        self.assertGreater(result.uncertainty_ps, 0.0)
