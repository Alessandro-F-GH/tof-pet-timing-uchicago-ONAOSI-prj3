import unittest

import numpy as np

from utils_fit import fit_ctr_ps


class HistogramCTRTests(unittest.TestCase):
    def test_fixed_bin_fwhm_recovers_gaussian_width(self):
        rng = np.random.default_rng(1)
        sigma = 42.0
        values = rng.normal(0.0, sigma, 40000)
        result = fit_ctr_ps(
            values,
            {"min_events": 100, "bin_width_ps": 5.0, "bootstrap_samples": 20},
            seed=11,
        )
        expected = 2.354820045 * sigma
        self.assertTrue(result.success)
        self.assertLess(abs(result.ctr_ps - expected) / expected, 0.08)
        self.assertAlmostEqual(result.bin_width_ps, 5.0)

    def test_half_max_crossings_define_ctr(self):
        rng = np.random.default_rng(2)
        result = fit_ctr_ps(
            rng.normal(12.0, 30.0, 10000),
            {"min_events": 100, "bin_width_ps": 5.0, "bootstrap_samples": 10},
            seed=12,
        )
        self.assertTrue(result.success)
        self.assertAlmostEqual(result.ctr_ps, result.right_half_ps - result.left_half_ps, places=12)
        self.assertAlmostEqual(result.center_ps, 0.5 * (result.left_half_ps + result.right_half_ps), places=12)
        self.assertGreater(result.half_max_events, 0.0)

    def test_bootstrap_reports_ctr_uncertainty(self):
        rng = np.random.default_rng(4)
        result = fit_ctr_ps(
            rng.normal(0.0, 30.0, 4000),
            {"min_events": 100, "bin_width_ps": 5.0, "bootstrap_samples": 30},
            seed=13,
        )
        self.assertEqual(result.bootstrap_samples, 30)
        self.assertGreater(result.bootstrap_successful, 1)
        self.assertTrue(np.isfinite(result.ctr_error_ps))
        self.assertGreater(result.ctr_error_ps, 0.0)

    def test_core_fwhm_is_not_forced_to_describe_far_tails(self):
        rng = np.random.default_rng(5)
        core = rng.normal(0.0, 25.0, 30000)
        tails = np.concatenate([rng.normal(-250.0, 20.0, 300), rng.normal(250.0, 20.0, 300)])
        result = fit_ctr_ps(
            np.concatenate([core, tails]),
            {"min_events": 100, "bin_width_ps": 5.0, "bootstrap_samples": 10, "max_abs_ps": 2000.0},
            seed=14,
        )
        expected_core_fwhm = 2.354820045 * 25.0
        self.assertLess(abs(result.ctr_ps - expected_core_fwhm) / expected_core_fwhm, 0.12)


if __name__ == "__main__":
    unittest.main()
