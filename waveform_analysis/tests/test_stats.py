import unittest

import numpy as np

from utils_fit import CORE_FWHM_METRIC_NAME, CTR_METRIC_NAME, fit_ctr_ps


class RobustCTRTests(unittest.TestCase):
    def test_gaussian_equivalent_shortest_90_interval_recovers_gaussian_fwhm(self):
        rng = np.random.default_rng(1)
        sigma = 42.0
        values = rng.normal(0.0, sigma, 40000)
        result = fit_ctr_ps(
            values,
            {
                "min_events": 100,
                "coverage_fraction": 0.90,
                "bin_width_ps": 5.0,
                "bootstrap_samples": 20,
            },
            seed=11,
        )
        expected = 2.354820045 * sigma
        self.assertTrue(result.success)
        self.assertLess(abs(result.ctr_ps - expected) / expected, 0.04)
        self.assertAlmostEqual(result.coverage_fraction, 0.90)
        metadata = result.as_dict()
        self.assertEqual(metadata["fit_metric"], CTR_METRIC_NAME)
        self.assertEqual(metadata["core_metric"], CORE_FWHM_METRIC_NAME)
        self.assertGreaterEqual(result.interval_events, int(np.ceil(0.90 * values.size)))
        self.assertAlmostEqual(
            result.ctr_ps,
            result.gaussian_equivalent_scale * result.interval_width_ps,
            places=12,
        )

    def test_interval_endpoints_define_robust_ctr(self):
        rng = np.random.default_rng(2)
        values = rng.normal(12.0, 30.0, 10000)
        result = fit_ctr_ps(
            values,
            {
                "min_events": 100,
                "coverage_fraction": 0.90,
                "bin_width_ps": 5.0,
                "bootstrap_samples": 10,
            },
            seed=12,
        )
        self.assertTrue(result.success)
        self.assertAlmostEqual(
            result.interval_width_ps,
            result.interval_high_ps - result.interval_low_ps,
            places=12,
        )
        self.assertAlmostEqual(
            result.center_ps,
            0.5 * (result.interval_low_ps + result.interval_high_ps),
            places=12,
        )

    def test_bootstrap_reports_robust_ctr_uncertainty(self):
        rng = np.random.default_rng(4)
        result = fit_ctr_ps(
            rng.normal(0.0, 30.0, 4000),
            {
                "min_events": 100,
                "coverage_fraction": 0.90,
                "bin_width_ps": 5.0,
                "bootstrap_samples": 30,
            },
            seed=13,
        )
        self.assertEqual(result.bootstrap_samples, 30)
        self.assertGreater(result.bootstrap_successful, 1)
        self.assertTrue(np.isfinite(result.ctr_error_ps))
        self.assertGreater(result.ctr_error_ps, 0.0)

    def test_multimodal_side_peaks_increase_robust_ctr(self):
        rng = np.random.default_rng(5)
        core = rng.normal(0.0, 2.0, 6000)
        left = rng.normal(-25.0, 2.0, 2000)
        right = rng.normal(25.0, 2.0, 2000)
        result = fit_ctr_ps(
            np.concatenate([core, left, right]),
            {
                "min_events": 100,
                "coverage_fraction": 0.90,
                "bin_width_ps": 1.0,
                "bootstrap_samples": 10,
            },
            seed=14,
        )
        self.assertTrue(result.success)
        self.assertGreater(result.ctr_ps, 20.0)
        self.assertLess(result.core_fwhm_ps, 10.0)
        self.assertLess(result.core_fraction, 0.70)

    def test_no_silent_absolute_residual_cut(self):
        rng = np.random.default_rng(6)
        values = np.concatenate([
            rng.normal(0.0, 20.0, 900),
            np.full(100, 5000.0),
        ])
        result = fit_ctr_ps(
            values,
            {
                "min_events": 100,
                "coverage_fraction": 0.90,
                "bin_width_ps": 5.0,
                "bootstrap_samples": 10,
            },
            seed=15,
        )
        self.assertEqual(result.n_valid, values.size)
        self.assertEqual(result.interval_events, 900)


if __name__ == "__main__":
    unittest.main()
