import unittest

import numpy as np

from utils_fit import double_gaussian_fwhm, fit_ctr_ps


class CTRTests(unittest.TestCase):
    def test_gaussian_equivalent_shortest_90_interval_recovers_gaussian_fwhm(self):
        rng = np.random.default_rng(1)
        sigma = 42.0
        values = rng.normal(0.0, sigma, 40000)
        result = fit_ctr_ps(
            values,
            {
                "coverage_fraction": 0.90,
                "bootstrap_samples": 20,
            },
            seed=11,
        )
        expected = 2.354820045 * sigma
        self.assertTrue(result.success)
        self.assertEqual(result.definition, "shortest_interval")
        self.assertLess(abs(result.ctr_ps - expected) / expected, 0.04)
        self.assertAlmostEqual(result.coverage_fraction, 0.90)
        self.assertGreaterEqual(result.interval_events, int(np.ceil(0.90 * values.size)))
        self.assertAlmostEqual(
            result.ctr_ps,
            result.gaussian_equivalent_scale * result.interval_width_ps,
            places=12,
        )

    def test_interval_endpoints_define_ctr(self):
        rng = np.random.default_rng(2)
        values = rng.normal(12.0, 30.0, 10000)
        result = fit_ctr_ps(
            values,
            {
                "coverage_fraction": 0.90,
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

    def test_bootstrap_reports_ctr_uncertainty(self):
        rng = np.random.default_rng(4)
        result = fit_ctr_ps(
            rng.normal(0.0, 30.0, 4000),
            {
                "coverage_fraction": 0.90,
                "bootstrap_samples": 30,
            },
            seed=13,
        )
        self.assertEqual(result.bootstrap_samples, 30)
        self.assertGreater(result.bootstrap_successful, 1)
        self.assertTrue(np.isfinite(result.ctr_error_ps))
        self.assertGreater(result.ctr_error_ps, 0.0)

    def test_double_gaussian_recovers_total_mixture_fwhm(self):
        rng = np.random.default_rng(7)
        n = 8000
        narrow_fraction = 0.68
        sigma_narrow = 28.0
        sigma_wide = 82.0
        narrow = rng.random(n) < narrow_fraction
        values = np.where(
            narrow,
            rng.normal(15.0, sigma_narrow, n),
            rng.normal(15.0, sigma_wide, n),
        )
        result = fit_ctr_ps(
            values,
            {"coverage_fraction": 0.90, "bootstrap_samples": 0},
            definition="double_gaussian",
            histogram_bins=40,
        )
        expected = double_gaussian_fwhm(
            sigma_narrow,
            sigma_wide,
            narrow_fraction,
        )
        self.assertEqual(result.definition, "double_gaussian")
        self.assertEqual(result.histogram_bins, 40)
        self.assertTrue(np.isnan(result.coverage_fraction))
        self.assertLess(abs(result.ctr_ps - expected) / expected, 0.08)
        self.assertLess(result.sigma_narrow_ps, result.sigma_wide_ps)
        self.assertGreater(result.narrow_fraction, 0.0)
        self.assertLess(result.narrow_fraction, 1.0)

    def test_double_gaussian_uses_existing_event_bootstrap(self):
        rng = np.random.default_rng(8)
        n = 2500
        narrow = rng.random(n) < 0.72
        values = np.where(
            narrow,
            rng.normal(0.0, 25.0, n),
            rng.normal(0.0, 70.0, n),
        )
        result = fit_ctr_ps(
            values,
            {"coverage_fraction": 0.90, "bootstrap_samples": 20},
            seed=81,
            definition="double_gaussian",
            histogram_bins=30,
        )
        self.assertEqual(result.bootstrap_samples, 20)
        self.assertGreater(result.bootstrap_successful, 1)
        self.assertTrue(np.isfinite(result.ctr_error_ps))
        self.assertGreater(result.ctr_error_ps, 0.0)

    def test_two_events_are_mathematically_sufficient(self):
        result = fit_ctr_ps(
            np.array([0.0, 10.0]),
            {
                "coverage_fraction": 0.90,
                "bootstrap_samples": 0,
            },
            seed=9,
        )
        self.assertTrue(result.success)
        self.assertEqual(result.n_valid, 2)

    def test_multimodal_side_peaks_increase_ctr(self):
        rng = np.random.default_rng(5)
        core = rng.normal(0.0, 2.0, 6000)
        left = rng.normal(-25.0, 2.0, 2000)
        right = rng.normal(25.0, 2.0, 2000)
        result = fit_ctr_ps(
            np.concatenate([core, left, right]),
            {
                "coverage_fraction": 0.90,
                "bootstrap_samples": 10,
            },
            seed=14,
        )
        self.assertTrue(result.success)
        self.assertGreater(result.ctr_ps, 20.0)

    def test_no_silent_absolute_residual_cut(self):
        rng = np.random.default_rng(6)
        values = np.concatenate([
            rng.normal(0.0, 20.0, 900),
            np.full(100, 5000.0),
        ])
        result = fit_ctr_ps(
            values,
            {
                "coverage_fraction": 0.90,
                "bootstrap_samples": 10,
            },
            seed=15,
        )
        self.assertEqual(result.n_valid, values.size)
        self.assertEqual(result.interval_events, 900)


if __name__ == "__main__":
    unittest.main()
