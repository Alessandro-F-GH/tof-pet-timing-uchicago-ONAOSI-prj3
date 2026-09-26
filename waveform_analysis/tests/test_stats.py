import unittest

import numpy as np

from utils_fit import (
    direct_fwhm_from_histogram,
    fit_ctr_ps,
    fixed_width_histogram_edges,
)


class CTRTests(unittest.TestCase):
    def test_direct_f1_recovers_gaussian_fwhm(self):
        rng = np.random.default_rng(1)
        sigma = 30.0
        values = rng.normal(12.0, sigma, 100000)
        result = fit_ctr_ps(
            values,
            {
                "histogram_bin_width_ps": 8.0,
                "bootstrap_samples": 0,
            },
        )
        expected = 2.354820045 * sigma
        self.assertTrue(result.success)
        self.assertAlmostEqual(result.histogram_bin_width_ps, 8.0)
        self.assertGreater(result.histogram_bins, 3)
        self.assertLess(abs(result.ctr_ps - expected) / expected, 0.05)
        self.assertLess(abs(result.center_ps - 12.0), 8.0)

    def test_direct_f1_uses_histogram_maximum_without_parabolic_peak(self):
        edges = np.arange(-4.5, 5.5, 1.0)
        counts = np.array([0, 1, 3, 7, 10, 8, 4, 2, 0], dtype=float)
        result = direct_fwhm_from_histogram(counts, edges)
        self.assertEqual(result.peak_height, 10.0)
        self.assertEqual(result.center_ps, 0.0)
        self.assertAlmostEqual(result.histogram_bin_width_ps, 1.0)
        self.assertLess(result.half_max_left_ps, result.center_ps)
        self.assertGreater(result.half_max_right_ps, result.center_ps)
        self.assertAlmostEqual(
            result.ctr_ps,
            result.half_max_right_ps - result.half_max_left_ps,
            places=12,
        )

    def test_direct_f1_uses_middlemost_maximum(self):
        edges = np.arange(-3.5, 4.5, 1.0)
        counts = np.array([0, 2, 8, 8, 8, 2, 0], dtype=float)
        result = direct_fwhm_from_histogram(counts, edges)
        self.assertEqual(result.center_ps, 0.0)
        self.assertEqual(result.peak_height, 8.0)

    def test_bootstrap_reports_ctr_uncertainty(self):
        rng = np.random.default_rng(4)
        result = fit_ctr_ps(
            rng.normal(0.0, 30.0, 5000),
            {
                "histogram_bin_width_ps": 10.0,
                "bootstrap_samples": 30,
            },
            seed=13,
        )
        self.assertEqual(result.bootstrap_samples, 30)
        self.assertGreater(result.bootstrap_successful, 1)
        self.assertTrue(np.isfinite(result.ctr_error_ps))
        self.assertGreater(result.ctr_error_ps, 0.0)

    def test_fixed_width_edges_are_outlier_invariant_locally(self):
        core = np.array([-21.0, -4.0, 0.0, 7.0, 24.0])
        base = fixed_width_histogram_edges(core, 10.0)
        extended = fixed_width_histogram_edges(np.append(core, 5000.0), 10.0)
        self.assertTrue(np.allclose(np.diff(base), 10.0))
        self.assertTrue(np.allclose(np.diff(extended), 10.0))
        self.assertTrue(np.any(np.isclose(0.5 * (base[:-1] + base[1:]), 0.0)))
        self.assertTrue(np.any(np.isclose(0.5 * (extended[:-1] + extended[1:]), 0.0)))
        common = extended[(extended >= base[0]) & (extended <= base[-1])]
        self.assertTrue(np.allclose(common, base))
        self.assertGreater(extended.size, base.size)

    def test_fit_rejects_removed_coverage_configuration(self):
        with self.assertRaisesRegex(ValueError, "Unknown CTR option"):
            fit_ctr_ps(
                np.linspace(-20.0, 20.0, 100),
                {
                    "coverage_fraction": 0.90,
                    "bootstrap_samples": 0,
                },
            )

    def test_fit_requires_enough_residuals(self):
        with self.assertRaisesRegex(ValueError, "at least 5"):
            fit_ctr_ps(
                np.array([-1.0, 0.0, 1.0, 2.0]),
                {
                    "histogram_bin_width_ps": 1.0,
                    "bootstrap_samples": 0,
                },
            )

    def test_no_silent_absolute_residual_cut(self):
        rng = np.random.default_rng(6)
        values = np.concatenate([
            rng.normal(0.0, 20.0, 900),
            np.full(100, 5000.0),
        ])
        result = fit_ctr_ps(
            values,
            {
                "histogram_bin_width_ps": 10.0,
                "bootstrap_samples": 10,
            },
            seed=15,
        )
        self.assertEqual(result.n_valid, values.size)


if __name__ == "__main__":
    unittest.main()
