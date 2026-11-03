# Portable Gaussian core and empirical-tail analysis

The standalone script `compare_energy_timing_residuals.py` analyzes event-matched energy and timing coincidence residuals **before** (LED) and **after** ML correction.

## Sharing data with a collaborator

Provide the standalone Python script and a directory containing both boards:

```text
predictions/
├── UC/
│   ├── energy_predictions.npz
│   └── timing_predictions.npz
└── FBK/
    ├── energy_predictions.npz
    └── timing_predictions.npz
```

The shorter names `energy.npz` and `timing.npz` are also accepted, provided there is exactly one matching file per channel. Files must contain one-dimensional arrays `event_id`, `led_residual_ps`, and `corrected_ps`. The script matches by event ID **within each board**; UC/FBK populations are independent and must never be joined.

Dependencies:
```bash
python -m pip install numpy scipy matplotlib
```

Run:
```bash
python compare_energy_timing_residuals.py --predictions-root predictions --output comparison_results
```

No prepared cache, waveforms, models or project-specific packages are required.

## Method

A single **bivariate Gaussian core** is estimated independently for LED and ML.

1. Initialize robustly using medians and compact central observations.
2. Repeatedly select observations with squared Mahalanobis distance `d² <= chi2.ppf(0.95, 2)`, update the mean and covariance, and correct covariance for radial truncation of a 2D Gaussian. Stop when the parameters stabilize. This is a robust iterative moment estimator, not an exact truncated-likelihood MLE.
3. After fixing the core, define empirical **outside-core** events with `d² > chi2.ppf(0.999, 2)` by default. This selection is in 2D, not separate marginal clipping. **No distribution is assumed or fitted to these outside-core observations.**
4. Calculate their sample standard deviations, correlation and covariance directly. Since the selection itself changes correlation, compare against a simulated pure-Gaussian core with exactly the same Mahalanobis cut. This simulated reference is a null benchmark, **not** a Gaussian assumption for the observed tails.
5. Compare the number of outside-core events against the theoretical Gaussian expectation, `N*(1-0.999)` by default. Some genuine Gaussian events fall outside this envelope naturally.

Set `--fit-fraction 0.95` to control the iterative central fit radius and `--tail-fraction 0.999` for the outside-core envelope. These are **separate choices** and are reported explicitly in results. Sensitivity checks across several tail-fraction values are recommended; changing either cutoff can change empirical correlations.

The truncation covariance correction is derived under a bivariate Gaussian core with fixed elliptical selection. It need not be exactly unbiased for contaminated samples or when the fitted parameters determine the selection.

## Outputs

The output directory contains:

- `UC/core_tail_summary.json` and `FBK/core_tail_summary.json`: per-board Gaussian core fits and empirical outside-core tail statistics.
- Per-board `led_vs_ml_core_distributions.png/.pdf`, `led_vs_ml_joint_scatter.png/.pdf`, `core_radial_exceedance.png/.pdf`.
- `uc_fbk_comparison.json`: UC−FBK differences in empirical core/outside-core correlations; conditional bootstrap percentile 95% confidence intervals, plus fitted core point estimates.
- `uc_vs_fbk_core_distributions.png/.pdf`: UC and FBK histograms and Gaussian core fits on shared per-channel x-limits, with 10-ps energy and 5-ps timing bins. Histograms are raw event counts; N is shown because the boards may have different sample sizes.
- `uc_vs_fbk_joint_scatter.png/.pdf`: LED and ML scatterplots for both boards using identical x and y limits, gray points only.

Use `--bootstrap-resamples 300` (default) to choose the number of bootstrap draws for correlation differences. The bootstrap resamples observations **after** their core/tail membership has been determined; it does not refit the core or recalculate membership. These intervals only measure conditional sampling uncertainty, not uncertainty from the iterative fitting procedure, thresholds, or systematic effects. Confidence intervals containing zero indicate an unresolved difference, **not statistical equivalence**. To establish equivalence, a scientifically justified tolerance would be needed.


The fitted `cov_energy_timing_ps2` is the core covariance, whereas outside-core covariance is an **empirical** statistic on a selected subset. Neither is automatically a physically identifiable common-noise variance; that interpretation requires equal common-error loadings and independent channel-specific errors.

Importantly, `numerically_converged` indicates iterative parameter stability, **not** a test of Gaussianity, and `tail_definition.nominal_gaussian_probability_outside` defines a reference fraction, not the fitted proportion of a second population. LED and ML classifications are fitted separately and are not themselves a paired correction-effect measure.

## Terminal report

The terminal output is intentionally verbose and self-contained. It explains the difference between the fit region and the outside-core detection region, reports numerical convergence and fitted core moments, then empirical outside-core statistics and the Gaussian-selection-only reference. Numerical convergence does **not** demonstrate Gaussian goodness of fit. The output names listed in the report are the only plots created by the current version; older plots from previous runs may remain if the same output directory is reused.
