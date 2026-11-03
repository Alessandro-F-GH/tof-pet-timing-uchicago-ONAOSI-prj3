# Portable Gaussian core and empirical-tail analysis

The standalone script `compare_energy_timing_residuals.py` analyzes event-matched energy and timing coincidence residuals **before** (LED) and **after** ML correction.

## Sharing data with a collaborator

Provide just:
- `compare_energy_timing_residuals.py`
- `energy_predictions.npz`
- `timing_predictions.npz`

The two NPZ files must contain `event_id`, `led_residual_ps`, and `corrected_ps` (1D arrays). The script matches by event ID and does not use the prepared dataset cache, waveforms, models or any project-specific packages. Both files should come from the same board and evaluation split; blind predictions are preferable to in-sample development diagnostics.

Dependencies:
```bash
python -m pip install numpy scipy matplotlib
```

Run:
```bash
python compare_energy_timing_residuals.py --energy energy_predictions.npz --timing timing_predictions.npz --output results
```

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

- `core_tail_summary.json`: core parameters, outside-core counts, empirical tail correlations/covariances, and selection-only Gaussian reference, LED and ML separately.
- `gaussian_core_distributions.png/.pdf`: measured energy/timing **event counts** and Gaussian-core extrapolation to the full range (no parametric tail fit).
- `core_and_empirical_tails.png/.pdf`: paired residual scatter, colored by whether events lie inside the fitted core envelope; all outside-core points shown with higher opacity.
- `radial_tail_exceedance.png/.pdf`: observed radial survival function versus `chi2(df=2)` survival for a Gaussian core.

The fitted `cov_energy_timing_ps2` is the core covariance, whereas outside-core covariance is an **empirical** statistic on a selected subset. Neither is automatically a physically identifiable common-noise variance; that interpretation requires equal common-error loadings and independent channel-specific errors.

Importantly, `numerically_converged` indicates iterative parameter stability, **not** a test of Gaussianity, and `tail_definition.nominal_gaussian_probability_outside` defines a reference fraction, not the fitted proportion of a second population. LED and ML classifications are fitted separately and are not themselves a paired correction-effect measure.
