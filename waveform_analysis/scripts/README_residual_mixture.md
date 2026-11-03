# Portable energy/timing residual-mixture analysis

The standalone script `compare_energy_timing_residuals.py` compares the same coincidence events before (LED) and after ML correction. It uses a full-covariance, two-component **bivariate** Gaussian mixture to identify a central component and a broad component, quantify their correlations, and estimate their shared covariance.

## Data to share with a collaborator

Copy these three files into a directory:

- `compare_energy_timing_residuals.py` (copy from `waveform_analysis/scripts/`)
- `energy_predictions.npz` (the desired model's saved prediction artifact)
- `timing_predictions.npz` (the corresponding model's saved prediction artifact)

Each NPZ must contain three one-dimensional arrays: `event_id`, `led_residual_ps`, and `corrected_ps`. These are already included in standard project prediction artifacts. The script matches events by `event_id`; it does **not** require a prepared waveform cache, model weights, or raw detector data.

The energy and timing predictions must come from corresponding populations (same board, same blind/development split, same events) and the same control/LED subtraction convention. For publications, prefer **blind** predictions. Development predictions from a final model are generally in-sample.

Install:
```bash
python -m pip install numpy scipy matplotlib scikit-learn
```

Run from the directory containing the three files:
```bash
python compare_energy_timing_residuals.py --energy energy_predictions.npz --timing timing_predictions.npz --output results
```

## Files generated

- `mixture_summary.json`: separate LED and ML bivariate fit parameters, BIC difference (one vs two Gaussians), core/tail correlations and covariance decomposition.
- `gaussian_mixture_distributions.png`/`.pdf`: experimental energy and timing residual histograms, overlaid with the corresponding marginal PDFs of each bivariate Gaussian and their sum, for LED and ML. Logarithmic **event-count** axis displays the broad tails; each fitted curve is normalized to the matching histogram bin width.
- `gaussian_mixture_joint.png`/`.pdf`: energy vs timing residuals colored by GMM posterior assignment (core blue, tail orange); all tail points are shown.

No waveform, per-event classification, beta test, percentile clipping or cache access is performed.

## Mathematical interpretation

For each component `k`, the bivariate covariance matrix is
```
Sigma_k = [[var_energy,k, cov_energy_timing,k],
           [cov_energy_timing,k, var_timing,k]]
```
The reported `cov_energy_timing_ps2` can be interpreted as the variance of a **unit-loading additive common timing error** only under

```
r_E = C + epsilon_E
r_T = C + epsilon_T
```

with `C`, `epsilon_E`, and `epsilon_T` pairwise uncorrelated. The script reports `sigma_common_if_unit_loading_ps` only if both specific variances remain nonnegative; `unit_loading_noise_decomposition_valid` is a check of this physical interpretation, not the statistical Gaussian fit.

The labels `core` and `tail` are assigned by increasing determinant of the fitted covariance matrix (the central ellipsoid has the smaller area). This is a descriptive two-Gaussian model; a lower BIC does not establish two physical mechanisms. Marginal LED and ML curves shown in the figures are **derived from the joint fits**, not separately optimized histograms.

Two independent mixtures are estimated on the *same event population*. Their fitted component weights and widths are not paired event-by-event improvement estimates. There is no preprocessing, retraining or test-set-driven model selection.

## Fit-quality diagnostics

The script additionally creates three plots (PNG and PDF):

- `gmm_fit_residuals`: signed Poisson deviance residuals, for energy, timing and energy-minus-timing. Adjacent bins are merged to roughly five or more expected events; dashed ±2 lines are visual guides, not formal confidence limits.
- `gmm_exceedance`: observed versus predicted `P(|residual|>threshold)` for LED and ML in all three projections. This emphasizes rare events rather than only the central peak.
- `gmm_energy_minus_timing`: observed counts versus the **analytic projection** of the fitted bivariate mixture onto energy-minus-timing. No new fit is performed.

`mixture_summary.json` includes `fit_quality`: descriptive KS distances, binned Poisson deviance and expected versus observed exceedance counts at 100, 200, 300, 500 and 1000 ps. A KS p-value is intentionally **not** reported since the same events were used to fit model parameters. BIC compares one versus two Gaussian components, while `em_numerically_converged` only confirms that the numerical EM stopping criterion was satisfied; neither proves good fit quality.

The summary distinguishes `cov_energy_timing_ps2` (a directly fitted covariance) from the conditional interpretation as a physical common-noise variance. For possible model mismatch, inspect the tail exceedance plot and the distribution of energy-minus-timing.
