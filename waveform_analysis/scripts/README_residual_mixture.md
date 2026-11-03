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
- `gaussian_mixture_distributions.png`/`.pdf`: experimental energy and timing residual histograms, overlaid with the corresponding marginal PDFs of each bivariate Gaussian and their sum, for LED and ML. Logarithmic ordinate displays the broad tails.
- `gaussian_mixture_joint.png`/`.pdf`: energy vs timing residuals with two-sigma covariance ellipses for each mixture component, separately for LED and ML.

No waveform, per-event classification, beta test, percentile clipping or cache access is performed.

## Mathematical interpretation

For each component `k`, the bivariate covariance matrix is
```
Sigma_k = [[var_energy,k, cov_energy_timing,k],
           [cov_energy_timing,k, var_timing,k]]
```
The reported `shared_variance_ps2 = cov_energy_timing,k` can be interpreted as the variance of a **unit-loading additive common timing error** only under

```
r_E = C + epsilon_E
r_T = C + epsilon_T
```

with `C`, `epsilon_E`, and `epsilon_T` pairwise uncorrelated. The script also reports `sigma_common_ps = sqrt(shared_variance_ps2)` for nonnegative covariance and checks that both inferred channel-specific variances are nonnegative. An invalid decomposition is flagged.

The labels `core` and `tail` are assigned by increasing determinant of the fitted covariance matrix (the central ellipsoid has the smaller area). This is a descriptive two-Gaussian model; a lower BIC does not establish two physical mechanisms. Marginal LED and ML curves shown in the figures are **derived from the joint fits**, not separately optimized histograms.

Two independent mixtures are estimated on the *same event population*. Their fitted component weights and widths are not paired event-by-event improvement estimates. There is no preprocessing, retraining or test-set-driven model selection.
