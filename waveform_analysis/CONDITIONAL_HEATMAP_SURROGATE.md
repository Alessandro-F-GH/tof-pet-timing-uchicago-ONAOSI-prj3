# Conditional Heatmap Ridge Surrogate

Purpose: reconstruct the **saved model correction** from pointwise conditional heatmap lookups, calibrated with an L2-regularized linear surrogate.

Use existing single-model heatmaps, prepared waveforms, and saved Development and Blind predictions. No model retraining or new inference.

## Command (Windows PowerShell)

```powershell
python -m waveform_analysis.scripts.evaluate_conditional_heatmap_surrogate `
  --model-dir "waveform_analysis/results/UC/multirocket_rdst/energy/wide/direct_multirocket_rdst" `
  --heatmap-root "C:/actual/path/to/heatmaps/direct_multirocket_rdst" `
  --cache-root "$HOME/UC_cache" `
  --output-root "$HOME/xai/UC/heatmap_surrogate/direct_multirocket_rdst"
```

The heatmap root must contain summary.json and representation directories (original, baseline_subtracted, etc.). Replace illustrative paths with actual artifact locations.

Reuse exactly the Blind events of a previous local TSHAP case study by adding:

```powershell
  --event-ids-file "C:/actual/path/to/local_tshap/selected_events.json"
```

Without an explicit file, the same TSHAP selector is used: good_positive, good_negative, bad_positive, bad_negative, and typical, two events per group.

## Protocol

1. Reconstruct the heatmap RNG split (seed 42, normally 500 tuning + 500 holdout events excluded from map reference).
2. Build per-detector, per-timepoint conditional mean deviation features for the Development calibration subset.
3. Fit a separate Ridge coefficient for each detector/timepoint; choose L2 alpha via 5-fold CV on calibration only.
4. Evaluate the frozen Ridge on the separate Development validation subset and then the full Blind population.
5. Produce original physical blind waveforms colorized once by raw heatmap associations and once by calibrated Ridge terms.

Out-of-range, sparse and cropped cells map to zero deviation only in the surrogate feature matrix; their validity is separately tracked. Gray segments in plots indicate no supported estimate, not a measured zero contribution.

## Mathematical interpretation

With mu the mean of saved predictions in the heatmap reference, the feature for event i, detector d and time t is a(i,d,t) = E[f(X) | X(d,t) in local bin] - mu.

The surrogate is y_tilde(i) = intercept + sum(d,t) w(d,t) * a(i,d,t).

Local calibrated terms c(i,d,t) = w(d,t) * a(i,d,t) sum exactly to the Ridge output minus its intercept, not necessarily to the original model prediction.

Fidelity metrics are RMSE, MAE, bias, R2 and Pearson correlation against saved **ML predictions**. They are not CTR or error versus physical timing target. These conditional associations do not establish causal model attribution.

## Outputs

Per representation: metrics.json, surrogate.npz, blind_fidelity.png, validation_fidelity.png; per selected TSHAP event: heatmap_colored_waveforms.png, ridge_colored_waveforms.png, local_profiles.npz.

Test: python -m pytest waveform_analysis/tests/test_conditional_heatmap_surrogate.py -q