# Digital CFD reference on frozen ML prepared inputs

This is an **independent classical baseline**, not an ML surrogate. The CLI
loads **only existing development and Blind prepared caches** from a selected
existing ML run. It does not read ROOT files, reconstruct absolute timestamps,
train/load an ML model, or call ML inference.

## Definition and signs

The repository stores the centered LED residual

`y = (LED_1-LED_2) - true_TOF - control_LED_mean`.

The *prepared* time axis is centered on the closest **sample** to the LED
crossing, not necessarily on the fractional crossing itself. Thus both the
original LED crossing at its saved baseline-relative threshold and the digital
CFD threshold crossing are linearly re-interpolated on the SAME prepared
waveform/time grid.

For detector `d`:

- `B_d`: mean waveform in specified LED-relative baseline interval;
- `A_d`: waveform maximum before subtracting `B_d`;
- `t_LED,d`: local fixed-threshold interpolation using the saved fixed LED
  threshold (mV converted by the saved prepared scaling);
- `t_CFD,d(alpha)`: local fractional threshold `B_d + alpha*(A_d-B_d)`,
  interpolated on the rising edge (before first occurrence of the maximum).

For `delta=t_1-t_2`:

```
prediction_CFD_raw = (t_LED,1 - t_CFD,1) - (t_LED,2 - t_CFD,2)
CFD_corrected_residual = LED_centered_target - prediction_CFD_raw - development_offset
```

The offset is **one constant**, fitted from the development target after
fraction selection, and frozen for Blind evaluation. Thus this is a
development-calibrated classical CFD, not a target-independent procedure in
the strict sense.

## Selection / evaluation protocol

CFD fraction candidates `0.05, 0.10, ..., 0.50` are scored on the same
finite-crossing development event subset, by the repository's native
CTR/FWHM or RMSE estimators. Reject scan if their common valid coverage is
below `--minimum-coverage`. Select the fraction on development **only**.
Use that fraction for a complete development diagnostic and ONE Blind
evaluation, with 100 event bootstrap replicates as in the ML benchmark.

The prepared waveform is previously clipped at the physical acquisition
limits; this benchmark is therefore a digital CFD *on prepared data*, not
necessarily identical to native-waveform CFD.

## Windows PowerShell

```powershell
git pull origin codex/artifact-store-refactor

python -m waveform_analysis.scripts.evaluate_cfd_led_correction `
  --model-dir "waveform_analysis/results/UC/FINAL/energy/wide/direct_multirocket_rdst" `
  --cache-root "$HOME/UC_cache" `
  --output-root "$HOME/xai/UC/energy/wide/cfd_led" `
  --metric ctr `
  --baseline-window-ns -1.9 -1.0 `
  --bootstrap 100
```

Output contains:

- `summary.json`: fraction scan, development metrics, Blind CTR/RMSE
  and 100-replicate paired LED/CFD bootstrap, coverage and provenance;
- `development_predictions.npz`, `blind_predictions.npz`: exact prepared
  `event_id`, `prediction_ps`, `corrected_ps`, `led_residual_ps`,
  `valid` and calibration metadata; invalid CFD predictions remain NaN;
- `blind_bootstrap.npz`: event-resampling draws.

The sign and units of `prediction_ps` match ML corrections:
`corrected_ps = led_residual_ps - prediction_ps`.
Do **not** compare raw array indices if a different ML model uses a different
prepared population: align on `event_id` and verify control/target metadata.

### Generate conditional ML-minus-CFD heatmaps

```powershell
python -m waveform_analysis.scripts.plot_conditional_target_heatmaps `
  --model-dir "waveform_analysis/results/UC/FINAL/energy/wide/direct_multirocket_rdst" `
  --comparison-predictions-npz "$HOME/xai/UC/energy/wide/cfd_led/development_predictions.npz" `
  --cache-root "$HOME/UC_cache" `
  --output-root "$HOME/xai/UC/energy/wide/ml_minus_cfd_heatmaps" `
  --representations original amplitude_normalized `
  --include-original-derivative
```

Only common, **finite** CFD events are used. The code checks metadata and
`event_id` alignment. Heatmap label is the **ML saved development prediction
minus CFD development prediction**, not a causal attribution. The heatmap
program currently creates conditional *mean* maps; conditional variance maps
are a separate prospective extension.
