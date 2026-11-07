# Full-signal windowed MultiRocket/RDST surrogate

Frozen per-window MultiRocket/RDST transformations are fitted exclusively on the prepared control (calibration) waveforms. The full-signal Ridge target is saved teacher predictions, with no teacher inference.

## Linux command

```bash
python -m waveform_analysis.scripts.evaluate_windowed_rocket_surrogate \
  --model-dir "$PWD/waveform_analysis/results/UC/multirocket_rdst/energy/wide/direct_multirocket_rdst" \
  --cache-root "$HOME/UC_cache" \
  --transform-cache-root "$HOME/windowed_rocket_cache/UC" \
  --output-root "$HOME/full_signal_distillation/UC_windowed_rocket" \
  --window-length 256 --features-per-window 256 --feature-family both \
  --lambda-min 1e2 --lambda-max 1e7 --n-lambda 7 \
  --transform-batch-size 128 --n-jobs 2
```

## PowerShell

```powershell
python -m waveform_analysis.scripts.evaluate_windowed_rocket_surrogate `
  --model-dir "$PWD\waveform_analysis\results\UC\multirocket_rdst\energy\wide\direct_multirocket_rdst" `
  --cache-root "$HOME\UC_cache" `
  --transform-cache-root "$HOME\windowed_rocket_cache\UC" `
  --output-root "$HOME\full_signal_distillation\UC_windowed_rocket" `
  --window-length 256 --features-per-window 256 --feature-family both `
  --lambda-min 1e2 --lambda-max 1e7 --n-lambda 7 `
  --transform-batch-size 128 --n-jobs 2
```

## Procedure

1. Use the same deterministic seed-42 training/tuning/holdout split (74,714/500/500 for UC).
2. Split the paired waveform into disjoint blocks; merge a final tail shorter than 11 samples.
3. Fit the two transformations without target labels on prepared control only. Keep aeon 1.6.0 and existing transform parameters.
4. Select a fixed total feature budget per block; with both families, allocate approximately half to each. Use deterministic evenly spaced columns, saved alongside fitted transforms.
5. Transform development events per window in bounded batches; cache matrices as disk-backed npy files.
6. Fit an intercept-enabled Ridge by a centered LSQR LinearOperator reading cached feature blocks. Evaluate each log-spaced lambda on tuning; select by tuning RMSE.
7. Evaluate the holdout once at the selected lambda.

## Cache and outputs

Transform cache: one hash-keyed directory per window containing transforms.pkl, transform_manifest.json, and a development_<identity>.npy feature matrix plus metadata. Completed blocks survive interruption; incomplete blocks rebuild. Changes to lambda grid reuse those blocks.

Output directory: evaluation.log with progress and every lambda, lambda_scan.json, surrogate_ridge.npz, diagnostics.npz, and summary.json.

Feature storage estimate: 4 bytes × development events × total selected features. The iterative Ridge scan can be slow due to repeated disk reads. Check the LSQR iteration-limit warnings.

Scope: full-waveform distillation only; no conditional masks, perturbation or TSHAP yet.