# Chronological Prefix Attribution (CPA)

CPA is an **exploratory, deterministic prefix-imputation** analysis of saved
TOF-PET timing-correction models. It is **not** exact conditional SHAP.

## Inputs and compatibility

Pass a canonical result root or one run directory, plus the **prepared base
waveform cache**. Models and their fitted transforms are loaded by the same
adapters used for TSHAP and model inference. Legacy main-branch runs with a
local `model/` and `feature_transform/transform.pkl` work without a
transformation cache. Recent runs whose models are stored externally require
their model artifact store; provide `--artifact-root` if it is not under the
cache root. The script applies the frozen transformation to each new imputed
waveform; it never reads or writes precomputed transformation-specific features.

Run **one board at a time**; the board and actual prepared-dataset identities
are included in output locations. Only wide-window models are considered.

Example (Linux):

```bash
python -m waveform_analysis.cli explain-cpa \
  --results "$PWD/waveform_analysis/results/UC" \
  --cache-root "$HOME/UC_cache" \
  --board UC \
  --output-root "$HOME/cpa_results" \
  --max-events 1024 --bootstrap 100 --seed 42
```

To analyze only one mode, add `--mode energy` or `--mode timing`. To explain
one saved run, replace `--results` with `--model-dir /path/to/run`.
Use `--models` to override the default selection:
- Energy wide: direct_multirocket_rdst, onishi_cnn, shared_linear_ridge.
- Timing wide: direct_multirocket, onishi_cnn, shared_linear_ridge.

Options include `--fit-events` (1024), `--validation-events` (128),
`--orders` (8 16 32 64 128), `--local-samples` (100),
`--predict-batch-size` (64), `--bootstrap` (100), and `--overwrite`.
The source correction models remain frozen.

## Procedure

1. A seeded development split fits conditional mean AR and standardized AR
   for each detector. Each candidate order must have enough contiguous observed
   samples and stable fitted roots; mean replacement always remains eligible.
2. Choose the minimum validation loss for **each detector and cutoff**:
   average of the local-horizon and full-next-valid-segment MSE ratios
   to the mean baseline. No correction-model information or blind events
   enters imputer selection.
3. Refit selected imputers on the designated development cohort, check
   stability again and fall back to the mean if necessary.
4. Select a reproducible, **model-specific** subset of up to 1024 blind
   events, with exactly the same events for the selected and mean-only
   imputers within that model.
5. Reveal both aligned input waveforms through the cutoff. Forecast the
   remaining current unmasked interval using the chosen AR imputer; from
   the next masked interval onward use the mean. Cutoffs inside masked
   regions are skipped (not shifted), as are out-of-grid cutoffs.
6. Recompute full-waveform predictions through the normal frozen-model
   inference adapter, including its fitted feature transform and mask.
7. Report centered timing-residual RMSE (primary), CTR using the pipeline's
   F1 histogram estimator, paired bootstrap intervals, signed and
   absolute increments, and convergence to the full-input prediction.
   Bootstrap resamples saved event predictions, not ML inference.

Cutoffs in ns, relative to the aligned ML-input grid:
`-1, 0, 2, 4, 6, 8, 10, 15, 20, 25, ...` until the window ends.
The complete-waveform prediction is a separate final reference endpoint.
At the first cutoff the incremental attribution is intentionally undefined.

## Outputs

`<output-root>/<board>/<dataset-identity>/<mode>/<model>/` contains
`metrics.csv`, `attribution.csv`, `imputer_validation.csv`,
`selected_imputers.csv`, `predictions.npz`, `imputer.npz`,
`summary.json`, `timing_resolution.png` and
`prediction_increments.png`. The dataset identity derives from the actual
prepared-cache population and run identity, avoiding collisions between UC
and FBK or identically named datasets.

Imputer selection is a pragmatic approximation. Increments can reflect changes
in the reconstruction rule as well as new observed samples. Mean-only prefix
predictions and complete candidate validation scores are retained as diagnostics
without making imputer comparisons the scientific focus.
