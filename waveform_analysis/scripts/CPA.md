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

Run boards sequentially with a single command, or use the legacy single-board invocation; the board and actual prepared-dataset identities
are included in output locations. Only wide-window models are considered.

Example for both boards (Linux), with independent prepared caches and shared outputs:

```bash
python -m waveform_analysis.cli explain-cpa \\
  --results "$PWD/waveform_analysis/results" \\
  --boards UC FBK \\
  --uc-cache-root "$HOME/UC_cache" \\
  --fbk-cache-root "$HOME/FBK_cache" \\
  --output-root "$HOME/cpa_results" \\
  --max-events 0 --bootstrap 100 --seed 1
```

Multi-board execution is **sequential**, not parallel. `--results` must contain
`UC/` and `FBK/` subdirectories; `--boards UC` or `--boards FBK`
runs just the requested board. Per-board cache roots are required only for
selected boards. No dataset or cache is reused across boards. With
`--max-events 0`, every available blind event is evaluated. Already complete
results with matching settings are reused. You can interrupt and restart the
same command without `--overwrite`; completed models are skipped.

Single-board example (Linux):

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
- Timing wide: direct_multirocket_rdst, onishi_cnn, shared_linear_ridge.

Options include `--fit-events` (1024), `--validation-events` (128),
`--orders` (8 16 32 64 128), `--local-samples` (100),
`--predict-batch-size` (64), `--bootstrap` (100), and `--overwrite`.
The source correction models remain frozen.


### Model scope and inference

CPA analyzes the frozen combined `direct_multirocket_rdst` model (plus the
optional Onishi CNN and shared Ridge baselines). Standalone
`direct_multirocket` and `direct_rdst` are excluded from CPA. Every model
uses its own saved fitted transformation, independently. There is no
cross-model feature reuse or compatibility check.

### Weighted k-NN causal forecasting

Use `--imputer weighted-knn` to replace AR imputer selection with
development-validated weighted k-NN. Candidate `k` and exponentially
decaying prefix-weight `tau` are selected independently for each detector
and cutoff on held-out development events; `tau=inf` (uniform prefix
distance) is always eligible. Selection minimizes the same balanced
local-horizon and full-unmasked-segment MSE ratios used in CPA. The chosen
`k` and `tau` are then applied to a development reference cohort, and
only the *observed blind prefix* participates in neighbor matching. The
neighbor mean supplies the future waveform up to the next mask; beyond
that boundary the regular development mean is used.

```bash
nohup python -m waveform_analysis.cli explain-cpa \
  --results "$PWD/waveform_analysis/results" \
  --boards UC FBK \
  --uc-cache-root "$HOME/UC_cache" \
  --fbk-cache-root "$HOME/FBK_cache" \
  --output-root "$HOME/cpa_weighted_knn_results" \
  --imputer weighted-knn \
  --knn-k 1 3 5 10 20 50 100 \
  --knn-tau-ns 0.25 0.5 1 2 5 \
  --max-events 1024 --bootstrap 100 --seed 42 \
  > cpa_weighted_knn.log 2>&1 < /dev/null &
```

The original `--imputer ar` remains available. Use different output roots
for meaningful side-by-side comparison. Model inference still uses the
frozen, saved transform and model; no timing-correction model retraining.

## Procedure

1. With `--imputer ar`, a development split fits mean AR and standardized AR\n   for each detector. With `--imputer weighted-knn`, the development split instead\n   chooses nearest-neighbor count and temporal decay for each cutoff. The mean\n   replacement remains the comparison baseline.
2. Choose the minimum validation loss for **each detector and cutoff**:
   average of the local-horizon and full-next-valid-segment MSE ratios
   to the mean baseline. No correction-model information or blind events
   enters imputer selection.
3. Refit selected AR candidates on development or use the development\n   reference population for weighted k-NN. Blind signals never enter fitting.
4. Select a reproducible, **model-specific** subset of up to 1024 blind
   events, with exactly the same events for the selected and mean-only
   imputers within that model.
5. Reveal both aligned input waveforms through the cutoff. Forecast the\n   remaining current unmasked interval using the selected imputer; from
   the next masked interval onward use the mean. Cutoffs inside masked
   regions are skipped (not shifted), as are out-of-grid cutoffs.
6. Recompute full-waveform predictions through the normal frozen-model
   inference adapter, including its fitted feature transform and mask.
7. Report centered timing-residual RMSE and CTR using the pipeline's
   F1 histogram estimator with paired bootstrap intervals. Compare both
   against the full-input reference; no prediction increments are computed.
   Bootstrap resamples saved event predictions, not ML inference.

Cutoffs in ns, relative to the aligned ML-input grid:
`-1, 0, 2, 4, 6, 8, 10, 15, 20, 25, ...` until the window ends.
For timing, the initial valid segment uses only its portion of the ordinary\nabsolute schedule. Subsequent valid segments use **only** cutoffs 1, 3, 5, 7,\n9, 14, 19, ... ns after their first retained sample. Absolute cutoffs must\nnot leak into later segments. Cutoffs inside masked regions are excluded, not\nshifted. The complete-waveform prediction is a separate final reference endpoint.

## Outputs

`<output-root>/<board>/<study-folder>/<mode>/<model>/` contains
`metrics.csv`, `imputer_validation.csv`,
`selected_imputers.csv`, `predictions.npz`, `imputer.npz`,
`summary.json` and `timing_resolution.png`. The study folder is the readable saved-result batch directory (for example
`benchmark_48V_final`). The full cohort fingerprint is stored in `summary.json`
and checked before reusing or overwriting a result. If two distinct datasets
share a study folder, CPA raises an explicit collision error instead of using
opaque hashes as directory names. Change the study-folder name or the output
root to store both.

Imputer selection is a pragmatic approximation. Performance changes with
cutoff may reflect both newly observed samples and altered reconstruction. Mean-only prefix
predictions and complete candidate validation scores are retained as diagnostics
without making imputer comparisons the scientific focus.
