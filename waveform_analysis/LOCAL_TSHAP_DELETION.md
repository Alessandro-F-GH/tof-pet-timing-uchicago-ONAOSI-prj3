# Local paired TSHAP and signed deletion

Entry point: `python -m waveform_analysis.scripts.local_tshap_deletion`

This is an independent, **local-only** experiment and does not modify model artifacts or the existing `explain-tshap-local` command. It uses frozen final-fit models, verified saved blind predictions, and a Development-only reference cohort.

## Example: Windows PowerShell

```powershell
git pull origin codex/artifact-store-refactor

$run = "waveform_analysis/results/UC/energy/wide/direct_multirocket_rdst"
$cache = "$env:USERPROFILE\UC_cache"
$out = "$env:USERPROFILE\xai\UC\local_tshap_deletion\multirocket_rdst"
$shared = "$env:USERPROFILE\xai\UC\frozen_imputer_cache"

python -m waveform_analysis.scripts.local_tshap_deletion `
  --model-dir "$run" `
  --cache-root "$cache" `
  --output-root "$out" `
  --imputer-cache-root "$shared" `
  --events-per-group 1 `
  --window-samples 80 --stride-samples 80 `
  --reference-events 1024 --ppca-components 32 `
  --knn-k 20 --knn-weighting distance `
  --predict-batch-size 64 --seed 42
```

**Use the actual saved run directory** containing `metadata/manifest.json`; the example path may differ on your machine.

A second model on the **same board and mode** can reuse the event selection and PPCA reference cache:

```powershell
python -m waveform_analysis.scripts.local_tshap_deletion `
  --model-dir "PATH_TO_OTHER_SAVED_RUN" `
  --cache-root "$cache" `
  --output-root "$env:USERPROFILE\xai\UC\local_tshap_deletion\other_model" `
  --imputer-cache-root "$shared" `
  --event-ids-file "$out\selected_events.json" `
  --window-samples 80 --stride-samples 80 `
  --reference-events 1024 --ppca-components 32 `
  --knn-k 20 --seed 42
```

Repeat selection independently for a different board or signal family: event IDs across UC/FBK or energy/timing are not generally interchangeable.

## What is measured

Windows are **paired** (both detector channels) and non-overlapping. TSHAP ranks only interior windows fully included in the final model's retained-sample mask.

For each imputation method (KNN / joint PPCA / timewise Development mean), compute

```text
phi(W) = 0.5 * [ f(impute(observed W)) - f(impute(observed none))
                 + f(original) - f(impute(observed complement W)) ]
```

The imputer approximates a **conditional waveform mean**; evaluating a nonlinear ML model on that mean is not generally the same as the conditional mean of model outputs. These scores are **Shapley-style scores for the defined point-imputation value function**, not guaranteed conditional-expectation Shapley values.

Two **complete progressive deletion sequences** are calculated for every method and event:

- **Positive → negative:** remove positive-attribution windows in descending score order, then negative-attribution windows from most negative toward zero
- **Negative → positive:** remove negative windows first (most negative to zero), then positive windows in descending score order

Zero-score windows are removed last. At every intermediate step, the union of the deleted windows is re-imputed using **only the original still-observed samples**, never earlier imputations. PPCA and KNN use the same frozen Development state/reference population, without refitting.

**Both complete curves end with all model-retained samples removed.** Existing TSHAP excludes boundary windows, so the final cleanup step also removes unranked retained samples. The terminal waveform uses the **same frozen timewise Development mean for all three methods**, even if PPCA's unconditional mean differs from that arithmetic mean. As a result, all curves have exactly the same endpoint model prediction. The final cleanup may cause an additional jump and is reported separately in the saved curve stage.

The figure `deletion.png` has two panels (positive→negative and negative→positive) with three method curves each; y-axis is the **signed change in the model prediction** from its untouched value (ps), and the x-axis is the fraction of model-retained input coordinates deleted. This is not the normalized dAUC/iAUC measure from the referenced article; it is a corresponding local signed output-deletion visualization.

**Important:** old results only contain partial sign-only deletion sequences. A new run with the model is necessary to compute the missing intermediate model outputs. `--replot-only` intentionally refuses old sign-only results; it cannot extrapolate the full curves.

## Reuse and computational cost

- **PPCA:** fit once on 1024 seeded Development waveforms (two detectors jointly), cache the factor-loading matrix and isotropic noise in a content-identified NPZ. Reuse for all events and subsequent model runs with identical reference waveforms and settings. For every new observed-mask pattern, calculate the Gaussian posterior using the **same** frozen PPCA parameters; Cholesky factors are cached by observed mask for reuse among events.
- **KNN:** fixed Development reference waveforms, independent time-point z-scoring with a variance floor to balance detectors. Each query gets **one blocked reference-distance pass** and squared per-window contributions. Complement and cumulative deletion distances are sums/differences of those contributions. KNN is never refitted; neighbors and weights are recomputed for each observed mask, using original observed values only. The selected k and weighting are fixed CLI hyperparameters, **not cross-validated in this experiment**.
- **Mean:** frozen timewise arithmetic mean of the same Development references, reused for every mask.
- The same event IDs, time grid, model, seed and reference cohort are used for all three methods *within a run*. An explicit `--event-ids-file` keeps event IDs fixed across models.

The runner re-evaluates the frozen model on the perturbed signals (unavoidable) but validates its untouched-prediction replay against saved blind outputs before explanation.

## Outputs

```text
output/
  selected_events.json
  reference_details.npz
  deletion_curves.csv
  summary.json
  event_<ID>/
    results.json
    event_waveforms.png
    tshap_profiles.png
    deletion.png
```

**Replot existing numerical results without running inference, PPCA or KNN:**

```powershell
python -m waveform_analysis.scripts.local_tshap_deletion `
  --replot-only `
  --output-root "$out"
```

This also removes obsolete `tshap_deletion.png` and `deletion_error.png` from each event folder. Existing *full-sequence* numerical deletion predictions can be replotted without inference. Sign-only results from previous versions need a new run. **Older runs lack persisted per-event waveforms.** Recover them without rerunning TSHAP or model inference using `--waveforms-only` and the original model run/cache: The two complete signed sequences appear in the upper and lower panels of one `deletion.png` figure, for three imputers; the common final point is the mean-imputed waveform prediction. The TSHAP figure shows three signed bar profiles with a common y scale and time in ns from the frozen waveform grid. The event waveform figure shows the two untouched ML inputs converted from frozen min-max units to physical mV. Each line segment is colored by the pointwise z-score relative to the same seeded Development reference cohort (separately for each detector and sample time): blue means below the Development mean, red above it, and white near the mean. The color map clips at ±3, with a detector-specific standard-deviation floor to avoid enormous z-scores on nearly constant samples. Actual amplitudes remain on the y-axis in mV. The z-score values and reference counts are persisted in each event JSON; reference mean/std arrays are stored in reference_details.npz.

```powershell
python -m waveform_analysis.scripts.local_tshap_deletion `
  --waveforms-only `
  --model-dir "$run" `
  --cache-root "$cache" `
  --output-root "$out"
```

This resolves saved event IDs in the frozen Blind prepared dataset, checks the saved model fingerprint and time grid, recovers the same Development reference cohort from saved reference event IDs, computes per-sample detector-wise z-score statistics, converts original model input waveforms back to physical mV, writes `waveform_mv` and `waveform_zscore` to each event JSON, and creates `event_waveforms.png`. No model, PPCA, or KNN inference is needed. Then `--replot-only` can redraw all figures.

Run unit tests:

```powershell
python -m pytest waveform_analysis/tests/test_local_tshap_deletion.py waveform_analysis/tests/test_local_tshap_deletion_plots.py -q
```

## Interpretation and limitations

The deletion benchmark tests the consistency of each method's ranking **under that method's own imputation**, not an imputation-independent comparison of ranking quality. Different perturbation methods may have substantially different induced waveform distributions, especially for nonlinear models. The 80-sample stride is intended for a low-cost pilot, not guaranteed to resolve sub-ns features. No performance claim should be made before inspecting actual plots and rerunning the test suite on the target machine.
