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

Two independent deletion curves are calculated **for each method and event**:
- Positive windows: descending strictly positive scores
- Negative windows: ascending strictly negative scores (most negative first)

At step `k`, the union of the first `k` deleted windows is re-imputed from the **original waveform and all remaining observed samples**, never from previously imputed samples. All original model-retained samples outside that union are available. No retraining/refitting occurs during deletion. Curves show signed output change, absolute error change against the calibrated timing target, and optionally squared error change.

There is no forced deletion of windows with the opposite sign. As a consequence, positive and negative curves can end at different percentages of all windows.

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
    tshap_deletion.png
    deletion_error.png
```

Run unit tests:

```powershell
python -m pytest waveform_analysis/tests/test_local_tshap_deletion.py -q
```

## Interpretation and limitations

The deletion benchmark tests the consistency of each method's ranking **under that method's own imputation**, not an imputation-independent comparison of ranking quality. Different perturbation methods may have substantially different induced waveform distributions, especially for nonlinear models. The 80-sample stride is intended for a low-cost pilot, not guaranteed to resolve sub-ns features. No performance claim should be made before inspecting actual plots and rerunning the test suite on the target machine.
