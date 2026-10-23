# Waveform timing analysis

One pipeline fits selection, preprocessing, timing calibration and model-input
rules on **control**, selects/fits models on **development**, and evaluates the
frozen final model on independent **blind/test** data. Blind events never select
hyperparameters or fit transformations. Results, reporting and XAI have one
canonical folder hierarchy. Historical results must be converted explicitly.

See [ARCHITECTURE.md](ARCHITECTURE.md) for ownership/interfaces and
[CLEANUP_NOTES.md](CLEANUP_NOTES.md) for removed code and external archive candidates.

## Running

From the repository root, with Python and `waveform_analysis/requirements.txt`
installed in your environment:

```bash
python -m waveform_analysis.cli check-batch --config waveform_analysis/config/batches/benchmark_UC_test.json
python -m waveform_analysis.cli batch --config waveform_analysis/config/batches/benchmark_UC_test.json
python -m waveform_analysis.cli report waveform_analysis/results/UC/BATCH
python -m waveform_analysis.cli plots --results waveform_analysis/results/UC/BATCH
```

`batch` plans dependency-scoped reuse/recomputation and asks before rebuilding
existing incompatible/CV-invalidated model scopes. Use a different batch output
folder for a different experiment that must coexist with the original. Shared
settings cannot silently overwrite retained models omitted from a new sweep.

`report` and `plots` read saved scalar metrics/prediction arrays. They do not
migrate files, refit a model or backfill predictions by inference. They support
`--exclude-models MODEL ...`. Missing diagnostics remain unavailable.

Each mode/window report also exports an `ensemble_ctr` heatmap and CSV matrix.
Off-diagonal cells give the CTR of the equal-weight arithmetic mean of two saved
blind predictions, aligned by event ID and evaluated with the original CTR
settings and saved LED residuals. No weights, calibration or models are fitted;
the averaged prediction is neither recentered nor clipped again. Diagonal cells
retain the saved single-model CTR on its original population. Colours use an
absolute CTR scale (`heatmap.ensemble_cmap`, default `viridis_r`), with values in
ps. `ensemble_ctr_n.csv` records each cell's event count; off-diagonal populations
can differ. Unavailable pairs are blank and explained in `ensemble_ctr.json`.
These matrices are regenerated from saved arrays even with `plots`, so older
runs can gain this report without training or access to acquisition files.

Explicit saved-model operations:

```bash
python -m waveform_analysis.cli diagnostics --results waveform_analysis/results/UC/BATCH
python -m waveform_analysis.cli xai --results waveform_analysis/results/UC/BATCH --models direct_hydra shared_hydra
```

`diagnostics` generates missing full-development predictions using the saved
final model and verified prepared caches. `xai` accepts `--cache-root`,
`--group-size-samples`, and `--max-events`. Neither command trains a model or
reads raw acquisitions. Original saved masks, scalers/transforms and prediction
limits are retained. RDST coefficient XAI uses persisted shapelets and model
coefficients; temporal occlusion uses the frozen per-detector control mean.

## Global waveform FFT (read-only)

Compute the full-window, population-average **event-wise** power spectral density
for detector 1, detector 2, and their event-wise difference. Uses the verified
prepared waveform cache and saved sample mask; no model predictions, model
inference, new feature transformations, synthetic signals or tail windows.
Event-wise mean subtraction and a Hann window precede each FFT. Requires a
uniformly sampled time grid after the model's sample mask.

```powershell
python -m waveform_analysis.scripts.global_fft --results PATH_TO_BATCH --models direct_rdst --role blind
```

Outputs under `<run>/explanations/global_fft/<identity>/`:
`spectra.png` (three overlaid PSDs), `spectra.npz` (frequency bins in Hz,
three spectra and time grid), and `summary.json` (sampling and provenance).
The PSDs are averaged over all events, not FFTs of mean waveforms.

## Correction-quantile waveform examples (read-only)

Select **real paired waveforms** nearest to the 10th, 30th, 50th, 70th and
90th percentiles of the **original model's saved timing corrections**. This is
output-space selection only: no clustering, feature weighting, Ridge reload,
new RDST/MultiRocket transforms, surrogate, or synthetic waveforms.

```powershell
python -m waveform_analysis.scripts.correction_examples --results PATH_TO_BATCH --models direct_rdst shared_rdst
```

The default population is `blind` and reads `artifacts/pred.npz`. For
already-saved full-development diagnostic predictions, add
`--role development` to use `artifacts/development_pred.npz`. Generate that
optional artifact first using the existing `diagnostics` CLI command if absent.
Use `--quantiles 0.1 0.3 0.5 0.7 0.9` to customize fractions and
`--cache-root PATH` to locate the original prepared data if moved. Predictions
are aligned to prepared waveforms by exact event ID; mismatches fail rather
than silently pairing unrelated events. The original universal timing sample
mask is honored in the plotted signal/time grid.

Outputs live under `<model_run>/explanations/correction_examples/<identity>/`:
`correction_distribution.png` shows saved predictions and selected examples;
`waveforms.png` shows the five normalized two-detector waveform pairs on common
axes; `examples.csv`, `examples.npz` and `summary.json` preserve event IDs,
quantiles, timing corrections and provenance. These examples represent different
**predicted correction levels**, not necessarily different pulse-morphology groups.
In addition to individual real-event examples, `--half-width 0.05` groups
all events in 5th–15th, 25th–35th, 45th–55th, 65th–75th and 85th–95th
**empirical prediction-percentile bands**. The band boundaries are based on
prediction rank (ties broken by event ID), so groups have comparable sizes even
if predictions repeat. Per-sample means and sample standard deviations (±1 SD)
are computed on the original prepared normalized signals, without synthetic
waveforms. Channel difference is computed **event by event**, retaining the
covariance of the two channels in its SD. Computation streams waveform batches
instead of loading all event waveforms into RAM.

```powershell
python -m waveform_analysis.scripts.correction_examples --results PATH_TO_BATCH --models direct_rdst --half-width 0.05 --zoom-max-ps 10000
```

New outputs are `groups_full.png` and `groups_zoom.png`, each with paired
waveforms on the left and detector-1-minus-detector-2 on the right for **every**
quantile band. The right-hand curve is centered by subtracting the **mean D1−D2 waveform across the entire selected population** (not just the plotted quantile bands), leaving the within-group ±1 SD unchanged. `groups.npz` retains the original difference means as `difference_mean`, the centered curves as `difference_centered_mean`, and the reference as `global_difference_mean`. `groups.csv` reports group sizes and correction distributions;
`groups.npz` contains numerical means/SDs, time grid, and member event IDs.
Existing individual example outputs remain available. Increasing
`--half-width` gives smoother group averages but mixes a wider range of
predicted corrections. The zoom uses the saved time grid up through
`--zoom-max-ps` (default 10000 ps).

The script also saves `temporal_association.png` and `temporal_association_zoom.png`: sample-wise Pearson correlation between the real event-level detector difference (D1−D2) and the original model prediction, using the **entire** selected population, not only the quantile groups. `groups.npz` stores `temporal_difference_correlation`, the corresponding difference mean/std, and prediction mean/std. Constant-difference time samples are marked NaN; association is descriptive and should not be interpreted as model attribution.

Derivative-based association is also exported as `temporal_association_derivative.png` and `temporal_association_derivative_zoom.png`: Pearson r of `np.gradient(D1-D2, time_ps)` with original predictions. Derivatives use the actual saved masked time grid and no smoothing. `groups.npz` contains the derivative correlation, mean and sample SD; constant derivative samples are NaN. Differentiation highlights local slope variations but can amplify noise.

Temporal association plots accept `--group-samples 1` (default), `2`, `3`, etc. For values above one, consecutive **nonoverlapping** samples of the real detector difference are averaged, with the incomplete final group discarded; the effective time coordinates are the means of the original time coordinates. Both raw-difference and derivative correlation are recalculated on these grouped event-level signals. Grouping does not alter the original model predictions, the group waveform plots, or saved spectra. The effective time grid and grouping factor are stored in `groups.npz` as `association_time_ps` and `group_samples`.

Nothing in the original run is modified.

Global continuous-output explanations can be generated from saved models:

```bash
python -m waveform_analysis.cli explain-regression --results waveform_analysis/results/UC/BATCH --models direct_rdst shared_rdst
```

This explains the **original frozen feature space**. No new shapelets, kernels,
scalers or transformations are constructed or fitted. It supports models with
an explicit saved feature transform (RDST, MiniRocket, MultiRocket, Hydra and
linear models). CNN/MLP models without one are rejected; use input-level `xai`
for those models. Original models, masks, clipping limits, benchmark predictions
and stage receipts are preserved.

Prepared development and blind caches must be available; use `--cache-root PATH`
if they moved. For maximum speed, also retain `model_inputs/` from that cache
root: matching transformed features are reused read-only after checking the
original transform and dataset identities. Otherwise the **saved** transform is
applied once per role in bounded batches. The same features feed the original
predictor, the tree and the contribution calculation. No raw ROOT files are read.

The primary linear-head explanation is an exact additive decomposition using
`original_feature * original_coefficient`, plus the original intercept. RDST
retains its original distance/location/occurrence columns and original direct
or shared formulation, scaling and shapelet IDs. Its three contributions are
summed per shapelet before taking absolute values; shapelets are ranked by mean
absolute net contribution over the explained blind cohort. Signed means,
population SDs, positive fractions and separate feature-type contribution means
are exported. Contributions describe the pre-clipping output; reconstruction
is checked against original predictions **after the original clipping** and
its numerical error and clipped-event count are recorded. These are additive
model explanations, not causal effects or independent credit for correlated
features.

A compact regression tree on those **same original features** remains a
secondary approximation, with depth selected on a 20% development holdout by
prediction fidelity. Only the tree is fitted; the final tree uses all sampled
development rows. Blind data never selects settings. Development predictions
come from the original final model, not out-of-fold predictions. Tree thresholds
are in the original transformed/scaled feature units; a shared RDST column is a
feature difference, not a detector-specific match. The tree need not preserve
the shared teacher's antisymmetry and does not replace its exact contributions.

Options: `--depths 2 3 4 5`, `--min-samples-leaf 20`, and optional `--max-events N`
for a reproducible cap per role (default: all). `--max-shapelets` and
`--shapelet-lengths` were removed: the original transformation is immutable.

Each model receives `explanations/regression_tree/IDENTITY/` with:

- `summary.json`: provenance, frozen transform identity, cache usage, selection,
  blind tree fidelity MAE/RMSE/R² and LED/original/tree timing CTR and RMSE.
- `contributions.npz`, `contributions.csv`, `contributions.png`: original
  linear-head population explanations in ps. RDST groups refer to original
  shapelets; other models use original feature indices.
- `performance.csv` and `performance.png`: paired blind comparison using the
  existing CTR fitter, with point estimates and no bootstrap uncertainty.
  An event cap means a subset comparison; benchmark metrics are never changed.
- `tree.png`, `tree.json`, `rules.txt`: approximate global rules using original
  feature names. For RDST, `shapelets.npz` and `used_shapelets.png` contain
  original shapelets used by the tree or ranked highest by contributions.
- `surrogate.pkl`: **tree only**, with the original transform identity. Its
  `predict` accepts original frozen feature matrices, not raw waveforms. It
  does not duplicate or replace the original saved transformation.
- `predictions.npz`: audited development/blind event IDs, original/tree blind
  predictions, residual targets and leaf assignments.

Identical settings reuse the explanation. The method identity changed, so old
independent-shapelet explanations are never reused. Poor tree fidelity means
its rules are not a faithful approximation; similar CTR alone is insufficient.
No single-event explanations are generated.

Progress logs identify cache reuse or the reason for recomputation, event counts,
completed-batch percentages and approximate ETAs. Long stages emit an elapsed-time
heartbeat every 30 seconds, including individual tree fits. Each candidate depth
reports validation fidelity and duration. The first transform batch can include
JIT compilation, so early ETAs may be inflated.

For headless plotting, set `MPLBACKEND=Agg` and a writable `MPLCONFIGDIR`.

## TSHAP post-training explanations

TSHAP uses the upstream `tshap==0.0.1` package (GPL-3.0) for overlapping-window aggregation and optional region-of-interest identification. Its perturbation scoring is evaluated in bounded batches to avoid constructing all synthetic events simultaneously. Both global and local TSHAP commands skip runs whose configured `window_name` is `onishi`; wide-window runs remain eligible. Skipped runs are logged and no new TSHAP outputs are generated for them.

```bash
python -m waveform_analysis.cli explain-tshap \
  --results waveform_analysis/results/UC/BATCH \
  --output-root ./xai \
  --models direct_linear_ridge direct_hydra \
  --max-events 100 \
  --n-baselines 8 \
  --seed 42 \
  --window-samples 20 --stride-samples 10 \
  --predict-batch-size 64 \
  --roi
```

The method replays **frozen** saved models on verified prepared control/blind caches; it does not refit models, transforms or calibration, nor read ROOT acquisitions. `--cache-root PATH` locates moved original prepared caches. Selects a reproducible, seed-controlled cap of blind events (`--max-events`; default 100) and control baseline waveforms (`--n-baselines`; default 8). The explicit `--seed` (default 42) controls both selections independently of model name, so models using the same control/blind populations and event limits receive the same blind event IDs and baseline IDs. Separate role-specific derived seeds prevent the blind and control selections from sharing the same random stream. All 2 detector inputs remain present for direct and shared models. Samples and stride refer to the **original prepared time grid**, even when the fitted model uses a sample mask. Check the saved `time_ps` grid before selecting settings. Only complete results with matching model/data/settings fingerprints are reused; use `--overwrite` to deliberately replace an existing matching explanation. A different event count, window or baseline count creates a different fingerprinted output, not a duplicate calculation. Incomplete results are not silently replaced.

Artifacts are saved outside the supplied results tree, by default under `./xai/BATCH/RUN/tshap/`; pass `--output-root PATH` to choose another dedicated output directory. The batch and run path components prevent collisions when explaining several batches or model runs. Each run has exactly one `tshap/` directory containing `summary.json`, `attributions.npz` (event IDs, control baseline IDs, time grid, original predictions and window/optional ROI maps), `global_importance.csv`, and `global_importance.png`. The identity fingerprint is stored in `summary.json`, not in the directory name. Matching fingerprints reuse the cache without inference; if the requested settings change, the old contents are replaced in place only after a successful recalculation. Legacy hash-named TSHAP folders are migrated into the stable directory and removed after a successful migration; stale TSHAP artifacts in the results tree are cleaned up after a successful cache hit or replacement. Other result artifacts are left untouched. The global plot uses separate detector waveform panels in physical mV, each coloured by unnormalized mean absolute TSHAP importance in ps/sample. Direct models use detector-specific importance curves and two aligned importance histograms; shared models pool absolute attribution across both detectors, colour both waveform panels with the same curve, and show one pooled histogram. Histogram bars have the configured TSHAP window width; their heights sum mean absolute sample attributions within non-overlapping bins, in ps. The Onishi interval (−1.5, 2 ns) is shaded on all panels. Detector identity is indicated by panel titles without a redundant legend. Plot versioning refreshes PNG/CSV/summary from cached `attributions.npz` without recalculating TSHAP values. The horizontal axis is nanoseconds (the stored grid is picoseconds). Model outputs and window scores are in ps, but the upstream library redistributes window scores over samples; exported per-sample attribution maps are therefore labelled ps/sample. Input waveforms are inverse-transformed to physical mV for plotting. Results are independent of training, blind evaluation, bootstrap and existing XAI receipts.

**Interpretation:** TSHAP computes a two-player SHAP score for each time window against its complement. Its overlapping-window attribution map redistributes those window scores and **is not** an exact additive Shapley decomposition of individual samples. Reference substitutions can also produce unphysical transition boundaries; inspect reference sensitivity before asserting physical importance. Upstream project/paper: https://github.com/mlgig/tshap (Nguyen and Ifrim, ECML-PKDD 2025).

### Local TSHAP case studies (separate from global explanations)

Use the distinct `explain-tshap-local` command to explain individual blind events. It does not overwrite or reuse the global `tshap/` directory; outputs are stored in the stable `xai/BATCH/RUN/tshap_local/` directory.

```bash
python -m waveform_analysis.cli explain-tshap-local \
  --results waveform_analysis/results/UC/BATCH \
  --output-root ./xai \
  --models direct_linear_ridge direct_hydra \
  --events-per-group 2 \
  --extreme-margin 0.05 \
  --n-baselines 8 \
  --seed 42 \
  --window-samples 20 --stride-samples 10 \
  --predict-batch-size 64
```

By default, the command explains 10 events per model: two good positive corrections, two good negative corrections, two bad positive corrections, two bad negative corrections, and two typical corrections. Positive/negative refers to the **predicted correction**. Signed improvement is `abs(target) - abs(target - prediction)` in ps: positive means the absolute timing error decreased; negative means it increased.

The default 5% extreme margin avoids selecting the absolute best/worst cases: good events are selected near the 90th and 95th percentiles of positive improvement within each correction-sign group; bad events are selected near the 5th and 10th percentiles of negative improvement. Typical events are the unused predictions closest to the median predicted correction, which need not be zero. If a category has too few eligible events, the command fails with a category-specific error rather than silently substituting another type of event.

Each event gets an individual three-panel plot titled with its group, event ID, predicted correction, signed improvement, and target correction. The first panel overlays the detector waveforms in mV with detector-specific LED reference levels. Each level is the calibrated baseline-relative threshold added to the baseline calculated from the original blind-native waveform using the same trigger-relative window and `baseline_level_mV` convention as the timing pipeline. Each reference line ends at time zero rather than extending across the full pulse. The second shows the detector difference d_j(t) = s1,j(t) − s2,j(t) after subtracting the per-sample dataset mean, estimated from up to 1,000 reproducibly sampled blind events: d_j(t) − mean_j[d_j(t)]. The third overlays the original TSHAP attribution curves. Existing normalized waveform caches are converted back through the frozen normalization transform to refresh plots without rerunning TSHAP; the plotted amplitudes are in mV and respect the configured normalization limits. The stable `tshap_local/` cache contains `summary.json`, `events.csv`, `local_attributions.npz`, and individual PNGs. Matching settings reuse completed outputs; changed settings replace the contents only after successful recomputation. `--overwrite` forces recalculation. The command predicts across the blind population to select cases but computes TSHAP only for the selected events. Event selection is model-specific because good/bad status depends on that model's predictions; the control baseline sample is reproducible and model-independent for a fixed seed.

## Source configuration

Shipped JSONs under `config/batches` are executable examples. A source batch owns:

- `control_dataset`, `development_dataset`, `blind_dataset`: acquisition paths,
  true TOF and identical channel/polarity definitions for the three roles;
- `results.root` and `results.folder`: batch destination;
- `protocol`: seed, preprocessing configuration, CV, bootstrap, timing-fit,
  model input/output, runtime and XAI settings;
- `sweep`: models, waveform modes, named numeric windows and optional exclusions;
- optional `plot_config` and `save_model`.

Acquisition paths resolve against `waveform_analysis`; referenced JSON paths
resolve against the owning configuration. `check-batch` validates these inputs
and prints common settings/model spaces once, with a compact run inventory.
The effective expanded configuration exists only in memory.

Modes are `energy_to_energy` and `timing_to_timing`. Windows use explicit
`start`/`end` in ns. Input `subsampling` is a positive integer. Sampling intervals
come from event metadata. `config/preprocessing/default_ctr.json` defines the
physical selection, amplitude limits, baseline window, timing grids and
control-fitted LED threshold selection. No implicit smoothing or alternative
CTR estimator is introduced by the result layout.

## Scientific execution

1. Fit control selection/preprocessing and LED calibration for the requested modes.
2. Apply the frozen rules to each dataset role and prepare the requested windows.
3. Timing inputs receive the same control-clipping mask on a given time grid;
   energy inputs keep their full grid. No model-specific training mask is fitted.
4. Fit feature transforms/scalers on control only where required, then freeze
   them for all downstream hyperparameter candidates and final fitting.
5. Select parameters on development, fit on full development, and predict blind.
6. Compute CTR/RMSE and paired LED improvements from the original blind residuals;
   estimate uncertainty by blind-event bootstrap without retraining.
7. Save optional XAI and full-development diagnostic predictions.

CTR uses the existing direct histogram FWHM definition in `utils_fit`. LED
residuals remain centered using the frozen control mean; they are not recentered
on development, test or bootstrap draws. Stored predictions are in ps and keep
the existing output clipping. No detector or single-photon rescaling is added.

Shared formulations preserve their existing detector-difference construction
and antisymmetry where declared; direct formulations use joint paired inputs.
Existing time-grid, mask, feature/scaler and intercept semantics remain
model-specific. Neural Onishi MSE/full-split training is intentionally separate
from the other RMSE/internal-holdout neural training procedure.

## Models and search

| Family | Direct / shared names | Selection |
|---|---|---|
| Linear ridge | `direct_linear_ridge`, `shared_linear_ridge` | Internal sklearn RidgeCV; no outer CV |
| MiniRocket | `direct_minirocket`, `shared_minirocket` | Outer CV, fixed control transform, ridge-alpha search |
| Dilated shapelets | `direct_rdst`, `shared_rdst` | Outer CV, fixed control RDST/scaling, ridge-alpha search |
| Hydra | `direct_hydra`, `shared_hydra` | Outer CV, fixed control Hydra/reference sparse scaling, ridge-alpha search |
| MultiRocket | `direct_multirocket`, `shared_multirocket` | Outer CV, fixed control transform/scaling, ridge-alpha search |
| MultiRocket + dilated shapelets | `direct_multirocket_rdst`, `shared_multirocket_rdst` | Outer CV, fixed control transforms/scaling, joint two-penalty grid |
| Dense MLP | `direct_mlp`, `antisymmetric_mlp` | Outer CV |
| Temporal CNN | `independent_cnn1d`, `shared_cnn1d` | Outer CV |
| Locally connected MLP | `locally_connected_mlp` | Outer CV |
| Onishi CNN | `onishi_cnn` | Outer CV, direct formulation |

Model spaces under `config/model_spaces` declare `fixed`, `grid`, or `optuna`
optimization and typed parameters. Transform parameters remain fixed; ridge
alpha is downstream. Linear RidgeCV uses its configured logarithmic/explicit
alpha grid and MSE objective, not outer CTR selection. A Ridge-only batch need
not define outer CV; mixed/other batches require it. No outer-CV CTR is invented
for internal RidgeCV models.

Outer CV keeps deterministic common folds for each development population. LED
pruning applies from the first candidate at the configured fold threshold;
startup protects only against incumbent pruning. If every candidate is pruned,
the best finite recorded partial score resumes for its remaining folds without
pruning, and that candidate is selected. Saved folds are reused on interruption.

MultiRocket keeps its current feature budget, float32/memory-mapped features,
bounded transformation batches and existing `lsqr` Ridge solver/ownership rules.
Parallelism and chunk settings are execution controls; this cleanup does not
change model precision or replace estimators. Aeon remains pinned to 1.6.0 for
its persisted transform representations.

### Combined MultiRocket/RDST models

`config/batches/test_multirocket_rdst.json` demonstrates both formulations.
Each constituent transform and its per-feature standard-deviation scaler is
fitted once on control, then frozen. The cached float32 feature matrix contains
MultiRocket columns followed by RDST columns; no extra feature-count weighting
or development-fitted scaling is applied. Shared models concatenate the two
families' detector-difference features and retain a zero intercept.

The model spaces search `multirocket_alpha` and `rdst_alpha` jointly on the
existing development folds. Defaults are `[1e3, 3e5, 1e8]` and
`[1e2, 1e4, 1e6]`, respectively: nine candidate pairs. Both penalties must be
finite and positive. Transform settings remain fixed throughout the search.
For each fit, the RDST block is multiplied by
`sqrt(multirocket_alpha / rdst_alpha)` and the existing LSQR Ridge fitter uses
`alpha=multirocket_alpha`. This gives independent squared-coefficient penalties
for the two standardized feature families. The scaling is folded into the saved
coefficients, so predictions use the original frozen features directly.

Feature extraction uses the existing memory-mapped cache and bounded batches.
Candidate scaling only modifies disposable training rows. XAI is unavailable
for these combined models: training and explicit `xai` replay skip them even
when the batch enables XAI for other models.

## Result ownership

```text
BATCH/
  config.json                      # Dataset roles, common channels and protocol
  plots.json                       # Presentation
  manifest.json                    # Execution status and planner receipts
  model_spaces/MODEL.json           # Frozen search/model settings, once
  tables/runs.csv                   # Batch run index
  artifacts/                       # Shared folds/LED statistics
  preprocessing/                   # Shared preprocessing figures
  timing/
    mode.json                      # Effective fit/input/output settings
    onishi/
      context.json                 # Named window and numeric bounds
      population.json              # Counts/IDs, calibration and shared mask
      direct_hydra/
        metadata/
          config.json              # {"model": "direct_hydra"}
          manifest.json            # Model status/provenance
          candidates.json
          best.json                # Selected parameters and validation statistics
          final_fit.json
          blind.json
          bootstrap.json
        tables/                    # Candidate/fold scores
        artifacts/                 # Predictions, bootstrap draws, XAI
        model/                     # Weights/regressor, mask, fitted transform
        plots/                     # Saved model diagnostics
  report/                          # Generated batch tables and figures
```

Each common field has one owner; there are no fully expanded per-model JSONs.
Moving a model independently requires copying its owning batch/mode/window
files too. Moving an entire batch preserves its configuration hierarchy; raw
acquisition/cache references may still require an explicit local cache binding.

`data.results.resolve_run_config`, `read_run_manifest`, `read_run`, and
`discover_runs` are the shared read-only APIs. They reconstruct inherited values
in memory without moving files or modifying source manifests. The conversion
script is the only historical format adapter.

## Group existing results

```bash
python -m waveform_analysis.scripts.group_results --sources waveform_analysis/results/UC waveform_analysis/results/FBK --output waveform_analysis/grouped_results --check
```

Remove `--check` to generate the grouped report. Only canonical converted/native
runs are accepted. Normal grouping and `--check` show compact duplicate choices;
append one `--select RUN_ID` per conflict. `--selection choices.json` accepts
`{"select": ["RUN_ID", ...]}`. Unique models are included automatically; there
is no selection by blind performance. `--json` prints machine-readable details.

A group contains one control/development/blind dataset triple, channel/TOF
configuration, and subsampling factor. Names include `UC_48V_sub1` and
`UC_48V_sub4`. Different datasets with the same board/voltage/factor get a hash
suffix. Different physical windows and modes remain separate. Split diagnostics
print the exact differing metadata fields. Missing subsampling is never guessed.

Saved acquisition paths can differ across machines. `--dataset-aliases FILE`
accepts explicit equivalent paths mapped to the same label:

```json
{
  "C:/old_repo/processed_data/10-03/48V-R1.root": "UC/development_R1",
  "/home/user/repo/processed_data/10-03/48V-R1.root": "UC/development_R1"
}
```

Aliases declare equivalence; they do not verify file content. Population-ID
mismatches warn rather than block scalar comparisons, since their hashes include
machine paths. Metric-definition/calibration incompatibilities still skip
comparison bars with explicit reasons. Models/masks/predictions are never refit.

Generated entries contain:

- `summary.csv`, `sources.csv`, source references and copied saved plots/tables;
- original reporting CTR/RMSE bars, ascending by the respective unrounded score,
  stable model codes, direct/shared colours, bootstrap errors and LED reference;
- ranked `blind_ctr.csv` and `blind_rmse.csv`;
- one caption-free `metrics.tex`: Model, CV CTR, Test CTR, Test RMSE, with a
  leading LED row and models ranked by unrounded test CTR. CV uses fold SD and
  test uses bootstrap SD. Missing values are `n/a`; missing SDs are not zeros.

LED CV statistics come from complete saved selected-candidate folds; otherwise
CV CTR is `n/a`. Test LED uncertainty comes from one deterministic persisted
reference run. Its run IDs are recorded in the entry metadata. No old separate
`blind_ctr.tex`/`blind_rmse.tex` is created in grouped entries.

The top grouping manifest is an index. Dataset/subsampling settings belong to
each group `config.json`; mode/window metadata belongs to its `group.json`.
Source model metadata/configuration and model files are not copied into grouped
output. Original sources remain necessary to regenerate a report.

Repeated commands append new datasets/subsampling/windows without overwriting
unrelated entries. `--overwrite` replaces only conflicting group/mode/windows;
include every model desired in each replaced entry. Publication is staged and
only script-owned canonical outputs can be extended. Regenerate old grouped
report folders in a new destination after converting their source runs.

Additional flags: `--no-summary-plots` (standard-library-only file/table grouping),
`--plot-config FILE` (default `config/plots/default.json`), and
`--copy-predictions` (explicit opaque array copies, never loaded for scalar plots).
Generated grouped output is Git-ignored.

## Historical result conversion

The separate `scripts/convert_results.py` tool is intentionally disposable.
It recovers saved results into a new canonical destination without training,
predicting, changing event arrays or recalculating statistics. Use its `--help`
for discovery, dry-run, artifact and optional cache-conversion options:

```bash
python -m waveform_analysis.scripts.convert_results --sources waveform_analysis/results --output waveform_analysis/recovered_results --check
python -m waveform_analysis.scripts.convert_results --sources waveform_analysis/results --output waveform_analysis/recovered_results
```

The destination must not exist. `--check` reads metadata only and lists every
recoverable/skipped run and destination. Full conversion recovers trusted saved
pickle/PyTorch models with canonical class paths. Use `--metrics-only` to omit
serialized models/samplers when only reporting is needed or a historical class
cannot be recovered. Missing model files are not fabricated.

Conflicting common settings are separated into experiment variants; old
model-specific input policies are retained as explicit overrides. Missing dataset
roles, numeric windows or recorded subsampling are listed as unrecoverable,
never guessed. Fully expanded old batch snapshots can recover absent model-level
configuration. Unknown optional provenance remains unknown.

`--cache-source PATH` explicitly copies and converts a cache tree into
`recovered_results/_caches` and binds recovered configurations to that location.
Without it, existing cache paths are retained and no cache tree is copied or
rebuilt. Compatible historical full-grid array layouts can be recovered without
altering array bytes. Original population/seed identities remain unchanged.

`conversion_manifest.json` records sources, destinations, skipped reasons and
file hashes. Shared batch artifacts (including persisted folds) and preprocessing
figures are retained once per migrated experiment. Model import references may
change serialized bytes; fitted numerical values do not. Saved metrics and
prediction arrays are copied without recalculation. Recovered runs can be grouped
or read without the converter. Old grouped report folders should be regenerated
from their converted source runs.

Keep source folders until conversion checks and your desired reports/XAI succeed.
No runtime module imports the converter, so deleting it afterward is supported.

## Validation

```bash
MPLBACKEND=Agg python -m pytest waveform_analysis/tests -q
```

Use the same scientific dependencies/hardware for exact numerical comparisons.
Tests cover independent signal references, reference/current model outputs,
actual miniature batches, masks, CV/pruning, feature caches, saved-model reload,
reporting, canonical ownership, conversion and resume. Real acquisition data and
GPU execution are not represented by these synthetic CPU checks.
