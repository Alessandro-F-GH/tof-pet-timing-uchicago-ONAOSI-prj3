# Waveform ML pipeline

Implementation packages are `core/`, `signal/`, `data/`, `models/`, `engine/`,
and `reporting/`. Existing `ml_pipeline` imports remain compatible. See
[ARCHITECTURE.md](ARCHITECTURE.md) for the package layout, typed interfaces,
pure NumPy timing/baseline APIs and exact numerical regression checks.

The pipeline uses three explicit scientific dataset roles:

1. **control** — fit preprocessing, event-selection and LED criteria once;
2. **development** — apply the frozen control rules, select/train the model using the method described below;
3. **blind** — apply the same frozen rules and evaluate the already-selected final model once.

The blind dataset never participates in preprocessing fitting, fold construction, hyperparameter tuning, pruning, model/window/formulation selection, or final training.

## Model input transforms

Model-input transforms are prepared outside the fold loop when they do not belong to the downstream estimator fit.

- Linear Ridge inputs are deterministic: `s1-s2` for the shared model and `[s1,s2]` concatenation for the direct model are materialized once per dataset; RidgeCV uses the entire development feature matrix.
- MiniRocket transforms (including their fitted scaling) are fitted once on the prepared control dataset, then frozen and applied once to development and blind. CV tunes/fits only the downstream estimator on the cached features.
- The frozen transform identity is part of the CV/final-fit fingerprints, so results produced with fold-fitted transforms are not resume-compatible.

## Linear RidgeCV

`direct_linear_ridge` and `shared_linear_ridge` use sklearn `RidgeCV` to select
lambda and fit all development events. They bypass the normal fixed/grid/Optuna
search, common outer folds and pruning. Direct Ridge retains an intercept;
shared Ridge remains a difference scorer without an intercept. MiniRocket's
Ridge estimator continues to use the existing outer validation pipeline.

Model-space settings:

```json
{"model": "direct_linear_ridge", "ridge_cv": {
  "alphas": {"low": 0.001, "high": 100.0, "num": 6},
  "cv": null, "gcv_mode": "auto", "scoring": "neg_mean_squared_error"
}}
```

`cv: null` selects efficient leave-one-out CV; an integer >= 2 selects sklearn
K-fold CV. `alphas.low` and `alphas.high` are inclusive endpoints;
`alphas.num` is the number of logarithmically spaced values. Explicit lists
remain supported for compatibility. The bundled model spaces specify 50 logarithmically spaced positive alphas from
1e-8 to 1e3. Direct API calls that omit the grid use the typed 23-value fallback. This intentionally changes lambda selection from CTR-based outer
validation to MSE on the estimator's unmodified predictions. Output clipping,
preprocessing, blind CTR/RMSE, bootstrap and XAI follow the common pipeline.
The independent control/blind populations never become Ridge training events.

`ridge_cv` is the canonical lambda configuration. The parser also accepts
`parameters.ridge_alpha` grids/ranges as input aliases; `ridge_cv` takes
precedence. Outer optimization/solver settings do not apply to these two models. Ridge-only batches may omit `protocol.cross_validation`;
mixed batches retain it for the other models.

Ridge outputs retain `best.json`, `final_fit.json`, saved models and blind
artifacts. Outer validation fields are null and no outer fold/candidate table
is generated. `report/tables/ridge_cv.csv` records lambda, internal MSE in ps²
and full development training count. Validation tables/plots and development
CV winner comparisons include only models using outer CV; Ridge window plots
show every window separately. Alpha-grid changes invalidate the Ridge fit, while unused outer-CV changes do not.

## Development CV and pruning

For models other than the two linear Ridge variants, every `(mode, window)` development population gets one deterministic K-fold definition derived from the single batch seed. Every participating model and candidate uses the same folds in the same order. CTR and RMSE are always computed, together with LED CTR/RMSE on the exact same validation events.

Pruning is controlled by `protocol.cross_validation.pruning`. Startup candidates complete all folds. Later candidates are compared only on folds already completed by that candidate. LED comparison is evaluated first; then, if available, the best fully evaluated incumbent is compared on those same fold IDs. Lower is better and pruning uses a strict `degradation_ps > tolerance_ps` rule, so equality does not prune. Tolerances may be scalar or fold-count mappings.

Optuna/TPE only proposes parameters. Fold execution and pruning are implemented by the repository evaluator, so fixed/grid/Optuna all follow the same protocol.

## Blind evaluation and uncertainty

After development selection, the chosen configuration is trained once on the complete development population and applied once to the prepared blind population. Blind CTR/RMSE central values are computed from the original non-resampled residual distribution.

Event bootstrap is then used only for uncertainty estimation. Each draw resamples blind event indices with replacement and applies the same indices to ML and LED residuals. The model is never retrained inside bootstrap.

- development CV std = fold-to-fold validation variability;
- blind bootstrap std = event-level uncertainty conditional on the final fitted model;
- blind bootstrap is not training-instability uncertainty.

## Configuration

Batch configs require `control_dataset`, `development_dataset`, `blind_dataset`, `results`, `protocol`, and `sweep`.

There is exactly one configured `protocol.seed`. Semantic seeds for folds, candidate fitting, Optuna, final fitting, bootstrap and XAI are derived from it.

Example pruning:

```json
"cross_validation": {
  "folds": 5,
  "shuffle": true,
  "metric": "ctr",
  "minimum_events_per_fold": 50,
  "pruning": {
    "enabled": true,
    "startup_complete_candidates": 3,
    "min_folds_before_prune": 1,
    "max_degradation_ps": 5.0,
    "prune_if_worse_than_led": true,
    "led_max_degradation_ps": 0.0
  }
}
```

Plot styling is centralized in `config/plots/default.json`; scientific thresholds do not belong there.

## Commands

From the repository root:

```bash
python -m waveform_analysis.cli check-batch --config waveform_analysis/config/batches/test_ridge.json
python -m waveform_analysis.cli batch --config waveform_analysis/config/batches/test_ridge.json
python -m waveform_analysis.cli plots --results waveform_analysis/results/FBK/test_ridge_48V_R1_R2
```

The batch command prints a `KEEP` / `RESUME` / `RUN` / `REBUILD` execution plan before destructive changes.

## Result layout

```text
results/<folder>/
├── manifest.json
├── config.json
├── plots.json
├── tables/
│   └── runs.csv
├── preprocessing/
│   ├── control/<mode>/
│   │   ├── plots/
│   │   └── tables/
│   ├── development/<mode>/
│   │   ├── plots/
│   │   └── tables/
│   └── blind/<mode>/
│       ├── plots/
│       └── tables/
├── artifacts/<mode>/<window>/...
├── <mode>/<window>/<model>/
│   ├── metadata/
│   │   ├── manifest.json
│   │   ├── state.json
│   │   ├── config.json
│   │   ├── candidates.json
│   │   ├── best.json
│   │   ├── final_fit.json
│   │   ├── blind.json
│   │   └── bootstrap.json
│   ├── tables/
│   │   ├── folds.csv
│   │   └── cv.csv
│   ├── artifacts/
│   │   ├── pred.npz
│   │   ├── bootstrap.npz
│   │   └── xai.npz
│   ├── model/
│   └── plots/
└── report/
    ├── tables/<mode>/<window>/
    └── plots/<mode>/<window>/
```

`pred.npz` persists blind event IDs explicitly for model-to-model alignment.
`folds.csv`, `cv.csv` and `candidates.json` belong to outer-CV runs;
RidgeCV records lambda selection in `metadata/best.json` and the report table
`report/tables/ridge_cv.csv`.

## Resume and dependency invalidation

Completed CV folds and candidates are persisted as they finish. Interrupted candidate CV resumes from the first missing fold. Persisted pruning decisions remain authoritative while scientific dependencies are unchanged.

Changes are scoped:

- development/CV/model-search changes invalidate CV and downstream stages for the affected model;
- blind-only changes preserve compatible development CV and selection;
- bootstrap-only changes reuse blind predictions and central values;
- XAI-only changes reuse CV, final selection and blind predictions;
- plot-only changes regenerate plots/reports from persisted results only.

Batch results must use the supported schema checked by the dependency planner.

## Reporting

Development CV plots use one-based candidate order numbers on the x-axis, sorted
by evaluation order. Optuna runs use saved trial numbers; new fixed/grid runs
record `candidate_order` in `metadata/candidates.json`. Older fixed/grid results
without that metadata use stable numbering from their saved CV table order.

Reporting consumes persisted numeric artifacts and produces validation/blind summaries, model-output correlations aligned by blind event ID, paired CTR/RMSE model-difference matrices with paired event bootstrap, blind RMSE-vs-CTR plots, validation-vs-blind plots, and window comparisons.

At model level, `plots/development.png` mirrors `plots/blind.png` for the complete
prepared development population: final-model corrected residuals and the
LED reference centered using the fixed control mean. Both use the same histogram styling.
This is a training-population diagnostic, not out-of-fold validation. Future runs
save its arrays in `artifacts/development_pred.npz` after scientific evaluation;
refitting invalidates that artifact, while bootstrap-only changes preserve it.
Diagnostic inference preserves Python, NumPy and PyTorch random-generator states.

For compatible runs, CLI reporting can backfill the development arrays by loading the
saved final model and a uniquely matching prepared development cache. It performs
inference only, using the existing preprocessing, sample mask, feature transform
and output limits. It never loads raw acquisitions, refits preprocessing or trains
a model. The control JSON is optional when the cache can be verified from the
saved development protocol, control-directory hash or neural final-fit fingerprint.
Saved cache paths are also resolved within a relocated `waveform_analysis` checkout.
Missing models/caches or ambiguous cache identities produce a warning
and leave the development plot unavailable; blind reporting remains available.
Once saved, subsequent plot regeneration reads the diagnostic arrays directly.

Blind distribution comparisons also include `plots/blind_boxplot.png` for each
model and a combined `report/plots/<mode>/<window>/blind_boxplot.png`. The combined
plot uses stable model codes, orders direct models before shared models and obeys
CLI reporting exclusions. Each model has adjacent LED and corrected boxes for
its paired finite blind events. LED residuals already subtract the fixed control
mean, exactly as in metrics and histograms. Neither series is recentered in plots.
Boxes show the 25th–75th percentiles and median. Whiskers extend to the most extreme
observations within 1.5 interquartile ranges of the box by default; every outlier
is displayed and the axis covers the full finite range. The `boxplot` section of
the plot configuration controls whiskers, dimensions, colours and outlier styling.
These plots help inspect central spread, bias and tails; quartile widths are not CTR.

All matrix axes list direct models first, then shared models, alphabetically within
each formulation. Numerical comparisons retain their original bootstrap seeds;
cached matrices are reordered or subsetted without recomputing their entries.
Standard-deviation matrices are no longer exported or plotted; obsolete outputs
are removed during regeneration. Blind metric uncertainties remain in summary
tables and plot error bars. Scatter model labels appear in an external legend,
with Pearson statistics in the title rather than inside the plotting area.

Each mode/window also has `blind_ctr` and `blind_rmse` bar charts sorted by ascending
blind CTR and RMSE respectively (best first), bootstrap error bars and a dashed LED
reference. Integer picosecond labels above the bars round only the displayed values;
the bar heights and exported metrics retain full precision. Stable codes such as
`D-MLP` and `S-CNN` identify bars; the external legend gives full model names.
Matching `blind_ctr.csv`/`.tex` and `blind_rmse.csv`/`.tex` files are written under
`report/tables/<mode>/<window>/`, with a code column for manuscript use. Each
result root is reported independently; boards are not pooled. Reference the
generated LaTeX table label explicitly when including the fragment in the paper.

To omit models from all regenerated summaries and comparisons:

```bash
python -m waveform_analysis.cli report waveform_analysis/results/FBK/benchmark_48V_R1_R2 --exclude-models direct_mlp independent_cnn1d
python -m waveform_analysis.cli plots --results waveform_analysis/results/FBK/benchmark_48V_R1_R2 --exclude-models direct_mlp independent_cnn1d
```

Exclusions use exact space-separated model names and affect reporting only.
Unknown names and excluding every completed model produce an error. Running again
without exclusions restores the full report from the saved runs.

Window comparison winners are selected from development CV only; their blind performance is then displayed together with the LED reference baseline. Aggregate blind RMSE-vs-CTR comparison plots also include the LED reference. CTR/RMSE bar annotations use integer-rounded picoseconds. XAI is grouped temporal occlusion and stores numeric importance separately from its plot.

## Scientific figure styling

All waveform report and preprocessing figures share
[`config/plots/default.json`](config/plots/default.json). The defaults use a
colour-blind-friendly palette, serif typography, subtle horizontal grids,
frameless legends and 300 dpi exports. Top/right spines are hidden. Bar-value
annotations are disabled by default to keep comparisons uncluttered; enable
`bar.show_values` to display them. Heatmaps grow with the number of models and
use contrasting text on dark cells.

Choose a batch-specific configuration using the top-level `plot_config` path.
The selected configuration is copied to the result root as `plots.json`.
Customize `font`, `palette`, `axes`, `ticks`, `legend`, `grid`, and the individual
plot sections (`cv`, `histogram`, `scatter`, `bar`, `heatmap`, `xai`,
`preprocessing`). `formulations` and `reference` set comparison colours;
`preprocessing.detector_colors` and `xai.detector_colors` set waveform colours.
The style is applied locally and does not modify global Matplotlib settings.
Omitted optional style settings receive defaults.

For publication, set `output.format` to `pdf` or `svg`; PDF embeds TrueType fonts.
Set `output.dpi` for PNG resolution, and `output.bbox_inches` / `pad_inches` for
export margins. To restyle existing model/report figures, edit the result root's
`plots.json` and run:

```bash
python -m waveform_analysis.cli plots --results waveform_analysis/results/FBK/test_ridge_48V_R1_R2
```

This regenerates plots from saved numerical artifacts without training. Frozen
preprocessing diagnostics receive the style when generated by the batch pipeline.
Styling does not change timing estimates, histogram binning, CTR/RMSE or model
selection.

## Selected-event LaTeX tables

Batch reporting also writes `report/tables/datasets/selected_events.csv` and one
`UC_selected_events.tex` or `FBK_selected_events.tex` fragment under the batch
result root. The shipped configurations identify the electronics board through
the `UC/` or `FBK/` component in `results.folder`. Batches without an unambiguous
board component still export CSV. Boards are not combined in these tables.

Rows give selected events for each acquisition, waveform mode and input window,
independently of the number of models. Control, train (the complete development
population) and test (blind) counts include frozen event selection, finite fixed-LED
crossings, the coincidence requirement and complete window availability. They
are not fold-training or neural internal-holdout counts. Population snapshots are
saved in each new run manifest; standalone reporting reads them without fitting.
For older runs, matching prepared caches can supply counts. Unavailable or ambiguous
historical counts are exported as a dash, not zero.

The LaTeX fragments use `booktabs`, `multirow`, `arydshln` and a two-column
`table*` environment. Mode and window labels span their grouped rows, with dashed
separators between groups. Copy the
matching board fragment into the manuscript's `report/tables/datasets/` directory
after checking the experiment identity. Reporting never overwrites manuscript sources.

LED calibration is fitted separately for each board and readout mode. After the
control threshold has been selected, the mean of the control LED time difference
minus its configured true TOF is computed on finite events within the existing
coincidence window, before ML input-window availability is checked. This fixed
mean is subtracted from every prepared control, development and blind residual,
and therefore from the supervised target. Event masks, waveform anchors and
threshold selection remain unchanged. RMSE is the square root of the mean
squared calibrated residual; it includes any remaining development/blind bias.
Neither validation folds, bootstrap draws nor plots estimate another mean.
Calibration is persisted in control/dataset/run manifests and prediction archives.
Older scientific-input policies are incompatible with run schema 53: rerun the batch to
rebuild caches and models before reporting. They cannot be repaired by relabelling
saved predictions or centering each dataset independently.

All ML models use one common control-fitted temporal mask in timing mode. A sample
is removed only when both detectors independently equal a single normalized
clipping limit (0 or 1) in at least 99% of selected control events. Ordinary
constant values inside the amplitude limits are retained. The shared mask is
fitted once per control population, physical input window and subsampling factor;
its provenance and values are cached and persisted with each prepared dataset.
It is applied before linear/MiniRocket feature transforms and neural processing,
and remains fixed across validation folds, final refitting and blind prediction.
Energy inputs retain every sample. Event selection, LED residuals and targets are
unchanged. Original waveform/time arrays remain available for reporting and XAI;
fully removed occlusion groups have zero effect on predictions. Saved models retain
the same mask for reloading and development diagnostics. Updated runs require
retraining; the batch planner invalidates incompatible model scopes automatically.

XAI temporal occlusion replaces each group with the detector-specific mean
normalized waveform from the prepared control cohort at those same sample
positions. This mean is cached per control input grid; blind waveforms contribute
no replacement statistics. Shared models replace both detector groups together;
direct models replace one detector at a time. Model masks and frozen feature
transforms are applied after replacement, using the saved inference rules.

To repeat XAI from saved models without retraining or recomputing blind metrics:

```bash
python -m waveform_analysis.cli xai \
  --results waveform_analysis/results/FBK/benchmark_48V_R1_R2 \
  --models direct_mlp shared_cnn1d \
  --group-size-samples 8 --max-events 4096
```

Omit `--models` to process all saved models. A subsampling experiment root is also
accepted; its nested batch indexes are discovered automatically. Retain the
original prepared control and blind caches, or specify their relocated directory
with `--cache-root`. Cache selection requires verified cohort identities and
rejects missing or ambiguous matches. Raw acquisitions, preprocessing fits and
model/feature-transform fitting are never invoked by this command. Compatible
older prepared arrays can be read for diagnostics while retaining the checkpoint's
original input mask; their scientific input policy is never upgraded. The command
works even when the original batch disabled XAI. New `artifacts/xai.npz`,
`metadata/xai.json` and model-level XAI plots record the replacement method, control
cohort and settings. Model weights, CV tables, predictions and blind/bootstrap
metrics remain intact. Existing manuscript XAI figures must be refreshed separately
from these regenerated plots.


## Random dilated shapelets + Ridge

`direct_rdst` and `shared_rdst` use aeon 1.6.0's
`RandomDilatedShapeletTransform`. Install `requirements.txt` before selecting
these models. The aeon import is lazy; other models do not require it at import time.

- **Direct:** fit multivariate shapelets on paired control waveforms, transform
  both detector channels jointly, then fit Ridge with an intercept.
- **Shared:** fit one univariate transform on pooled control detector waveforms;
  predict from `phi(s1)-phi(s2)` with Ridge without an intercept. Detector swap
  negates the prediction, including the common symmetric output clipping.

The shapelet transform and `StandardScaler(with_mean=False)` are fitted on
control data without timing targets or class labels. Shapelets, thresholds and
scales remain frozen across development folds, final fitting and blind evaluation.
The existing control-fitted sample mask is applied before RDST. Lengths and
dilations refer to retained sample indices, which need not be uniformly spaced
in physical time. Lengths exceeding the retained input size raise an error.

Model spaces are `config/model_spaces/direct_rdst.json` and `shared_rdst.json`.
All RDST transform parameters are fixed; only `ridge_alpha` is grid-searched.
These models retain the normal outer development CV, common folds, metric,
pruning, final full-development fitting and blind event bootstrap. They do not
use the special `RidgeCV` selection of the two linear waveform models. Do not
make transform parameters categorical: the frozen cache rejects that configuration.

From the repository root:

```bash
python -m waveform_analysis.cli check-batch --config waveform_analysis/config/batches/test_rdst.json
python -m waveform_analysis.cli batch --config waveform_analysis/config/batches/test_rdst.json
```

The batch requires the same external acquisitions as `test_ridge.json`.

With `protocol.xai.enabled=true`, RDST produces coefficient XAI instead of
waveform occlusion. Each shapelet's feature columns are ordered as minimum
absolute distance, match location and occurrence count. All three feature types
are divided by their **per-feature control standard deviation** before Ridge
fitting (a constant feature uses scale 1). Ranking is the sum of the absolute
coefficients on these standardized features, in ps. This avoids comparing raw
coefficients on different distance/location/count scales. Shared coefficients
act on the difference of the two standardized detector feature vectors, using
scales fitted on pooled control waveforms. They are linear-model weights rather
than a measure of realized prediction contribution, accuracy gain or causality;
correlated features can share coefficient weight. Output clipping is a later
nonlinear operation and is not included in the coefficient score.

Saved artifacts under each run include:

- `model/model.pkl` and `model/feature_transform/transform.pkl`: prediction reload.
- `model/feature_transform/shapelets.npz`: pickle-free values, length, dilation,
  startpoint, threshold, normalization, means/stds, feature mapping, scaling,
  aeon version, control provenance and retained time/sample grids.
- `artifacts/xai.npz`: signed scaled and raw coefficients, scores and stable ranks.
- `tables/shapelet_importance.csv`: all shapelets in descending score order.
- `plots/shapelets/ranked_*.png` (or the configured output format): ranked pages
  of 12 templates, including both channels for direct shapelets. Normalized
  templates are shown as z scores; others use the pipeline's normalized amplitude.
  Time coordinates follow the retained grid at the original sampled source start.

`plots` regenerates the figures and CSV from numeric XAI artifacts. `xai --results
... --models direct_rdst shared_rdst` regenerates coefficient XAI from saved
models without prepared datasets or raw acquisitions. It does not refit, rerun
selection, or alter blind metrics. Temporal group-size/max-event options have
no effect on coefficient ranking. aeon is pinned because the portable export
adapter validates the library's stored shapelet layout.


## Hydra + Ridge

`direct_hydra` and `shared_hydra` use aeon 1.6.0's `HydraTransformer` followed
by Ridge regression. The existing requirements already include aeon and PyTorch.
Hydra imports aeon/PyTorch lazily when selected. Execution uses CPU tensors.

- **Direct:** initialize multivariate Hydra on paired control waveforms and fit
  sparse scaling on their features; transform each detector pair jointly and
  fit Ridge with an intercept.
- **Shared:** initialize one univariate Hydra on pooled control detector
  waveforms and fit one sparse scaler on their features. Transform and scale
  each detector separately, then fit Ridge without an intercept on
  `phi(s1)-phi(s2)`. Swapping detectors negates the prediction, including the
  common symmetric output clipping.

Hydra's random competing kernels use length 9 and several dilations, extracting
weighted maximum responses and minimum-winning counts from the input and its
first differences. Kernel initialization depends on input dimensions and the
semantic seed, not on targets. Both initialization and sparse-scaler fitting
use control data only. Neither development nor blind events update kernels or
scaling statistics. The shared model subtracts **after** nonlinear scaling;
scaling a detector-feature difference would be a different model.

Scaling matches aeon's reference Hydra scaler: clamp features to nonnegative
values, take their square root, compute the control mean and sample standard
deviation, and add `(fraction_of_zeros ** 4) + 1e-8` to that standard deviation.
Subtract the frozen mean only at nonzero entries, divide by the frozen scale,
and preserve zeros. This is not ordinary `StandardScaler`. At least two control
waveforms are required to estimate the sample standard deviation.

The existing control-fitted sample mask applies before Hydra; inputs require
at least 9 retained samples. Dilations and first differences operate on the
retained sample sequence, which may have nonuniform physical spacing. No
resampling or new scientific preprocessing is introduced.

Model spaces are `config/model_spaces/direct_hydra.json` and
`shared_hydra.json`. Defaults are 8 kernels per group, 64 groups and a maximum
channel-sampling count of 8. `n_groups` must be 1 or even, avoiding aeon's silent
truncation of odd group counts. These transform settings stay fixed. Only
`ridge_alpha` is grid-searched through the existing outer development CV, common
folds, metric, pruning and resume logic; `protocol.cross_validation` is required.
Final fitting uses all development events, followed by the existing blind
metrics, output clipping and event bootstrap.

`transform.n_jobs` defaults to 2. `transform.batch_size` defaults to 128 and
bounds the number of waveforms entering each convolution call, including control
feature generation. The scaler is still fitted on the complete control feature
matrix. Existing prediction chunking and memory-mapped feature caches remain
in use. PyTorch's random state and thread count are restored after Hydra calls,
including failures, so mixed batches retain the other models' runtime state.

Saved artifacts include `model/model.pkl`,
`model/feature_transform/transform.pkl`, and
`model/feature_transform/hydra.npz`. The pickle files support prediction reload;
the NumPy export records kernels, sampled channel indices, dilation and padding,
feature-to-kernel ordering, sparse-scaler statistics, seed, aeon/transform version,
control provenance and retained time/sample grids. Kernels are stored explicitly:
aeon's internal kernel tensors are not registered in a PyTorch `state_dict`.

Hydra retains the existing temporal-occlusion XAI: replace waveform groups with
the detector-specific control mean, apply the frozen mask/transform/scaler and
measure prediction changes. Direct explanations are detector-specific; shared
explanations intervene on both detectors together. `plots` and `xai` replay
use the existing saved-results workflow. XAI replay requires matching prepared
control/blind caches and never refits kernels, scaling or Ridge. Ranked kernel
plots are not part of this implementation.

From the repository root:

```bash
python -m waveform_analysis.cli check-batch --config waveform_analysis/config/batches/test_hydra_FBK.json
python -m waveform_analysis.cli batch --config waveform_analysis/config/batches/test_hydra_FBK.json
```

The example uses the same external FBK acquisitions as `test_ridge.json`.


## MultiRocket + Ridge

`direct_multirocket` and `shared_multirocket` use aeon 1.6.0's `MultiRocket`.
The existing requirements already include aeon and Numba. One aeon class handles
both univariate and multivariate data; there is no separate multivariate class.

- **Direct:** fit a multivariate transform on paired control waveforms, then
  apply it jointly to each two-detector event. Ridge includes an intercept.
- **Shared:** fit one univariate transform on pooled control detector waveforms,
  apply the same transform and scaler independently to each detector, and use
  `phi(s1)-phi(s2)` as the Ridge input, without an intercept. Detector swap
  negates predictions, including the common symmetric output clipping.

MultiRocket extends MiniRocket with raw-waveform and first-difference features,
using four pooling summaries: proportion of positive values, mean positive value,
mean positive index and longest positive stretch. Dilations, biases and channel
selections are fitted on control data without timing targets. The feature scaler
is `StandardScaler(with_mean=False)`, matching aeon's reference MultiRocket
classifier. It is fitted incrementally over **all** control features (pooled
across detectors for shared), then frozen. Scaling happens before detector
subtraction. Neither CV folds nor development/blind data refit the transform or
scaler.

Model spaces are `config/model_spaces/direct_multirocket.json` and
`shared_multirocket.json`. Defaults match aeon: `n_kernels=6250`,
`max_dilations_per_kernel=32`, `n_features_per_kernel=4`, `normalise=false`.
`n_kernels` is a feature-budget parameter; aeon rounds it down to a multiple
of 84 per representation. The default yields **49,728 features per event**
for either formulation, after combining two representations and four pooling
operators. Shared features are subtracted rather than concatenated. Smaller
fixed budgets (at least 84) can be used for memory-limited experiments.

Only `ridge_alpha` is grid-searched. These models require
`protocol.cross_validation` and retain the common development folds, metric,
pruning, full-development final fitting, blind metrics and event bootstrap.
All transform parameters must remain fixed in the model space. The pipeline's
existing normalization and control mask apply first; `normalise=false` avoids
adding per-waveform z normalization. Inputs require at least 10 retained samples
so the first-difference representation also has at least 9 samples. No padding
or resampling is introduced; differencing follows the retained sample sequence.

`transform.n_jobs` defaults to 2 and `transform.batch_size` to 128. Control
feature scaling streams batches and does not allocate an unused full control
feature matrix. Development and blind features use the existing chunked,
memory-mapped cache. Their generation and prediction batches are automatically
capped at `transform.batch_size`, even when `runtime.prediction_chunk_size` is
larger. During engine fitting, Ridge reuses the owned training-row buffer rather
than making another full matrix copy; cached features remain unchanged. These
matrices can still be large: 15,000 events at the
default budget require about 3 GB per float32 feature matrix, and Ridge fitting
still materializes one dense matrix of selected training rows. Changing batch
size cannot shrink that training matrix or feature budget. NumPy's global random state and Numba's thread count are restored after
transform calls, including failures.

Saved `model/model.pkl` and `model/feature_transform/transform.pkl` support
prediction reload and resume. `model/feature_transform/multirocket.npz` exports
raw and differenced transform parameters, scaling statistics, seed, versions,
control provenance and the retained time/sample grid. Hydra/RDST artifacts and
existing models retain their existing formats.

XAI uses the existing temporal occlusion with control-mean replacement: direct
explanations are detector-specific, while shared explanations intervene on both
detectors together. Saved-model XAI replay requires matching prepared control
and blind caches; it does not refit transformations or regressors.

From the repository root:

```bash
python -m waveform_analysis.cli check-batch --config waveform_analysis/config/batches/test_multirocket.json
python -m waveform_analysis.cli batch --config waveform_analysis/config/batches/test_multirocket.json
```

The example references external UC acquisitions configured in `test_multirocket.json`.


## Group existing results across batches

Use the standalone script to organize completed results from multiple batch
folders. It never loads models, runs inference, refits preprocessing, regenerates
XAI, migrates legacy source layouts, or changes source files. It accepts current
and legacy run metadata and copies existing figures, CSV/LaTeX tables and JSON
metadata into a separate output directory. Model files and feature caches are
not copied. Original saved XAI figures may be copied like any other saved plot.

From the repository root, inspect the proposed groups first:

```powershell
python -m waveform_analysis.scripts.group_results --sources waveform_analysis/results/UC waveform_analysis/results/FBK --output waveform_analysis/grouped_results --check
```

You can provide batch folders instead of board folders, or a single parent
containing all batches. Overlapping input directories are deduplicated. Each
source directory must exist. The file can also be executed directly with
`python waveform_analysis/scripts/group_results.py ...`.

**Each top-level group contains exactly one subsampling factor.** Groups are
named, for example, `UC_48V_sub1`, `UC_48V_sub4`, or `FBK_48V_sub1`. Different
control/development/blind acquisition triples with the same board/voltage/factor
receive distinct dataset-hash suffixes; they are never merged by filename alone.
Within a group, waveform modes and numeric window bounds have separate folders.
Window labels may differ between batches if the numeric bounds match. Dataset
channels/polarities and true TOF are part of the dataset grouping identity.
Missing or inconsistent subsampling is reported and skipped, never guessed as 1.
When one board/voltage/subsampling family splits, the console lists only the
differing dataset fields (including nested channels/polarities) and their values
for each group. This explanation is also included as `dataset_splits` in
`--json` output and `grouping_manifest.json`.

If the same model appears twice in one group/mode/window, `--check` lists a
compact conflict block with stable source-run IDs, for example:

```text
Choice 1: UC_48V_sub1 | timing_to_timing | -1.5 to 2 ns | direct_rdst
  Select                     Batch / run
  --select a12bc345de67  batch_a/timing/onishi/direct_rdst
  --select f98ed765cb43  batch_b/timing/onishi/direct_rdst
```

Copy one `--select ID` option per block into your command. These same choices
appear automatically if a normal grouping command encounters duplicates, so
rerunning with `--check` is optional. Full machine-readable output remains
available with `--json`. Choose exactly one explicitly:

```powershell
python -m waveform_analysis.scripts.group_results --sources waveform_analysis/results/UC --output waveform_analysis/grouped_results --select RUN_ID
```

Repeat `--select` for each conflict, or pass `--selection choices.json` with:

```json
{"select": ["run_id_from_check", "another_run_id"]}
```

Unique models are included automatically; the script never chooses by blind
performance or modification time. IDs depend on source paths, so repeat the
check if source folders move. Without conflicts, simply omit `--check` to write
output. `--check` writes nothing and exits with code 2 when conflicts remain or no
groupable runs were found.

Example output:

```text
grouped_results/
├── grouping_manifest.json
├── sources.csv
├── UC_48V_sub1/
│   └── timing_to_timing/window_-1.5_2ns/
│       ├── group.json
│       ├── sources.csv
│       ├── summary.csv
│       ├── blind_ctr.png
│       ├── blind_rmse.png
│       └── runs/<model>/
│           ├── metadata/
│           ├── tables/
│           └── plots/
└── UC_48V_sub4/
    └── ...
```

Summary tables contain the saved blind metrics/uncertainties, available
validation scores and historical mask provenance. No predictions or bootstrap
statistics are recomputed. Summary bars reuse the original reporting `blind_metric_bar` renderer and ranked
metric-row builder: ascending CTR/RMSE order, stable model codes and legend names,
direct/shared formulation colours, bootstrap error bars, and the saved LED
reference line. Matching ranked `blind_ctr.csv/.tex` and `blind_rmse.csv/.tex`
tables accompany the plots. No models or prediction arrays are loaded.

Summary bars use saved metrics and require matching metric definitions and
control LED calibration across the compared runs. Missing or differing
control/development/blind population IDs produce a terminal warning and are saved
in `group.json` as `comparison_plot_warnings`; they do not block plots, since
these IDs can differ across machines. Identical event cohorts are not verified
in that case. Optional model-specific input
masks and older schema versions do not themselves prevent grouping. When
comparison metadata is missing or incompatible, the CSV and existing figures
are still copied; `group.json` explains why combined bars were skipped. Missing
files and ungroupable runs are recorded in the output manifest.

Use `--no-summary-plots` for a pure file/CSV grouping operation requiring only
the Python standard library. Otherwise the repository reporting dependencies are used for summary bars.
The style defaults to `waveform_analysis/config/plots/default.json`; use
`--plot-config path/to/plots.json` to reuse a batch’s saved style. The resolved
style is saved as `plots.json` in the grouped output, including its output format.
Missing or different saved LED references skip the affected metric plot and are
recorded in `group.json`; missing bootstrap uncertainties remain unknown.
`--copy-predictions` additionally copies existing prediction/bootstrap/development
arrays as opaque files; even then, no arrays are loaded or inference performed.

Saved acquisition paths are matched lexically, including Windows drive paths.
For acquisitions moved between checkouts, provide `--dataset-aliases aliases.json`
with explicit equivalent paths mapped to the same label, for example:

```json
{
  "C:/old_repo/processed_data/48V-R1.root": "UC/development_R1",
  "D:/new_repo/processed_data/48V-R1.root": "UC/development_R1"
}
```

Aliases are user declarations, not file-content verification. They do not bypass
population warnings or calibration checks for comparison plots. Unknown board or
voltage labels appear as `UNKNOWN` or `unknownV` rather than being guessed.

You can run the command repeatedly with the same output directory. New datasets,
subsampling factors, and mode/window entries are added without `--overwrite`;
existing unrelated results are preserved. Group names remain stable across runs.
If a different dataset would reuse an existing group name, it gets a dataset-hash
suffix automatically.

A conflict occurs only when the target group/mode/window already exists.
`--check` lists these paths. Use `--overwrite` to replace those entries only;
include all models you want in each replaced entry, since it is regenerated as a
whole. Other groups/windows are preserved. The combined manifest and `sources.csv`
include both retained and newly generated results. Each generated entry also saves
its own `plots.json` when summary plots are enabled.

Only outputs marked as created by this script can be extended or overwritten;
unrelated folders and all source runs are protected. Publication is staged, so a
failed grouping operation leaves existing output unchanged. Output directories
are excluded from subsequent discovery. `waveform_analysis/grouped_results/`
is Git-ignored.
