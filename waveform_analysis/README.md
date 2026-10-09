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
preprocessing, blind CTR/RMSE, bootstrap and XAI keep their existing definitions.
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

The LaTeX fragments use `booktabs` and a two-column `table*` environment. Copy the
matching board fragment into the manuscript's `report/tables/datasets/` directory
after checking the experiment identity. Reporting never overwrites manuscript sources.
