# Waveform ML pipeline

The pipeline uses three explicit scientific dataset roles:

1. **control** — fit preprocessing, event-selection and LED criteria once;
2. **development** — apply the frozen control rules, run deterministic common-fold K-fold CV, tune/select the model, and train the final selected configuration;
3. **blind** — apply the same frozen rules and evaluate the already-selected final model once.

The blind dataset never participates in preprocessing fitting, fold construction, hyperparameter tuning, pruning, model/window/formulation selection, or final training.

## Development CV and pruning

Every `(mode, window)` development population gets one deterministic K-fold definition derived from the single batch seed. Every model and every candidate uses the same folds in the same order. CTR and RMSE are always computed, together with LED CTR/RMSE on the exact same validation events.

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

```bash
python -m waveform_analysis.cli check-batch --config config/batches/test_ridge.json
python -m waveform_analysis.cli batch --config config/batches/test_ridge.json
python -m waveform_analysis.cli plots --results results/FBK/test_ridge_48V_R1_R2
```

The batch command prints a `KEEP` / `RESUME` / `RUN` / `REBUILD` execution plan before destructive changes.

## Result layout

```text
results/<folder>/
├── manifest.json
├── config.json
├── plots.json
├── runs.csv
├── preprocessing/
│   ├── control/<mode>/
│   ├── development/<mode>/
│   └── blind/<mode>/
├── artifacts/<mode>/<window>/...
├── <mode>/<window>/<model>/
│   ├── manifest.json
│   ├── config.json
│   ├── folds.csv
│   ├── cv.csv
│   ├── best.json
│   ├── final_fit.json
│   ├── blind.json
│   ├── pred.npz
│   ├── bootstrap.json
│   ├── bootstrap.npz
│   ├── xai.npz
│   ├── model/
│   └── plots/
└── report/
    ├── tables/
    └── <mode>/<window>/
```

`pred.npz` persists blind event IDs explicitly for model-to-model alignment.

## Resume and dependency invalidation

Completed CV folds and candidates are persisted as they finish. Interrupted candidate CV resumes from the first missing fold. Persisted pruning decisions remain authoritative while scientific dependencies are unchanged.

Changes are scoped:

- development/CV/model-search changes invalidate CV and downstream stages for the affected model;
- blind-only changes preserve compatible development CV and selection;
- bootstrap-only changes reuse blind predictions and central values;
- XAI-only changes reuse CV, final selection and blind predictions;
- plot-only changes regenerate plots/reports from persisted results only.

Old fixed-validation and replica result schemas are intentionally unsupported. There are no compatibility readers or legacy execution modes.

## Reporting

Reporting consumes persisted numeric artifacts and produces validation/blind summaries, model-output correlations aligned by blind event ID, paired CTR/RMSE model-difference matrices with paired event bootstrap, blind RMSE-vs-CTR plots, validation-vs-blind plots, and window comparisons.

Window comparison winners are selected from development CV only; their blind performance is then displayed. CTR/RMSE bar annotations use integer-rounded picoseconds. XAI is grouped temporal occlusion and stores numeric importance separately from its plot.
