# Waveform pipeline architecture

There is one active implementation and one result layout. Historical readers,
import aliases and pickle-name rewrites are confined to the disposable
`scripts/convert_results.py` conversion tool.

## Responsibilities

| Package | Responsibility |
|---|---|
| `core` | Input validation, settings, stage dependencies, logging and I/O |
| `signal` | Numerical waveform and timing kernels |
| `data` | Acquisition/prepared inputs, masks, frozen features, shared splits and result persistence |
| `models` | Estimator definitions, capabilities, feature transforms and model serialization |
| `engine` | Control preprocessing, model selection, final fit, evaluation and explicit diagnostics |
| `reporting` | Saved-result readers, statistics, table exporters and plot renderers |
| `scripts` | Grouped reporting, acquisition merging and the temporary converter |

`ModelSpec` is the estimator interface. `BaseTorchModel` supplies tensor inference
without owning training policy. The registry imports each model lazily. Feature
fit role, required frozen features, bounded transformation batches, disposable
fit-input ownership and artifact format are declared capabilities rather than
repeated model-name lists.

## Configuration ownership

Source batch JSONs still define the same scientific experiment. Runtime run
configurations are expanded in memory. Persisted results have separate owners:

- Batch `config.json`: datasets, one common channels/polarities definition,
  common protocol settings and declared board. No expanded run dictionaries.
- Batch `plots.json`: presentation only.
- Batch `model_spaces/<model>.json`: frozen model/search configuration, once per model.
- Mode `mode.json`: waveform mode and effective fit/input/output settings.
- Window `context.json`: named window and numerical bounds.
- Window `population.json`: population counts/IDs, calibration and shared mask provenance.
- Model `metadata/config.json`: model identity only.
- Model `metadata/manifest.json`: model execution state and artifact fingerprints.
- Selection, blind and bootstrap JSONs: their own numerical outputs.

`data.results.resolve_run_config()` reconstructs the effective configuration.
`read_run_manifest()` joins inherited metadata and saved selection results in
memory. `read_run()` and `discover_runs()` are side-effect-free and are shared by
reporting, grouping and post-training TSHAP. A changed owner cannot silently relabel retained
models. Use a new experiment folder when changing shared scientific settings
without rebuilding all of their owners.

`RunStore` writes model artifacts and stage state. Its constructor never migrates
files. Conversion is explicit. Identifiers retained in numerical caches and seed
payloads describe scientific algorithms, not alternate active pipelines.

## Execution and dependencies

The study runner calls explicit stages for outer-CV selection/final fitting,
blind evaluation, event-bootstrap uncertainty and publication. Linear
RidgeCV remains a separate selection/fitting procedure. Onishi's MSE/full-split
training remains distinct from RMSE/internal-holdout neural training.

`core.dependencies` defines planning fingerprints, stage identity payloads and
invalidation edges. Invalidation preserves independent work: bootstrap changes
do not invalidate model weights; plot changes do not retrain models.

Prepared datasets are reused within a batch with a key including dataset role,
mode, window, subsampling, preprocessing, control artifact and empty-population
policy. Frozen features remain memory mapped. MultiRocket retains bounded
transform batches, float32 feature storage and its existing Ridge solver.

## Reporting and historical artifacts

Ordinary reporting uses saved arrays and metrics. It does not load models to
backfill predictions, fit preprocessing, migrate files or alter scientific source
artifacts. `cli diagnostics` requests development predictions. Post-training TSHAP
loads frozen models and verified caches without refitting. A converted historical model-specific mask is an explicit
model-owned input-policy override, never relabelled as a modern shared mask.

Grouped reports refer to original runs, copy requested saved figures/tables, and
render comparisons from saved scalar metrics. They do not copy source model
configuration or models. Each group has one dataset triple and subsampling;
numeric modes/windows have separate entries. Population-ID differences warn
rather than block scalar comparisons. Metric/calibration/LED conflicts remain
visible, and unavailable values are never invented.

## Non-regression

Preserve event IDs/order, clipping masks, time grids, feature fit roles/scalers,
seed construction, CV folds, pruning and fallback decisions, selection metric,
optimizers, precision, output clipping, prediction order, CTR definition and
bootstrap sampling. Changing path-dependent population IDs is outside this
cleanup because they currently affect CV seeds.

The regression suite includes independent frozen signal kernels, isolated
reference/current model fits, actual Ridge/shapelet/Hydra/MultiRocket batch
execution, model reloads, reporting, scoped invalidation and resume. Numerical
arrays are compared exactly where supported on the same CPU/dependencies.
Conversion tests additionally compare source hashes, persisted array bytes and
predictions from recovered models. Production acquisition data and GPU runs
require their own validation; synthetic tests are not a substitute for those.

## Interpretability

The only maintained interpretability procedure is post-training TSHAP.
The legacy XAI stage is retained as a disabled metadata marker for saved
run layouts. Cache loading lives in `data/prepared_cache.py`.
