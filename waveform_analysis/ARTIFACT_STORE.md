# Cache and persistence

## Current implementation

This repository is being simplified on `codex/lean-cache-xai-cleanup`. It does **not** support automatic migration of older cache or result layouts. Start filtered benchmark experiments in fresh result and cache roots.

### Run-local results

Each completed run owns its model and numerical outputs:

```text
results/<board>/<batch>/<mode>/<window>/<model>/
  model/
  artifacts/
    predictions.npz
    development_predictions.npz
    bootstrap.npz
  tables/
  metadata/
```

The full model, including any bundled fitted transformation, is serialized in `model/`. Prediction and bootstrap arrays no longer require resolving external immutable product references.

The batch runner checks stage completion and required files. It reuses complete runs and resumes incomplete work. It does not compare configuration hashes to decide whether to invalidate outputs. **The user must create a new result directory or explicitly clear stale outputs when changing scientific inputs or hyperparameters.**

### Shared CV assignments

Common CV folds and LED fold reference metrics are stored at:

```text
<batch-results>/artifacts/cv/<mode>/<window>/seed_<seed>/
  folds.npz
  led_folds.json
```

An explicit integer seed determines shuffled assignments. No population-name or file-name seed derivation is performed.

### Remaining preparation and feature caches

Waveform selection, native waveform materialization, prepared datasets, fitted sample masks, frozen feature transforms and feature matrices **still use the existing ArtifactStore implementation** with readable aliases. This shared-cache subsystem has not yet been converted to fully readable direct storage. Its internal implementation currently uses content hashes.

Do not delete these remaining store components while any training, TSHAP or cache-preparation consumer still relies on them. In particular, `data/preparation.py`, `data/feature_cache.py`, `data/sample_mask.py`, `data/preprocessing.py`, `engine/control_preprocessing.py` and `engine/event_selection.py` must be migrated together with their readers.

## Next refactoring block

Replace the remaining shared-cache backing store with direct, semantic paths while retaining safe write completion, reusable frozen transformations and resume. Then consolidate conditional PPCA/kNN imputation and TSHAP. Do not retrain the acquisition models or mix old cache content with the simplified layout.
