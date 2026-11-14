# Waveform pipeline architecture

## Scientific pipeline

One batch configuration expands into runs by input mode, named temporal window and model. The scientific workflow is:

1. Fit Control waveform/event-selection and LED calibration rules.
2. Apply those frozen rules to Development and Blind.
3. Materialize paired, normalized waveform inputs with a common sample mask.
4. Fit reusable control-scoped model feature transformations when the registered model requires them.
5. Select hyperparameters by Development cross-validation, reusing the same folds across models on the same population.
6. Fit the selected estimator on all Development events.
7. Evaluate frozen model corrections on Blind events, then bootstrap Blind events without retraining.

A native Butterworth low-pass, if enabled in preprocessing configuration, is applied before signal selection, threshold detection, or cropping. The final comparison uses order 5 and cutoff 2 GHz for UC and FBK.

## Main modules

| Module | Role |
|---|---|
| `core/config.py` | Batch validation, expansion into model/mode/window runs |
| `engine/batch.py` | Batch planning, execution and reporting |
| `engine/study.py` | CV, model selection, final fitting, blind metrics and bootstrap |
| `engine/preprocessing.py` | Control and dataset preparation |
| `data/feature_cache.py` | Fitted transforms and feature matrices |
| `data/shared_artifacts.py` | Common CV folds and LED fold metrics |
| `data/storage.py` | Run-local model, predictions, scalar results and resume records |
| `models/registry.py` | Model capabilities and estimators |
| `engine/tshap_explainer.py` | Existing global/local paired-window TSHAP |
| `engine/local_tshap_deletion.py` | Local-importance deletion kernels |
| `scripts/evaluate_progressive_windows.py` | Separate prefix/window-length CV study |

## Randomness and numerical operations

The batch protocol defines an explicit integer `seed`. CV splits, fitted model training, control-fitted random transformations, Optuna selection, metrics and bootstrap are seeded explicitly rather than constructing seeds from a filename, model name or hashed population identifier.

Ridge regularization is selected using development CV and a `Ridge` estimator; the Onishi CNN follows its separately configured training procedure. A saved fitted feature transform is reused for model inference.

## Results and caches

Model run results use the nested `<batch>/<mode>/<window>/<model>/` layout. The fitted model and transform live in `model/`, and saved blind/development predictions plus bootstrap draws live in `artifacts/` with readable filenames. Incomplete CV work can be resumed from saved candidate/fold outputs, and stage state is stored under `metadata/`.

Shared CV assignments use `<batch>/artifacts/cv/<mode>/<window>/seed_<seed>/`. Input preparation and frozen feature caches still have an ArtifactStore implementation under a named cache bundle. Those caches are not yet unified with the simpler run-local storage. New protocols should use new cache/result directories.

## Interpretability

Interpretability is post-training and does not retrain the timing model. Current paired-window TSHAP provides global and local interpretations. Conditional kNN and original reference substitution remain separate implementations; a benchmark selecting frozen imputers for remove-window and keep-only tasks is planned but is not yet wired into the CLI. Independent temporal/progressive-window experiments are intentionally retained.

## Validation

Use `python -m pytest waveform_analysis/tests` for synthetic and integration coverage. The repository does not include experimental ROOT files, so end-to-end acquisition-based reproducibility must be tested in an environment with those files.
