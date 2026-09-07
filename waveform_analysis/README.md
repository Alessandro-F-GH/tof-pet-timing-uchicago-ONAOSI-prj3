# Compact waveform timing pipeline

`waveform_analysis` evaluates how much TOF-PET timing information is available in oscilloscope waveforms while keeping the scientific protocol explicit and small.

## Protocol

For each prepared ROOT dataset:

`events -> development/blind -> training/validation`

The blind set is created first and is not passed to standard-method optimization, hyperparameter search, early stopping or model selection. LED/CFD parameters and ML candidates are selected with the single validation holdout. The selected ML candidate is then refit on the complete development population and evaluated once on blind data.

There is no K-fold, nested CV, cross-validation or pooled OOF path.

## Methods

Standard timing:

- LED: threshold selected from `standard_methods.led_thresholds_mV`;
- CFD: fraction selected from `standard_methods.cfd_fractions` when enabled for a mode.

Waveform ML:

- `linear_svr`: linear shared-pair correction;
- `cnn`: nonlinear shared 1-D convolutional scorer.

Both models enforce

`correction = g(detector_1) - g(detector_2)`.

New model families can be added by dropping a module with a `MODEL_SPEC` into `ml_pipeline/models/`; the registry discovers it without modifying `study.py`.

## CTR metric

`ml_pipeline.stats.ctr_fwhm` measures the full width at half maximum of the dominant smoothed timing histogram directly. The smoothing kernel stabilizes the histogram only; it is not a Gaussian fit. Blind uncertainty is obtained by event-resampling bootstrap with the same FWHM estimator.

## CLI

From the repository root:

```bash
python -m waveform_analysis.cli check --config waveform_analysis/config/experiments/complete_new.json
python -m waveform_analysis.cli prepare --config waveform_analysis/config/experiments/complete_new.json
python -m waveform_analysis.cli run --config waveform_analysis/config/experiments/complete_new.json
python -m waveform_analysis.cli report --run-dir waveform_analysis/results/studies/complete
```

Use `--overwrite` for a fresh run and `--rebuild-preprocessing` only when the physical preparation must be rebuilt.

## Outputs

A run stores:

- `manifest.json`: resolved configuration and protocol;
- `results.csv`: validation and final blind CTR rows;
- `splits/`: exact deterministic indices;
- `search/`: candidate scores and selected parameters;
- `models/`: only final development-refit models;
- `artifacts/`: blind residuals and XAI arrays;
- `plots/`: generated CTR/XAI figures.

## Tests

```bash
python -m unittest discover -s waveform_analysis/tests -v
```

The tests cover split isolation/determinism, holdout candidate selection, direct FWHM robustness, model-registry extension and exact detector-swap antisymmetry.
