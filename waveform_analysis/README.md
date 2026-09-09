# Waveform timing pipeline

The waveform pipeline separates **selection**, **physical preprocessing**, **ML dataset construction**, and **model fitting**. Every experiment has exactly one configured `mode`: `energy_to_energy`, `energy_to_timing`, or `timing_to_timing`. Optional CFD evaluation is controlled by the experiment-level boolean `cfd`.

## 1. Event selection

The ROOT entry population is split into permanent **development** and **test** sets before any fitted selection. Using development only, the pipeline fits the two energy photopeaks, detects threshold hits on the waveform families required by the selected mode, derives timing ToT limits when needed, and optionally derives a baseline-RMS limit. Frozen cuts are applied unchanged to test.

Selection, native-preprocessing and ML-prepared caches are mode-scoped, so two studies on the same ROOT source cannot reuse incompatible waveform families.

## 2. Native-time preprocessing

Only selected events and waveform families required by the experiment mode are materialized. For each waveform the pipeline decodes/orients native samples, clamps them to detector-specific `vertical_scale_limit_mV`, crops around the selected main trigger, preserves native acquisition timing, and stores the rising-edge interval used by LED/CFD.

There is **no denoising** and no event-wise baseline subtraction.

## 3. ML dataset preparation

LED thresholds are scanned on development and evaluated with the common Gaussian CTR fitter in `utils_fit`; CFD is treated the same way when `cfd: true`. The native sample closest to the selected LED threshold becomes the ML anchor.

The ML window is materialized with `t_anchor = 0` and `ml_input.subsampling` is applied. Waveforms are then scaled globally to `[0, 1]` using the detector-specific physical limits from `preprocessing.<family>.vertical_scale_limit_mV`. The transform is fixed by configuration: it is not fitted per event, per sample, or from the training population, and its inverse is persisted for physical-mV reporting.

## 4. ML and final test

Linear SVR and CNN candidates are trained on training and ranked **only by validation RMSE**. CNN early stopping uses the same validation RMSE. The selected candidate is refit on complete development. The permanent test population is evaluated once after selection.

CTR values and their bootstrap uncertainty use the repository-wide Gaussian fitter from `utils_fit`; the ML pipeline contains no independent CTR implementation.

Before a rebuild or result overwrite, the CLI preflights every ROOT file and every relevant cache. All overwrite targets are shown once and a single terminal confirmation is requested before the batch begins. Stale caches are reported before processing starts.

## Results

Each study stores `ctr_vs_voltage.pdf` directly in the study directory. Detailed reporting is grouped under:

- `plots/corrections/`
- `plots/train_distribution/`
- `plots/test_distribution/`
- `plots/xai/`

## CLI

```bash
python -m waveform_analysis.cli check --config waveform_analysis/config/experiments/complete_energy.json
python -m waveform_analysis.cli prepare --config waveform_analysis/config/experiments/complete_energy.json
python -m waveform_analysis.cli run --config waveform_analysis/config/experiments/complete_energy.json --overwrite
python -m waveform_analysis.cli report --run-dir waveform_analysis/results/studies/complete_energy
```

The former multi-mode `complete.json` was intentionally removed. Use `complete_energy.json` and `complete_timing.json` as independent studies.
