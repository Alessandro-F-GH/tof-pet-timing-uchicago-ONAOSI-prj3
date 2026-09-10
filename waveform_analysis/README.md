# Waveform timing pipeline

The waveform pipeline separates **selection**, **physical preprocessing**, **ML dataset construction**, and **model fitting**. Every experiment has exactly one configured `mode`: `energy_to_energy` or `timing_to_timing`. Optional CFD evaluation is controlled by the experiment-level boolean `cfd`.

## 1. Event selection

The ROOT entry population is split into permanent **development** and **test** sets before any fitted selection. Using development only, the pipeline fits the two energy photopeaks, detects threshold hits on the waveform family required by the selected mode, derives timing ToT limits for `timing_to_timing`, and optionally derives a baseline-RMS limit. Frozen cuts are applied unchanged to test.

Selection, native-preprocessing and ML-prepared caches are mode-scoped, so two studies on the same ROOT source cannot reuse incompatible waveform families.

## 2. Native-time preprocessing

Only selected events and the waveform family required by the experiment mode are materialized. For each waveform the pipeline decodes/orients native samples, clamps them to detector-specific `vertical_scale_limit_mV`, crops around the selected main trigger, preserves native acquisition timing, and stores the rising-edge interval used by LED/CFD.

There is **no denoising** and no event-wise baseline subtraction.

## 3. ML dataset preparation

LED thresholds are scanned on development and ranked by the common robust CTR estimator from `utils_fit`; CFD is treated the same way when `cfd: true`. The canonical metric is the Gaussian-equivalent shortest interval containing the configured fraction of finite residuals, with 90% coverage by default. Bootstrap is skipped during candidate ranking because uncertainty is not part of threshold selection. LED crossing times are linearly interpolated. Each ML waveform is then resampled by linear interpolation on a common continuous time grid relative to that crossing, so `t=0` is exactly the selected LED threshold for every event and detector.

The fixed channel calibration is estimated from training only as

`C_hat_12 = mean_training(Delta t_LED) - TOF`.

The canonical supervised target is the calibrated LED residual itself:

`y_target = Delta t_LED - TOF - C_hat_12`.

There is no anchor correction and no separate stored target array: the model target is computed directly from calibrated LED.

The continuous ML grid uses the native sampling interval multiplied by `ml_input.subsampling`, but every coordinate is evaluated by interpolation relative to the exact LED crossing. The fixed crossing coordinate at `t=0` is removed. A second development-only feature mask removes time coordinates that are constant in at least 99% of development events on both detectors; this removes cropped/clipped dead regions without looking at the blind test set. The learned mask is frozen and applied unchanged to validation and test. Waveforms are then scaled globally to `[0, 1]` using the detector-specific physical limits from `preprocessing.<family>.vertical_scale_limit_mV`.

### Concatenated-dataset experiments

Set `experiment.concatenate_datasets: true` to train one model on all configured bias-voltage datasets instead of fitting one model per ROOT dataset. Each ROOT file still goes through the normal selection and native preprocessing independently, preserving its own frozen development/blind-test split. A single LED threshold must be fixed in the experiment with `experiment.fixed_led_threshold_mV`; the source prepared datasets therefore use that same LED threshold before concatenation.

The source training partitions are concatenated into one training set, source validation partitions into one validation set, and source blind-test partitions into one blind-test set. The fixed channel calibration and ML target are then recomputed globally on the concatenated training population. For concatenated studies, source preparation removes only the fixed t=0 crossing coordinate; the 99% dead-region mask is learned once from the pooled development population after concatenation and is then frozen for the pooled test set. Source prepared caches are stored separately from ordinary per-voltage prepared caches.

Example:

```json
"experiment": {
  "concatenate_datasets": true,
  "concatenated_dataset_name": "timing_all_bias",
  "fixed_led_threshold_mV": 15.0
},
"standard_methods": {
  "led_thresholds_mV": [15.0]
}
```

A ready timing configuration is available at `config/experiments/timing_concatenated.json`.

## 4. ML and final test

Models may use paired or waveform-difference representations; their exact prediction definition is recorded in per-model metadata. Candidate hyperparameters are trained on the training split and ranked **only by full-validation RMSE** of `y_target - y_theta`. The configured training target ranges are part of that same hyperparameter search: for a candidate range `R`, fitting uses only training events satisfying `|y_target| <= R`, while validation is never filtered. Model-internal early stopping also uses the full validation set where applicable. **There is no final refit:** the validation-selected trained model is used directly for final evaluation. Predictions are limited to the configured physical range; the default is `±2000 ps`.

The permanent test population is evaluated once after model and training-target-range selection. The selected model predicts every blind/test event; the target-range filter is never applied to validation or test. The LED reference residual is

`Delta t_LED - TOF - C_hat_12`,

while the ML residual used for CTR is

`y_target - y_theta`.

CTR is the **Gaussian-equivalent shortest empirical coverage interval**. With coverage fraction `p`, the sorted residuals are scanned for the narrowest interval containing `ceil(p*N)` finite events; the interval width is multiplied by the Gaussian conversion factor that maps the corresponding central Gaussian coverage width to FWHM. The default is `p = 0.90`, configured with `fit.coverage_fraction`. CTR uncertainty is the event-bootstrap standard deviation of this robust estimator; the default is `500` resamples configured with `fit.bootstrap_samples`.

All finite residuals are included in the canonical CTR calculation. There is no internal `fit.max_abs_ps` rejection. The fixed-bin histogram FWHM is retained only as a secondary `core_fwhm_ps` diagnostic, together with `core_fraction`; `fit.bin_width_ps` controls only that diagnostic histogram.

Before a rebuild or result overwrite, the CLI preflights every ROOT file and every relevant cache. All overwrite targets are shown once and a single terminal confirmation is requested before the batch begins. Stale caches are reported before processing starts.

## Results

For ordinary per-voltage studies, the study root contains:

- `ctr_vs_voltage.pdf`: grouped test CTR bars with bootstrap error bars;
- `relative_improvement_vs_voltage.pdf`: relative CTR improvement over LED, with uncertainty obtained from a **paired bootstrap** using the same resampled event indices for LED and each ML model;
- `relative_improvement.csv`: numerical values used in that paired-improvement plot.

The paired relative improvement is computed as

`100 * (CTR_LED - CTR_model) / CTR_LED`

for every common bootstrap resample. This accounts for covariance between LED and ML CTR estimates from the same event population.

Concatenated-dataset studies intentionally do **not** produce voltage-comparison plots because only one pooled dataset/model is evaluated.

Detailed reporting is grouped under:

- `plots/corrections/`
- `plots/train_distribution/`: one LED-versus-model CTR distribution per model;
- `plots/test_distribution/`: one LED-versus-model CTR distribution per model;
- `plots/model_output/<model>/`
- `plots/xai/<model>/`: XAI outputs grouped by model

The reporting histograms are presentation views with one model compared against LED per figure. Legends are placed in the upper-right with extra plot headroom and a wider residual display range to avoid obscuring the distributions. The canonical robust CTR itself is bin-free; `fit.bin_width_ps` is used only for the secondary core-FWHM diagnostic.

## CLI

```bash
python -m waveform_analysis.cli check --config waveform_analysis/config/experiments/timing.json
python -m waveform_analysis.cli prepare --config waveform_analysis/config/experiments/timing.json
python -m waveform_analysis.cli run --config waveform_analysis/config/experiments/timing.json --overwrite
python -m waveform_analysis.cli report --run-dir waveform_analysis/results/studies/complete_timing
```

For the concatenated timing study:

```bash
python -m waveform_analysis.cli run \
  --config waveform_analysis/config/experiments/timing_concatenated.json \
  --overwrite
```
