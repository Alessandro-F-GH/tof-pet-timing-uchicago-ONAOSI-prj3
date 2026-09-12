# Waveform timing pipeline

The waveform pipeline separates **selection**, **physical preprocessing**, **ML dataset construction**, and **model fitting**. Every experiment has exactly one configured `mode`: `energy_to_energy` or `timing_to_timing`. Optional CFD evaluation is controlled by the experiment-level boolean `cfd`.

## 1. Event selection

The ROOT entry population is split into permanent **development** and **test** sets before any fitted selection. Using development only, the pipeline fits the two energy photopeaks, detects threshold hits on the waveform family required by the selected mode, derives timing ToT limits for `timing_to_timing`, and optionally derives a baseline-RMS limit. Frozen cuts are applied unchanged to test.

Selection, native-preprocessing and ML-prepared caches are mode-scoped, so two studies on the same ROOT source cannot reuse incompatible waveform families.

## 2. Native-time preprocessing

Only selected events and the waveform family required by the experiment mode are materialized. For each waveform the pipeline decodes/orients native samples, clamps them to detector-specific `vertical_scale_limit_mV`, crops around the selected main trigger, preserves native acquisition timing, and stores the rising-edge interval used by LED/CFD.

There is **no denoising** and no event-wise baseline subtraction.

## 3. ML dataset preparation

LED thresholds are scanned on development and ranked by the common robust CTR estimator from `utils_fit`; CFD is treated the same way when `cfd: true`. The canonical metric is the Gaussian-equivalent shortest interval containing the configured fraction of finite residuals, with 90% coverage by default. Bootstrap is skipped during candidate ranking because uncertainty is not part of threshold selection. LED crossing times are linearly interpolated. Waveforms remain on the native acquisition grid, so the ML anchor `t_a` is still the native sample nearest in time to the interpolated selected LED crossing for window materialization only.

The fixed channel calibration is estimated from training only as

`C_hat_12 = mean_training(Delta t_LED) - TOF`.

The canonical supervised target is the calibrated LED residual itself:

`y_target = Delta t_LED - TOF - C_hat_12`.

No anchor-shift term is subtracted from the target.

The ML window is materialized with `t_anchor = 0` and `ml_input.subsampling` is applied. Waveforms are scaled globally to `[0, 1]` using the detector-specific physical limits from `preprocessing.<family>.vertical_scale_limit_mV`. The transform is fixed by configuration: it is not fitted per event, per sample, or from the training population, and its inverse is persisted for physical-mV reporting.

### Concatenated-dataset experiments

Set `experiment.concatenate_datasets: true` to train one model on all configured bias-voltage datasets instead of fitting one model per ROOT dataset. Each ROOT file still goes through the normal selection and native preprocessing independently, preserving its own frozen development/blind-test split. A single LED threshold must be fixed in the experiment with `experiment.fixed_led_threshold_mV`; the source prepared datasets therefore use that same LED threshold before concatenation.

The source training partitions are concatenated into one training set, source validation partitions into one validation set, and source blind-test partitions into one blind-test set. The fixed channel calibration and ML target are then recomputed globally on the concatenated training population. Source prepared caches are stored separately from ordinary per-voltage prepared caches.

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

CNN training is reproducible for a fixed candidate seed: NumPy, PyTorch CPU/CUDA RNGs, DataLoader shuffling, deterministic PyTorch algorithms, deterministic cuDNN, and deterministic cuBLAS workspace configuration are fixed before model initialization. The exact training seed is saved in model metadata. This may reduce GPU throughput slightly but prevents run-to-run kernel nondeterminism.

The standard CNN comparison uses two paired-input architectures. `cnn` is the shared 1-D scorer: the same 1-D network scores each detector waveform and the correction is `score(s1)-score(s2)`, enforcing detector-swap antisymmetry. `cnn_2d` is one joint network over the stacked `[2,time]` detector pair. Its first temporal block preserves the two detector rows with a height-1 kernel; a configurable later convolution spans both rows once and fuses the detector axis before the remaining temporal blocks. The network then predicts one correction directly. The two model-space configs use the same temporal channels, kernels, strides, dilations, pooling and dense head so the comparison isolates the shared-1-D versus joint-2-D structure. Their exact prediction definition is recorded in per-model metadata. Candidate hyperparameters are trained on the **entire training split** and ranked **only by validation CTR** of `y_target - y_theta`. No events are removed according to the magnitude of `y_target`. Model-internal early stopping may still use validation RMSE where appropriate. **There is no final refit:** the validation-selected trained model is used directly for final evaluation. Predictions are limited to the configured physical range; the default is `±2000 ps`.

A third optional architecture, `cnn_heteroscedastic`, keeps the shared single-waveform 1-D structure but predicts both a mean score and a single-signal uncertainty. For a detector pair,

`mu_pair = mu(s1) - mu(s2)`

and, assuming conditionally independent single-signal noise contributions,

`sigma_pair = sqrt(sigma(s1)^2 + sigma(s2)^2)`.

It is trained with a Gaussian heteroscedastic negative log-likelihood. The configurable candidate parameter `sigma_max_ps` acts as an abstention threshold: when `sigma_pair > sigma_max_ps`, the returned timing correction is exactly `0 ps`; otherwise the returned correction is `mu_pair`. Because the uncertainty is symmetric under detector exchange while the mean is antisymmetric, the final gated prediction remains detector-swap antisymmetric. A comparison configuration is available at `config/experiments/timing_heteroscedastic.json`.

The permanent test population is evaluated once after model selection. The LED reference residual is

`Delta t_LED - TOF - C_hat_12`,

while the ML residual used for CTR is

`y_target - y_theta`.

CTR is the **Gaussian-equivalent shortest empirical coverage interval**. With coverage fraction `p`, the sorted residuals are scanned for the narrowest interval containing `ceil(p*N)` finite events; the interval width is multiplied by the Gaussian conversion factor that maps the corresponding central Gaussian coverage width to FWHM. The default is `p = 0.90`, configured with `fit.coverage_fraction`. CTR uncertainty is the event-bootstrap standard deviation of this robust estimator; the default is `500` resamples configured with `fit.bootstrap_samples`.

All finite residuals are included in the canonical CTR calculation. There is no internal `fit.max_abs_ps` rejection. The fixed-bin histogram FWHM is retained only as a secondary `core_fwhm_ps` diagnostic, together with `core_fraction`; `fit.bin_width_ps` controls only that diagnostic histogram.

Before a rebuild or result overwrite, the CLI preflights every ROOT file and every relevant cache. All overwrite targets are shown once and a single terminal confirmation is requested before the batch begins. Stale caches are reported before processing starts.

## Results

For ordinary per-voltage studies, the study root contains:

Study outputs are type-separated from creation time:

- `csv/results.csv`: canonical study results table;
- `csv/relative_improvement.csv`: numerical values used in the paired-improvement plot;
- `plots/ctr_vs_voltage.pdf`: grouped test CTR bars with bootstrap error bars;
- `plots/relative_improvement_vs_voltage.pdf`: relative CTR improvement over LED, with uncertainty obtained from a **paired bootstrap** using the same resampled event indices for LED and each ML model;
- `plots/corrections/<dataset>/`: top/worst correction figures;
- `csv/corrections/<dataset>/`: corresponding correction ranking tables;
- `plots/xai/<model>/`: model-grouped XAI plots; CNN and 2-D CNN importance is input-gradient importance aggregated onto the exact waveform time axis;
- `csv/xai/<model>/`: tabular XAI/shapelet exports only when present;
- `plots/model_output_diagnostics/`: prediction-vs-target diagnostics and model-output correlation matrices. Correlation matrices are plot-only diagnostics; redundant matrix/count CSV exports are not written.

Reporting directories are created lazily, so absent diagnostics (for example XAI for a model without an explainer) do not leave empty folders behind.

The paired relative improvement is computed as

`100 * (CTR_LED - CTR_model) / CTR_LED`

for every common bootstrap resample. This accounts for covariance between LED and ML CTR estimates from the same event population.

Concatenated-dataset studies intentionally do **not** produce voltage-comparison plots because only one pooled dataset/model is evaluated.

Detailed reporting is grouped under:

- `plots/corrections/`
- `plots/train_distribution/`
- `plots/test_distribution/`
- `plots/model_output/<model>/`
- `plots/xai/`

The reporting histograms are presentation views and use a compact display interval with about 20 bins. The canonical robust CTR itself is bin-free; `fit.bin_width_ps` is used only for the secondary core-FWHM diagnostic.

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

### Study summary analysis

A completed per-voltage study can be summarized with:

```bash
python -m waveform_analysis.scripts.analyze_study_results \
  --run-dir waveform_analysis/results/studies/complete_energy
```

By default this writes two files under `<run-dir>/analysis_summary/`:

- `study_summary.tex`: ready-to-include LaTeX table with selected LED threshold, blind-test LED CTR, and blind-test CTR for every ML model at each voltage;
- `study_summary_vs_voltage.pdf`: selected LED threshold and blind-test CTR versus bias voltage, including bootstrap CTR error bars.

Use `--plot-format png` for raster plots or `--output-dir <path>` to redirect the report.

### Cross-voltage generalization

To test whether a model trained at one bias voltage generalizes to the blind/test set of another voltage, run:

```bash
python -m waveform_analysis.scripts.cross_voltage_evaluation \
  --run-dir waveform_analysis/results/studies/complete_energy
```

By default the script evaluates every ML model that has a trained artifact for every voltage in the study. It reuses the selected saved model from each training voltage, including its saved temporal sample mask and output clipping, and applies it to the destination voltage's prepared blind/test split and calibrated ML target.

Outputs are written under `<run-dir>/cross_voltage/`:

- `cross_voltage_results.csv`: long-form numeric results for every `model × V_train × V_predict` combination;
- `cross_voltage_<model>.csv`: matrix with rows = training voltage, columns = prediction/blind-test voltage, and cells = `CTR ± bootstrap uncertainty`;
- `cross_voltage_<model>.pdf`: annotated CTR heatmap for the same matrix.

Diagonal cells use the CTR and uncertainty already stored in the original study after first reloading the saved model and verifying that its recomputed diagonal CTR agrees within 0.1 ps. Off-diagonal cells are newly evaluated on the destination blind/test set using the study's configured CTR estimator and bootstrap settings. Use `--models cnn cnn_2d` to restrict the analysis or `--diagonal-tolerance-ps <value>` to change the consistency tolerance.
