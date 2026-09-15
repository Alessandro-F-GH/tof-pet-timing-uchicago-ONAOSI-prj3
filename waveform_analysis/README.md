# Waveform timing pipeline

The waveform pipeline separates **selection**, **physical preprocessing**, **ML dataset construction**, and **model fitting**. Every experiment has exactly one configured `mode`: `energy_to_energy` or `timing_to_timing`. Optional CFD evaluation is controlled by the experiment-level boolean `cfd`. The active ML registry contains only `mlp` and `onishi_cnn`.

## 1. Event selection

The ROOT entry population is split into permanent **development** and **test** sets before any fitted selection. Using development only, the pipeline fits the two energy photopeaks, detects threshold hits on the waveform family required by the selected mode, derives timing ToT limits for `timing_to_timing`, and optionally derives a baseline-RMS limit. Frozen cuts are applied unchanged to test.

Selection, native-preprocessing and ML-prepared caches are mode-scoped, so two studies on the same ROOT source cannot reuse incompatible waveform families.

## 2. Native-time preprocessing

Only selected events and the waveform family required by the experiment mode are materialized. For each waveform the pipeline decodes/orients native samples, clamps them to detector-specific `vertical_scale_limit_mV`, crops around the selected main trigger, preserves native acquisition timing, and stores the rising-edge interval used by LED/CFD.

There is **no denoising** and no event-wise baseline subtraction.

## 3. ML dataset preparation

LED thresholds are scanned on development and ranked by the common CTR estimator from `utils_fit`; CFD is treated the same way when `cfd: true`. The canonical metric is the Gaussian-equivalent shortest interval containing the configured fraction of finite residuals, with 90% coverage by default. Bootstrap is skipped during candidate ranking because uncertainty is not part of threshold selection. LED crossing times are linearly interpolated. Waveforms remain on the native acquisition grid, so the ML anchor `t_a` is still the native sample nearest in time to the interpolated selected LED crossing for window materialization only.

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

## 4. Final ML protocol

The final study compares two deliberately different paired-waveform models.

### Proposed model: antisymmetric MLP

The proposed model is `mlp`: one shared dense scorer `g_theta` is applied independently to the two aligned detector waveforms and the timing correction is

`y_theta = g_theta(s1) - g_theta(s2)`.

This enforces exact detector-swap antisymmetry. A dense model is used intentionally because the waveforms are aligned to the LED crossing and absolute temporal position is physically meaningful; translation equivariance is therefore not treated as a useful prior for the proposed model.

The MLP hyperparameter grid is defined in `config/model_spaces/mlp.json`. Every candidate is trained using training data only and ranked by CTR on the validation split. The validation-selected trained checkpoint is used directly: **there is no refit**.

### Reference model: Onishi CNN

The paired CNN reference is registered as `onishi_cnn` and labelled **Onishi CNN**. It follows the paired-waveform architecture used for LED timing correction by Onishi et al. (Phys. Med. Biol. 67 (2022) 04NT01, DOI 10.1088/1361-6560/ac508f): the first convolution spans both detector rows and the network predicts one joint correction.

The fixed reference configuration is stored in `config/model_spaces/onishi_cnn.json`:

- Conv2D 2x5, 32 channels, ReLU, max-pool 1x3;
- Conv2D 1x3, 64 channels, ReLU, max-pool 1x3;
- Conv2D 1x3, 64 channels, ReLU, max-pool 1x3;
- flatten, dense 256, ReLU, scalar output;
- Adam, MSE, batch size 32, initial learning rate 1e-4;
- 100 epochs, learning-rate factor 0.1 at epochs 30 and 60.

The Onishi CNN has one fixed candidate. Validation is therefore not used to tune its architecture; it only passes through the same model-selection interface. The trained checkpoint is not refitted.

### Validation and blind-test policy

For every model with tunable hyperparameters:

1. fit candidate models using training data only;
2. select the candidate with the best **validation CTR**;
3. freeze that exact trained checkpoint;
4. evaluate final performance on the permanent blind/test split.

Validation metrics are selection diagnostics, not final performance results. Final CTR values and uncertainties are computed only on blind data. CTR uncertainty is the event-bootstrap standard deviation of the canonical Gaussian-equivalent shortest-coverage-interval estimator.

No target-magnitude filtering is used and no model is refitted after validation selection.

## 5. Experiment types

The experiment JSON explicitly declares `experiment.type`. Final analyses no longer require combining LED-threshold and window scans inside one ordinary study.

### `model_comparison`

This is the main final experiment. It requires exactly:

`models: ["mlp", "onishi_cnn"]`.

It also requires:

- one fixed LED threshold;
- an `onishi` window fixed to `[-1.5, 2.0]` ns relative to the LED crossing;
- one configured `wide` window.

The runner creates two complete sub-runs:

`<output>/onishi/`

and

`<output>/wide/`.

Each sub-run iterates over every configured bias-voltage ROOT dataset and produces the normal study outputs: blind CTR versus voltage, paired-bootstrap improvement over LED, residual distributions, model outputs, prediction-target correlations, model-output correlations, correction examples, XAI, saved searches, models, splits and numerical artifacts.

After both sub-runs finish, the experiment root additionally compares **Antisymmetric MLP vs Onishi CNN directly on the blind population**. The same event indices are resampled for the two models in every bootstrap replicate. The numerical comparison is stored in:

`csv/paired_model_comparison.csv`.

The experiment is intended to be run separately for each board dataset configuration and for each channel mode (`energy_to_energy` and `timing_to_timing`), keeping the board-specific input source explicit in `data_config`.

Minimal experiment-specific section:

```json
{
  "experiment": {
    "type": "model_comparison",
    "name": "BOARD_MODE_model_comparison",
    "output_dir": "results/studies/BOARD_MODE_model_comparison",
    "fixed_led_threshold_mV": 15.0,
    "windows": {
      "onishi": {"start": -1.5, "end": 2.0},
      "wide": {"start": -2.0, "end": 30.0}
    }
  },
  "models": ["mlp", "onishi_cnn"],
  "mode": "timing_to_timing",
  "cfd": false
}
```

The threshold and wide-window values in a real experiment must be set deliberately for that dataset; the example above is only a configuration example.

### `threshold_scan`

This experiment studies LED-threshold dependence for the antisymmetric MLP only. It requires:

- `models: ["mlp"]`;
- one explicit `experiment.voltage_V`;
- the threshold candidates in `standard_methods.led_thresholds_mV`.

For every threshold independently, the MLP hyperparameter grid is selected using validation CTR. The selected checkpoint is then evaluated on blind data without refitting. The scan **does not select a winning LED threshold** and never uses blind data for hyperparameter selection.

Reported scan quantities are blind-only:

- LED CTR with bootstrap uncertainty;
- MLP CTR with bootstrap uncertainty;
- relative MLP improvement over LED;
- paired-bootstrap uncertainty of that improvement.

Outputs are written to `csv/threshold_scan.csv`, threshold-specific artifacts, and blind-only threshold plots.

Minimal experiment-specific section:

```json
{
  "experiment": {
    "type": "threshold_scan",
    "name": "BOARD_MODE_threshold_scan",
    "output_dir": "results/studies/BOARD_MODE_threshold_scan",
    "voltage_V": 46
  },
  "models": ["mlp"],
  "mode": "timing_to_timing",
  "cfd": false
}
```

Run separate threshold-scan configs for UC/FBK and energy/timing as required.

## 6. Ordinary/legacy study runner

`experiment.type: "standard"` remains the low-level single-window study runner used internally by the model-comparison experiment. It still supports the existing study/report artifacts, but final paper comparisons should use the explicit experiment types above rather than mixing several questions in one run.

## Results

For ordinary per-voltage studies, the study root contains:

Study outputs are type-separated from creation time:

- `csv/results.csv`: canonical study results table;
- `csv/relative_improvement.csv`: numerical values used in the paired-improvement plot;
- `plots/ctr_vs_voltage.pdf`: publication-style CTR curves versus bias voltage with bootstrap error bars;
- `plots/relative_improvement_vs_voltage.pdf`: relative CTR improvement over LED, with uncertainty obtained from a **paired bootstrap** using the same resampled event indices for LED and each ML model;
- `plots/corrections/<dataset>/`: top/worst correction figures;
- `csv/corrections/<dataset>/`: corresponding correction ranking tables;
- `plots/xai/<model>/`: model-grouped XAI plots; MLP and Onishi CNN importance is input-gradient importance aggregated onto the exact waveform time axis;
- `csv/xai/<model>/`: tabular XAI exports only when present;
- `plots/model_output_diagnostics/`: publication-style prediction-vs-target and model-output correlation figures reconstructed from saved model-output/residual arrays.

Reporting directories are created lazily, so absent diagnostics (for example XAI for a model without an explainer) do not leave empty folders behind.

All report figures use one centralized publication style (`ml_pipeline/plot_style.py`): fixed single/double-column dimensions, embedded TrueType PDF fonts, color-vision-friendly model identities, line/marker redundancy for grayscale printing, restrained grids, and no plot titles or experiment-context text. Context belongs in the report caption.

The experiment persists the numerical data needed to redraw the figures: blind residuals, model outputs, XAI arrays, split/event identifiers and experiment-specific comparison CSV files. The first report render also caches the selected top/worst waveform examples inside the run artifacts. Therefore figure styling can be changed later without retraining models or rerunning the experiment.

Plot generation is intentionally separated from training. The `--remake-plots` run option deletes and recreates only plot directories from persisted numerical artifacts; it does not rerun event selection, preprocessing, model selection, fitting, or blind evaluation. Model-comparison root CTR-vs-voltage figures use the same canonical paper style as ordinary studies and include LED as the reference curve alongside Antisymmetric MLP and Onishi paired CNN.

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

The reporting histograms are presentation views and use a compact display interval with about 20 bins. CTR itself is bin-free and does not depend on the presentation histogram.

## CLI

```bash
python -m waveform_analysis.cli check --config waveform_analysis/config/experiments/timing.json
python -m waveform_analysis.cli prepare --config waveform_analysis/config/experiments/timing.json
python -m waveform_analysis.cli run --config waveform_analysis/config/experiments/timing.json --overwrite

# recreate figures only from saved CSV/NPY/model-output artifacts
python -m waveform_analysis.cli run \
  --config waveform_analysis/config/experiments/timing.json \
  --remake-plots

# equivalent run-directory interface; works for standard, model-comparison,
# and threshold-scan experiment roots
python -m waveform_analysis.cli report \
  --run-dir waveform_analysis/results/studies/complete_timing

# optionally render into a separate directory without changing the run data
python -m waveform_analysis.cli report \
  --run-dir waveform_analysis/results/studies/complete_timing \
  --output-dir waveform_analysis/results/paper_figures/complete_timing
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

Diagonal cells use the CTR and uncertainty already stored in the original study after first reloading the saved model and verifying that its recomputed diagonal CTR agrees within 0.1 ps. Off-diagonal cells are newly evaluated on the destination blind/test set using the study's configured CTR estimator and bootstrap settings. Use `--models mlp onishi_cnn` to restrict the analysis or `--diagonal-tolerance-ps <value>` to change the consistency tolerance.
