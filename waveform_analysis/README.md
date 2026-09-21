# Waveform timing pipeline

The waveform pipeline separates **selection**, **physical preprocessing**, **ML dataset construction**, and **model fitting**. Every experiment has exactly one configured `mode`: `energy_to_energy` or `timing_to_timing`. Optional CFD evaluation is controlled by the experiment-level boolean `cfd`. The active ML registry contains `mlp`, `locally_connected_mlp`, and `onishi_cnn`.

## 1. Event selection

The ROOT entry population is split into permanent **development** and **test** sets before any fitted selection. Using development only, the pipeline fits the two energy photopeaks, detects threshold hits on the waveform family required by the selected mode, derives timing ToT limits for `timing_to_timing`, and optionally derives a baseline-RMS limit. Frozen cuts are applied unchanged to test.

Selection, native-preprocessing and ML-prepared caches are mode-scoped, so two studies on the same ROOT source cannot reuse incompatible waveform families.

## 2. Native-time preprocessing

Only selected events and the waveform family required by the experiment mode are materialized. For each waveform the pipeline decodes/orients native samples, clamps them to detector-specific `vertical_scale_limit_mV`, crops around the selected main trigger, preserves native acquisition timing, and stores the rising-edge interval used by LED/CFD.

There is **no denoising** and no event-wise baseline subtraction. LED thresholds are nevertheless baseline-relative: for each event and detector, the configured LED value is added to the mean pre-trigger baseline measured in `preprocessing.selection.baseline_noise.window_ns`; the waveform itself is left unchanged.

## 3. ML dataset preparation

LED thresholds are scanned on development and ranked by the common CTR estimator from `utils_fit`; CFD is treated the same way when `cfd: true`. The canonical metric is the Gaussian-equivalent shortest interval containing the configured fraction of finite residuals, with 90% coverage by default. Bootstrap is skipped during candidate ranking because uncertainty is not part of threshold selection. LED crossing times are linearly interpolated at the event-specific level `baseline + configured LED threshold`. Waveforms remain on the native acquisition grid, so the ML anchor `t_a` is still the native sample nearest in time to the interpolated selected LED crossing for window materialization only.

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

The MLP hyperparameter grid is defined in `config/model_spaces/mlp.json`. Weight optimization uses stochastic gradient descent with Nesterov momentum (momentum 0.9 by default) and RMSE loss. When multiple candidates are configured, each candidate is trained on the training split and ranked by CTR on the validation split. After the parameters are selected, a fresh final MLP is trained on the full development population; the MLP itself reserves its configured internal holdout from that development population for early stopping. If the model space contains only one candidate, validation-based model selection is skipped and the final development fit starts immediately.

### Locally connected MLP

The parallel model `locally_connected_mlp` keeps the same detector-shared antisymmetric form but replaces the fully connected first stage with a 1-D locally connected layer. Each node receives one contiguous receptive field of waveform samples and produces one scalar local representation. Neighboring fields may overlap. Unlike a CNN, weights and biases are not shared across temporal positions, so absolute position remains explicit.

For receptive-field width `K` and overlap `O`, the stride is `K - O`. There is exactly one learned node per receptive field: no bank of multiple kernels is applied to the same window. The local outputs are activated and then passed to a conventional dense stack. PyTorch's standard `Tensor.unfold` operation extracts the overlapping windows; the per-position weights are ordinary `nn.Parameter` tensors.

Because local receptive fields require consecutive samples, this model retains the complete configured ML time grid instead of applying the training-derived constant-sample mask. Receptive-field width and overlap are selected on validation CTR together with the other configured hyperparameters. The ready configuration is `config/experiments/model_study_locally_connected_mlp.json`.

### Reference model: Onishi CNN

The paired CNN reference is registered as `onishi_cnn` and labelled **Onishi CNN**. It follows the paired-waveform architecture used for LED timing correction by Onishi et al. (Phys. Med. Biol. 67 (2022) 04NT01, DOI 10.1088/1361-6560/ac508f): the first convolution spans both detector rows and the network predicts one joint correction.

The fixed reference configuration is stored in `config/model_spaces/onishi_cnn.json`:

- Conv2D 2x5, 32 channels, ReLU;
- Conv2D 1x3, 32 channels, ReLU;
- Conv2D 1x3, 64 channels, ReLU;
- no pooling layers;
- flatten, dense 256, ReLU, scalar output;
- Adam, MSE, batch size 128, initial learning rate 1e-3;
- 600 epochs, learning rate reduced to 1e-4 and 1e-5 at epochs 180 and 360 (30% and 60% of training).

The Onishi CNN has one fixed reference configuration. Model selection is therefore skipped entirely and the reference model is fitted directly on the development population before blind evaluation.

### Validation and blind-test policy

For every model with tunable hyperparameters:

When more than one candidate is configured, candidate models are fitted on the training split and the best hyperparameters are selected by validation CTR. The search checkpoints are then discarded. A fresh final model is trained on the full development population and evaluated on the permanent blind/test split. If only one candidate exists, the validation-selection stage is skipped and the final development fit is performed directly. Validation metrics are selection diagnostics, not final performance results. Final CTR values and uncertainties are computed only on blind data. CTR uncertainty is the event-bootstrap standard deviation of the canonical Gaussian-equivalent shortest-coverage-interval estimator.

## 5. Experiment types

The experiment JSON explicitly declares `experiment.type`. Final analyses no longer require combining LED-threshold and window scans inside one ordinary study.

### `model_study`

Final ML runs are **single-model experiments**. A model study requires exactly one
configured model, for example `mlp` or `onishi_cnn`, plus one fixed LED
threshold. The waveform windows are not defined in the experiment file: they
live in the shared profile under `ml_input.windows`.

The default profile currently defines:

```json
"ml_input": {
  "windows": {
    "onishi_window": {"start": -1.5, "end": 2.0},
    "wide_window": {"start": -2.0, "end": 30.0}
  },
  "default_window": "wide_window",
  "subsampling": 1
}
```

A model study creates one complete standard sub-run for every profile window:

```text
<output>/
├── onishi_window/
└── wide_window/
```

Each window is independently complete and portable. It contains the saved
split identities, LED residuals, model residuals/output, model/search metadata,
XAI and publication plots. In particular each window produces:

- `plots/ctr_vs_voltage.pdf`: LED reference plus the studied model;
- `plots/improvement_vs_led.pdf`: absolute paired-bootstrap improvement
  `CTR_LED - CTR_model` in ps;
- `plots/relative_improvement_vs_voltage.pdf`: paired-bootstrap relative
  improvement in percent;
- XAI and the ordinary correction/distribution/model-output diagnostics.

This makes expensive models independent: MLP and Onishi CNN can be trained on
different computers, operating systems or GPUs. The complete result directory
is the portable analysis unit; no preprocessing cache or checkpoint from the
other machine is required.

Example MLP experiment:

```json
{
  "experiment": {
    "type": "model_study",
    "name": "timing_mlp",
    "output_dir": "results/studies/timing_mlp",
    "fixed_led_threshold_mV": 15.0
  },
  "models": ["mlp"],
  "mode": "timing_to_timing",
  "cfd": false
}
```

The corresponding Onishi experiment differs only in model/name/output:

```json
{
  "experiment": {
    "type": "model_study",
    "name": "timing_onishi_cnn",
    "output_dir": "results/studies/timing_onishi_cnn",
    "fixed_led_threshold_mV": 15.0
  },
  "models": ["onishi_cnn"],
  "mode": "timing_to_timing",
  "cfd": false
}
```

### Comparing completed model studies

After the run directories have been copied onto one machine, combine them
without retraining:

```bash
python -m waveform_analysis.cli compare-runs \
  --runs waveform_analysis/results/studies/timing_mlp \
         waveform_analysis/results/studies/timing_onishi_cnn \
  --output-dir waveform_analysis/results/comparisons/timing_models
```

The comparison first checks a portable compatibility signature covering data
definition, preprocessing, split policy, LED threshold, CTR settings and
profile windows. It then verifies the exact persisted blind `event_index`
arrays for every dataset/window before any paired model comparison.

The combined report contains one CTR-vs-voltage plot per window with a single
LED reference and all supplied models, paired-bootstrap improvement versus LED
for each model, and numerical pairwise model comparisons. Because pairing is
verified from persisted event identities, runs produced independently on
different machines can be safely compared.

### `threshold_scan`

This experiment studies LED-threshold dependence for the antisymmetric MLP only. It requires:

- `models: ["mlp"]`;
- one explicit `experiment.voltage_V`;
- the threshold candidates in `standard_methods.led_thresholds_mV`.

For every threshold independently, the MLP hyperparameter grid is selected using validation CTR when more than one candidate is configured. The selected parameters are then used for a fresh final fit on the full development population before blind evaluation. If only one candidate exists, selection is skipped and the final development fit is performed directly. The scan **does not select a winning LED threshold** and never uses blind data for hyperparameter selection.

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

## 6. Low-level standard study runner

`experiment.type: "standard"` remains the low-level single-window runner used internally by `model_study`. Final paper model runs should use `model_study`; `standard` remains useful for focused development/debugging.

## Results

For ordinary per-voltage studies, the study root contains:

Study outputs are type-separated from creation time:

- `csv/results.csv`: canonical study results table;
- `csv/relative_improvement.csv`: numerical values used in the paired-improvement plot;
- `plots/ctr_vs_voltage.pdf`: publication-style CTR curves versus bias voltage;
- `plots/relative_improvement_vs_voltage.pdf`: relative CTR improvement over LED, with uncertainty obtained from a **paired bootstrap** using the same resampled event indices for LED and each ML model;
- `plots/corrections/<dataset>/`: top/worst correction figures;
- `csv/corrections/<dataset>/`: corresponding correction ranking tables;
- `plots/xai/<model>/`: model-grouped XAI plots; MLP and Onishi CNN importance is input-gradient importance aggregated onto the exact waveform time axis;
- `csv/xai/<model>/`: tabular XAI exports only when present;
- `plots/model_output_diagnostics/`: publication-style prediction-vs-target and model-output correlation figures reconstructed from saved model-output/residual arrays.

Reporting directories are created lazily, so absent diagnostics (for example XAI for a model without an explainer) do not leave empty folders behind.

All report figures use one centralized publication style (`ml_pipeline/plot_style.py`): fixed single/double-column dimensions, embedded TrueType PDF fonts, color-vision-friendly model identities, line/marker redundancy for grayscale printing, restrained grids, and no plot titles or experiment-context text. Context belongs in the report caption.

The experiment persists the numerical data needed to redraw the figures: blind residuals, model outputs, XAI arrays, split/event identifiers and experiment-specific comparison CSV files. The first report render also caches the selected top/worst waveform examples inside the run artifacts. Therefore figure styling can be changed later without retraining models or rerunning the experiment.

Plot generation is intentionally separated from training. The `--remake-plots` run option deletes and recreates only plot directories from persisted numerical artifacts; it does not rerun event selection, preprocessing, model selection, fitting, or blind evaluation. For a `model_study`, every configured window is rebuilt independently from its saved artifacts. Plot axes use concise paper-style labels with explicit units, and voltage axes show integer tick labels only.
CTR-vs-voltage figures show central CTR estimates without error bars; uncertainty bars are reserved for paired-bootstrap improvement/comparison plots, while numerical CTR uncertainties remain stored in CSV results and tables.

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

## Dataset-table export from existing preprocessing caches

Dataset tables for the report can be generated directly from the frozen event-selection caches. This export does **not** read waveform ROOT contents, rerun event selection, rebuild preprocessing, prepare ML data, or train a model. It only uses the configured source filenames to locate the corresponding cache directories and reads each cache's `manifest.json` and `selection_summary.csv`.

Example:

```bash
python -m waveform_analysis.cli dataset-table \
  --config waveform_analysis/config/experiments/model_study_mlp.json \
  --output-file report/tables/uc_board_dataset.tex \
  --caption "UC-board waveform dataset." \
  --label tab:uc-board-dataset
```

The generated rows contain bias voltage, collected events, two-detector photopeak events, and final selected events. Bias voltage is parsed with the repository-wide `voltage_from_name()` helper; cache locations are resolved with `dataset_cache_dir()`; persisted JSON/CSV files are read with the common I/O helpers. If any required selection cache is missing or inconsistent, the command fails rather than rebuilding it.

Use the corresponding configuration for each board/source dataset to create its report table.

## CLI

Run the two expensive models independently:

```bash
python -m waveform_analysis.cli check \
  --config waveform_analysis/config/experiments/model_study_mlp.json

python -m waveform_analysis.cli run \
  --config waveform_analysis/config/experiments/model_study_mlp.json \
  --overwrite

python -m waveform_analysis.cli run \
  --config waveform_analysis/config/experiments/model_study_onishi.json \
  --overwrite
```

They may be run on different machines. After copying both completed study
directories onto one machine:

```bash
python -m waveform_analysis.cli compare-runs \
  --runs waveform_analysis/results/studies/timing_mlp \
         waveform_analysis/results/studies/timing_onishi_cnn \
  --output-dir waveform_analysis/results/comparisons/timing_models
```

Recreate a completed model study's figures without training:

```bash
python -m waveform_analysis.cli run \
  --config waveform_analysis/config/experiments/model_study_mlp.json \
  --remake-plots
```

Or rebuild from the run directory directly:

```bash
python -m waveform_analysis.cli report \
  --run-dir waveform_analysis/results/studies/timing_mlp

python -m waveform_analysis.cli report \
  --run-dir waveform_analysis/results/studies/timing_mlp \
  --output-dir waveform_analysis/results/paper_figures/timing_mlp
```

For the concatenated timing study:

```bash
python -m waveform_analysis.cli run \
  --config waveform_analysis/config/experiments/timing_concatenated.json \
  --overwrite
```

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
