# Waveform timing ML pipeline

This package implements the waveform-based timing studies used in the project. It keeps event selection, native-time preprocessing, standard timing methods, ML training, blind evaluation and reporting in one reproducible workflow.

The final ML workflow currently compares two model families:

- an antisymmetric shared MLP, `g(s1) - g(s2)`;
- the paired 2-D CNN based on Onishi et al., implemented with the fixed paper architecture/training configuration.

## Final evaluation policy

Final model performance is always evaluated on a permanent blind/test split that is not used for model selection. The MLP uses training data plus a validation split for hyperparameter selection and freezes the selected trained checkpoint without refitting. The Onishi CNN has no model-selection grid and is trained with its fixed reference configuration.

No target-magnitude filtering is used in training or final evaluation.

## Experiment types

### `model_study`

Final ML runs are single-model experiments. A model study requires exactly one configured model, one fixed LED threshold and one channel mode. Waveform windows are defined centrally in the shared profile under `ml_input.windows`, rather than duplicated across model-specific experiment files.

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

Each window is independently complete and portable. It contains the saved split identities, LED residuals, model residuals/output, model/search metadata, XAI and publication plots. In particular each window produces:

- `plots/ctr_vs_voltage.pdf`: LED reference plus the studied model;
- `plots/improvement_vs_led.pdf`: absolute improvement `CTR_LED - CTR_model` in ps;
- `plots/relative_improvement_vs_voltage.pdf`: relative improvement in percent;
- XAI and the ordinary correction/distribution/model-output diagnostics.

At the model-study root, `plots/ctr_vs_voltage_windows.pdf` overlays the model result from every configured waveform window against the common LED reference.

This makes expensive models independent: MLP and Onishi CNN can be trained on different computers, operating systems or GPUs. The complete result directory is the portable analysis unit; no preprocessing cache or checkpoint from the other machine is required.

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

After the run directories have been copied onto one machine, combine them without retraining:

```bash
python -m waveform_analysis.cli compare-runs \
  --runs waveform_analysis/results/studies/timing_mlp \
         waveform_analysis/results/studies/timing_onishi_cnn \
  --output-dir waveform_analysis/results/comparisons/timing_models
```

The comparison checks that the runs use the same window definitions and voltage set before combining their persisted blind/test results.

The combined report contains, for every waveform window:

- `plots/ctr_vs_voltage_<window>.pdf`: one LED reference plus all supplied models;
- absolute and relative model improvement plots;
- pairwise model comparison plots;
- `plots/model_output_correlation_<window>.pdf`: Pearson correlation between model correction outputs versus bias voltage.

Numerical output correlations are saved in `csv/model_output_correlations.csv`. Correlation is computed only on the blind/test population. The comparison reads each model's persisted `*_test_model_output_ps.npy` artifact, checks the corresponding `test_event_index`, and aligns one run to the other by event identity before computing Pearson `r`. A correlation is rejected if the two runs do not contain the same blind events, so runs produced independently on different machines remain safely comparable.

### `threshold_scan`

This experiment studies LED-threshold dependence for the antisymmetric MLP only. It requires:

- `models: ["mlp"]`;
- one explicit `experiment.voltage_V`;
- the threshold candidates in `standard_methods.led_thresholds_mV`.

For every threshold independently, the MLP hyperparameter grid is selected using validation CTR. The selected checkpoint is then evaluated on blind data without refitting. The scan does not select a winning LED threshold and never uses blind data for hyperparameter selection.

Reported scan quantities are blind-only:

- LED CTR;
- MLP CTR;
- absolute/relative MLP improvement over LED.

## Plotting policy

CTR-vs-voltage figures show central CTR estimates without error bars. Numerical CTR uncertainties remain stored in CSV/table outputs. Error bars are kept only on comparison/improvement plots where the displayed quantity explicitly carries an uncertainty estimate.

All report figures use the centralized publication style in `ml_pipeline/plot_style.py`: fixed single/double-column dimensions, embedded TrueType PDF fonts, color-vision-friendly model identities, line/marker redundancy for grayscale printing, restrained grids and concise axes with explicit units.

## CLI

Run the two expensive models independently:

```bash
python -m waveform_analysis.cli run \
  --config waveform_analysis/config/experiments/model_study_mlp.json \
  --overwrite

python -m waveform_analysis.cli run \
  --config waveform_analysis/config/experiments/model_study_onishi.json \
  --overwrite
```

They may be run on different machines. After copying both completed study directories onto one machine:

```bash
python -m waveform_analysis.cli compare-runs \
  --runs waveform_analysis/results/studies/timing_mlp \
         waveform_analysis/results/studies/timing_onishi_cnn \
  --output-dir waveform_analysis/results/comparisons/timing_models
```

The standalone reporting entry point is equivalent:

```bash
python -m waveform_analysis.scripts.compare_model_runs \
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
```
