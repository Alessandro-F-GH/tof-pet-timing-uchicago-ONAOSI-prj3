# Waveform ML pipeline

The waveform analysis uses a fixed-control, repeated-holdout protocol. A resolved run is one `dataset + mode + model + window`; a batch generates and organizes those runs while owning model-independent experiment artifacts.

## Statistical protocol

For every repeated-holdout replica, preprocessing and LED selection remain fixed, candidate selection uses only train/validation data, the winning model is refit on train+validation, and the blind split is evaluated once. CTR extraction is a point estimate on each replica: the CTR fit itself is not bootstrapped. Study variation is estimated from the distribution across replica seeds.

Cross-model paired bootstrap is allowed only when analysis dataset, resolved analysis-protocol identity, mode, window, and replica seed match. Energy and timing modes are never pooled.

## Compact batch configuration

Comparison studies should normally use one batch JSON instead of one JSON per model/mode/window combination. Shared protocol choices are written once and the batch expands the Cartesian product of models, modes, and named windows. Mode-specific protocol values are supported when scientifically required, for example different CTR histogram bin widths for energy and timing.

```json
{
  "name": "benchmark_49V",
  "reference_dataset": "../datasets/control.json",
  "analysis_dataset": "../datasets/analysis_49V.json",
  "output_dir": "results/studies/benchmark_49V",
  "protocol": {
    "preprocessing_config": "../preprocessing/default_ctr.json",
    "resampling": {
      "policy": "repeated_holdout",
      "seed": 1001,
      "n_replicas": 50,
      "validation_fraction": 0.20,
      "test_fraction": 0.50,
      "minimum_events_per_split": 50
    },
    "fit": {
      "energy_to_energy": {"histogram_bin_width_ps": 20.0},
      "timing_to_timing": {"histogram_bin_width_ps": 10.0}
    },
    "ml_input": {"subsampling": 1},
    "ml_output": {"max_abs_ps": 2000.0}
  },
  "sweep": {
    "models": ["mlp", "locally_connected_mlp", "independent_cnn1d"],
    "modes": ["energy_to_energy", "timing_to_timing"],
    "windows": {
      "onishi": {"start": -1.5, "end": 2.0},
      "wide": {"start": -2.0, "end": 30.0}
    }
  }
}
```

Optional `sweep.exclude` entries remove exceptional combinations without enumerating the remaining runs. Legacy explicit study-list batches remain readable; legacy `resampling.n_bootstrap` is interpreted as `n_replicas`, while `fit.bootstrap_samples` is ignored because fit-level bootstrap is no longer part of the ML pipeline.

## Shared batch artifacts

A compact batch stores model-independent artifacts once under `artifacts/populations/`. A population is identified by the resolved analysis protocol; each replica stores one positional `split.npz`, one blind LED reference, and one LED baseline point estimate. Every model using that same population and seed reuses those files. Model run directories therefore keep only model-specific results, residuals/predictions, fitted models, and diagnostics.

The event-population identity and analysis-protocol identity are distinct. Split generation is tied to the event population, while paired model comparison requires the stricter analysis-protocol identity so studies with coincidentally identical event IDs but different preprocessing/control definitions are not treated as paired.

Control LED-selection cache identity includes the mode-specific CTR fit definition. This makes the energy/timing LED threshold selection independent of batch execution order. When preprocessing is explicitly rebuilt, the batch rebuilds each shared control/mode/window scope once rather than once per model.

## Commands

Validate a batch:

```bash
python -m waveform_analysis.cli check-batch --config config/batches/benchmark_49V.json
```

Run it:

```bash
python -m waveform_analysis.cli batch --config config/batches/benchmark_49V.json
```

Report one batch:

```bash
python -m waveform_analysis.cli report --batch-config config/batches/benchmark_49V.json
```

Report multiple completed studies or batch result roots:

```bash
python -m waveform_analysis.cli report \
  --studies results/studies/study_A results/studies/study_B \
  --output-dir results/reports/A_vs_B
```

Reports are grouped by model, mode, and window. They contain seed-level mean/std summaries, replica-level paired bootstrap comparisons where pairing is valid, one paired LED-to-ML improvement plot per mode/window, and one model CTR comparison plot per mode/window.

A compact batch writes a root `manifest.json`, `runs.csv`, and shared `artifacts/`. Each resolved model run stores its own `manifest.json`, `resolved_config.json`, `results.csv`, candidates, fitted models, model-dependent blind residuals, and plots; it references the shared replica artifacts instead of duplicating split and LED arrays.
