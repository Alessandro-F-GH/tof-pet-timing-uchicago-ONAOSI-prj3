# Waveform ML pipeline

The waveform analysis now uses a fixed-control, repeated-holdout protocol.

## Study unit

Each study resolves exactly one independent reference/control ROOT file, one analysis ROOT file, one mode, one registered ML model, and one waveform window.

1. Fit photopeak, baseline-noise, baseline-clipping and timing-ToT rules on the complete control dataset.
2. Apply those frozen rules to the control dataset and select the LED threshold once for every supported mode.
3. Persist a versioned/fingerprinted control artifact.
4. Apply the frozen rules and fixed LED threshold to the analysis dataset. Analysis data never refit selection rules.
5. Materialize one fixed ML population with normalization bounds taken from configured oscilloscope vertical limits.
6. For every `resampling_seed`, split that same population into train, validation and blind test. Split identity depends only on the analysis population and seed, not model/window/candidate.
7. One candidate: fit directly on train+validation and evaluate blind once. Multiple candidates: train each on train, select by validation CTR, refit the winner from scratch on train+validation, then evaluate blind once.

The blind set never affects preprocessing, LED selection, hyperparameter selection, sample masking or early stopping outside the legal fit population.

## Configuration

Old `standard`, `model_study`, `threshold_scan`, voltage-scan and concatenated-dataset schemas are intentionally incompatible.

A study JSON contains:

```json
{
  "name": "timing_lcmlp",
  "reference_dataset": "data/control.json",
  "analysis_dataset": "data/analysis.json",
  "preprocessing_config": "preprocessing/default_ctr.json",
  "mode": "timing_to_timing",
  "model": "locally_connected_mlp",
  "window": {"start": -1.5, "end": 2.0},
  "resampling": {
    "policy": "repeated_holdout",
    "seeds": [11, 22, 33],
    "validation_fraction": 0.20,
    "test_fraction": 0.20,
    "minimum_events_per_split": 50
  },
  "fit": {"histogram_bin_width_ps": 10.0, "bootstrap_samples": 500},
  "ml_input": {"subsampling": 1},
  "ml_output": {"max_abs_ps": 2000.0},
  "output_dir": "results/timing_lcmlp"
}
```

The values above illustrate one resolved study; split fractions and resampling seeds are intentionally study-level choices. The shared preprocessing file retains the previously used trigger, fit, ToT and LED-candidate settings unless explicitly changed for the new acquisition.

Dataset configs explicitly define `root_file`, `true_tof_ps`, and `channels`. Reference and analysis must be different files.

## Batch execution

```bash
python -m waveform_analysis.cli batch --config config/batches/main.json
```

Batch JSON:

```json
{"studies": ["../studies/a.json", "../studies/b.json"]}
```

Studies run sequentially; compatible preprocessing caches are reused.

## Outputs

A completed study contains `manifest.json`, `candidates.json`, lean `results.csv`, exact split/event IDs under `splits/`, final refitted models, `study.log`, and `hyperparameter_validation.png` only for multi-candidate studies. Control and analysis preprocessing caches contain compact stage-count CSVs and selection diagnostics. `results.csv` stores corrected and uncorrected fixed-LED CTR on the same evaluated events so later cross-study reporting can use paired seeds without rerunning training.
