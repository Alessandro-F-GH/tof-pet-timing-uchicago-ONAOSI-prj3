# Waveform ML pipeline

The waveform ML benchmark uses one fixed hyperparameter-selection split followed by repeated blind replicas.

## Protocol

For each resolved `dataset + mode + model + window` run:

1. A **fixed validation set** is drawn once from the prepared population using the batch seed.
2. Every hyperparameter candidate is trained once on the complementary tuning-training pool and evaluated only on that fixed validation set.
3. The best candidate is selected by fixed-validation CTR and its configuration is frozen.
4. The validation set stops being an evaluation set. For replica `r`, the blind test is sampled only from the original non-validation pool, using a deterministic replica seed derived from the same batch seed.
5. The replica model is fit **once** on `fixed validation + all non-blind events` and evaluated once on that replica's blind test.

The hyperparameter-tuning pass is not a replica and is not included in replica uncertainty.

With `validation_fraction = 0.10` and `blind_fraction = 0.50`, 10% of the full population is fixed validation, each blind test contains 50% of the full population sampled from the remaining 90%, and each replica model is trained on the other 50%.

The protocol requires

```text
validation_fraction + blind_fraction < 1
```

so every replica retains a variable non-validation training subset.

## Compact batch configuration

```json
{
  "name": "benchmark_49V",
  "reference_dataset": "../datasets/control.json",
  "analysis_dataset": "../datasets/analysis_49V.json",
  "output_dir": "results/studies/benchmark_49V",
  "save_models": "first",
  "protocol": {
    "seed": 1001,
    "preprocessing_config": "../preprocessing/default_ctr.json",
    "model_selection": {
      "validation_fraction": 0.10
    },
    "evaluation": {
      "n_replicas": 50,
      "blind_fraction": 0.50,
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

The configuration schema is strict: only the fields shown above are accepted. There is no compatibility path for previous experiment layouts.

## Seeds and shared artifacts

`protocol.seed` is the only sampling seed supplied by the batch.

- fixed validation seed: derived from `protocol.seed + population identity`
- replica seed: derived from `protocol.seed + population identity + replica index`

Models sharing the same prepared population and sampling protocol therefore reuse exactly the same fixed validation split and replica blind tests.

Shared artifacts are written once under:

```text
<batch>/artifacts/populations/<population>/sampling/<sampling-id>/
    fixed_validation.npz
    manifest.json
    replicas/
        replica_001_seed_<seed>/
            split.npz
            blind_reference.npz
            baseline_<fit-id>.json
        ...
```

`fixed_validation.npz` contains `tuning_train` and `validation`. A replica `split.npz` contains only `train` and `test`; the fixed validation indices are already included in replica `train`.

## Results

Each run writes one `results.csv` with two explicit phases:

- `phase = hyperparameter_validation`: one row per candidate, evaluated on the one fixed validation set; `replica_index` is empty.
- `phase = replica`: one row per blind replica using the selected frozen configuration; `replica_index = 1..N`.

`selected_hyperparameters.json` records the chosen candidate and the fixed-validation selection rule.

Once hyperparameters are frozen, each replica consists of one fit and one blind evaluation.

## Model persistence

Batch `save_models` accepts:

- `all`: save the fitted model from every replica.
- `first`: save only the fitted model from replica 1.
- `none`: save no fitted replica model.

Hyperparameter-tuning candidate models are never saved. `first` always means the first blind replica, never the fixed-validation tuning fit.

Blind residuals are retained independently of model persistence, so reporting and output-correlation analysis work with `save_models: "none"`.

## Reporting

Reports use only `phase = replica` rows for uncertainty and model comparison.

Paired model comparisons require the same:

- analysis dataset and analysis-protocol identity,
- sampling identity,
- mode,
- waveform window,
- replica index.

Energy and timing modes are never pooled.

The report includes:

- replica CTR/RMSE summaries,
- paired LED-to-ML improvements,
- paired model CTR comparisons,
- model-output Pearson-correlation matrices.

Model-output correlation is computed event-by-event inside each matched blind replica and combined across replicas with a Fisher-z mean.

## Commands

```bash
python -m waveform_analysis.cli check-batch --config config/batches/benchmark_other_models_49V.json
python -m waveform_analysis.cli batch --config config/batches/benchmark_other_models_49V.json
python -m waveform_analysis.cli report --batch-config config/batches/benchmark_other_models_49V.json
```
