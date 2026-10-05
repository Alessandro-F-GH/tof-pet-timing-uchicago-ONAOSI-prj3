# Waveform ML pipeline

This package evaluates waveform-based timing corrections for paired TOF-PET detector signals.

Scientifically, each study:

1. preprocesses and selects valid events using a fixed control-derived protocol;
2. builds the requested waveform window and detector mode;
3. selects model hyperparameters on one fixed validation split using RMSE;
4. freezes the selected configuration;
5. repeats train/blind evaluation over deterministic replicas;
6. compares corrected timing against the LED reference using CTR and RMSE;
7. produces per-study plots and cross-model reports.

The code supports only the current configuration and result formats. Old result schemas are not migrated or interpreted.

## Models

Current registered models include:

- `linear_ridge`
- `antisymmetric_mlp`
- `locally_connected_mlp`
- `shared_cnn1d`
- `shared_minirocket`
- `direct_mlp`
- `independent_cnn1d`
- `onishi_cnn`
- `direct_minirocket`

Models are classified as either **shared/antisymmetric** or **direct/non-shared** formulations and are compared on the same event population, validation split, and replica seeds whenever the study context matches.

## Configuration

Configuration files are under:

```text
config/
├── batches/
├── datasets/
├── model_spaces/
├── preprocessing/
└── reporting.json
```

A batch defines the analysis dataset, protocol, models, detector modes, waveform windows, number of replicas, output directory, and model-save policy. Model-specific hyperparameter spaces are stored in `config/model_spaces/`.

## Main commands

Validate a batch configuration:

```bash
python -m waveform_analysis.cli check-batch --config config/batches/test.json
```

Run a batch:

```bash
python -m waveform_analysis.cli batch --config config/batches/test.json --overwrite
```

Resume an interrupted batch:

```bash
python -m waveform_analysis.cli batch --config config/batches/test.json --resume
```

Rebuild preprocessing artifacts:

```bash
python -m waveform_analysis.cli batch \
  --config config/batches/test.json \
  --overwrite \
  --rebuild-preprocessing
```

Generate the cross-model report:

```bash
python -m waveform_analysis.cli report --batch-config config/batches/test.json
```

Recreate plots without retraining:

```bash
python -m waveform_analysis.cli remake-batch-plots --config config/batches/test.json
```

## Outputs

Each study stores its resolved configuration, selected hyperparameters, replica results, blind residuals, summary plots, and optional fitted models.

The batch report summarizes model performance across modes and waveform windows, including:

- blind CTR and RMSE;
- LED-to-ML improvement;
- paired model comparisons across replicas;
- RMSE-vs-CTR relation;
- CTR-vs-computation-time comparison;
- waveform-window comparison;
- best model by formulation and mode;
- model-output correlation matrices when matched blind outputs are available.

Hyperparameters are selected only from the fixed validation split. Replica statistics are computed only from blind evaluation rows; the validation set is never reused in replica training or testing.
