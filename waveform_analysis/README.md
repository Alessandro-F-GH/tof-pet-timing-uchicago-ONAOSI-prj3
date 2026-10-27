# Waveform timing analysis

This package implements waveform-based coincidence timing correction for two-detector TOF-PET measurements. It covers event selection, LED calibration, waveform preprocessing, model training, blind evaluation, reporting and post-training interpretability.

The workflow separates three dataset roles:

- **Control:** determine event-selection, normalization, feature-transform and LED-calibration settings.
- **Development:** select hyperparameters and fit models using the frozen control-derived rules.
- **Blind/test:** evaluate final models and estimate event-bootstrap uncertainty without refitting.

## Data and installation

**Experimental ROOT waveform files are not publicly provided because of their size and GitHub hosting limits.** Batch JSON files contain paths to local acquisitions. To run a real experiment, provide the corresponding data and update those paths as needed. The code and configurations can be inspected without the data; model fitting and numerical reproduction require them.

From the repository root:

```bash
python -m pip install -r waveform_analysis/requirements.txt
```

The requirements specify supported minimum versions for most libraries and pin `aeon==1.6.0` and `tshap==0.0.1`. They do not constitute a complete snapshot of the thesis training environment.

## Run an experiment

Check the configuration before launching a batch:

```bash
python -m waveform_analysis.cli check-batch --config waveform_analysis/config/batches/benchmark_UC.json
```

Run the configured experiments after the required ROOT files are available:

```bash
python -m waveform_analysis.cli batch --config waveform_analysis/config/batches/benchmark_UC.json
```

The supplied UC and FBK benchmark JSON files define dataset locations, waveform modes, observation windows, model sweeps, cross-validation and bootstrap settings. Other configurations under `config/batches/` cover additional experiments; in particular, combined MultiROCKET–RDST models have separate example configurations. A configuration is not automatically an exact manifest of every reported thesis result.

## Models

Supported families include:

| Family | Implementations |
| --- | --- |
| Linear Ridge | Direct and shared, using scikit-learn RidgeCV |
| MiniROCKET | Direct and shared, using sktime |
| MultiROCKET | Direct and shared, using aeon with Ridge regression |
| Random dilated shapelets (RDST) | Direct and shared, using aeon with Ridge regression |
| MultiROCKET + RDST | Direct and shared, with separately regularized feature blocks |
| Neural networks | Direct and shared MLPs, an Onishi-inspired CNN, and additional research architectures |

Direct models learn from the detector pair together. Shared models apply a common detector-level function and combine its outputs as a difference. The model registry also contains experimental alternatives not all included in the final thesis comparisons.

Most feature-based models use development cross-validation to select the Ridge penalty after a control-fitted transformation. The linear Ridge baselines use their separate RidgeCV procedure. The Onishi-inspired CNN has a separate fixed-epoch training procedure. Model settings are recorded in `config/model_spaces/`.

## Results and reporting

Training writes run metadata, model artifacts, predictions, metrics and figures under the configured results root (by default within `waveform_analysis/results/`, which is not tracked in Git).

Generate reports or remake plots from existing results without retraining:

```bash
python -m waveform_analysis.cli report waveform_analysis/results/UC/benchmark_48V_final
python -m waveform_analysis.cli plots --results waveform_analysis/results/UC/benchmark_48V_final
```

The paths above are examples: the corresponding results directory must already exist locally. Reporting reads saved results; it does not recreate missing trained models or original predictions. Commands for additional saved-model diagnostics and grouped reporting are available in the CLI and `scripts/`.

## TSHAP explanations

The repository supports global and local TSHAP explanations of saved, frozen models. The explanations require the relevant saved model artifacts and prepared data caches; they do not retrain a model or re-read the original ROOT files.

For the global settings used in the thesis (64 blind events, 32 control baselines, 80-sample non-overlapping windows), run:

```bash
python -m waveform_analysis.cli explain-tshap \
  --results waveform_analysis/results/UC/benchmark_48V_final \
  --models direct_multirocket direct_rdst shared_linear_ridge \
  --max-events 64 \
  --n-baselines 32 \
  --window-samples 80 \
  --stride-samples 80
```

Adjust the results path and model list to the experiment. The command processes eligible wide-window runs; it skips runs configured with the Onishi window. Use `--cache-root` if prepared caches were moved, or `--output-root` to choose where explanations are written. The default output is a separate `xai/` directory. See `python -m waveform_analysis.cli --help` and `python -m waveform_analysis.cli explain-tshap --help` for options.

TSHAP attributions indicate model sensitivity to temporal-segment replacement, not independent physical timing information or causal effects.

## Other commands

```bash
# Saved-model diagnostic predictions
python -m waveform_analysis.cli diagnostics --results waveform_analysis/results/UC/benchmark_48V_final

# Input-level occlusion explanations
python -m waveform_analysis.cli xai --results waveform_analysis/results/UC/benchmark_48V_final --models direct_rdst

# Frozen-feature regression explanations
python -m waveform_analysis.cli explain-regression --results waveform_analysis/results/UC/benchmark_48V_final --models direct_rdst

# Local temporal explanations
python -m waveform_analysis.cli explain-tshap-local --results waveform_analysis/results/UC/benchmark_48V_final --models direct_multirocket
```

These commands require the corresponding saved artifacts and, where applicable, verified prepared caches. They are not substitutes for the unavailable acquisition files when starting an experiment from scratch.

## Development and technical details

- [`ARCHITECTURE.md`](ARCHITECTURE.md) describes model interfaces, result ownership, frozen-feature caching and pipeline dependencies.
- [`CLEANUP_NOTES.md`](CLEANUP_NOTES.md) records legacy cleanup decisions and candidate archival utilities.
- [`tests/`](tests/) contains regression and integration tests for pipeline components.

From the repository root, with test dependencies available, run:

```bash
python -m pytest waveform_analysis/tests
```

The tests do not establish numerical reproduction of the original acquisition-based experiments. For research results, retain the exact batch configuration, package versions, source-data identities and saved evaluation artifacts.
