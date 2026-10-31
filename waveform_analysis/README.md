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

## Post-training TSHAP

The maintained interpretability workflow is TSHAP on frozen saved models, using
verified control/blind caches. Model training and blind evaluation do not run
any alternative XAI procedure.

```bash
python -m waveform_analysis.cli explain-tshap \
  --results waveform_analysis/results/UC/benchmark_48V_final \
  --models direct_multirocket direct_rdst shared_linear_ridge \
  --cache-root /path/to/cache_bundle \
  --ml-window wide \
  --imputation hermite_ppca

# Explain one saved-model run directly (no batch result-root scan).
python -m waveform_analysis.cli explain-tshap \
  --model-dir waveform_analysis/results/UC/benchmark_48V_final/energy/wide/direct_multirocket \
  --cache-root /path/to/cache_bundle \
  --max-events 100 \
  --window-samples 20 \
  --stride-samples 20
```

TSHAP supports two perturbation modes: `--imputation reference` performs symmetric two-player window substitution against a reproducible sample of independent control waveforms; `--imputation hermite_ppca` reconstructs each masked segment from that waveform's own observed context using detector-specific Hermite-PPCA models. Reference substitution is the default; select Hermite-PPCA explicitly when desired. Both modes use multi-baseline scoring and the upstream temporal aggregation/ROI implementation. The control reference count is configurable with `--n-baselines` (default: 8).

The imputer cache stores fitted detector-specific PPCA models and position-dependent blend weights as immutable ArtifactStore products, exposed through a stable named alias. Its identity includes the control population, board, channel/mode, time grid, window configuration, and selected reference-baseline cohort. Changes in baseline selection therefore cannot silently reuse an incompatible imputer. Global and local TSHAP outputs remain in the dedicated XAI tree, separate from run results.

Conditional temporal occlusion retains signed prediction differences in ps. Window scores distributed over samples are densities in ps/sample, not exact individual-sample Shapley values. This conditional reconstruction approach is not mathematically equivalent to reference-baseline SHAP. The
existing absolute-importance plots are provisional, pending the final
signed local, response-conditioned and global analyses. Use
`explain-tshap-local` for the current local diagnostic.

## Imputation diagnostics

`scripts/train_ppca_interpolation_blend.py` and
`scripts/evaluate_imputation_benchmark.py` are retained for quality checks.
The benchmark compares reference substitution, interpolation, PPCA, and the
position-specific blend, with RMSE measured in mV. These diagnostics are not
part of the fitted timing-correction model.

## Other commands

The CLI retains `diagnostics`, `explain-tshap` and
`explain-tshap-local`. Old grouped-occlusion, shapelet-coefficient ranking,
and regression-surrogate XAI commands have been removed.

## Development and technical details

- [`ARCHITECTURE.md`](ARCHITECTURE.md) describes model interfaces, result ownership, frozen-feature caching and pipeline dependencies.
- [`CLEANUP_NOTES.md`](CLEANUP_NOTES.md) records legacy cleanup decisions and candidate archival utilities.
- [`tests/`](tests/) contains regression and integration tests for pipeline components.

From the repository root, with test dependencies available, run:

```bash
python -m pytest waveform_analysis/tests
```

The tests do not establish numerical reproduction of the original acquisition-based experiments. For research results, retain the exact batch configuration, package versions, source-data identities and saved evaluation artifacts.

## Two-stage timing correction

The two-stage experiment applies the existing shared linear Ridge correction to a
short Onishi window on a fixed 25% development subset, then trains the prescribed
direct MultiRocket model on the complete wide window and the out-of-sample residual
target for the remaining 75%. The split is deterministic and saved per board/mode.
The fitted wide-window transformation is loaded from the matching completed
full-development benchmark run; it is never refitted for this experiment.

Run one board and acquisition mode at a time, using the corresponding existing
wide-window benchmark configuration:

```bash
python -m waveform_analysis.cli two-stage \
  --config waveform_analysis/config/batches/benchmark_UC.json \
  --board UC --mode energy_to_energy --split-fraction 0.25 \
  --cache-root "$HOME/UC_cache"
```

For UC timing use `--mode timing_to_timing`; for FBK use
`benchmark_FBK.json`, `--board FBK` and
`--cache-root "$HOME/FBK_cache"`. The board-specific cache defaults are
`$HOME/UC_cache` and `$HOME/FBK_cache`, respectively, so the explicit
`--cache-root` option is only needed to override them. The command uses
`preprocessing/` and `artifact_store/` inside that cache bundle. Results are written under a dedicated
`two_stage/<board>/<mode>/` namespace and do not overwrite single-stage runs.
The command requires a completed matching wide-window source run and its saved
model/transform artifact in the configured artifact store. It stops if the source
population, transformation parameters or seed do not match.
