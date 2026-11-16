# Waveform timing analysis

The pipeline corrects two-detector TOF-PET coincidence timing using digitized energy or timing waveforms. It includes native waveform filtering, LED/event selection, control-fitted preprocessing, model selection by development cross-validation, full-development fitting, blind evaluation and event bootstrap uncertainty.

## Dataset roles

- **Control**: fit event-selection, LED-calibration and waveform preprocessing rules, and control-fitted feature transforms.
- **Development**: select model hyperparameters using common CV folds and train the final model.
- **Blind**: evaluate the frozen final model. Bootstrap resamples blind events without retraining.

No ROOT acquisition files are included in this repository.

## Definitive model comparison

| Board | Preprocessing | Onishi window | Wide window |
|---|---|---|---|
| UC | Butterworth low-pass, 2 GHz, order 5 | −1.5 to 2 ns | −2 to 50 ns |
| FBK | Butterworth low-pass, 2 GHz, order 5 | −1.5 to 2 ns | −2 to 60 ns |

Filtering acts on **both detector channels** on complete native traces, before triggering and cropping. The same filter settings apply to energy and timing.

The five comparison models are `direct_linear_ridge`, `shared_linear_ridge`, `onishi_cnn`, `direct_rdst` and `direct_multirocket`. Both observation windows and both input modes are evaluated for each model, for 40 runs total.

Additional registered estimators remain available for exploratory work; they are not part of the definitive comparison. Ridge alpha is selected by the generic development CV, using `sklearn.linear_model.Ridge`. The Onishi CNN has its separate MSE training procedure. All selected estimators are refitted on the full development population before blind prediction.

```bash
python -m waveform_analysis.cli check-batch --config waveform_analysis/config/batches/UC_lowpass_final.json
python -m waveform_analysis.cli batch --config waveform_analysis/config/batches/UC_lowpass_final.json

python -m waveform_analysis.cli check-batch --config waveform_analysis/config/batches/FBK_lowpass_final.json
python -m waveform_analysis.cli batch --config waveform_analysis/config/batches/FBK_lowpass_final.json
```

Each batch has its own result directory. An explicitly configured integer seed controls the stochastic steps; seeds are not derived from filenames, model names or hash-based population identifiers.

## Preparation, results and resume

A batch can reuse precomputed preparation and frozen features using a dedicated cache bundle:

```bash
python -m waveform_analysis.cli prepare-cache \
  --config waveform_analysis/config/batches/UC_lowpass_final.json \
  --cache-root /path/to/UC_cache

python -m waveform_analysis.cli batch \
  --config waveform_analysis/config/batches/UC_lowpass_final.json \
  --cache-root /path/to/UC_cache
```

Keep UC and FBK caches separate. The execution runner maintains per-stage completion records, completed CV candidate/fold rows and an Optuna sampler for resuming interrupted model selection. Cached transforms/features are shared where appropriate; they are never refitted for blind evaluation.

Within each model's result directory, `model/` contains the saved model and its fitted transform, `artifacts/predictions.npz` holds blind predictions, `artifacts/development_predictions.npz` holds development predictions, and `artifacts/bootstrap.npz` holds bootstrap draws. CV results and stage metadata are recorded in `tables/` and `metadata/`. Model and prediction arrays are stored directly under each run. The batch runner can resume completed CV folds, and never needs an external artifact registry.

**Cache implementation:** Control selection, native waveforms, prepared populations, control-derived timing sample masks, frozen transforms and feature matrices now use direct, human-readable cache paths. Incomplete cache builds cannot be reused: an atomic publication step writes a `.complete` marker only after successful materialization. See [CACHE.md](CACHE.md) for the layout and the explicit cache-isolation rules. Use new cache/result roots for scientific configuration changes; old artifact-store directories are not migrated automatically.

## Interpretation and diagnostics

Model training does not fit an XAI surrogate. The current post-training commands are:

```bash
python -m waveform_analysis.cli explain-tshap \
  --results waveform_analysis/results/UC/filtered_2GHz_order5_benchmark \
  --models direct_linear_ridge shared_linear_ridge \
  --windows wide --mode energy --max-events 100

python -m waveform_analysis.cli explain-tshap-local \
  --results waveform_analysis/results/UC/filtered_2GHz_order5_benchmark \
  --models direct_linear_ridge shared_linear_ridge \
  --windows wide --mode energy
```

### Frozen imputer benchmark (recommended for XAI)

Fit, tune, independently evaluate, and save waveform imputers **before** running TSHAP.
The benchmark requires the already-prepared wide Development cache (no model retraining),
and uses disjoint Development fitting/tuning/evaluation subsets. It compares joint PPCA
(with candidate latent dimensions), conditional kNN (with one global k/weighting per
task), Development median, Control reference substitution, and local linear
interpolation for short missing windows. Selection minimizes pooled hidden-sample
RMSE in physical mV over all eligible temporal positions. The two complementary
coalitions can select different imputers. Independent evaluation scores are
reported but never used for winner selection. No Blind events are used.

Windows PowerShell (run from repository root):

```powershell
python -m waveform_analysis.cli benchmark-imputers `
  --results waveform_analysis/results/UC/filtered_2GHz_order5_benchmark `
  --cache-root UC_lowpass_rebuilt `
  --output-root xai/fitted_imputers `
  --modes energy `
  --window-samples 20 --stride-samples 20
```

The benchmark writes `xai/fitted_imputers/UC/energy/wide/w20_s20/` with
`selection.json`, `comparison.csv`, `fitted_imputers.npz`, and `.complete`.
Repeat with FBK results and its own prepared cache for FBK; the same output root
separates boards and modes automatically. A completed frozen selection is reused
rather than silently refitted. Change the window settings/output root to trigger
a distinct benchmark; `--overwrite` explicitly replaces the selected artifact.

Run TSHAP with **the saved winners**, without any imputer or model fitting:

```powershell
python -m waveform_analysis.cli explain-tshap `
  --results waveform_analysis/results/UC/filtered_2GHz_order5_benchmark `
  --models direct_linear_ridge `
  --windows wide --mode energy `
  --cache-root UC_lowpass_rebuilt `
  --imputer-root xai/fitted_imputers `
  --output-root xai/UC_lowpass_linear `
  --max-events 10 `
  --window-samples 20 --stride-samples 20
```

The separate `tshap_frozen/` output includes per-run plots, CSVs, score arrays,
and metadata identifying both selected imputers. Use `explain-tshap-local` with
the same `--imputer-root` for representative local explanations. The frozen
benchmark currently targets **wide** waveform input only; it fails explicitly
if the selected cache or analysis grid does not match. The previous reference
and conditional-kNN TSHAP commands remain available without `--imputer-root`.

Local-importance deletion and the separate temporal/progressive-window study
are retained. The paired-window scores correspond to a two-player Shapley game,
not exact Shapley values of individual time samples.

Source-domain diagnostics (FFT, acquisition phase and filtering sensitivity) remain available in `scripts/`.

## Development

```bash
python -m pytest waveform_analysis/tests
```

Production ROOT data are required to verify actual timing performance. The code and synthetic tests alone do not demonstrate numerical equivalence of training outcomes across library, OS or GPU environments.
