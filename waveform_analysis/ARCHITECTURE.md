# Waveform architecture migration

## 1. Migration plan and file map

Implemented against Git baseline `59a095a4cbc8b699c5bce16f656d4156483ce86b`.
The complete current-path/new-path table is in [MIGRATION.md](MIGRATION.md),
with its machine-readable equivalent in [module_migration.json](module_migration.json).

1. Freeze original signal calculations and reproduce baseline test outcomes.
2. Move implementations into responsibility packages while retaining old imports
   and serialized definition identities.
3. Extract pure numerical kernels and shared runtime operations without changing
   arithmetic, RNG consumption, fitted state or exported schemas.
4. Compare baseline and migrated implementations using identical dependencies,
   synthetic inputs and seeds, including separate-process model experiments.

| Original location | Implementation location | Responsibility |
| --- | --- | --- |
| `ml_pipeline/config.py`, `common.py`, `progress.py` | `core/` | Configuration, hashing/atomic I/O and progress policy. |
| `ml_pipeline/energy_io.py`, `data.py`, `dataset.py`, `prepared_data.py` | `data/` | Acquisition I/O and prepared waveforms. |
| `ml_pipeline/splits.py`, feature/shared caches, `storage.py`, `view.py` | `data/` | Population splits, caching and persistence. |
| Timing/baseline numerical kernels and selection pulse/calibration kernels | `signal/` | Pure NumPy operations independent of Torch and file I/O. |
| `utils/peak.py`, `utils/photopeak.py` | `signal/peak.py`, `signal/photopeak.py` | Original Gaussian photopeak/ToT fits. |
| `ml_pipeline/models/*linear_ridge*` | `models/linear/` | Linear estimators and feature definitions. |
| `ml_pipeline/models/*minirocket*` | `models/kernel/` | Frozen MiniRocket transforms and downstream estimators. |
| MLP/CNN definitions | `models/neural/` | Architectures and artifact definitions. |
| Batch, study, search, validation, training, preprocessing and XAI orchestration | `engine/` | Scientific execution order. |
| Plotting, reports, postprocessing and aggregate statistics | `reporting/` | Persisted-result rendering and aggregation. |

The legacy implementation files are module aliases, rather than copied code.
Old and new implementation imports resolve to the same module; the legacy
model-registry package re-exports the same callbacks and specifications.
Migrated classes keep their old `__module__` paths so existing pickle references
still resolve.

## 2. Interfaces and typed configuration

[core/config.py](core/config.py) remains the canonical batch parser. Its existing
JSON normalization, validation, exception messages, project-root resolution,
public dictionaries and configuration fingerprints are retained. Frozen typed
settings live in [core/settings.py](core/settings.py) and are re-exported there:

- `BaselineConfig`: required trigger-relative window and clipping margin.
- `TimingConfig`: LED offsets from preprocessing JSON and explicit CFD fractions.
- `MLPTrainingConfig`: original RMSE/Nesterov/internal-holdout defaults: 300
  epochs, patience 10, min_delta 0.01 ps, holdout fraction 0.20, gradient clipping
  10.0 and momentum 0.9.
- `OnishiTrainingConfig`: separate full-split/MSE defaults: 600 epochs, decay
  milestones (180, 360) and factor 0.1.
- `RuntimeConfig` and `XAIConfig`: existing prediction and occlusion defaults.

```python
from waveform_analysis.core.config import TimingConfig, MLPTrainingConfig

timing = TimingConfig.from_preprocessing(preprocessing_json, fractions=(0.2, 0.5))
training = MLPTrainingConfig.from_mapping(model_space_json.get('training', {}))
```

Existing JSON keys override settings through the existing CLI `--config` option.
Typed views do not replace dictionaries in persisted artifacts or hashes. No
YAML parser or new CLI override semantics were introduced. Sampling intervals
remain per-event acquisition metadata. Required physical windows are explicit;
the documented example baseline window is (-2, -1) ns. Detector count, native
sample indexing and unit conversions retain their scientific meaning.

[models/base.py](models/base.py) defines `BaseWaveformModel[ArtifactT]`, an adapter
over the existing `ModelSpec`, and `BaseTorchModel(nn.Module, ABC)`. The Torch base
allocates no layers, buffers, parameters or random draws. Physical waveform
networks inherit it without changing forward passes or state_dict keys.
`RegisteredWaveformModel` delegates fitting, prediction and saving to the same
callbacks and returns existing artifacts, including sklearn/sktime fitted state.

Dense scorer components are in `models/neural/dense.py`. The existing common
RMSE/internal-holdout training procedure is in `engine/neural_training.py`;
Onishi's distinct MSE/full-split procedure is in `engine/onishi_training.py`.
Shared tensor inference, checkpoint serialization and input-gradient attribution
are in `models/torch_runtime.py`. Grouped temporal occlusion remains in
`engine/xai.py`, retaining its existing formulation-specific interventions.

`WaveformDataset` retains the original contiguous float32 input conversion,
float32 targets, TensorDataset indexing and seeded DataLoader behavior.
CLI logging uses the same logger name, level and message format through
`core/logging.py`.

Precise types were added to settings, signal kernels, dataset boundaries, model
interfaces and shared trainers. Free-form legacy engine/report payloads retain
their dictionaries; this migration does not claim strict static typing of every
legacy payload.

## 3. Pure signal-processing APIs

[signal/timing.py](signal/timing.py) accepts waveforms `[event, 2, sample]`,
starts/intervals/rising bounds `[event, 2]`, selected indices `[event]`, and
threshold/fraction grids `[parameter]`. Output is `[selected_event, 2, parameter]`
in ps. [signal/baseline.py](signal/baseline.py) accepts `[sample]` arrays and
explicit sampling, window and rail parameters.

```python
from waveform_analysis.signal.timing import led_times_ps, cfd_times_ps

led = led_times_ps(waves, starts, intervals, rising_start, rising_stop,
                   indices, thresholds_mV, baseline_window_ns=(-2.0, -1.0),
                   materialized_before_ns=3.0)
cfd = cfd_times_ps(waves, starts, intervals, rising_start, rising_stop,
                   indices, fractions)
```

The sample masks are vectorized. Per-waveform reductions and scalar interpolation
retain their original order; event/detector/threshold loops preserve exact
floating-point behavior. The implementation retains:

- The last strict below-to-at-least crossing, with equal-to-above fallback only
  when no strict crossing exists.
- `(start + (lower + fraction) * interval) * 1e12`, float64 conversion, original
  rising bounds and invalid-result NaNs.
- Baseline floor/ceil window bounds, inclusive upper sample, finite-sample
  filtering and the original mean/RMS reductions.
- CFD's `fraction * nanmax(signal[a:b+1])`, with no new baseline subtraction,
  smoothing or peak definition.
- Original anchor rounding and tie-breaking in `data/timing_adapter.py`.
- The direct F1 histogram FWHM definition in `utils_fit`, with no Gaussian-sigma
  replacement or single-detector rescaling. Photopeak/ToT Gaussian fits retain
  their original `curve_fit` procedure.
- The distinct finite-residual reporting RMSE and training RMSE behavior.

ADC conversion, pulse finding, robust baseline scale and frozen amplitude
selection are in `signal/pulses.py`. Baseline rail clearance equal to the
clipping margin still counts as clipped.

## 4. Exact non-regression strategy

`tests/fixtures/legacy_signal_reference.py` freezes the original numerical
kernels independently of migrated imports. Tests compare shapes, dtypes,
NaN masks and bytes of non-NaN values, including signed zero. Synthetic signals
cover float32/float64, repeated/reordered indices, invalid windows/bounds,
missing/multiple crossings, equality fallback, calibration and clipping.

```python
import numpy as np
from waveform_analysis.signal.timing import crossing_ps
from waveform_analysis.tests.fixtures import legacy_signal_reference as original

def test_led_crossing_matches_original_exactly():
    waveform = np.array([0.0, 2.0, 0.0, 2.0], dtype=np.float64)
    args = (waveform, -3.713e-9, 2.5e-12, 0, 3, 1.0)
    actual = crossing_ps(*args)
    expected = original._crossing_ps(*args)
    assert np.float64(actual).tobytes() == np.float64(expected).tobytes()
```

`test_architecture_regression.py` extracts the untouched baseline using local
`git archive`, then fits all 10 registered models in separate reference/current
processes. Real MiniRocket transforms are exercised. Comparisons cover trained
weights/coefficients, predictions, feature arrays, gradient explanations,
metadata, logs, filenames, checkpoint payloads, legacy pickle loading, common
CV folds and paired blind bootstrap arrays. The baseline Git object must remain
available; these tests use no network. Following the requested RidgeCV change,
the two linear Ridge variants are compared with sklearn in `test_ridge_cv.py`
instead of requiring the old outer-search Ridge outputs. The other eight model
families and scientific protocol retain exact baseline comparisons.

`test_package_compatibility.py` checks module identity, serialized class names,
typed defaults, tensor conversion, the model adapter, Torch-free signal imports,
and exact normalized configuration/fingerprint equality.

From the repository root, using the installed virtual environment:

```bash
source /workspace/venvs/pet-timing/bin/activate
export MPLBACKEND=Agg MPLCONFIGDIR=/workspace/.cache/matplotlib
export XDG_CACHE_HOME=/workspace/.cache NUMBA_CACHE_DIR=/workspace/.cache/numba
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
python -m pytest waveform_analysis/tests/test_signal_regression.py waveform_analysis/tests/test_package_compatibility.py waveform_analysis/tests/test_architecture_regression.py -q
python -m pytest waveform_analysis/tests -q --continue-on-collection-errors
```

Original architectural migration verification: 59 added checks passed. The full suite ran **101 passing tests, 2 failing
tests and 1 collection error**. The untouched baseline ran 42 passing tests with
the same two failures and error:

- `test_models.py` imports nonexistent `models.mlp`.
- The MiniRocket test's scaler mock rejects `copy=False`, which the existing
  implementation uses.
- The registry test requires `preserve_temporal_grid=False` for all models,
  contradicting the existing CNN setting.

Those existing tests were not weakened, skipped or rewritten. Deliberately
all-NaN synthetic signals emit the same NumPy warning in both CFD implementations.

Validation uses synthetic inputs and CPU execution with identical installed
dependencies. GPU behavior and end-to-end production acquisition datasets were
not verified. Exact equality here establishes the tested cases rather than a
proof over all possible inputs, hardware or library versions.

The subsequent linear RidgeCV selection change is intentional and described in
[README.md](README.md#linear-ridgecv). Its tests also cover full development
fitting, blind isolation, batch reports, cache invalidation and saved-fit resume.

RidgeCV verification: 16 dedicated checks pass, including Ridge-only and mixed
Ridge/MLP batches with real fitting, persistence, blind evaluation and reporting.
The outer-search configuration tests now validate RidgeCV spaces separately.
