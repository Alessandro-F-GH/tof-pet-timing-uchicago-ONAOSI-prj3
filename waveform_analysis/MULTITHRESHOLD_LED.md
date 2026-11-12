# Multi-threshold leading-edge reference

The standalone CFD experiment was removed. The library's pre-existing
`waveform_analysis.signal.timing.cfd_times_ps` is untouched because other code
may still use it.

The multi-threshold benchmark uses **existing ML prepared waveforms** with a
frozen LED preprocessing calibration. No ROOT reads, no timestamp regeneration,
no ML inference and no target-dependent fitting of per-event pickoff.

## Method and protocol

Each prepared waveform retains an approximate sample anchor of the frozen
interpolated LED crossing. On that same grid, search a short rising-edge window
for crossings at absolute, baseline-relative mV thresholds. Convert mV to the
prepared units using each detector's stored affine vertical scale.

For each event/detector, linearly interpolate the upward crossing time at
every selected threshold and fit `t(V) = t0 + slope * V` by unweighted OLS.
The multi-threshold timestamp is the intercept at `V=0`. Compute the
LED-relative correction using the native `detector1-detector2` convention:

`prediction_raw = (LED1 - t0_1) - (LED2 - t0_2)`

The final prediction adds a single constant development-calibrated offset.
Thus the timestamp extraction itself is target-independent, whereas fraction
selection/offset are tuned using development timing labels.

Threshold groups must contain >=3 strictly increasing positive absolute
thresholds, with **no threshold over 50 mV**. The default scan covers
10–50 mV with short/medium/five-threshold fits. The CTR/RMSE selection
uses the same finite-crossing development subset for every candidate and the
repository's established FWHM estimator. Blind is opened after selection.
A fixed development offset is applied unchanged on Blind; 100 bootstrap
resamplings use repository functions and do not refit.

A numerical crossing failure remains NaN with `valid=False`. Do not
silently substitute LED or drop different sets of events across compared
candidates. Saved timing diagnostics include intercept, local slope
and fit RMSE for each detector, but those are **not** used to choose events.

## PowerShell example

```powershell
git pull origin codex/artifact-store-refactor

python -m waveform_analysis.scripts.evaluate_multithreshold_led `
  --model-dir "waveform_analysis/results/UC/FINAL/energy/wide/direct_multirocket_rdst" `
  --compare-model-dir "waveform_analysis/results/UC/FINAL/energy/wide/direct_multirocket_rdst" `
  --cache-root "$HOME/UC_cache" `
  --output-root "$HOME/xai/UC/energy/wide/multithreshold" `
  --metric ctr --bootstrap 100
```

Optional threshold scan override (groups are comma-separated):

```powershell
  --threshold-sets 10,15,20 15,25,35 20,35,50 10,20,30,40,50
```

Output artifacts:
- `summary.json`: selected thresholds, scan, metrics, saved-model paired
  comparisons on development and Blind (Pearson r, bias, RMSE, MAE, paired CTR);
- `threshold_scan.csv`: full threshold search;
- `development_predictions.npz`, `blind_predictions.npz`: matching ML-style
  `event_id`, `prediction_ps`, `corrected_ps`, `led_residual_ps`, `valid`
  plus per-detector crossing-fit diagnostics;
- `blind_bootstrap.npz`: paired event bootstrap draws for MT and LED.

The ML correlation comparison is a **post-selection** diagnostic only:
existing ML prediction arrays are loaded using each run's frozen provenance;
matching uses event IDs plus calibrated target agreement, and never feeds
the ML output into MT selection. Correlation of correction outputs does not
establish a shared causal correction mechanism.

## Conditional heatmaps of ML minus multi-threshold

```powershell
python -m waveform_analysis.scripts.plot_conditional_target_heatmaps `
  --model-dir "waveform_analysis/results/UC/FINAL/energy/wide/direct_multirocket_rdst" `
  --comparison-predictions-npz "$HOME/xai/UC/energy/wide/multithreshold/development_predictions.npz" `
  --cache-root "$HOME/UC_cache" `
  --output-root "$HOME/xai/UC/energy/wide/ml_minus_multithreshold" `
  --representations original amplitude_normalized `
  --include-original-derivative
```

Uses finite event-ID intersections and checks calibration metadata; no
re-prediction. The current heatmap routine visualizes conditional means;
the requested conditional variance extension is separate.

## Caveats

The reference is a linear extrapolation within a user-configured window; it is
not guaranteed to remove all amplitude walk or to isolate pure time walk.
The prepared input may be physically clipped or cropped. The initial
implementation supports `energy_to_energy`; timing channels may saturate.
The selected threshold set must be frozen before inspecting Blind metrics.

## Compare ALL saved models without rerunning the experiment

If the multi-threshold experiment was already completed, reuse its existing
`development_predictions.npz` and `blind_predictions.npz` and produce
correlations for all saved model runs discovered recursively under a result
folder. This **does not repeat threshold search, timing extraction, Blind
bootstrap, or model inference**.

```powershell
python -m waveform_analysis.scripts.evaluate_multithreshold_led `
  --model-dir "waveform_analysis/results/UC/FINAL/energy/wide/direct_multirocket_rdst" `
  --compare-model-root "waveform_analysis/results/UC/FINAL/energy/wide" `
  --cache-root "$HOME/UC_cache" `
  --compare-only-root "$HOME/xai/UC/energy/wide/multithreshold_v2"
```

The run folder given in `--compare-only-root` must be the **actual existing
completed output**. The original reference run and prepared data are verified.
Every matching saved model is checked separately for mode, channels, fixed
control calibration, event ID alignment and equal calibrated targets.

Outputs are updated `summary.json` with `comparison_development` and
`comparison_blind`, and a new `model_correlations.csv` for the Blind
population. Incompatible or unavailable discovered runs appear in
`comparison_skipped_*` with reasons. An explicitly passed
`--compare-model-dir` fails loudly instead of being silently skipped.

The individual `--compare-model-dir` argument still accepts one or many
explicit run paths. For a fresh MT experiment, `--compare-model-root` may be
used without `--compare-only-root`; for existing results the compare-only
mode is preferable.

## Wide heatmaps with Onishi prediction outputs

For the descriptive comparison, the *waveforms displayed* must always be
loaded from a Wide model run. The prediction source may instead be Onishi.
Match populations using shared development event IDs and verify the calibrated
LED target vector on the intersection. Window-specific protocol identities
must **not** be required to match.

```powershell
$wide = "waveform_analysis/results/UC/FINAL/energy/wide/direct_multirocket_rdst"
$onishi = "waveform_analysis/results/UC/FINAL/energy/onishi/direct_multirocket"
$mt = "$HOME/xai/UC/energy/wide/multithreshold_all_models/development_predictions.npz"

# MT alone, displayed on the Wide waveform grid.
python -m waveform_analysis.scripts.plot_conditional_target_heatmaps `
  --model-dir $wide `
  --comparison-predictions-npz $mt `
  --comparison-target comparison `
  --cache-root "$HOME/UC_cache" `
  --output-root "$HOME/xai/UC/heatmaps/mt_wide" `
  --representations original amplitude_normalized

# Onishi ML minus MT, also displayed on the Wide waveform grid.
python -m waveform_analysis.scripts.plot_conditional_target_heatmaps `
  --model-dir $wide `
  --prediction-model-dir $onishi `
  --comparison-predictions-npz $mt `
  --comparison-target difference `
  --cache-root "$HOME/UC_cache" `
  --output-root "$HOME/xai/UC/heatmaps/onishi_minus_mt_wide" `
  --representations original amplitude_normalized
```

Change `$onishi` to an **existing** Onishi saved-model run; it is an
illustrative path. The two plots use the same Wide waveform time coordinates,
but their event populations may differ: the comparison intersects MT,
Onishi and Wide event IDs and discards invalid MT crossings. For directly
comparable heatmaps of MT and Onishi−MT, use the same three-way event subset
(e.g., save an explicit shared event-ID list in a future single-run plotting
mode) rather than assuming the separate commands are perfectly paired.
