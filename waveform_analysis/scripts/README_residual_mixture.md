# UC/FBK paired residual analysis: empirical IQR outliers

The standalone `compare_energy_timing_residuals.py` compares energy/timing coincidence residuals for UC and FBK, before (LED) and after ML correction. It uses **no Gaussian fit or Gaussian assumption** for either the central population or the outliers.

## Expected input layout

```text
predictions/
  UC/
    energy_predictions.npz
    timing_predictions.npz
  FBK/
    energy_predictions.npz
    timing_predictions.npz
```

The shorter names `energy.npz` / `timing.npz` are also supported, but each folder must have exactly one matching file per channel. Each NPZ contains `event_id`, `led_residual_ps`, and `corrected_ps`. Energy and timing predictions are matched **within their board by event ID**; never align UC events with FBK.

Dependencies: `numpy` and `matplotlib`.

```powershell
python -m pip install numpy matplotlib
python compare_energy_timing_residuals.py --predictions-root "$HOME\predictions" --output "$HOME\iqr_results"
```

For module execution from the repository root:

```powershell
python -m waveform_analysis.scripts.compare_energy_timing_residuals --predictions-root "$HOME\predictions" --output "$HOME\iqr_results"
```

## Nonparametric selection

For each board and each correction state (UC/LED, UC/ML, FBK/LED, FBK/ML), compute the 25th and 75th percentiles independently for energy and timing. Extreme Tukey fences are `[Q1 − k*IQR, Q3 + k*IQR]`, with `k=3` by default. An event is marked as an outlier if **either** its energy or timing residual falls outside its respective interval. This is an axis-aligned rule, and does not require a distributional model.

Use `--iqr-multiplier 3` to adjust the threshold. It is stricter (fewer flagged events) than the conventional exploratory `1.5*IQR` rule. Samples are not assumed normal and no second component is fitted to the outliers.

The inlier and outlier correlations/covariances are **empirical** on the selected subsets. The cut itself affects correlations; do not interpret them as fitted parameters of physically separated populations. Membership is independently determined per board and correction state, so the LED and ML flagged events need not be identical.

## Plots

- `UC/led_vs_ml_distributions.png/.pdf` and `FBK/led_vs_ml_distributions.png/.pdf`: LED and ML raw-count histograms on a **linear y-axis** and shared per-channel x-range spanning the respective IQR fences. Energy uses 10 ps bins; timing 5 ps bins.
- `UC/led_vs_ml_iqr_scatter.png/.pdf` and `FBK/led_vs_ml_iqr_scatter.png/.pdf`: **gray inliers, orange IQR outliers**. Same x/y limits for LED and ML within each board.
- `uc_vs_fbk_distributions.png/.pdf`: UC and FBK count histograms on shared axes per state/channel. The boards can have different event counts; compare shapes with that caveat.
- `uc_vs_fbk_iqr_scatter.png/.pdf`: four-panel UC/FBK and LED/ML scatterplot with **the same x and y limits** for all four panels, gray inliers and orange outliers.

All plots show classifications without imposing a Gaussian core or tail shape. Outliers are all drawn; the background inliers may be downsampled to at most 20,000 per panel to keep output small.

## Numerical output

- `UC/iqr_summary.json` and `FBK/iqr_summary.json`: quartiles, IQR fence boundaries, observed outlier counts/fractions, and empirical correlations, covariances, and widths for inliers, outliers and all events.
- `uc_fbk_iqr_comparison.json`: UC−FBK differences for those statistics, including conditional percentile bootstrap 95% intervals for **correlation differences** (default `--bootstrap-resamples 300`).
- Verbose terminal report detailing the method, fence values, empirical statistics, and board comparisons.

The bootstrap draws resample fixed IQR-selected populations without recalculating fences, so confidence intervals omit threshold-estimation uncertainty. A confidence interval that includes zero does not demonstrate UC/FBK equivalence; formal equivalence requires a scientifically chosen tolerance.
