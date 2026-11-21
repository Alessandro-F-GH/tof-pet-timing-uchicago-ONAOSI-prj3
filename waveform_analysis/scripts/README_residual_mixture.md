# Definitive Energy–Timing correlation analysis

Use `waveform_analysis.scripts.analyze_energy_timing_correlation` on canonical saved blind predictions of matching Energy and Timing models (UC and FBK). No waveform loading, inference, or refitting.

```powershell
$base = "waveform_analysis/results"
python -u -m waveform_analysis.scripts.analyze_energy_timing_correlation `
  --uc-energy-run "$base/UC/filtered_2GHz_order5_benchmark/energy/wide/shared_linear_ridge" `
  --uc-timing-run "$base/UC/filtered_2GHz_order5_benchmark/timing/wide/shared_linear_ridge" `
  --fbk-energy-run "$base/FBK/filtered_2GHz_order5_benchmark/energy/wide/shared_linear_ridge" `
  --fbk-timing-run "$base/FBK/filtered_2GHz_order5_benchmark/timing/wide/shared_linear_ridge" `
  --output-dir xai/shared_linear_correlations/definitive `
  --iqr-multiplier 3 --bootstrap 300 --acquisition-bins 20
```

Each board/state estimates Q1/Q3 separately for Energy and Timing. An event is an outlier if E **or** T is beyond Q1 − 3 IQR or Q3 + 3 IQR. The conservative multiplier 3 corresponds to ~4.72 standard deviations for a Gaussian marginal, but **Gaussianity is not assumed**.

For LED and ML separately, report the outlier rate, Pearson correlation and covariance separately for core and tails, their confidence intervals, plus the fraction of full-sample centered sum of squares contributed by outliers (descriptive variance share). Also report energy-only, timing-only and simultaneous outlier counts. For ML model outputs, use only the intersection of LED and ML cores (same event population in both channels). No mixed global correlation/covariance is reported.

Results:
- `UC/summary.json`, `FBK/summary.json`: state-specific fences, group statistics, full-population outlier frequencies, contributions to variance, outputs/core correlations, and acquisition-order bins.
- `UC/scatter_led_ml.png/pdf`, `FBK/scatter_led_ml.png/pdf`: scatterplots highlighting ALL outliers; gray inliers may be subsampled for readability.
- `UC/outlier_acquisition_order.png/pdf` and corresponding FBK plots: binned counts in ascending event ID.
- `board_differences.csv`, `summary.json`: UC minus FBK differences with bootstrap intervals.

**Interpretation:** Bootstrap intervals condition on the original IQR cut and population memberships; they do not re-estimate fences. The event-id order is only an acquisition-order proxy until its relationship to the source acquisition is verified. The binned chi-square test screens for nonuniform outlier rate, assumes event independence, and cannot establish general stationarity. Correlations cannot identify a physical common source on their own.
