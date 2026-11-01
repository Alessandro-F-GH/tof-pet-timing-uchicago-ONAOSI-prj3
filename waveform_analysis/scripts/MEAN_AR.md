# Standalone mean-conditioned AR prefix benchmark

This experiment is **not part of TSHAP** and does not fit timing-correction
models. It uses existing named prepared caches and compares deterministic
future-waveform reconstruction against a mean-waveform baseline.

## Run from the repository root

```bash
python -m waveform_analysis.scripts.evaluate_mean_ar \
  --uc-cache-root "$HOME/UC_cache" \
  --fbk-cache-root "$HOME/FBK_cache" \
  --output-root "$HOME/mean_ar_results" \
  --ar-orders 8 16 32 64 128 \
  --n-cutoffs 10 \
  --max-train-events 512 \
  --max-eval-events 128
```

Zero for either event limit selects all available events. Both cohorts are
selected reproducibly, and they come from distinct **development** and
**blind** prepared dataset roles. Named-cache ambiguities must be resolved
using `--development-name` and `--blind-name`.

For each board (UC/FBK), mode (energy/timing), and detector (1/2), the
script independently fits:

- **Mean replacement**: suffix equals the development-set per-sample mean.
- **Mean AR**: Ridge autoregression of `x[t] - mean[t]`.
- **Standardized AR**: Ridge autoregression of
  `(x[t] - mean[t]) / std[t]`; standard deviations are floored.

Each AR order has its own regression coefficients. Ridge uses a regularization
parameter scaled to mean input energy (`--ridge-alpha`, default 0.001).
The same training target positions are used for all orders. The script does
not tune alpha against blind data.

Cutoffs are derived from the **development population mean**, densely
distributed near its largest positive slope, plus a small number of early and
tail cutoffs. They are the same for all AR orders. Ridge training pools lagged samples from each contiguous retained region,
without creating training examples that cross masked boundaries. Cutoffs are
concentrated around the strongest rising edge, with some cutoffs in valid
post-mask intervals when those are long enough. For each cutoff, AR predicts
only until the next masked boundary; from that boundary onward (including any
later valid regions), the mean waveform is substituted. When a cutoff lies
after a mask, the AR uses only the contiguous observed history in that region.
Metrics cover the **entire valid suffix** after the cutoff, excluding
artificially masked positions; `ar_predicted_samples` separately records the
part actually handled by AR.

The original prefix is never replaced. Missing samples are predicted
recursively without reading the test suffix. Predicted voltage is clipped to
the frozen normalized `[0,1]` input range. Evaluation scores only missing
samples and reports RMSE/MAE in normalized units and mV (using the frozen
per-detector input transform), along with percentage MSE change relative
to the mean baseline.

## Output

Each `OUTPUT/BOARD/MODE/detector_N/` contains:

- `metrics.csv`: held-out errors for every cutoff, order and method.
- `model.json`: population provenance, selected event indices,
  per-position mean/std, fit settings, AR coefficients and characteristic-root
  stability.
- `rmse_vs_cutoff.png`: error versus cutoff for mean baseline and the
  largest tested AR order.
- `waveform_example.png`: one representative held-out waveform with mean
  and standardized-AR suffix reconstruction.

**Limits:** deterministic forecasts approach the learned mean when residual
memory vanishes; low reconstruction MSE is not proof of realistic stochastic
imputation. Stationary AR coefficients may struggle with event-specific time
shifts, multimodal futures and very long generation horizons. This experiment
does not validate a conditional-sampling distribution or causal SHAP values.

Synthetic tests:

```bash
python -m pytest -q waveform_analysis/tests/test_mean_ar.py
```
