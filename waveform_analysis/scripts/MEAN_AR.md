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

Optional **prefix k-NN forecaster** (no learned model or additional dependencies):

```bash
python -m waveform_analysis.scripts.evaluate_mean_ar \
  --uc-cache-root "$HOME/UC_cache" \
  --fbk-cache-root "$HOME/FBK_cache" \
  --output-root "$HOME/mean_ar_knn_results" \
  --ar-orders 8 16 32 64 128 \
  --n-cutoffs 10 --max-train-events 1024 --max-eval-events 256 \
  --with-knn --knn-k 1 5 20
```

At each cutoff, find the nearest **development-set** events by squared
Euclidean distance over the same observed, contiguous prefix. For `k=1`,
copy the nearest event's *real future*; for `k>1`, take the uniform mean
of the selected events' futures. Only the real prefix of a blind event
is used to select neighbors: its future never enters the distance.
Predictions terminate at the next masked boundary, with the same
development-population mean fallback as AR and GRU. The method is evaluated
on the exact same blind events and valid-suffix positions. `--with-knn`
works independently of `--with-gru`. k-NN estimates are deterministic,
and the comparison plot shows the largest requested `k`.

Optional **cutoff-specific GRU encoder + direct forecasting decoder**:

```bash
python -m waveform_analysis.scripts.evaluate_mean_ar \
  --uc-cache-root "$HOME/UC_cache" \
  --fbk-cache-root "$HOME/FBK_cache" \
  --output-root "$HOME/mean_ar_gru_results" \
  --ar-orders 8 16 32 64 128 \
  --n-cutoffs 10 --max-train-events 1024 --max-eval-events 256 \
  --with-gru --gru-hidden 32 \
  --gru-epochs 8 --gru-steps-per-epoch 32
```

With `--with-gru`, one independent GRU is fitted **per cutoff, board, mode,
and detector** on development waveforms. The GRU encoder reads the entire
real contiguous prefix (not only the last value). Its decoder combines the
encoded prefix with target-time coordinates and the development mean,
then regresses the normalized waveform voltage **directly** (not residuals).
No autoregressive rollout or future waveform values are used during evaluation.
Training evaluates MSE at **every future position in the current valid interval**
at each optimization step, not a subset of future positions. Following a masked boundary, predictions revert
to the mean as in the AR baseline. This is a deterministic conditional-mean
forecaster, not a probabilistic imputer.

Each cutoff has a saved `gru_cutoff_XXXXX.pt` checkpoint. On rerun, matching
settings and development indices allow reuse, avoiding unnecessary retraining.
The checkpoints from the previous one-step GRU implementation cannot be
reused with this architecture. Training cost scales with number of cutoffs.

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

- `metrics.csv`: held-out errors for every cutoff, order and method (including kNN/GRU when enabled).\n- `gru_cutoff_XXXXX.pt`: per-cutoff GRU encoder-decoder checkpoints (when enabled).
- `model.json`: population provenance, selected event indices,
  per-position mean/std, fit settings, AR coefficients and characteristic-root
  stability.
- `rmse_vs_cutoff.png`: error versus cutoff for mean baseline and the
  largest tested AR order.
- `waveform_examples/waveform_cutoff_XXXXX.png`: one held-out waveform reconstruction
  per tested cutoff, always using the same blind event; plots show the original,\n  mean replacement, standardized AR (largest tested order), optional GRU and kNN, the cutoff, and
  the next mask boundary when applicable.

**Limits:** deterministic forecasts approach the learned mean when residual
memory vanishes; low reconstruction MSE is not proof of realistic stochastic
imputation. Stationary AR coefficients may struggle with event-specific time
shifts, multimodal futures and very long generation horizons. This experiment
does not validate a conditional-sampling distribution or causal SHAP values.

Synthetic tests:

```bash
python -m pytest -q waveform_analysis/tests/test_mean_ar.py
```
