# Conditional KNN TSHAP

Run on saved frozen models using the same preprocessed paired waveforms, model input transforms and sample mask as the original pipeline. No retraining or model modifications.

The value function is deliberately **hybrid**, not exact conditional-expectation Shapley:

```text
v(empty) = mean(f(reference training/development waveforms))
v(W) = f(KNN-reconstructed waveform with W observed)
v(not W) = f(KNN-reconstructed waveform with complement observed)
phi_W = 0.5 * (f(x) - v(not W) + v(W) - v(empty))
```

The two detector channels share each temporal window. For every window and its complement, choose neighbor count and weighting (uniform or inverse Euclidean distance) independently on a disjoint Development validation subset, minimizing ordinary mean squared reconstruction error on the missing model-visible samples. Validation never includes Blind events. The reference cohort consists of remaining Development waveforms. The input scale is the **frozen model preprocessing normalization**, without another standardization.

For each query-reference pair, the KNN engine computes temporal contributions to squared Euclidean distance once and uses prefix sums for every window; the complement distance is full minus window distance. One sorted top-k list serves all candidate k and weightings. Reference processing is blocked to avoid materializing an enormous event-reference-time tensor.

A separate diagnostic computes the weighted mean of **model predictions on real KNN neighbors** and compares it against the **model prediction on the KNN-mean reconstructed waveform** for both observed-region types. This diagnostic never enters Shapley computation. Model reference predictions are computed once per model.

## CLI

Global:

```bash
python -m waveform_analysis.cli explain-tshap \
  --model-dir /path/to/model-run \
  --method conditional-knn \
  --cache-root "$HOME/UC_cache" \
  --output-root "$HOME/xai/UC" \
  --max-events 256 \
  --window-samples 20 --stride-samples 20 \
  --knn-k 1 3 5 10 20 50 100 \
  --knn-tuning-events 512
```

Local:

```bash
python -m waveform_analysis.cli explain-tshap-local \
  --model-dir /path/to/model-run \
  --method conditional-knn \
  --cache-root "$HOME/UC_cache" \
  --output-root "$HOME/xai/UC" \
  --events-per-group 2 --window-samples 20 --stride-samples 20
```

`--results`, `--models`, `--windows` and `--mode` can also select multiple runs. `--knn-reference-events` optionally caps the reference population, otherwise all Development events outside the validation subset are used. `--knn-reference-block` bounds reference batching. `--seed` makes the split and Blind selection deterministic. `--overwrite` replaces matching explanations. Default `--method reference` retains the existing non-conditional baseline method.

## Outputs

Separate `tshap_conditional_knn/` or `tshap_conditional_knn_local/` directories under the corresponding run contain:

- `summary.json`: model/method identities, tuning artifact reference, selected k/weight, physical and normalized reconstruction metrics, output-gap statistics.
- `knn_tuning_selected.csv`: per-window per-coalition validation MSE, RMSE, MAE and mV metrics.
- `knn_diagnostics_per_window.csv`: for each window and complement, Blind reconstruction error and `f(reconstructed) - weighted_mean(f(neighbors))` statistics.
- `attributions.npz` or `local_attributions.npz`: raw paired scores, redistributed maps, source event IDs, full predictions, imputed predictions, neighbor-prediction means, per-event gaps and errors.
- Global: `global_importance.csv`, `signed_window_importance.csv`, `global_importance.png`, `global_importance.pdf` and `knn_prediction_gap.png`.
- Local: `events.csv` and a waveform/attribution plot for each automatically selected good/bad/typical event.

Logs contain tuning progress and each selected k/weight with validation errors; explanation progress with per-event output differences; per-window diagnostic metrics; and final aggregated reconstruction and output-gap statistics.

The KNN tuning itself is immutable in the Artifact Store and shared across compatible saved models; the final model and its feature transform are always loaded from the existing frozen run. No stored prediction is modified.

## Caveats

- This is conditional **mean imputation**. `f(E[X|S])` need not equal `E[f(X)|S]`; only the former enters Shapley. The latter is estimated through neighbor predictions strictly as a diagnostic.
- Inverse-distance weighting uses distance only on observed samples of both detectors; identical neighbors receive equal weights.
- The normalised Euclidean metric does not necessarily equal physical mV Euclidean distance if detector amplitude scales differ. mV-domain error is reported separately.
- Blended waveforms may still be out-of-distribution for the frozen model; inspect the output-gap diagnostics before interpreting TSHAP peaks.
- Computational cost remains substantial for wide waveforms and large reference cohorts despite reuse of distance contributions.
