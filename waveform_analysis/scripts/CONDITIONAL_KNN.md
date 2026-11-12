# Conditional KNN benchmark (saved predictions only)

The experiment estimates the conditional model output using **existing** prepared
development waveform pairs and **existing** saved development model predictions.
It does not train, load, or evaluate any ML model and does not use model feature
transforms. It does not impute or generate waveforms.

## Example

```bash
python -m waveform_analysis.scripts.evaluate_conditional_knn \
  --model-dir "$PWD/waveform_analysis/results/UC/<batch>/<mode>/<window>/<model>" \
  --cache-root "$HOME/UC_cache" \
  --output-root "$HOME/conditional_knn/UC/model" \
  --window-samples 20 --stride-samples 20 \
  --k 5 10 20 50 100 200 500 \
  --weighted \
  --tuning-queries 500 --holdout-queries 500 \
  --reference-batch 2048 --log-every 10
```

Replace the example model path with an actual complete run directory containing
`metadata/config.json`. The dataset is its **development** population, not
the blind population. The output directory must be separate from the source run.

Both a saved prepared development artifact and a development prediction artifact
are required. The script checks array length, unique event IDs and their
alignment, final-fit fingerprint, dataset role and analysis-protocol identity.
Missing or incompatible predictions **fail**, with no fallback to inference.

The development predictions typically live at the run-registered
`development_predictions/predictions.npz` artifact or legacy
`artifacts/development_pred.npz`. If the saved predictions have not previously
been materialized, produce them in a **separate**, explicitly authorized
pipeline job; this benchmark will never calculate them.

## Computation

For each query, the script processes **all** development events as references,
excluding only the query itself (leave-one-out). It computes sample-level
squared Euclidean differences over both detectors, integrates time with a prefix
sum, and derives both the window distance and its complementary distance
using the same distance pass. Windows follow the retained **interior** TSHAP
window geometry and frozen timing mask (if applicable).

Distances are normalized by the count of observed coordinates. A single
exact top-`k_max` search is reused for all smaller k values. Optional
inverse-distance weights reuse the same nearest neighbors; zero-distance
neighbors are given all the weight. The output is a mean of the **stored**
`prediction_ps` values for these neighbors.

Queries are divided, using one seed, into disjoint tuning and holdout query
groups, while **all** development events remain eligible references except
the same event (LOO). On tuning queries, both a single k shared by the two
coalitions and separate k for window/complement are selected by mean squared
prediction error. Holdout metrics are produced only after selection.

The RMSE is relative to a frozen full-waveform prediction, not a physical
timing target. This is a diagnostic of conditional-mean prediction quality:
even an optimal conditional expectation may differ from each full-waveform
prediction because the masked waveform contains less information.

## Outputs

- `evaluation.log`: input artifact identities, data shapes, algorithm,
  progress, query throughput, ETA, selected hyperparameters and diagnostics.
- `metrics.csv`: RMSE and bias for each coalition, k and weighting strategy,
  on tuning and holdout query populations.
- `diagnostics.npz`: per-window RMSE, nearest-neighbor and kth-neighbor
  distance summaries, queried row indices and window starts.
- `summary.json`: input provenance, settings, selected configurations,
  holdout comparison and explicit model-inference-disabled fields.

To run tests:

```bash
python -m pytest waveform_analysis/tests/test_conditional_knn.py -q
```

## Scope and limitations

- This is a standalone **validation experiment**, not yet integrated into
  TSHAP or the deletion benchmark.
- It requires previously saved development predictions for the *same final fit*;
  it never reuses predictions from an incompatible run.
- It does not cap the number of reference events.
- The number of **query** events is configurable for computational efficiency.
- All full-waveform values and KNN estimates are deterministic after the
  query selection seed is fixed.
- Near-ties in exact nearest-neighbor distances can be sensitive to floating
  point rounding; double precision and stable sorting of the retrieved top-k
  neighbors mitigate this.
