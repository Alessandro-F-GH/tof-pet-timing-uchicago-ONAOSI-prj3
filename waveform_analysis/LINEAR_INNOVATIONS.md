# Linear innovations benchmark

The experimental models `direct_linear_innovations` and
`shared_linear_innovations` use the standard feature-transform pipeline.
For each acquisition mode/window, a fixed transform is fitted **only** to
control waveforms, pooled across detectors, independently of timing targets.
It is then applied separately to both detectors.

The transform subtracts the control mean, orders samples chronologically
(or in reverse), and solves the lower-triangular Cholesky system of a
shrinkage-regularized control covariance. Each output represents a normalized
linear innovation with respect to the preceding samples in that order.
Features are written in original temporal order. Neither detector mixing nor
feature selection is performed.

The transform is invertible under positive diagonal regularization. Thus
unregularized OLS on the complete raw and transformed feature sets has
equivalent prediction capacity. **Ridge with a fixed alpha is not invariant**
to this transformation: comparisons against the original Ridge models are
empirical and do not guarantee equal CTR.

## Generate and evaluate both boards

From repository root:

```bash
python -m waveform_analysis.scripts.linear_innovations_benchmark \
  --output-root "$HOME/linear_innovations"
python -m waveform_analysis.scripts.linear_innovations_benchmark \
  --output-root "$HOME/linear_innovations" --execute
```

The generated batch files cover UC/FBK, energy/timing and their configured
Onishi/Wide windows, using the standard control/development/blind pipeline.
Both original Ridge and transformed variants are compared. Original batches
and datasets are not modified. To assess ordering dependence, use a separate
output root and `--order reverse`. Adjust input ROOT paths if required.

Fast smoke test:

```bash
python -m pytest waveform_analysis/tests/test_linear_innovations.py -q
python -m waveform_analysis.cli check-batch \
  --config "$HOME/linear_innovations/UC_forward.json"
```

## TSHAP

Run the existing TSHAP CLI on **saved** innovation-model runs. The explainer
perturbs the **original waveforms** using its existing imputation procedure,
then applies the frozen innovation transform before Ridge prediction.
It is *not* correct to occlude transformed innovations independently.
Example:

```bash
python -m waveform_analysis.cli explain-tshap \
  --results "$HOME/linear_innovations/UC/linear_innovations_forward" \
  --models direct_linear_innovations shared_linear_innovations \
  --windows wide --mode energy \
  --max-events 100 --window-samples 20 --stride-samples 20 \
  --output-root "$HOME/xai/linear_innovations"
```

Supply `--cache-root` pointing to a matching prepared cache bundle if needed.
The control imputer is reusable across models that share the same raw waveform
protocol. Predictions are made from the original occlusion and then transformed.

## Scientific interpretation

TShap profiles obtained by occluding raw waveform regions represent the
sensitivity of the complete *raw-input predictor*. They are **not direct
importance measurements of individual innovations**. Changing the feature
representation while retaining the same linear prediction function should not
change raw-input occlusion effects; changes observed after refitting Ridge can
reflect the new regularization geometry. Comparing forward and reverse
transformations is therefore mainly a test of estimator sensitivity, not an
independent localization of nonredundant physical timing information.

The covariance matrix requires O(p²) memory and a Cholesky factorization
requires O(p³) time. `max_samples` prevents accidentally processing excessive
windows; adjust deliberately. The transform applies in batches using
triangular solves. Near-singular covariance is stabilized by shrinkage and a
strictly positive diagonal jitter.
