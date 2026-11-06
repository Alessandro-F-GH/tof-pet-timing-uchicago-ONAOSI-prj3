# NN-Olsen-style conditional surrogate benchmark

Reference: Olsen et al. (2024), *A comparative study of methods for estimating
model-agnostic Shapley value explanations*, Sect. 3.6.2 and Appendix A.

## Method

- The surrogate is a fully connected neural network using **ELU**, batch
  normalization, and summed residual/skip connections (depth 3, width 64 by
  default). These components and defaults follow the published NN-Olsen
  architecture. It is not a custom Transformer, CNN, or attention model.
- Each input is the **prepared original** paired waveform with hidden
  coordinates zero-filled, concatenated with an explicit binary observed-mask
  for both detector channels.
- Each training event is assigned one random **TSHAP temporal window** per
  epoch/visit. Both the window and its complement are evaluated in the same
  minibatch, with the **same saved original model prediction** as their target.
  This is a **deliberate adaptation** of NN-Olsen's paired MCAR masks; it
  restricts the mask distribution to the two TSHAP coalitions rather than the
  original author's Bernoulli(0.5) coalition sampling.
- Training target is standardized using the training cohort prediction mean
  and SD (the original waveform input is not transformed). At inference, only
  surrogate predictions are converted back to ps.
- The original model is **never loaded** or called. No feature transforms
  (including MultiRocket/RDST) are used and no missing predictions are
  backfilled. The script shares the KNN benchmark's existing strict
  input loader and prepared prediction alignment.
- The estimator is **model-agnostic in architecture**, but the learned
  function is specific to the saved original model predictions.

## Windows PowerShell

```powershell
python -m waveform_analysis.scripts.evaluate_nn_olsen `
  --model-dir ".\\waveform_analysis\\results\\UC\\multirocket_rdst\\energy\\wide\\direct_multirocket_rdst" `
  --cache-root "$HOME\\UC_cache" `
  --output-root "$HOME\\conditional_nn\\UC" `
  --window-samples 20 `
  --stride-samples 20 `
  --tuning-queries 500 `
  --holdout-queries 500 `
  --width 64 `
  --depth 3 `
  --epochs 50 `
  --batch-size 256 `
  --learning-rate 0.001 `
  --patience 8
```

Run from the root of the repository. A development predictions artifact must
already exist; if not, the script exits with an informative error and does not
invoke saved ML models or process unprepared waveforms.

For published-paper-style architecture experiments, change `--width` to
32 or 128 and `--learning-rate` to 0.01 or 0.0001. The paper used
500 epochs for its hyperparameter comparisons, whereas this diagnostic
defaults to 50 epochs with early stopping to avoid unnecessary runtime.
These are **experiment choices**, not a claim of exact published training
replication.

## Data selection and evaluation

A deterministic sample of events is partitioned into **disjoint** tuning
and holdout query sets (500 each by default). Unlike the KNN leave-one-out
baseline, the neural surrogate is trained on all **remaining** development
events. No artificial limit is imposed on this training set; training event
count is `N - tuning_queries - holdout_queries`.

At each epoch, model selection uses the average of W/R tuning MSE.
The holdout set is evaluated only after selecting the best checkpoint.

To directly compare NN and KNN, use the same prepared run, mask geometry,
seed, query count and holdout cohort. Their reference/training protocols
differ: KNN reference includes all development events except the query;
NN explicitly excludes the two query cohorts during surrogate training.
The reported differences in RMSE must be interpreted with this distinction.

## Diagnostics

- `evaluation.log`: sources, counts, seeds, mask starts, model setup,
  epoch-level training MSE and W/R tuning RMSE, checkpoint selection,
  holdout errors, target and estimated standard deviations.
- `summary.json`: full experimental settings, data provenance, epoch
  history, final tuning/holdout RMSE and bias, **original prediction
  variance and standard deviation**, conditional estimate variance and
  standard deviation for W and R, their standard deviation ratios, and
  per-window reconstruction diagnostics.
- `diagnostics.npz`: original saved predictions and W/R conditional
  estimates on tuning/holdout, all window starts and split indices.
- `nn_olsen.pt`: frozen surrogate weights, input dimensions, mask geometry,
  target output scaling and data provenance.

Note: for a true conditional mean,
`Var(E[f(X)|X_S]) <= Var(f(X))`. This makes output-variance diagnostics
important, but variance alone does not establish local calibration or
causal importance. Small predicted variance could mean the masks carry
little information, or simply that the surrogate is underfitting.

Tests:
```powershell
python -m pytest waveform_analysis/tests/test_nn_olsen.py -q
```
