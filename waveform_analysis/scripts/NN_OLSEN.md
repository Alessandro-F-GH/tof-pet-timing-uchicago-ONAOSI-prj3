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


## Parallel temporal CNN variant (experimental)

The original `--architecture mlp` remains unchanged. The separate
`--architecture temporal_cnn` variant tests whether temporal inductive
bias and higher capacity improve conditional fidelity.

- Shared **five-channel temporal input**: two masked original waveforms,
  two explicit presence masks, and a deterministic absolute-time coordinate.
- Initial Conv1d(kernel=9, stride=4), then residual temporal blocks using
  kernels 7 and 5, progressive downsampling and increasing dilation.
- GroupNorm (instead of BatchNorm) avoids mixing running statistics from
  the radically different W/R mask distributions.
- Adaptive average and max pooling preserve **eight ordered temporal bins**
  (by default) before a 3-layer dense regression head. No global
  permutation-invariant pooling; window position remains identifiable.
- The exact same `augment`, `paired_batch`, `coalitions`,
  `predict_coalitions`, saved targets, disjoint train/tuning/holdout split,
  early stopping criterion, and variance diagnostics are reused.
- This is an **original temporal architectural variant** of the
  NN-Olsen-style objective, **not** the published Olsen architecture.
  Any gain should be attributed to the architecture in this comparison,
  not presented as a replication of published NN-Olsen results.

Run after `git pull` from repository root:

```powershell
python -m waveform_analysis.scripts.evaluate_nn_olsen `
  --architecture temporal_cnn `
  --model-dir ".\\waveform_analysis\\results\\UC\\multirocket_rdst\\energy\\wide\\direct_multirocket_rdst" `
  --cache-root "$HOME\\UC_cache" `
  --output-root "$HOME\\conditional_temporal_cnn\\UC" `
  --window-samples 20 `
  --stride-samples 20 `
  --tuning-queries 500 `
  --holdout-queries 500 `
  --width 64 `
  --depth 3 `
  --temporal-bins 8 `
  --epochs 50 `
  --batch-size 128 `
  --learning-rate 0.0003 `
  --patience 8
```

The CNN costs more memory and computation than the MLP. This example uses
a smaller batch and learning rate for stability; for an architecture-only
comparison, keep all hyperparameters identical. Run results in **separate
output roots**; `temporal_conditional_cnn.pt` is the CNN checkpoint.
No existing NN-Olsen output is overwritten.

Tests:
```powershell
python -m pytest waveform_analysis/tests/test_nn_olsen.py waveform_analysis/tests/test_temporal_conditional_cnn.py -q
```


## Optional streaming per-time-sample standardization

Enable `--input-normalization per_sample` on either architecture. For each
detector channel and temporal index `t`, fit `mean[c,t]` and `std[c,t]`
on **training events only** in two float64 streaming passes. Use
`safe_std[c,t] = max(std[c,t], 0.05 * detector_rms_std[c])` by default
(the floor is configurable with `--normalization-scale-floor-ratio`).
Then form masked inputs **per minibatch** as
`((wave - mean) / safe_std) * observed_mask`. Missing entries stay exactly
zero and explicit masks remain unchanged. Never normalize after masking:
that would expose a mean-derived pattern in hidden positions. We do not
create or persist a second full waveform dataset.

To compare against the original CNN, specify a **new output directory**
since this is a different trained model. Fitted statistics and the scale-floor
metadata are stored inside the checkpoint; summary.json records settings.

Example:

```bash
python -m waveform_analysis.scripts.evaluate_nn_olsen \
  --architecture temporal_cnn \
  --input-normalization per_sample \
  --model-dir "$PWD/waveform_analysis/results/UC/multirocket_rdst/energy/wide/direct_multirocket_rdst" \
  --cache-root "$HOME/UC_cache" \
  --output-root "$HOME/conditional_temporal_cnn/UC_per_sample" \
  --window-samples 20 --stride-samples 20 \
  --tuning-queries 500 --holdout-queries 500 \
  --width 64 --depth 3 --temporal-bins 8 \
  --epochs 50 --batch-size 256 --learning-rate 0.001 --patience 8
```

Run `python -m pytest waveform_analysis/tests/test_sample_normalization.py -q`.


## Full-signal CNN distillation control (no temporal windows)

Before relying on the conditional W/R surrogate, test whether the **same
architecture** can reproduce the cached original-model predictions with
the *entire paired waveform observable*. This is a **separately trained
model** (do not extrapolate the W/R-trained model to an unseen full mask).

Uses exactly the same random train/tuning/holdout event selection as the
conditional experiment (same seed and cohort sizes), cached predictions,
train-only two-pass normalization, CNN settings, normalized-MSE training,
and early stopping on tuning RMSE. The input is all observed and every
mask bit equals one; no temporal coalitions, window sampling, or feature
transform from the original model.

```bash
python -m waveform_analysis.scripts.evaluate_full_signal_distillation \
  --model-dir "$PWD/waveform_analysis/results/UC/multirocket_rdst/energy/wide/direct_multirocket_rdst" \
  --cache-root "$HOME/UC_cache" \
  --output-root "$HOME/full_signal_distillation/UC" \
  --input-normalization per_sample \
  --tuning-queries 500 --holdout-queries 500 \
  --width 64 --depth 3 --temporal-bins 8 \
  --epochs 50 --batch-size 256 --learning-rate 0.001 --patience 8
```

Outputs: `summary.json` (tuning and untouched holdout RMSE, MAE, error
quantiles, bias, correlation and prediction spread), `diagnostics.npz`
with per-event predictions/targets, `temporal_cnn_full_signal.pt`
including normalization statistics, and three holdout diagnostic PNGs
(scatter, residual scatter, residual histogram).

A high full-signal holdout error would flag approximation/optimization
limits of this CNN; a low error would strengthen the case for investigating
masked conditional estimation separately. Low full-signal error alone does
not establish accurate Shapley attributions.


## Full-signal InceptionTime student (black-box)

Run the **independent** multiscale InceptionTime student against saved full-signal
teacher predictions, with the same seed, train/tuning/holdout cohorts,
train-only per-time-sample normalization, MSE, metrics, and early stopping
as the full-signal temporal CNN control:

```bash
python -m waveform_analysis.scripts.evaluate_full_signal_distillation \
  --architecture inception_time \
  --model-dir "$PWD/waveform_analysis/results/UC/multirocket_rdst/energy/wide/direct_multirocket_rdst" \
  --cache-root "$HOME/UC_cache" \
  --output-root "$HOME/full_signal_distillation/UC_inception_time" \
  --input-normalization per_sample \
  --tuning-queries 500 --holdout-queries 500 \
  --inception-filters 16 --inception-bottleneck 32 \
  --inception-depth 6 --inception-kernel-size 39 --inception-bins 8 \
  --epochs 50 --batch-size 32 --eval-batch 32 \
  --learning-rate 0.001 --patience 8
```

This is an independent PyTorch adaptation of InceptionTime with three
different convolutional kernels per module, a max-pooling branch, bottleneck
1x1 convolutions, BN/ReLU, residual links every three modules, and position-aware ordered\n8-bin average/max pooling followed by a regression head. The two waveform channels plus two explicit full-observation
mask channels are provided in the same `[B,4*T]` interface as the previous
CNN so the pipeline can later be adapted to masked coalitions. In this
experiment all mask bits are one. It does **not** use or inspect any internal
teacher features or transformations.

The reference InceptionTime architecture is for classification; the final
head here is scalar regression. Activation memory scales with waveform length
(4159 samples) and batch size, so the example starts at batch 32.

Outputs are separated from the previous CNN run, and the checkpoint is named
`inception_time_full_signal.pt`. Tests:
`python -m pytest waveform_analysis/tests/test_inception_time_surrogate.py waveform_analysis/tests/test_full_signal_distillation.py -q`.

**Position-aware change (2026-11-06):** This implementation now uses ordered
adaptive average and max temporal bins rather than the classic global average
pooling. This overwrites the previous InceptionTime implementation: its
checkpoints are incompatible because the output head shape changed. Use a
fresh output directory when comparing experiments, or explicitly overwrite the
old run knowing the original outputs will be lost. The original CNN is unchanged.


## Normalized full-signal kNN (PowerShell/CPU)

Unlike the older conditional kNN, this diagnostic uses the **same train-only
per-time-sample mean and scale** as the neural full-signal experiments. Each
query is matched to the 74,714 training reference events only, never itself
or other tuning/holdout events. Entire paired waveforms of 4,159 samples
per detector are used; no W/R windows. All nearest neighbours are computed
by exact Euclidean distance in standardized waveform space, with reference
blocks to limit RAM. Tuning selects k and inverse-distance versus uniform
weights, holdout is evaluated only with that frozen choice.

```powershell
python -m waveform_analysis.scripts.evaluate_full_signal_knn `
  --model-dir ".\\waveform_analysis\\results\\UC\\multirocket_rdst\\energy\\wide\\direct_multirocket_rdst" `
  --cache-root "$HOME\\UC_cache" `
  --output-root "$HOME\\full_signal_knn\\UC_normalized" `
  --k 1 3 5 10 20 50 100 `
  --weighted `
  --reference-batch 1024 `
  --query-batch 16
```

Saved: `summary.json`, `metrics.csv`, `diagnostics.npz`, and
`evaluation.log`. This is a computationally intensive exact kNN
baseline on CPU; adjust query/reference batch sizes to fit RAM.
Tests: `python -m pytest waveform_analysis/tests/test_full_signal_knn.py -q`.


## Full-signal temporal CNN + independent first-difference channels

Ablation: compare the original `TemporalConditionalCNN` with an otherwise
identical architecture receiving **two additional channels**, one temporal
first difference per detector (`d[:, :, 0] = 0`,
`d[:, :, t] = x[:, :, t] - x[:, :, t-1]` for `t >= 1`).
Differences are calculated from **raw** waveform samples first, and
standardized independently using per-detector/per-position training-only
mean and scale; they are *not* computed from normalized waveforms.
The network sees seven channels: two standardized signals, two full
observation masks, two standardized differences and the absolute time
coordinate. No additional model-feature extraction, teacher inference,
windowing or TSHAP coalition estimation is involved.

```bash
python -m waveform_analysis.scripts.evaluate_full_signal_distillation \
  --architecture temporal_cnn --use-derivative \
  --model-dir "$PWD/waveform_analysis/results/UC/multirocket_rdst/energy/wide/direct_multirocket_rdst" \
  --cache-root "$HOME/UC_cache" \
  --output-root "$HOME/full_signal_distillation/UC_temporal_derivative" \
  --input-normalization per_sample \
  --tuning-queries 500 --holdout-queries 500 \
  --width 64 --depth 3 --temporal-bins 8 \
  --epochs 50 --batch-size 256 --eval-batch 256 \
  --learning-rate 0.001 --patience 8
```

Outputs include `temporal_cnn_derivative_full_signal.pt`,
`summary.json`, `diagnostics.npz` and holdout PNGs. Derivative
statistics and the feature flag are saved in the checkpoint.
The default (flag omitted) preserves the original CNN and its input format.
This derivative option currently applies **only to full-signal CNN
distillation**, not the conditional W/R NN-Olsen workflow. For future
masked experiments, derivative values must be hidden unless both adjacent
samples are observed, to avoid boundary information leakage.


## Wider NN-Olsen MLP with masked first differences

Run conditional W/R regression with `--architecture mlp --width 256 --use-derivative`.
The original 64-unit 4T-input network remains the default when the flag is
not supplied. The 256-unit variant has **six T-length channels** per event:
two normalized detector waveform channels, two binary observation masks and
two separately normalized first-difference channels. Each derivative sample
at t is observed only when both t and t-1 are retained; boundary derivatives
are hard zeroed to avoid revealing hidden waveform samples.

For an informative comparison, use the **same cached teacher predictions**,
`seed=42`, 500 tuning/500 holdout, width 20/stride 20, input standardization
`per_sample`, and identical other training settings:

```bash
python -m waveform_analysis.scripts.evaluate_nn_olsen \
  --architecture mlp --width 256 --depth 3 --use-derivative \
  --model-dir "$PWD/waveform_analysis/results/UC/multirocket_rdst/energy/wide/direct_multirocket_rdst" \
  --cache-root "$HOME/UC_cache" \
  --output-root "$HOME/conditional_nn/UC_mlp256_derivative" \
  --input-normalization per_sample \
  --window-samples 20 --stride-samples 20 \
  --tuning-queries 500 --holdout-queries 500 \
  --epochs 50 --batch-size 256 --eval-batch 256 \
  --learning-rate 0.001 --patience 8
```

Checkpoint `nn_olsen_derivative.pt` includes the derivative normalization
metadata and flags. This is a model-input modification; it does not alter
the underlying teacher or teacher features. Test with
`python -m pytest waveform_analysis/tests/test_nn_olsen.py -q`.


### Matching full-signal Olsen MLP experiment

To measure ability to reproduce the original teacher **before** testing
conditional expectations, run the same expanded Olsen MLP on whole waveforms:

```bash
python -m waveform_analysis.scripts.evaluate_full_signal_distillation \
  --architecture olsen_mlp --width 256 --depth 3 --use-derivative \
  --model-dir "$PWD/waveform_analysis/results/UC/multirocket_rdst/energy/wide/direct_multirocket_rdst" \
  --cache-root "$HOME/UC_cache" \
  --output-root "$HOME/full_signal_distillation/UC_olsen256_derivative" \
  --input-normalization per_sample \
  --tuning-queries 500 --holdout-queries 500 \
  --epochs 50 --batch-size 256 --eval-batch 256 \
  --learning-rate 0.001 --patience 8
```

This experiment uses **all-ones masks** and two standardized raw first
difference channels; the same data split/targets and teacher-free procedure
as the CNN full-signal experiment. Checkpoint name:
`olsen_mlp_derivative_full_signal.pt`.


## Temporal CNN capacity: deliberate 256-event overfitting

Diagnostic for whether the **same temporal CNN with derivatives** can memorize
a small fixed set of original-model outputs. The teacher is never loaded.

```bash
python -m waveform_analysis.scripts.test_temporal_cnn_capacity \
  --model-dir "$PWD/waveform_analysis/results/UC/multirocket_rdst/energy/wide/direct_multirocket_rdst" \
  --cache-root "$HOME/UC_cache" \
  --output-root "$HOME/full_signal_distillation/UC_capacity_256" \
  --fit-events 256 --use-derivative \
  --width 64 --depth 3 --temporal-bins 8 \
  --epochs 400 --batch-size 64 --eval-batch 64 \
  --learning-rate 0.0005 --report-every 5
```

The exact seed-42 74,714-event training cohort from full-signal distillation
is retained as the **normalization and target-standardization reference**;
256 fixed distinct events are sampled from it for optimization. All other
events have no effect on weight updates. No tuning-based checkpoint selection,
holdout evaluation, early stopping or weight decay. The best checkpoint is
selected only by the measured RMSE on the very same memorization events.
Normalizing with the wider original training population avoids making the
small-subset test easier or harder merely by changing the input scaling.

Per-epoch online train MSE and **eval-mode, same-event RMSE and MAE** are logged.
Files: `learning_curve.csv`, `memorization_rmse.png`,
`best_capacity_model.pt`, `diagnostics.npz`, `summary.json`.

This tests **finite-sample interpolation**, not out-of-sample function
approximation. Nearly zero training error demonstrates that the network and
optimizer can memorize this subset; it does not prove faithful generalization
to new waveforms or accuracy of conditional W/R expectations.

Test: `python -m pytest waveform_analysis/tests/test_temporal_cnn_capacity.py -q`.
