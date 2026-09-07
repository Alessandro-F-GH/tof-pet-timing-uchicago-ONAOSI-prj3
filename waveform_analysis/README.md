# Waveform timing pipeline

The waveform pipeline separates **selection**, **physical preprocessing**, **ML dataset construction**, and **model fitting**.

## 1. Event selection

The ROOT entry population is split into permanent **development** and **test** sets before any fitted selection. Using development only, the pipeline fits the two energy photopeaks, detects every threshold hit on waveform families required by the experiment, derives detector-specific ToT limits from development median and scaled MAD, selects the longest acceptable pulse, and optionally derives a baseline-RMS limit from a trigger-relative development region. Frozen cuts are applied unchanged to test.

The selection cache stores indices, split labels, all hit metadata and selected main-hit indices. It does not copy waveforms. Energy used only for photopeak selection is not persisted downstream.

Diagnostics include photopeak, ToT and optional baseline-noise plots plus `selection_summary.csv`.

## 2. Native-time preprocessing

Only selected events and waveform families required by the configured ML modes are materialized. For each waveform the pipeline decodes/orients native samples, clamps them to detector-specific vertical limits, crops around the selected main trigger, preserves the original acquisition time through `window_start_time_s` and `sample_interval_s`, and stores a rising-edge interval ending at the selected pulse peak.

There is **no denoising** and no event-wise baseline subtraction. No relative-time conversion, LED or CFD is performed here.

## 3. ML dataset preparation

For each waveform family needed as a source or target, LED thresholds are scanned only inside the stored rising intervals and the best threshold is selected on complete development. CFD fraction is selected on development when active. The native sample whose voltage is closest to the selected LED threshold becomes the anchor. The target is

`true_tof_ps - (anchor_time_1_ps - anchor_time_2_ps)`.

The ML window is then materialized with `t_anchor = 0`, the single scalar `ml_input.subsampling` is applied, selected development is split into training/validation, and `mean[detector, sample]` / `std[detector, sample]` are fit on **training only**. The frozen transform and its inverse are persisted.

## 4. ML and final test

Linear SVR and CNN candidates are trained on training and ranked only on validation. The selected candidate is refit on complete development. The permanent test population is evaluated once after selection.

## CLI

```bash
python -m waveform_analysis.cli check --config waveform_analysis/config/experiments/complete.json
python -m waveform_analysis.cli prepare --config waveform_analysis/config/experiments/complete.json
python -m waveform_analysis.cli run --config waveform_analysis/config/experiments/complete.json --overwrite
python -m waveform_analysis.cli report --run-dir waveform_analysis/results/studies/complete
```

A small experiment with subsampling 4 is provided in `config/experiments/small_subsampling4.json`.
