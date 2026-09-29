# PET detector timing analysis

Analysis software for the ONAOSI/UChicago TOF-PET project.

- `waveform_analysis/`: oscilloscope event selection, fixed-control preprocessing and waveform ML.
- `janus_data_analysis/`: Pico-TDC / Janus timing analysis.
- `utils_fit/`: repository-wide timing-resolution utilities.

The waveform ML path follows one scientific order:

`independent control -> fit/freeze preprocessing + LED threshold -> apply frozen rules to independent analysis data -> fixed ML population -> repeated train/validation/blind holdout -> model selection/refit -> blind evaluation`

The control sample can determine photopeak, baseline-noise, baseline-clipping, timing-ToT and LED-selection rules, but contributes no ML training or test events. The analysis sample can only be filtered with those frozen rules. For a given analysis population and resampling seed, train/validation/blind event identities are independent of model and waveform window so later studies can be paired by seed.

Waveform studies use one mode per study: `energy_to_energy` or `timing_to_timing`. A study also resolves exactly one registered model and one waveform window. Hyperparameter selection is generic: a single candidate is fitted directly on train+validation; multiple candidates are trained on train only, selected by validation CTR, then refitted from scratch on train+validation before blind evaluation.

Run from the repository root, for example:

```bash
python -m waveform_analysis.cli check --config waveform_analysis/config/studies/example.json
python -m waveform_analysis.cli run --config waveform_analysis/config/studies/example.json
python -m waveform_analysis.cli batch --config waveform_analysis/config/batches/main.json
```

Old voltage-scan, threshold-scan, concatenated-dataset and multi-model/multi-window experiment schemas are intentionally incompatible with the current pipeline.

See `waveform_analysis/README.md` for the study configuration, output schema and leakage invariants.
