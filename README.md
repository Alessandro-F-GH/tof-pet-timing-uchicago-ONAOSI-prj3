# PET detector timing analysis

Analysis software for the ONAOSI/UChicago TOF-PET project.

- `waveform_analysis/`: oscilloscope event selection, control-fitted preprocessing and waveform ML.
- `janus_data_analysis/`: Pico-TDC / Janus timing analysis.
- `utils_fit/`: shared timing-resolution utilities.
- `report/` and `update_report/`: scientific manuscripts and study-specific figures/tables.

The waveform pipeline uses three independent dataset roles:

1. **Control:** fit event-selection, preprocessing and LED-threshold rules once.
2. **Development:** apply the frozen rules, select model parameters and fit the final model.
3. **Blind:** evaluate the final model and estimate event-bootstrap uncertainty.

The blind dataset never participates in selection or training. Most models use
common deterministic development CV folds and fixed/grid/Optuna parameter
selection. The two linear Ridge models use sklearn RidgeCV on all development
events with a configurable logarithmic lambda grid and MSE scoring; they bypass
outer CV and pruning. MiniRocket retains the outer development-CV pipeline.

Batches sweep models, waveform modes and windows. Modes are `energy_to_energy`
and `timing_to_timing`; each resolved run has one model and one window. Reports
compare predictions on matched blind event identities and provide paired
bootstrap uncertainty. Figure styling is configured in JSON.

Run from the repository root:

```bash
python -m waveform_analysis.cli check-batch --config waveform_analysis/config/batches/test_ridge.json
python -m waveform_analysis.cli batch --config waveform_analysis/config/batches/test_ridge.json
python -m waveform_analysis.cli plots --results waveform_analysis/results/FBK/test_ridge_48V_R1_R2
```

See [waveform_analysis/README.md](waveform_analysis/README.md) for configuration,
result layout, resume behavior, scientific invariants and plot customization.
Manuscript methods describe their specific experiments; executable pipeline
instructions are maintained in these software READMEs.
