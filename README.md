# PET detector timing analysis

Analysis software for the ONAOSI/UChicago TOF-PET project.

- `waveform_analysis/`: oscilloscope event selection, native-time preprocessing, LED/CFD calibration, waveform ML and XAI.
- `janus_data_analysis/`: Pico-TDC / Janus timing analysis.

The waveform path has one scientific order:

`ROOT -> raw development/test split -> development-fitted event selection -> native-time preprocessing -> LED/CFD + ML dataset preparation -> training/validation model selection -> final test evaluation`

The permanent test population never determines the photopeak, pulse-duration cuts, baseline-noise cuts, LED threshold, CFD fraction, input normalization, ML hyperparameters or early stopping.

Waveform ML models are Linear SVR and CNN. Both use one shared detector scorer and a pair correction `g(s1) - g(s2)`. CTR is measured with the direct FWHM estimator in `ml_pipeline.stats`.

Run from the repository root:

```bash
python -m waveform_analysis.cli check --config waveform_analysis/config/experiments/complete.json
python -m waveform_analysis.cli prepare --config waveform_analysis/config/experiments/complete.json
python -m waveform_analysis.cli run --config waveform_analysis/config/experiments/complete.json
```

See `waveform_analysis/README.md` for the detailed data contract.
