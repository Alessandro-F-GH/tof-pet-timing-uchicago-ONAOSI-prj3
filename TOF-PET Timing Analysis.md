# TOF-PET Timing Analysis

The repository contains an oscilloscope waveform pipeline and a separate Janus/Pico-TDC pipeline.

The waveform protocol is selection-first:

`raw ROOT entries -> development/test -> development-fitted physical selection -> native-time materialization -> development LED/CFD calibration -> training/validation ML dataset -> model selection -> test`

Only LED, CFD, Linear SVR and CNN are supported in the waveform study. Waveform preprocessing uses native samples with configured vertical clipping and no denoising. ML waveforms are aligned only during dataset preparation to the native sample closest to the selected LED threshold, then independently standardized for the two detector positions using training data only.

The ML correction remains `g(s1) - g(s2)`, and waveform CTR uses the direct FWHM estimator rather than a Gaussian timing-distribution fit.
