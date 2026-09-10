# TOF-PET Timing Analysis

The repository contains an oscilloscope waveform pipeline and a separate Janus/Pico-TDC pipeline.

The waveform protocol is selection-first:

`raw ROOT entries -> development/test -> development-fitted physical selection -> native-time materialization -> development LED/CFD calibration -> training/validation ML dataset -> model selection -> test`

Waveform preprocessing uses native samples with configured vertical clipping and no denoising. ML waveforms are aligned during dataset preparation to the native sample closest in time to the selected interpolated LED crossing. Their amplitudes are globally MinMax-scaled with the detector-specific physical voltage limits from configuration; the transform is not fitted from individual events or the training population.

The supervised ML target is the calibrated LED residual, `Delta t_LED - TOF - C_hat_12`; native-grid anchor offsets are not subtracted from the target. CTR throughout the timing analysis is the Gaussian-equivalent shortest empirical interval containing the configured fraction of finite residuals, 90% by default. Final CTR uncertainty is estimated with 500 event-bootstrap resamples by default. A fixed-bin dominant-peak FWHM is retained only as a secondary core diagnostic.
