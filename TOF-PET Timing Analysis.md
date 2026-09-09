# TOF-PET Timing Analysis

The repository contains an oscilloscope waveform pipeline and a separate Janus/Pico-TDC pipeline.

The waveform protocol is selection-first:

`raw ROOT entries -> development/test -> development-fitted physical selection -> native-time materialization -> development LED/CFD calibration -> training/validation ML dataset -> model selection -> test`

Only LED, CFD, Linear SVR and CNN are supported in the waveform study. Waveform preprocessing uses native samples with configured vertical clipping and no denoising. ML waveforms are aligned during dataset preparation to the native sample closest in time to the selected interpolated LED crossing. Their amplitudes are globally MinMax-scaled with the detector-specific physical voltage limits from configuration; the transform is not fitted from individual events or the training population.

The ML correction remains `g(s1) - g(s2)`. CTR throughout the timing analysis is measured as the direct FWHM of a fixed-bin timing histogram. The default metric bin width is 5 ps, half-maximum crossings are linearly interpolated between neighboring bin centres, and final CTR uncertainty is estimated from 100 event-bootstrap resamples.
