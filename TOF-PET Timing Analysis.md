# TOF-PET Timing Analysis

This repository contains the ONAOSI/UChicago analysis tools for SiPM timing and coincidence time resolution (CTR).

## Waveform analysis

The oscilloscope pipeline keeps one compact protocol:

1. physical event selection and permanent waveform preparation;
2. deterministic development/blind split;
3. one training/validation holdout inside development;
4. LED/CFD and ML candidate selection on validation only;
5. refit of the selected ML candidate on all development events;
6. one final evaluation on the untouched blind population.

The retained waveform models are **Linear SVR** and **1-D CNN**. Both implement a shared single-detector scorer and the exact antisymmetric pair correction

\[
y(s_1,s_2)=g(s_1)-g(s_2).
\]

The standard methods are **LED** and **CFD**. The waveform ML pipeline reports CTR from a direct FWHM measurement of the dominant timing peak, with bootstrap uncertainty computed from the same estimator.

Run from the repository root:

```bash
python -m waveform_analysis.cli check --config waveform_analysis/config/experiments/complete_new.json
python -m waveform_analysis.cli run --config waveform_analysis/config/experiments/complete_new.json
```

## Pico-TDC / Janus

`janus_data_analysis/` remains the independent reduced-readout analysis path for Pico-TDC data, including event matching, threshold scans and CTR studies.

## Data

Experimental data are not included in the repository.
