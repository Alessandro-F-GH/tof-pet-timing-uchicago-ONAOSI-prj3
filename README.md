# PET detector timing analysis

This repository contains the analysis software developed to characterize timing performance in a TOF-PET detector setup using two complementary acquisition systems:

- **oscilloscope waveform acquisition**, used as a full-information reference for conventional timing and waveform-based machine learning;
- **Pico-TDC / Janus timestamp acquisition**, used to study a more compact and scalable threshold-based timing readout.

The project focuses on coincidence timing resolution (CTR), waveform-dependent timing corrections, reduced multithreshold representations, and interpretation of the timing information encoded in detector signals.

## Main analysis components

### Oscilloscope waveform analysis

`waveform_analysis/` contains the waveform preprocessing and ML pipeline, including:

- photopeak/event preparation and LED/CFD timing references;
- physics-constrained pair corrections of the form `g(s1) - g(s2)`;
- linear SVR, CNN, and reduced multithreshold studies;
- development/blind evaluation with a common CTR fitter;
- explainability tools for studying where timing-relevant information is located in the waveform.

See [`waveform_analysis/README.md`](waveform_analysis/README.md) for the detailed protocol and commands.

### Pico-TDC / Janus analysis

`janus_data_analysis/` contains the Pico-TDC analysis workflow, including:

- trigger-matching acquisition analysis;
- ToT-based energy/photopeak selection;
- timing-hit matching and mismatch rejection;
- timing-threshold scans;
- CTR extraction and comparison with the oscilloscope reference.

### Supporting tools

- `trc_converter/` — conversion utilities for oscilloscope `.trc` data;
- `tools/` — repository-level analysis/support utilities;
- `docs/` — additional project documentation.

## Scientific goal

The two readout paths are used together to address two complementary questions:

1. **What timing performance is achievable when the complete detector waveform is available?**
2. **How much of that performance can be retained with a practical, reduced-data readout?**

The waveform analysis also uses interpretable ML models to investigate which parts of the detector signal carry timing-relevant information and to motivate future timing strategies or electronics designs.

## Data availability

**The experimental datasets used in this project are not publicly available yet.**

The repository currently provides the analysis code, configuration files, and documentation, but does **not** include the raw or processed oscilloscope and Pico-TDC datasets required to reproduce the numerical results.

Data availability will be updated if the experimental datasets can be released publicly.

## License

This repository is released under the MIT License. See [`LICENSE`](LICENSE).
