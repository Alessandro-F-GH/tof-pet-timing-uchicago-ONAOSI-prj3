# ML waveform timing report

This directory contains the LaTeX source for the waveform-based TOF-PET timing report.

The report is organized around three questions:

1. Can a detector-shared antisymmetric correction retain the timing performance of a general joint detector-pair CNN?
2. Which regions of the detector waveform contain useful information for timing correction?
3. How does the same framework perform on the dedicated timing-channel waveform compared with the energy-channel waveform?

Structure:

- `main.tex`: document entry point and abstract
- `references.bib`: bibliography
- `sections/01_overview.tex`: motivation, scope, and research questions
- `sections/02_experimental_setup.tex`: detector system and waveform datasets
- `sections/02_pipeline.tex`: waveform analysis framework
- `sections/03_models.tex`: linear baseline, detector-shared CNN, joint CNN, and interpretability
- `sections/04_results.tex`: results organized by scientific question
- `sections/05_limitations.tex`: discussion, limitations, and future work
- `figures/`: report figures
- `tables/`: LaTeX table fragments

Numerical tables and result summaries are taken from the study outputs used for the reported analysis.
