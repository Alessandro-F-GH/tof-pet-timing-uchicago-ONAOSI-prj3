# Paper-oriented ML waveform timing report template

This folder contains a modular LaTeX sketch for a possible paper on physically constrained and interpretable waveform-based timing correction.

The scientific narrative is organized around three questions:

1. Can a detector-shared antisymmetric correction retain the timing performance of a general joint detector-pair CNN?
2. Where in the waveform is the information used for timing correction, as tested with window studies / ablations and XAI?
3. How does the same framework perform when moving from the energy-channel waveform to the dedicated timing-channel waveform?

Structure:

- main.tex: paper title, abstract placeholder, and section assembly
- sections/01_overview.tex: motivation, research questions, intended contributions
- sections/02_experimental_setup.tex: detector system and UC/FBK waveform datasets
- sections/02_pipeline.tex: leakage-controlled waveform analysis framework
- sections/03_models.tex: linear baseline, detector-shared CNN, joint CNN, information-localization tools
- sections/04_results.tex: results organized by scientific question
- sections/05_limitations.tex: discussion, limitations, and next experiments
- figures/: paper figures
- tables/: reusable LaTeX table fragments

The main paper comparison intentionally excludes exploratory k-NN/shapelet models unless they later become necessary to support a specific scientific claim.

Numerical claims remain TODO placeholders until a frozen study run is selected. Tables and plots should be populated directly from run CSV/JSON outputs whenever possible.