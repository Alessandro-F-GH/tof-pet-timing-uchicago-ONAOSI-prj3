# ML pipeline report template

This folder contains a modular LaTeX template for documenting the waveform-analysis pipeline from event selection through final ML CTR results.

Structure:

- main.tex: document preamble and section assembly
- sections/: report body, ordered according to the analysis pipeline
- figures/: supporting plots
- tables/: reusable LaTeX table fragments

The template intentionally contains TODO placeholders instead of numerical claims. It is based on the current main-branch workflow and should be edited to match the exact frozen experiment configuration used for the final report.

Compile from the report directory, for example:

    pdflatex main.tex
    pdflatex main.tex

The template is organized around:

1. general overview;
2. step-by-step pipeline description with scientific motivation and diagnostic figures;
3. ML model descriptions and motivations;
4. CTR-focused results with tables and plots;
5. limitations and possible improvements.

Before finalizing the report, remove unused optional sections and ensure every table/plot can be traced to a frozen run directory or manifest.
