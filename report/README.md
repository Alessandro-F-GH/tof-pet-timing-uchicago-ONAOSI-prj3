# ML waveform timing report

This directory contains the LaTeX source for the waveform-based TOF-PET timing report.

The report follows the final waveform-analysis code and is centered on one model comparison:

1. **Proposed model:** detector-shared antisymmetric MLP, with prediction `g(s1) - g(s2)`.
2. **Reference model:** paired Onishi CNN, using the fixed literature-inspired architecture implemented as `onishi_cnn`.

The comparison is repeated for the two waveform windows defined by the shared profile:

- `onishi_window`: -1.5 ns to +2.0 ns;
- `wide_window`: -2.0 ns to +30.0 ns.

Energy-versus-timing-channel behavior, board dependence, MLP LED-threshold sensitivity, model-output correlation, and XAI are supporting analyses around this central comparison.

Structure:

- `main.tex`: document entry point and abstract.
- `references.bib`: bibliography, including the Onishi reference.
- `sections/01_overview.tex`: motivation, scope, and research questions.
- `sections/02_experimental_setup.tex`: detector system and waveform datasets.
- `sections/02_pipeline.tex`: selection, preprocessing, split policy, target definition, model-study protocol, and CTR estimator.
- `sections/03_models.tex`: antisymmetric MLP, Onishi paired CNN, controlled comparison, and XAI.
- `sections/04_results.tex`: results organized around MLP versus Onishi for the two input windows, followed by supporting analyses.
- `sections/05_limitations.tex`: interpretation, protocol limitations, and next experiments.
- `figures/`: report figures.
- `tables/`: LaTeX table fragments.

The final paper model runs should come from `experiment.type = model_study`. MLP and Onishi runs are produced independently and combined with `compare-runs`, which checks compatible windows/voltage sets and aligns persisted blind-test event identities before computing model-output correlations.

The standalone `threshold_scan` experiment belongs only to the antisymmetric MLP and is treated as a supporting sensitivity study rather than as part of the main architecture comparison.
