# Waveform timing manuscript

This is a two-column academic draft for the current waveform-analysis pipeline.
It describes the ten implemented estimators, detector-shared and direct formulations,
control-fitted preprocessing, development validation, the linear RidgeCV exception,
independent blind evaluation and temporal occlusion. Pico-TDC is outside its scope.

The UC and FBK boards have separate dataset, result and figure placeholders.
Results and conclusions contain no numerical results or performance claims.
The `figures/` directory is reserved for the completed analysis.

## Build

From this directory:

```bash
latexmk -pdf -interaction=nonstopmode -halt-on-error -file-line-error main.tex
```

This runs BibTeX and repeats LaTeX until citations and references are resolved.
If `latexmk` is unavailable, use the equivalent sequence:

```bash
pdflatex -interaction=nonstopmode -halt-on-error main.tex
bibtex main
pdflatex -interaction=nonstopmode -halt-on-error main.tex
pdflatex -interaction=nonstopmode -halt-on-error main.tex
```

The output is `main.pdf`. The template uses Times-style text and mathematics,
with `newtx` when installed and `mathptmx` as a fallback. Tables use `booktabs`, `multirow` and `arydshln` for grouped labels and dashed separators;
references use numerical citations and BibTeX.

Float barriers at every section and subsection keep tables within
the heading that includes them. Wide tables retain their two-column layout and
can continue onto subsequent pages within that section or subsection. A final
barrier keeps appendix tables before the bibliography.

### Build troubleshooting

Undefined citations and references on the first LaTeX pass are expected; they
should disappear after BibTeX and the subsequent LaTeX passes. An underfull box
is a layout diagnostic and does not cause a failed build.

If the build fails, inspect the **first error** in `main.log`, rather than only
the final unresolved-reference summary. With `-file-line-error`, this includes
the source file and line. After correcting the error, clear stale intermediate
files and rebuild from this directory:

```bash
latexmk -c main.tex
latexmk -pdf -interaction=nonstopmode -halt-on-error -file-line-error main.tex
```

The preamble loads `amsmath` before the font packages and avoids loading
`amssymb` after `newtxmath`, which supplies its own mathematical symbols.

## Sources and structure

- `main.tex`: template, abstract and section assembly.
- `sections/01_introduction.tex`: motivation and related waveform-timing literature.
- `sections/02_experimental_setup.tex`: detector hardware, acquisitions and dataset roles.
- `sections/03_methods.tex`: current preprocessing, F1 CTR, architectures, validation and evaluation.
- `sections/04_results.tex`: board-specific table and plot placeholders.
- `sections/05_conclusions.tex`: conclusion placeholder.
- `sections/06_hyperparameters_libraries.tex`: configured spaces and declared library requirements.
- `tables/<type>/<name>.tex`: every table, separated from section prose.
- `references.bib`: literature cited in the draft, including the five supplied papers.

Hardware and acquisition descriptions were checked against `update_report/` and
`presentation/` and incorporate the supplied experimental clarification: the two boards
use the same crystals and SiPMs and each coincidence event contains four waveforms
(two energy and two timing), the first-stage energy output retains the full pulse
amplitude for selection, and the additional timing stage uses high gain with an
intentionally narrow oscilloscope vertical range. The slides' shared-scorer explanation motivates the antisymmetric
formulation. Pipeline descriptions follow `waveform_analysis/` and its current
`benchmark_UC.json`, `benchmark_FBK.json`, preprocessing and model-space files;
older split policies, architecture settings and results are not carried forward.
The UC benchmark evaluates timing waveforms with windows -1 to +2 ns and -2 to +30 ns.
The FBK benchmark evaluates energy and timing waveforms with windows -1.5 to +2 ns
and -2 to +30 ns. The blind bootstrap settings are 1000 draws for UC and 100 for FBK.

## Populate tables after experiments

Batch reporting writes selected-event tables under
`<batch-result-root>/report/tables/datasets/UC_selected_events.tex` or
`FBK_selected_events.tex`, together with `selected_events.csv`. Copy the matching
board fragment into `report/tables/datasets/` after verifying the experiment identity.
Control, train (complete development) and test (independent blind) counts include
fixed-LED coincidence and input-window availability. Counts are deduplicated across
models. Missing historical metadata is a dash, never an inferred count.

CTR tables currently reserve all ten models for each board/mode/window in the current
benchmark. Populate them from the corresponding validation and blind exports;
linear ridge outer-validation CTR remains not applicable. Distinguish CV fold
dispersion from blind bootstrap uncertainty. Add XAI figures from the same frozen
experiments, keeping boards separate.

From the repository root, `python report/generate_appendix.py` refreshes appendix
spaces from model configurations. Add `--result-placeholders` to recreate blank
result templates from the benchmark axes; this explicit option replaces the
placeholders, including any manually inserted values. Library minimum versions
are recorded separately; actual runtime versions belong to the completed study.
