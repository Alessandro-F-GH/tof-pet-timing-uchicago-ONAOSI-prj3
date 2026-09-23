# Research-group update report

Two-column LaTeX draft for sharing the locally connected waveform-timing results.

## Structure

- `main.tex`: two-column document entry point and abstract.
- `sections/`: introduction, experimental setup, preprocessing, ML pipeline, results, conclusion.
- `tables/`: dataset and CTR result placeholders.
- `figures/`: expected plot filenames and figure placeholders.
- `references.bib`: TOF-PET, locally connected network, and robust-statistics references.

## Compile

From `update_report/`:

```bash
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

The source intentionally contains TODO markers for numerical results and final model/training details that should only be filled once the corresponding runs are frozen.
