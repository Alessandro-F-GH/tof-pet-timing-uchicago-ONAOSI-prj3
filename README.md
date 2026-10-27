# TOF-PET detector timing analysis

Python analysis software developed for the ONAOSI / University of Chicago TOF-PET project. The waveform study investigates whether machine-learning corrections based on digitized detector pulses can improve coincidence timing resolution beyond a conventional leading-edge discriminator (LED) reference.

The experiments compare linear regression, neural networks, and time-series feature transformations (MiniROCKET, MultiROCKET and random dilated shapelets). They examine direct models of detector pairs, shared detector-level models, short and extended observation windows, and time-series SHAP (TSHAP) explanations.

## Repository contents

| Directory | Purpose |
| --- | --- |
| [`waveform_analysis/`](waveform_analysis/) | Waveform preprocessing, model training, evaluation, reporting and interpretability |
| [`janus_data_analysis/`](janus_data_analysis/) | Pico-TDC / Janus timing studies |
| [`utils_fit/`](utils_fit/) | Shared timing-resolution and fitting utilities |
| [`trc_converter/`](trc_converter/) | Oscilloscope waveform conversion tools |
| [`report/`](report/), [`update_report/`](update_report/) | Research reports and associated figures/tables |
| [`presentation/`](presentation/) | Presentation materials |

The main thesis-related workflow is in **[`waveform_analysis/`](waveform_analysis/README.md)**.

## Data availability

**The experimental waveform datasets are not publicly available in this repository because of their size and GitHub hosting limits.** The benchmark configurations reference local ROOT acquisition files, which must be supplied separately to execute the experiments. The published code and configurations document the analysis workflow, but cloning the repository alone will not reproduce the numerical results.

## Getting started

Use a Python environment with the dependencies in `waveform_analysis/requirements.txt`:

```bash
python -m pip install -r waveform_analysis/requirements.txt
```

From the repository root, inspect a supplied batch configuration:

```bash
python -m waveform_analysis.cli check-batch --config waveform_analysis/config/batches/benchmark_UC.json
```

To run a batch **after providing the referenced acquisition files and configuring their paths**:

```bash
python -m waveform_analysis.cli batch --config waveform_analysis/config/batches/benchmark_UC.json
```

See the [waveform analysis guide](waveform_analysis/README.md) for models, configuration files, saved-result reporting and TSHAP commands. [`ARCHITECTURE.md`](waveform_analysis/ARCHITECTURE.md) describes the software internals for maintainers.

## Reproducibility scope

The pipeline separates **control**, **development**, and **blind/test** data: control data determine preprocessing and timing-calibration rules, development data are used for model selection and fitting, and blind data are used for final evaluation. The repository provides source code, model configurations and tests, but not the large experimental acquisitions or an exact lockfile for every software dependency. Only the configuration and environment actually used for a particular experiment should be treated as its reproducibility record.
