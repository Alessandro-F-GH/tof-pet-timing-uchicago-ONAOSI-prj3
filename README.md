# TOF-PET waveform timing analysis

Python analysis for the ONAOSI / University of Chicago TOF-PET project. The main workflow studies coincidence timing corrections from two-detector digitized waveforms, with leading-edge discrimination (LED) as the reference.

## Repository

- [waveform_analysis](waveform_analysis/README.md): waveform preparation, correction models, cross-validation, blind evaluation, reporting, and interpretability.
- `janus_data_analysis`: Janus / Pico-TDC timing analyses.
- `trc_converter`: oscilloscope trace conversion.
- `utils_fit`: fitting utilities.
- `report`, `update_report`, `presentation`: research material.

Acquisition ROOT files are not included. Set their local paths in the dataset configurations before running an experiment.

## Final filtered comparison

The two checked-in comparison configurations are:

- `waveform_analysis/config/batches/UC_lowpass_final.json`
- `waveform_analysis/config/batches/FBK_lowpass_final.json`

Both use a fifth-order Butterworth low-pass filter with 2 GHz cutoff on the native energy and timing waveforms of both detector channels, before LED selection and waveform cropping.

Each board configuration evaluates the same five models—direct/shared Ridge, Onishi CNN, direct RDST and direct MultiRocket—on energy/timing signals and two windows: Onishi (−1.5 to 2 ns), and wide (−2 to 50 ns UC; −2 to 60 ns FBK). The models are independent: there is no MultiRocket–RDST combined model in this comparison.

Install dependencies and inspect the configuration before running:

```bash
python -m pip install -r waveform_analysis/requirements.txt
python -m waveform_analysis.cli check-batch --config waveform_analysis/config/batches/UC_lowpass_final.json
python -m waveform_analysis.cli batch --config waveform_analysis/config/batches/UC_lowpass_final.json
```

See [waveform_analysis/README.md](waveform_analysis/README.md) for the runtime layout, cache and resume commands. The supplied datasets, configuration, environment and saved outputs are required for numerical reproducibility; the repository alone is not sufficient.
