# PET detector timing analysis

Analysis software for the ONAOSI/UChicago TOF-PET project. The repository contains two complementary paths:

- `waveform_analysis/`: oscilloscope preprocessing, LED/CFD baselines, waveform ML and XAI;
- `janus_data_analysis/`: Pico-TDC / Janus timing analysis.

The waveform pipeline is intentionally compact. It uses one deterministic holdout protocol:

`prepared events -> development/blind -> training/validation -> select -> refit on development -> evaluate blind once`

Only **Linear SVR** and **CNN** are retained as waveform models. Both use the physically constrained pair correction

`g(s1) - g(s2)`

so swapping the two detectors negates the prediction exactly. Standard timing is limited to **LED** and **CFD**. CTR is measured with a direct FWHM estimator of the dominant timing peak; no Gaussian distribution fit is used in the ML pipeline.

Run the waveform pipeline from the repository root:

```bash
python -m waveform_analysis.cli check --config waveform_analysis/config/experiments/complete_new.json
python -m waveform_analysis.cli prepare --config waveform_analysis/config/experiments/complete_new.json
python -m waveform_analysis.cli run --config waveform_analysis/config/experiments/complete_new.json
```

Generate result plots from an existing run with:

```bash
python -m waveform_analysis.cli report --run-dir waveform_analysis/results/studies/complete
```

See `waveform_analysis/README.md` for the current scientific protocol and configuration structure.

Experimental datasets are not included in the repository.
