# Cleanup decisions and external archive candidates

The supported runtime uses only the canonical hierarchy documented in
ARCHITECTURE.md. The conversion script is removable after the desired historical
results and optional caches have been recovered. It is not imported by runtime
modules.

Removed from the active package:

- `ml_pipeline/`, `utils/` import aliases and `module_migration.json`;
- `core/compat.py` and legacy Python class/module-name rewriting;
- the unused `BaseWaveformModel` / `RegisteredWaveformModel` callback adapter;
- the unused training-derived constant-sample mask helper (the active
  control-clipping mask is preserved);
- hard-coded `sample_mask.py` inspection script;
- `config/experiments/subsampling.json`, which had no consumer;
- implicit result-file migration and implicit diagnostic inference in reporting.

`engine/development_diagnostic.py` is the explicit diagnostic operation formerly
invoked during reporting. Numerical kernels, scientific fitting utilities,
independent regression fixtures and model-specific training procedures stay.

External candidates to review separately (no external files removed):

| Candidate | Why review it | Why it is not automatically obsolete |
|---|---|---|
| `report/generate_results_tables.py` | Overlaps grouped comparison export and reads historical layouts | May reproduce manuscript-specific frozen outputs |
| `make_dataset_tables.py` | Has historical waveform paths and a separate dataset-table implementation | Also supports Pico-TDC datasets, outside waveform scope |
| `report/`, `update_report/`, `presentation/` generated tables/figures | Multiple publication snapshots | Scientific outputs, not duplicate runtime implementations |

`janus_data_analysis`, `trc_converter`, `comparison_pico_scope` and `utils_fit`
have independent or shared responsibilities. Do not delete them based on their
age or filenames. Validate actual external usage before archiving scripts.

## October 2026 XAI consolidation

Removed alternative grouped temporal occlusion, coefficient-based shapelet
rankings, regression surrogate explanations and their command interfaces.
The TSHAP adapter and separate imputation training/benchmark scripts remain.
No model-training dependency was identified as exclusive to legacy XAI.
