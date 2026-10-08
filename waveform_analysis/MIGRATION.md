# Complete module migration map

All paths are relative to `waveform_analysis/`. Old paths retain compatibility aliases.

| Current path | New path | Responsibility |
| --- | --- | --- |
| `ml_pipeline/models/__init__.py` | `models/__init__.py` | Architectures, feature transforms and model contracts. |
| `ml_pipeline/config.py` | `core/config.py` | Configuration and shared infrastructure. |
| `ml_pipeline/common.py` | `core/io.py` | Configuration and shared infrastructure. |
| `ml_pipeline/progress.py` | `core/progress.py` | Configuration and shared infrastructure. |
| `ml_pipeline/data.py` | `data/preprocessing.py` | Acquisition I/O, datasets, splits, caches and persistence. |
| `ml_pipeline/dataset.py` | `data/dataset.py` | Acquisition I/O, datasets, splits, caches and persistence. |
| `ml_pipeline/energy_io.py` | `data/root_io.py` | Acquisition I/O, datasets, splits, caches and persistence. |
| `ml_pipeline/feature_cache.py` | `data/feature_cache.py` | Acquisition I/O, datasets, splits, caches and persistence. |
| `ml_pipeline/prepared_data.py` | `data/preparation.py` | Acquisition I/O, datasets, splits, caches and persistence. |
| `ml_pipeline/shared_artifacts.py` | `data/shared_artifacts.py` | Acquisition I/O, datasets, splits, caches and persistence. |
| `ml_pipeline/splits.py` | `data/splits.py` | Acquisition I/O, datasets, splits, caches and persistence. |
| `ml_pipeline/storage.py` | `data/storage.py` | Acquisition I/O, datasets, splits, caches and persistence. |
| `ml_pipeline/view.py` | `data/view.py` | Acquisition I/O, datasets, splits, caches and persistence. |
| `ml_pipeline/timing.py` | `data/timing_adapter.py` | Acquisition I/O, datasets, splits, caches and persistence. |
| `ml_pipeline/sample_mask.py` | `signal/sample_mask.py` | Pure numerical processing. |
| `ml_pipeline/batch.py` | `engine/batch.py` | Training and scientific orchestration. |
| `ml_pipeline/control_preprocessing.py` | `engine/control_preprocessing.py` | Training and scientific orchestration. |
| `ml_pipeline/event_selection.py` | `engine/event_selection.py` | Training and scientific orchestration. |
| `ml_pipeline/preprocessing.py` | `engine/preprocessing.py` | Training and scientific orchestration. |
| `ml_pipeline/search.py` | `engine/search.py` | Training and scientific orchestration. |
| `ml_pipeline/study.py` | `engine/study.py` | Training and scientific orchestration. |
| `ml_pipeline/train.py` | `engine/train.py` | Training and scientific orchestration. |
| `ml_pipeline/validation.py` | `engine/validation.py` | Training and scientific orchestration. |
| `ml_pipeline/xai.py` | `engine/xai.py` | Training and scientific orchestration. |
| `ml_pipeline/plotting.py` | `reporting/plotting.py` | Plots, metrics and persisted-result reporting. |
| `ml_pipeline/preprocessing_plots.py` | `reporting/preprocessing_plots.py` | Plots, metrics and persisted-result reporting. |
| `ml_pipeline/stats.py` | `reporting/stats.py` | Plots, metrics and persisted-result reporting. |
| `ml_pipeline/report.py` | `reporting/report.py` | Plots, metrics and persisted-result reporting. |
| `ml_pipeline/report_engine.py` | `reporting/report_engine.py` | Plots, metrics and persisted-result reporting. |
| `ml_pipeline/postprocess.py` | `reporting/postprocess.py` | Plots, metrics and persisted-result reporting. |
| `ml_pipeline/models/shared_linear_ridge.py` | `models/linear/shared_linear_ridge.py` | Architectures, feature transforms and model contracts. |
| `ml_pipeline/models/_torch_common.py` | `models/torch_runtime.py` | Architectures, feature transforms and model contracts. |
| `ml_pipeline/models/shared_cnn1d.py` | `models/neural/shared_cnn1d.py` | Architectures, feature transforms and model contracts. |
| `ml_pipeline/models/locally_connected_mlp.py` | `models/neural/locally_connected_mlp.py` | Architectures, feature transforms and model contracts. |
| `ml_pipeline/models/spec.py` | `models/spec.py` | Architectures, feature transforms and model contracts. |
| `ml_pipeline/models/direct_mlp.py` | `models/neural/direct_mlp.py` | Architectures, feature transforms and model contracts. |
| `ml_pipeline/models/direct_minirocket.py` | `models/kernel/direct_minirocket.py` | Architectures, feature transforms and model contracts. |
| `ml_pipeline/models/shared_minirocket.py` | `models/kernel/shared_minirocket.py` | Architectures, feature transforms and model contracts. |
| `ml_pipeline/models/independent_cnn1d.py` | `models/neural/independent_cnn1d.py` | Architectures, feature transforms and model contracts. |
| `ml_pipeline/models/onishi_cnn.py` | `models/neural/onishi_cnn.py` | Architectures, feature transforms and model contracts. |
| `ml_pipeline/models/registry.py` | `models/registry.py` | Architectures, feature transforms and model contracts. |
| `ml_pipeline/models/_linear_ridge_common.py` | `models/linear/linear_ridge_common.py` | Architectures, feature transforms and model contracts. |
| `ml_pipeline/models/antisymmetric_mlp.py` | `models/neural/antisymmetric_mlp.py` | Architectures, feature transforms and model contracts. |
| `ml_pipeline/models/_cnn1d_common.py` | `models/neural/cnn1d_common.py` | Architectures, feature transforms and model contracts. |
| `ml_pipeline/models/_mlp_common.py` | `engine/neural_training.py` | Training and scientific orchestration. |
| `ml_pipeline/models/direct_linear_ridge.py` | `models/linear/direct_linear_ridge.py` | Architectures, feature transforms and model contracts. |
| `utils/peak.py` | `signal/peak.py` | Pure numerical processing. |
| `utils/photopeak.py` | `signal/photopeak.py` | Pure numerical processing. |

Further extractions: timing kernels to `signal/timing.py`, baseline reductions to `signal/baseline.py`, ADC/hit/selection kernels to `signal/pulses.py`, dense scorer components to `models/neural/dense.py`, the Onishi trainer to `engine/onishi_training.py`, and shared checkpoints/gradient attribution to `models/torch_runtime.py`.
