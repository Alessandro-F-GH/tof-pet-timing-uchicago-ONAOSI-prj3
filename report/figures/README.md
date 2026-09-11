# Report figures

Place publication-quality figures used by report/main.tex in this directory.

The LaTeX template compiles even when a referenced figure is absent: it shows a labeled placeholder box instead. Replace placeholders by adding files with the expected names, or edit the corresponding section file.

Suggested filenames used by the template include:

- pipeline_overview.pdf
- data_split.pdf
- photopeak_selection.pdf
- tot_selection.pdf
- baseline_noise_selection.pdf
- native_preprocessing_example.pdf
- led_threshold_selection.pdf
- ml_input_and_sample_mask.pdf
- model_selection_flow.pdf
- ctr_metric_example.pdf
- linear_svr_importance.pdf
- cnn_architecture.pdf
- cnn_2d_architecture.pdf
- xai_summary.pdf
- validation_model_selection.pdf
- ctr_vs_voltage.pdf
- relative_improvement_vs_voltage.pdf
- test_residual_distributions.pdf
- model_output_diagnostics.pdf
- xai_results.pdf

Prefer PDF for vector plots. Copy figures from a frozen study run rather than regenerating them manually with different settings.
