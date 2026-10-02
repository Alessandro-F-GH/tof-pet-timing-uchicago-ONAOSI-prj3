from __future__ import annotations

import json

from waveform_analysis.ml_pipeline.report_naming import compact_report_filenames


def test_compact_report_filenames_uses_directory_context(tmp_path):
    root = tmp_path / "report"
    files = [
        root / "tables" / "study_summary.csv",
        root / "tables" / "paired_model_comparisons.csv",
        root / "tables" / "best_by_formulation.csv",
        root / "tables" / "correlations" / "model_output_correlations.csv",
        root / "tables" / "correlations" / "model_output_correlation__energy__onishi.csv",
        root / "tables" / "correlations" / "model_output_correlation__energy__onishi__n_replicas.csv",
        root / "plots" / "ctr" / "ctr_comparison__energy__onishi.png",
        root / "plots" / "rmse" / "rmse_comparison__energy__onishi.png",
        root / "plots" / "tradeoffs" / "ctr_vs_time" / "ctr_vs_time__energy__onishi.png",
        root / "plots" / "window_comparison" / "ctr" / "ctr_by_window__energy.png",
        root / "plots" / "correlations" / "model_output_correlation__energy__onishi.png",
        root / "reporting_config_resolved.json",
    ]
    for path in files:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("x", encoding="utf-8")
    (root / "manifest.json").write_text(
        json.dumps({"reporting_config": "reporting_config_resolved.json", "plots_and_tables": []}),
        encoding="utf-8",
    )

    compact_report_filenames(root)

    assert (root / "tables" / "summary.csv").is_file()
    assert (root / "tables" / "paired.csv").is_file()
    assert (root / "tables" / "best.csv").is_file()
    assert (root / "tables" / "correlations" / "pairs.csv").is_file()
    assert (root / "tables" / "correlations" / "energy__onishi.csv").is_file()
    assert (root / "tables" / "correlations" / "energy__onishi__n.csv").is_file()
    assert (root / "plots" / "ctr" / "energy__onishi.png").is_file()
    assert (root / "plots" / "rmse" / "energy__onishi.png").is_file()
    assert (root / "plots" / "tradeoffs" / "ctr_vs_time" / "energy__onishi.png").is_file()
    assert (root / "plots" / "window_comparison" / "ctr" / "energy.png").is_file()
    assert (root / "plots" / "correlations" / "energy__onishi.png").is_file()
    assert (root / "reporting.json").is_file()

    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["reporting_config"] == "reporting.json"
    assert "plots/ctr/energy__onishi.png" in manifest["plots_and_tables"]
    assert "tables/correlations/energy__onishi__n.csv" in manifest["plots_and_tables"]
    assert "filename_policy" in manifest
