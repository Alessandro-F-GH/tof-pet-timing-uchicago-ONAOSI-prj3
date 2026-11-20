"""Publication report: stable decision-tree codes and ranked summary table."""
import csv

from waveform_analysis.reporting.latex_tables import export_ranked_ctr_summary
from waveform_analysis.reporting.model_labels import model_label


def _run(root, model, ctr, cv, candidate):
    folder = root / model
    (folder / "tables").mkdir(parents=True)
    with (folder / "tables" / "folds.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=["candidate_id", "fold_id", "led_ctr_ps"])
        writer.writeheader()
        for index, value in enumerate((125.0, 126.0, 127.0), start=1):
            writer.writerow({"candidate_id": candidate, "fold_id": index, "led_ctr_ps": value})
    return {
        "directory": folder,
        "manifest": {"model": model},
        "best": {
            "candidate_id": candidate, "folds": 3,
            "validation_ctr_mean_ps": cv, "validation_ctr_std_ps": 2.0,
        },
        "blind": {
            "ctr_ps": ctr, "rmse_ps": 60.0,
            "led_ctr_ps": 125.0, "led_rmse_ps": 63.0,
        },
        "bootstrap": {"ctr_bootstrap_std_ps": 1.3, "led_ctr_bootstrap_std_ps": 1.5},
    }


def test_decision_tree_codes():
    assert model_label("direct_decision_tree_regressor")[0] == "D-DT"
    assert model_label("shared_decision_tree_regressor")[0] == "S-DT"


def test_summary_orders_models_and_separates_led(tmp_path):
    group = [
        _run(tmp_path, "direct_decision_tree_regressor", 117.0, 118.0, "tree"),
        _run(tmp_path, "shared_decision_tree_regressor", 112.0, 113.0, "shared"),
    ]
    path = tmp_path / "ctr_summary.tex"
    export_ranked_ctr_summary(path, group, mode="energy", window="onishi")
    text = path.read_text(encoding="utf-8")
    lines = text.splitlines()
    assert lines.index(r"\midrule") < next(i for i, line in enumerate(lines) if line.startswith("LED & "))
    assert lines.index(r"\midrule", lines.index(r"\midrule") + 1) > next(
        i for i, line in enumerate(lines) if line.startswith("LED & ")
    )
    assert text.index("S-DT &") < text.index("D-DT &")
    assert "$126.0 \pm 1.0$" in text
    assert "$125.0 \pm 1.5$" in text
    assert "$112.0 \pm 1.3$" in text
    assert "Test RMSE [ps]" in text
