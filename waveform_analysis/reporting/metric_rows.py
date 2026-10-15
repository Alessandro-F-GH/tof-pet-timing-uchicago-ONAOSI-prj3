"""Rank persisted metrics for both batch and cross-batch reports."""
from typing import Any
import numpy as np
from waveform_analysis.reporting.model_labels import model_label


def blind_metric_rows(group: list[dict[str, Any]], metric: str) -> list[dict[str, Any]]:
    """Rank stored blind metrics for display only; codes remain stable."""
    rows = []
    for run in group:
        name = run["manifest"]["model"]
        code, display_name = model_label(name)
        rows.append({
            "model": name, "code": code, "display_name": display_name,
            "formulation": run["manifest"]["estimator_formulation"],
            f"{metric}_ps": run["blind"][f"{metric}_ps"],
            f"{metric}_std_ps": run["bootstrap"][f"{metric}_bootstrap_std_ps"],
        })
    return sorted(
        rows,
        key=lambda r: (
            not np.isfinite(r[f"{metric}_ps"]),
            r[f"{metric}_ps"] if np.isfinite(r[f"{metric}_ps"]) else np.inf,
            r["model"],
        ),
    )
