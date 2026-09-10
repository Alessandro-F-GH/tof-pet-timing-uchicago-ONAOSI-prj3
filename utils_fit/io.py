from __future__ import annotations

import csv
import json
import os
from pathlib import Path

import numpy as np

from .histogram import FitResult

FIT_FIELDS = [
    "fit_metric",
    "ctr_definition",
    "ctr_uncertainty_definition",
    "core_metric",
    "method",
    "parameter",
    "success",
    "n_total",
    "n_selected",
    "n_valid",
    "crossing_efficiency",
    "ctr_ps",
    "ctr_error_ps",
    "center_ps",
    "coverage_fraction",
    "interval_events",
    "interval_low_ps",
    "interval_high_ps",
    "interval_width_ps",
    "gaussian_equivalent_scale",
    "core_fwhm_ps",
    "core_fwhm_error_ps",
    "core_fraction",
    "left_half_ps",
    "right_half_ps",
    "half_max_events",
    "bin_width_ps",
    "bootstrap_samples",
    "bootstrap_successful",
    "core_bootstrap_successful",
    "message",
    "edges_ps",
    "counts",
]


def write_fit_csv(path: str | Path, fit: FitResult, diagnostic_mode: str = "compact") -> None:
    del diagnostic_mode
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    row = fit.as_dict()
    row.update(
        {
            "success": int(fit.success),
            "edges_ps": json.dumps(fit.edges_ps.tolist(), separators=(",", ":")),
            "counts": json.dumps(fit.counts.tolist(), separators=(",", ":")),
        }
    )
    row.pop("n_rejected", None)
    with temporary.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIT_FIELDS)
        writer.writeheader()
        writer.writerow({field: row.get(field, "") for field in FIT_FIELDS})
    os.replace(temporary, output)


def load_fit_csv(path: str | Path) -> FitResult:
    with Path(path).open("r", encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) != 1:
        raise RuntimeError(f"Expected one CTR row in {path}")
    row = rows[0]

    def f(name: str) -> float:
        value = row.get(name, "")
        return float(value) if value not in ("", None) else float("nan")

    def i(name: str) -> int:
        value = row.get(name, "")
        return int(value) if value not in ("", None) else 0

    return FitResult(
        method=str(row["method"]),
        parameter=f("parameter"),
        success=bool(int(row["success"])),
        n_total=int(row["n_total"]),
        n_selected=int(row["n_selected"]),
        n_valid=int(row["n_valid"]),
        crossing_efficiency=f("crossing_efficiency"),
        ctr_ps=f("ctr_ps"),
        ctr_error_ps=f("ctr_error_ps"),
        center_ps=f("center_ps"),
        coverage_fraction=f("coverage_fraction"),
        interval_events=i("interval_events"),
        interval_low_ps=f("interval_low_ps"),
        interval_high_ps=f("interval_high_ps"),
        interval_width_ps=f("interval_width_ps"),
        gaussian_equivalent_scale=f("gaussian_equivalent_scale"),
        core_fwhm_ps=f("core_fwhm_ps"),
        core_fwhm_error_ps=f("core_fwhm_error_ps"),
        core_fraction=f("core_fraction"),
        left_half_ps=f("left_half_ps"),
        right_half_ps=f("right_half_ps"),
        half_max_events=f("half_max_events"),
        bin_width_ps=f("bin_width_ps"),
        bootstrap_samples=i("bootstrap_samples"),
        bootstrap_successful=i("bootstrap_successful"),
        core_bootstrap_successful=i("core_bootstrap_successful"),
        message=str(row.get("message", "")),
        edges_ps=np.asarray(json.loads(row.get("edges_ps") or "[]"), dtype=np.float64),
        counts=np.asarray(json.loads(row.get("counts") or "[]"), dtype=np.int64),
    )
