from __future__ import annotations

import csv
import json
import os
from pathlib import Path

import numpy as np

from .histogram import FitResult

FIT_FIELDS = [
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
    "left_half_ps",
    "right_half_ps",
    "half_max_events",
    "bin_width_ps",
    "bootstrap_samples",
    "bootstrap_successful",
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
        left_half_ps=f("left_half_ps"),
        right_half_ps=f("right_half_ps"),
        half_max_events=f("half_max_events"),
        bin_width_ps=f("bin_width_ps"),
        bootstrap_samples=int(row.get("bootstrap_samples") or 0),
        bootstrap_successful=int(row.get("bootstrap_successful") or 0),
        message=str(row.get("message", "")),
        edges_ps=np.asarray(json.loads(row.get("edges_ps") or "[]"), dtype=np.float64),
        counts=np.asarray(json.loads(row.get("counts") or "[]"), dtype=np.int64),
    )
