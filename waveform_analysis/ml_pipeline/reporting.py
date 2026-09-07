from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import numpy as np

MODEL_ORDER = ("led", "cfd", "linear_svr", "cnn")
LABELS = {"led": "LED", "cfd": "CFD", "linear_svr": "Linear SVR", "cnn": "CNN"}


def read_results(run_dir: str | Path) -> list[dict[str, Any]]:
    with (Path(run_dir) / "results.csv").open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def make_plots(run_dir: str | Path, output_dir: str | Path | None = None) -> list[Path]:
    import matplotlib.pyplot as plt

    run = Path(run_dir).resolve()
    output = Path(output_dir).resolve() if output_dir else run / "plots"
    output.mkdir(parents=True, exist_ok=True)
    rows = [row for row in read_results(run) if row.get("stage") == "blind"]
    paths: list[Path] = []

    for mode in sorted({row["mode"] for row in rows}):
        subset = [row for row in rows if row["mode"] == mode]
        fig, ax = plt.subplots(figsize=(8.2, 4.6))
        for method in MODEL_ORDER:
            points = sorted(
                [row for row in subset if row["method"] == method],
                key=lambda row: float(row["voltage_V"]),
            )
            if not points:
                continue
            voltage = np.asarray([float(row["voltage_V"]) for row in points])
            ctr = np.asarray([float(row["ctr_ps"]) for row in points])
            error = np.asarray([float(row["ctr_uncertainty_ps"]) for row in points])
            ax.errorbar(voltage, ctr, yerr=error, marker="o", label=LABELS[method])
        ax.set_xlabel("Bias voltage [V]")
        ax.set_ylabel("CTR [ps]")
        ax.set_title(mode.replace("_", " "))
        ax.grid(True, alpha=0.25)
        ax.legend()
        fig.tight_layout()
        target = output / f"ctr_vs_voltage_{mode}.pdf"
        fig.savefig(target)
        plt.close(fig)
        paths.append(target)

    for artifact in sorted((run / "artifacts").glob("*/*/*_xai.npz")):
        with np.load(artifact) as data:
            time_ns = np.asarray(data["time_ps"], dtype=np.float64) / 1000.0
            importance = np.asarray(data["importance"], dtype=np.float64)
            pair = np.asarray(data["example_pair_mV"], dtype=np.float64)
        if importance.size and np.nanmax(importance) > 0:
            importance = importance / np.nanmax(importance)
        fig, (top, bottom) = plt.subplots(2, 1, figsize=(8.4, 5.8), sharex=True, height_ratios=(2, 1))
        top.plot(time_ns, pair[0], label="detector 1")
        top.plot(time_ns, pair[1], label="detector 2")
        top.set_ylabel("Signal [mV]")
        top.legend()
        top.grid(True, alpha=0.25)
        bottom.plot(time_ns, importance)
        bottom.set_xlabel("Time [ns]")
        bottom.set_ylabel("normalized importance")
        bottom.grid(True, alpha=0.25)
        fig.tight_layout()
        target = output / f"xai_{artifact.parent.parent.name}_{artifact.parent.name}_{artifact.stem.removesuffix('_xai')}.pdf"
        fig.savefig(target)
        plt.close(fig)
        paths.append(target)
    return paths
