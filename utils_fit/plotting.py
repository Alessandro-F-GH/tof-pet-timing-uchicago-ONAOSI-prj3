from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from .histogram import FitResult


def plot_ctr_histogram(
    result: FitResult | None,
    path: str | Path,
    *,
    dpi: int = 180,
    title: str | None = None,
    xlabel: str = "Time difference [ps]",
) -> None:
    """Plot a NEMA CTR histogram with peak and half-maximum crossings."""
    if result is None or not result.success:
        return

    fig, ax = plt.subplots(figsize=(9.2, 6.1))
    if result.edges_ps.size >= 2 and result.counts.size == result.edges_ps.size - 1:
        edges = np.asarray(result.edges_ps, dtype=np.float64)
        counts = np.asarray(result.counts, dtype=np.float64)
        ax.bar(edges[:-1], counts, width=np.diff(edges), align="edge", alpha=0.65)

    if np.isfinite(result.peak_height) and np.isfinite(result.center_ps):
        ax.plot(result.center_ps, result.peak_height, marker="o", linestyle="none", label="NEMA peak")
    if np.isfinite(result.half_max_events):
        ax.hlines(
            result.half_max_events,
            result.left_half_ps,
            result.right_half_ps,
            linestyles="--",
            linewidth=1.3,
            label="NEMA half maximum",
        )
    if np.isfinite(result.left_half_ps):
        ax.axvline(result.left_half_ps, ls=":", lw=1.1)
    if np.isfinite(result.right_half_ps):
        ax.axvline(result.right_half_ps, ls=":", lw=1.1)

    error_text = f" ± {result.ctr_error_ps:.1f}" if np.isfinite(result.ctr_error_ps) else ""
    ax.plot([], [], label=f"CTR = {result.ctr_ps:.1f}{error_text} ps")

    ax.set_xlabel(xlabel)
    ax.set_ylabel("Events / bin")
    ax.set_title(title or f"{result.method} — parameter {result.parameter:g} — NEMA CTR")
    ax.grid(alpha=0.2)
    ax.legend(loc="upper right")

    selected_fraction = 100.0 * result.n_selected / result.n_total if result.n_total else 0.0
    ax.text(
        0.02,
        0.96,
        (
            f"Selected: {result.n_selected} ({selected_fraction:.1f}%)\n"
            f"Finite timing pairs: {result.n_valid}\n"
            f"Bin width: {result.histogram_bin_width_ps:.2f} ps\n"
            f"Histogram bins: {result.histogram_bins}\n"
            f"Peak: {result.peak_height:.2f} events\n"
            f"Bootstrap: {result.bootstrap_successful}/{result.bootstrap_samples}"
        ),
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=10.0,
        bbox={"boxstyle": "round,pad=0.35", "facecolor": "white", "alpha": 0.9},
    )
    fig.tight_layout()
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=int(dpi), bbox_inches="tight")
    plt.close(fig)


def plot_ctr_comparison(
    pico_rows: list[dict[str, Any]],
    scope_rows: list[dict[str, Any]],
    path: str | Path,
    *,
    dpi: int = 180,
    title: str = "Pico-TDC vs oscilloscope LED",
) -> None:
    fig, ax = plt.subplots(figsize=(9.2, 6.0))
    if scope_rows:
        ordered = sorted(scope_rows, key=lambda row: float(row["voltage_V"]))
        x = np.asarray([float(r["voltage_V"]) for r in ordered])
        y = np.asarray([float(r["ctr_ps"]) for r in ordered])
        e = np.asarray([float(r.get("ctr_error_ps", np.nan)) for r in ordered])
        if np.any(np.isfinite(e)):
            ax.errorbar(x, y, yerr=np.where(np.isfinite(e), e, 0.0), marker="o", capsize=3, linewidth=1.8, label=str(ordered[0]["series_label"]))
        else:
            ax.plot(x, y, marker="o", linewidth=1.8, label=str(ordered[0]["series_label"]))

    groups: dict[str, list[dict[str, Any]]] = {}
    for row in pico_rows:
        threshold = row.get("timing_threshold_mV", "")
        key = "unknown" if threshold in ("", None) or not np.isfinite(float(threshold)) else f"{float(threshold):g}"
        groups.setdefault(key, []).append(row)

    for threshold, rows in sorted(groups.items()):
        ordered = sorted(rows, key=lambda row: float(row["voltage_V"]))
        x = np.asarray([float(r["voltage_V"]) for r in ordered])
        y = np.asarray([float(r["ctr_ps"]) for r in ordered])
        e = np.asarray([float(r.get("ctr_error_ps", np.nan)) for r in ordered])
        label = "Pico-TDC timing channels" if threshold == "unknown" else f"Pico-TDC timing channels · T_th={threshold} mV"
        if np.any(np.isfinite(e)):
            ax.errorbar(x, y, yerr=np.where(np.isfinite(e), e, 0.0), marker="s", capsize=3, linewidth=1.6, label=label)
        else:
            ax.plot(x, y, marker="s", linewidth=1.6, label=label)

    ax.set_xlabel("Bias voltage [V]")
    ax.set_ylabel("CTR [ps]")
    ax.set_title(title)
    ax.grid(alpha=0.2)
    ax.legend(loc="best")
    fig.tight_layout()
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=int(dpi), bbox_inches="tight")
    plt.close(fig)
