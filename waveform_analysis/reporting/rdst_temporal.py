"""Compact global temporal importance figures for frozen RDST models."""
from __future__ import annotations

from pathlib import Path
import numpy as np


def plot_temporal_importance(directory, temporal):
    import matplotlib.pyplot as plt

    folder = Path(directory)
    time = np.asarray(temporal["time_ps"], dtype=float) / 1000.
    weight = np.asarray(temporal["combined_absolute_ps"], dtype=float)
    fraction = np.asarray(temporal["combined_fraction"], dtype=float)
    frequency = np.asarray(temporal["combined_match_count_per_event"], dtype=float)

    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    axes[0].plot(time, weight, label="Contribution-weighted matches")
    axes[0].set_ylabel("Absolute contribution [ps / event / sample]")
    axes[1].plot(time, fraction, label="Normalized importance")
    axes[1].set_ylabel("Fraction of total / sample")
    axes[1].set_xlabel("Time relative to alignment [ns]")
    for ax in axes:
        ax.axvspan(-1.5, 2., color="orange", alpha=.12, label="Onishi interval")
        ax.grid(alpha=.15)
        ax.legend()
    fig.suptitle("RDST global temporal importance (best-match starts)")
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(folder / f"temporal_importance.{ext}", dpi=170, bbox_inches="tight")
    plt.close(fig)

    if len(temporal["detector_names"]) == 2:
        fig, ax = plt.subplots(figsize=(10, 3.7))
        values = np.asarray(temporal["absolute_ps"], dtype=float)
        for detector, label in enumerate(temporal["detector_names"]):
            ax.plot(time, values[detector], label=str(label))
        ax.axvspan(-1.5, 2., color="orange", alpha=.12, label="Onishi interval")
        ax.set(xlabel="Time relative to alignment [ns]",
               ylabel="Absolute contribution [ps / event / sample]",
               title="Shared RDST best-match positions by detector")
        ax.grid(alpha=.15)
        ax.legend()
        fig.tight_layout()
        for ext in ("png", "pdf"):
            fig.savefig(folder / f"temporal_importance_detectors.{ext}",
                        dpi=170, bbox_inches="tight")
        plt.close(fig)
