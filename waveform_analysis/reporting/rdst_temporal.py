"""Publication-ready temporal RDST explanation plot from saved contribution maps."""
from __future__ import annotations

import numpy as np


def plot_temporal_importance(directory, temporal):
    import matplotlib.pyplot as plt
    from pathlib import Path

    folder = Path(directory)
    time = np.asarray(temporal["time_ps"], dtype=float) / 1000.0
    names = ("All blind events", "Low (2-10%)", "Central (46-54%)", "High (90-98%)")
    importance = np.asarray(temporal["combined_absolute_ps"], dtype=float)
    signed = np.asarray(temporal["combined_signed_ps"], dtype=float)
    distribution = np.asarray(temporal["combined_fraction"], dtype=float)
    counts = np.asarray(temporal["combined_match_count_per_event"], dtype=float)

    fig, axes = plt.subplots(3, 1, figsize=(11, 9), sharex=True)
    for g, name in enumerate(names):
        axes[0].plot(time, importance[g], label=name, linewidth=1.2)
        axes[1].plot(time, distribution[g], label=name, linewidth=1.2)
    axes[0].set(ylabel="Absolute contributions [ps / event / sample]",
                title="RDST contribution-weighted best-match starts")
    axes[1].set(ylabel="Fraction of total absolute contribution / sample",
                title="Normalized temporal distribution")
    axes[2].plot(time, counts[0], color="gray", label="Match count (all blind)")
    axes[2].plot(time, signed[0], label="Signed net contribution (all blind)")
    axes[2].set(ylabel="Matches/event or signed ps", xlabel="Time relative to alignment [ns]",
                title="Unweighted match density and signed contributions")
    for ax in axes:
        ax.axvspan(-1.5, 2.0, color="orange", alpha=.10, label="Onishi interval")
        ax.axvline(-1.5, color="gray", linestyle=":", linewidth=.8)
        ax.axvline(2., color="gray", linestyle=":", linewidth=.8)
        ax.legend(fontsize=8, loc="best")
        ax.grid(True, alpha=.15)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(folder / f"temporal_importance.{ext}", dpi=170, bbox_inches="tight")
    plt.close(fig)

    if len(temporal["detector_names"]) == 2:
        fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
        values = np.asarray(temporal["absolute_ps"], dtype=float)
        for d, ax in enumerate(axes):
            for g, name in enumerate(names):
                ax.plot(time, values[g, d], linewidth=1.1, label=name)
            ax.axvspan(-1.5, 2.0, color="orange", alpha=.12)
            ax.set(ylabel="Absolute contributions [ps / event / sample]",
                   title=f"Shared RDST: {temporal['detector_names'][d]} matches")
            ax.legend(fontsize=8)
        axes[-1].set_xlabel("Time relative to alignment [ns]")
        fig.tight_layout()
        for ext in ("png", "pdf"):
            fig.savefig(folder / f"temporal_importance_detectors.{ext}", dpi=170, bbox_inches="tight")
        plt.close(fig)
