"""PPV-only MultiRocket relevance at contributing convolution-response times."""
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np


def plot_rocket_ppv(directory, data):
    folder = Path(directory)
    t = np.asarray(data["time_ps"], dtype=float) / 1000
    a = np.asarray(data["combined_absolute_ps"], dtype=float)
    q = np.asarray(data["combined_fraction"], dtype=float)
    fig, axes = plt.subplots(2, 1, sharex=True, figsize=(10, 6))
    axes[0].plot(t, a)
    axes[0].set_ylabel("PPV absolute contribution [ps / event / sample]")
    axes[1].plot(t, q)
    axes[1].set_ylabel("Normalized localized PPV relevance")
    axes[1].set_xlabel("Time relative to alignment [ns]")
    for ax in axes:
        ax.axvspan(-1.5, 2., alpha=.12, color="orange")
        ax.grid(alpha=.15)
    fig.suptitle("ROCKET-LRP PPV pooling: response-position localization")
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(folder / f"ppv_temporal_importance.{ext}", dpi=170, bbox_inches="tight")
    plt.close(fig)

    rep = np.asarray(data["absolute_ps"], dtype=float).mean(axis=1)
    fig, ax = plt.subplots(figsize=(10, 3.5))
    for i, name in enumerate(data["representation_names"]):
        ax.plot(t, rep[i], label=str(name))
    ax.axvspan(-1.5, 2, alpha=.12, color="orange")
    ax.set(xlabel="Time relative to alignment [ns]",
           ylabel="PPV absolute contribution [ps / event / sample]",
           title="PPV attribution by MultiRocket representation")
    ax.legend()
    ax.grid(alpha=.15)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(folder / f"ppv_temporal_representations.{ext}", dpi=170, bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    axes[0].plot(t, data["combined_input_absolute_ps"])
    axes[1].plot(t, data["combined_input_fraction"])
    axes[0].set_ylabel("Input-tap PPV relevance [ps/event/sample]")
    axes[1].set_ylabel("Fraction of allocated relevance / sample")
    axes[1].set_xlabel("Time relative to alignment [ns]")
    for ax in axes:
        ax.axvspan(-1.5, 2., color="orange", alpha=.12)
        ax.grid(alpha=.15)
    fig.suptitle("MultiRocket PPV relevance allocated to dilated kernel input taps")
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(folder / f"ppv_input_importance.{ext}", dpi=170, bbox_inches="tight")
    plt.close(fig)
