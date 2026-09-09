from __future__ import annotations

from pathlib import Path

import numpy as np

from .timing import anchor_grid, family_arrays, led_grid


def _output_path(directory: Path, prefix: str, source: str | Path, family: str) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    return directory / f"{prefix}_{Path(source).stem}_{family}.pdf"


def plot_missing_led_example(
    data,
    family: str,
    event_row: int,
    threshold_mV: float,
    directory: Path,
) -> Path:
    """Plot one event that lacks an LED crossing on at least one detector."""
    import matplotlib.pyplot as plt

    waves, _starts, intervals, rising_start, rising_stop = family_arrays(data, family)
    crossings = led_grid(
        data,
        family,
        np.asarray([event_row], dtype=np.int64),
        np.asarray([threshold_mV], dtype=np.float64),
    )[0, :, 0]
    event_index = int(np.asarray(data.event_index)[event_row])
    fig, axes = plt.subplots(2, 1, figsize=(9.0, 6.2), squeeze=False)
    for detector in range(2):
        ax = axes[detector, 0]
        signal = np.asarray(waves[event_row, detector], dtype=np.float64)
        a = int(rising_start[event_row, detector])
        b = int(rising_stop[event_row, detector])
        dt_ns = float(intervals[event_row, detector]) * 1e9
        time_ns = (np.arange(signal.size, dtype=np.float64) - a) * dt_ns
        ax.plot(time_ns, signal)
        ax.axhline(float(threshold_mV), linestyle="--", label=f"LED {threshold_mV:g} mV")
        if 0 <= a < b < signal.size:
            ax.axvspan(0.0, (b - a) * dt_ns, alpha=0.12, label="LED search interval")
        status = "crossing found" if np.isfinite(crossings[detector]) else "NO LED CROSSING"
        ax.set_title(f"Detector {detector + 1}: {status}")
        ax.set_xlabel("Time relative to LED-search start [ns]")
        ax.set_ylabel("Signal [mV]")
        ax.grid(True, alpha=0.2)
        ax.legend()
    fig.suptitle(f"Discarded event #{event_index} · {family} · missing LED coverage")
    fig.tight_layout()
    target = _output_path(directory, "missing_led", data.manifest["source"], family)
    fig.savefig(target, bbox_inches="tight")
    plt.close(fig)
    return target


def plot_ml_window_exceeds_example(
    data,
    family: str,
    event_row: int,
    threshold_mV: float,
    window_ns: dict,
    directory: Path,
) -> Path:
    """Plot one source waveform whose requested ML window exceeds stored samples."""
    import matplotlib.pyplot as plt

    waves, _starts, intervals, _rising_start, _rising_stop = family_arrays(data, family)
    anchor_index, _anchor_time = anchor_grid(data, family, float(threshold_mV))
    event_index = int(np.asarray(data.event_index)[event_row])
    requested_start = float(window_ns["start"])
    requested_end = float(window_ns["end"])
    fig, axes = plt.subplots(2, 1, figsize=(9.0, 6.2), squeeze=False)
    for detector in range(2):
        ax = axes[detector, 0]
        signal = np.asarray(waves[event_row, detector], dtype=np.float64)
        anchor = int(anchor_index[event_row, detector])
        dt_ns = float(intervals[event_row, detector]) * 1e9
        time_ns = (np.arange(signal.size, dtype=np.float64) - anchor) * dt_ns
        ax.plot(time_ns, signal, label="materialized waveform")
        ax.axvline(0.0, linestyle="--", label="LED anchor")
        ax.axvspan(requested_start, requested_end, alpha=0.12, label="requested ML window")
        available_start = float(time_ns[0])
        available_end = float(time_ns[-1])
        ax.set_xlim(min(available_start, requested_start), max(available_end, requested_end))
        ax.set_title(
            f"Detector {detector + 1}: available [{available_start:.2f}, {available_end:.2f}] ns"
        )
        ax.set_xlabel("Time relative to LED anchor [ns]")
        ax.set_ylabel("Signal [mV]")
        ax.grid(True, alpha=0.2)
        ax.legend()
    fig.suptitle(
        f"Discarded event #{event_index} · {family} · ML window "
        f"[{requested_start:g}, {requested_end:g}] ns exceeds materialized waveform"
    )
    fig.tight_layout()
    target = _output_path(directory, "ml_window_exceeds_materialized", data.manifest["source"], family)
    fig.savefig(target, bbox_inches="tight")
    plt.close(fig)
    return target
