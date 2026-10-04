from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


_DISPLAY_QUANTILES = (0.005, 0.995)
_HIST_ALPHA = 0.22


def _save(fig, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def _finite(values):
    values = np.asarray(values, float).reshape(-1)
    return values[np.isfinite(values)]


def _robust_range(series, important=(), quantiles=_DISPLAY_QUANTILES):
    arrays = [_finite(values) for values in series]
    nonempty = [values for values in arrays if values.size]
    if not nonempty:
        return None

    joined = np.concatenate(nonempty)
    qlo, qhi = np.quantile(joined, quantiles)
    lo = float(qlo)
    hi = float(qhi)

    finite_important = [float(value) for value in important if np.isfinite(value)]
    if finite_important:
        lo = min(lo, min(finite_important))
        hi = max(hi, max(finite_important))

    if not np.isfinite(lo) or not np.isfinite(hi):
        return None
    if hi <= lo:
        pad = max(1e-9, abs(lo) * 0.01, 1e-3)
        lo -= pad
        hi += pad
    else:
        pad = 0.02 * (hi - lo)
        lo -= pad
        hi += pad
    return lo, hi


def _selected(values, selection):
    values = _finite(values)
    if selection is None:
        return values

    lo, hi = selection
    mask = np.ones(values.size, dtype=bool)
    if lo is not None and np.isfinite(lo):
        mask &= values >= float(lo)
    if hi is not None and np.isfinite(hi):
        mask &= values <= float(hi)
    return values[mask]


def _hist_with_selection(ax, values, bins, label, display_range, selection, color):
    values = _finite(values)
    if not values.size:
        return

    counts, edges = np.histogram(values, bins=bins, range=display_range)
    selected = _selected(values, selection)
    selected_counts, _ = np.histogram(selected, bins=edges)

    ax.stairs(
        selected_counts,
        edges,
        fill=True,
        color=color,
        alpha=_HIST_ALPHA,
        linewidth=0,
        zorder=1,
    )
    ax.stairs(
        counts,
        edges,
        color=color,
        linewidth=1.35,
        label=f"{label} ({selected.size}/{values.size} selected)",
        zorder=2,
    )


def _legend_outside(ax):
    ax.legend(
        loc="upper left",
        bbox_to_anchor=(1.02, 1.0),
        borderaxespad=0.0,
        frameon=True,
        title="Filled histogram = selected range",
    )


def plot_photopeak(amplitudes, intervals, path, title):
    amplitudes = np.asarray(amplitudes, float)
    series = [amplitudes[:, detector] for detector in range(2)]
    important = [float(value) for interval in intervals for value in interval]
    display = _robust_range(series, important)

    fig, ax = plt.subplots(figsize=(7.2, 4.5))
    for detector in range(2):
        lo, hi = map(float, intervals[detector])
        _hist_with_selection(
            ax,
            series[detector],
            100,
            f"detector {detector + 1}",
            display,
            (lo, hi),
            f"C{detector}",
        )

    if display is not None:
        ax.set_xlim(*display)
    ax.set_xlabel("Energy-channel amplitude [mV]")
    ax.set_ylabel("Events")
    ax.set_title(title)
    _legend_outside(ax)
    _save(fig, path)


def plot_baseline_noise(rms, candidate, limits, path, title):
    rms = np.asarray(rms, float)
    candidate = np.asarray(candidate, bool)
    series = [rms[candidate, detector] for detector in range(2)]
    display = _robust_range(series, [float(value) for value in limits])

    fig, ax = plt.subplots(figsize=(7.2, 4.5))
    for detector in range(2):
        _hist_with_selection(
            ax,
            series[detector],
            80,
            f"detector {detector + 1}",
            display,
            (None, float(limits[detector])),
            f"C{detector}",
        )

    if display is not None:
        ax.set_xlim(*display)
    ax.set_xlabel("Baseline RMS [mV]")
    ax.set_ylabel("Events")
    ax.set_title(title)
    _legend_outside(ax)
    _save(fig, path)


def plot_baseline_clipping(clearance, candidate, margin_mV, path, title):
    clearance = np.asarray(clearance, float)
    candidate = np.asarray(candidate, bool)
    series = [clearance[candidate, detector] for detector in range(2)]
    display = _robust_range(series, [float(margin_mV)])

    fig, ax = plt.subplots(figsize=(7.2, 4.5))
    for detector in range(2):
        _hist_with_selection(
            ax,
            series[detector],
            80,
            f"detector {detector + 1}",
            display,
            (float(np.nextafter(float(margin_mV), np.inf)), None),
            f"C{detector}",
        )

    if display is not None:
        ax.set_xlim(*display)
    ax.set_xlabel("Baseline clearance to vertical-scale limit [mV]")
    ax.set_ylabel("Events")
    ax.set_title(title)
    _legend_outside(ax)
    _save(fig, path)


def plot_tot(hits, photo_mask, limits, path, title):
    photo_mask = np.asarray(photo_mask, bool)
    series = []
    for detector in range(2):
        values = []
        for row in np.flatnonzero(photo_mask):
            values.extend(float(hit.duration_ns) for hit in hits[row][detector])
        series.append(np.asarray(values, float))

    important = [float(value) for interval in limits for value in interval]
    display = _robust_range(series, important)

    fig, ax = plt.subplots(figsize=(7.2, 4.5))
    for detector in range(2):
        lo, hi = map(float, limits[detector])
        _hist_with_selection(
            ax,
            series[detector],
            100,
            f"detector {detector + 1}",
            display,
            (lo, hi),
            f"C{detector}",
        )

    if display is not None:
        ax.set_xlim(*display)
    ax.set_xlabel("ToT [ns]")
    ax.set_ylabel("Hits")
    ax.set_title(title)
    _legend_outside(ax)
    _save(fig, path)


def plot_materialized_event(preprocessed, family, channel_numbers, path, title):
    family = str(family)
    if family not in {"energy", "timing"}:
        raise ValueError("family must be 'energy' or 'timing'")

    waveforms = (
        preprocessed.energy_windows_mV
        if family == "energy"
        else preprocessed.timing_windows_mV
    )
    intervals = (
        preprocessed.energy_sample_interval_s
        if family == "energy"
        else preprocessed.timing_sample_interval_s
    )
    if waveforms is None or intervals is None or preprocessed.n_events <= 0:
        return None

    row = preprocessed.n_events // 2
    event_waveforms = np.asarray(waveforms[row], float)
    event_intervals = np.asarray(intervals[row], float).reshape(-1)
    channels = np.asarray(channel_numbers).reshape(-1)
    if event_waveforms.shape[0] != 2 or event_intervals.size != 2 or channels.size != 2:
        raise ValueError("materialized event plot expects two detector channels")

    before_ns = float(preprocessed.manifest["materialized_window_ns"]["before"])
    event_index = int(np.asarray(preprocessed.event_index)[row])

    fig, axes = plt.subplots(2, 1, figsize=(7.2, 6.0), sharex=False)
    for detector, ax in enumerate(np.asarray(axes).reshape(-1)):
        waveform = event_waveforms[detector]
        dt_ns = float(event_intervals[detector]) * 1e9
        trigger_offset = int(np.ceil(before_ns / dt_ns))
        time_ns = (np.arange(waveform.size, dtype=float) - trigger_offset) * dt_ns

        ax.plot(time_ns, waveform, color=f"C{detector}", linewidth=1.2)
        ax.axvline(0.0, color="0.35", linestyle=":", linewidth=1.0)
        ax.set_ylabel("Amplitude [mV]")
        ax.set_title(f"Detector {detector + 1} — channel {int(channels[detector])}")
        ax.grid(alpha=0.2)

    axes[-1].set_xlabel("Time relative to selected trigger [ns]")
    fig.suptitle(f"{title} — event {event_index}")
    _save(fig, path)
    return Path(path)


def plot_led_selection(scan, selected, path, title):
    rows = list(scan)
    x = [float(row["threshold_mV"]) for row in rows]
    y = [float(row["ctr_ps"]) for row in rows]
    fig, ax = plt.subplots()
    ax.plot(x, y, marker="o")
    ax.axvline(float(selected), linestyle="--")
    ax.set_xlabel("LED threshold [mV]")
    ax.set_ylabel("Control CTR [ps]")
    ax.set_title(title)
    _save(fig, path)
