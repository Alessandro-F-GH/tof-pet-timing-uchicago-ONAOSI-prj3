"""Persist an illustrative excluded Blind waveform for fixed MT reference.

The diagnostics use the same crossing times computed during the MT pickoff.
They do not change event selection or use blind targets to tune the method.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg", force=False)
import matplotlib.pyplot as plt
import numpy as np

from waveform_analysis.core.io import atomic_json
from waveform_analysis.data.view import model_target


def _failure_reasons(raw, target, diagnostics):
    """Exclusive, prioritized reasons for MT invalidity on each event."""
    residual = np.asarray(raw, dtype=float)
    target = np.asarray(target, dtype=float)
    n = residual.size
    if target.shape != (n,):
        raise ValueError("MT diagnostic target and correction lengths disagree")
    crossings = np.asarray(diagnostics["crossing_times_ps"], float)
    led = np.asarray(diagnostics["led_crossing_ps"], float)
    t0 = np.asarray(diagnostics["intercept_ps"], float)
    if crossings.shape[0:2] != (n, 2) or led.shape != (n, 2) or t0.shape[0:2] != (n, 2):
        raise ValueError("MT crossing diagnostic dimensions disagree")

    # Categorize the first failing condition only, so counts add up to total
    # excluded events. Preserve a separate distinction for target validity.
    reasons = np.full(n, "valid", dtype="<U32")
    reasons[~np.isfinite(target)] = "invalid_target"
    pending = reasons == "valid"
    missing = ~np.isfinite(crossings).all(axis=(1, 2))
    reasons[pending & missing] = "missing_threshold_crossing"
    pending = reasons == "valid"
    monotonic = np.all(np.diff(crossings, axis=2) > 0, axis=(1, 2))
    reasons[pending & ~monotonic] = "nonmonotonic_crossings"
    pending = reasons == "valid"
    reasons[pending & ~np.isfinite(led).all(axis=1)] = "missing_led_crossing"
    pending = reasons == "valid"
    reasons[pending & ~np.isfinite(t0).all(axis=(1, 2))] = "invalid_linear_fit"
    pending = reasons == "valid"
    reasons[pending & ~np.isfinite(residual)] = "invalid_correction"
    if np.any((reasons == "valid") != (np.isfinite(residual) & np.isfinite(target))):
        raise AssertionError("MT coverage reasons do not match prediction validity")
    return reasons


def _plot_example(path, dataset, event_idx, diagnostics, options, reason):
    grid = np.asarray(dataset.energy_time_ps, np.float64) / 1000.
    transform = dataset.energy_transform
    # The stored normalized waveform is transformed back into physical mV.
    # Thresholds are baseline-relative, so display the baseline-subtracted
    # physical waveform, not an arbitrary normalized amplitude.
    scale = np.asarray(transform.maximum - transform.minimum, np.float64).reshape(2)
    waves = np.asarray(dataset.energy_windows[event_idx], np.float64)
    baseline_idx = np.flatnonzero(
        (grid >= float(options["baseline_window_ns"][0]) - 1e-9)
        & (grid <= float(options["baseline_window_ns"][1]) + 1e-9)
    )
    if len(baseline_idx) < 2:
        raise ValueError("Baseline window too short for MT coverage plot")
    physical = waves * scale[:, None]
    physical -= physical[:, baseline_idx].mean(axis=1)[:, None]
    thresholds = np.asarray(diagnostics["crossing_levels_mV"], np.float64)
    times = np.asarray(diagnostics["crossing_times_ps"], np.float64)[event_idx] / 1000.
    led_times = np.asarray(diagnostics["led_crossing_ps"], np.float64)[event_idx] / 1000.
    t0 = np.asarray(diagnostics["intercept_ps"], np.float64)[event_idx, :, 0] / 1000.

    fig, axes = plt.subplots(2, 1, figsize=(9, 7), sharex=True, constrained_layout=True)
    palette = plt.get_cmap("tab10")
    for detector, ax in enumerate(axes):
        ax.plot(grid, physical[detector], color="#2874A6", linewidth=1.5,
                label="Baseline-subtracted signal")
        ax.axvspan(*options["baseline_window_ns"], color="#B0B0B0",
                   alpha=.20, label="Baseline window")
        ax.axvspan(*options["crossing_window_ns"], color="#66BB6A",
                   alpha=.11, label="Crossing search window")
        for k, threshold in enumerate(thresholds):
            color = palette(k % 10)
            ax.axhline(float(threshold), color=color, linestyle=":", alpha=.7,
                       label=f"{threshold:g} mV")
            if np.isfinite(times[detector, k]):
                ax.scatter(times[detector, k], threshold, color=color,
                           s=45, edgecolor="black", linewidth=.3, zorder=4)
        if np.isfinite(led_times[detector]):
            ax.axvline(led_times[detector], color="#7B3294", linestyle="--",
                       linewidth=1, label="Local LED crossing")
        if np.isfinite(t0[detector]):
            ax.axvline(t0[detector], color="#D95F02", linestyle="-.",
                       linewidth=1, label="MT extrapolated t0")
        ax.set_ylabel(f"Detector {detector+1} [mV]")
        ax.grid(alpha=.2)
        ax.legend(fontsize=7, ncol=3, loc="best")
    axes[-1].set_xlabel("Time relative to prepared LED anchor [ns]")
    fig.suptitle(
        f"Blind MT excluded event {int(dataset.event_index[event_idx])}  |  "
        f"reason: {reason}", fontsize=11)
    path = Path(path)
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def save_coverage_diagnostic(destination, dataset, raw, diagnostics, options, *, role="blind"):
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    target = np.asarray(model_target(dataset, "energy_to_energy"), np.float64)
    reasons = _failure_reasons(raw, target, diagnostics)
    invalid = np.flatnonzero(reasons != "valid")
    summary = {
        "role": role, "total_events": int(reasons.size),
        "valid_events": int(np.count_nonzero(reasons == "valid")),
        "excluded_events": int(invalid.size),
        "failure_counts": {
            reason: int(np.count_nonzero(reasons == reason))
            for reason in (
                "invalid_target", "missing_threshold_crossing",
                "nonmonotonic_crossings", "missing_led_crossing",
                "invalid_linear_fit", "invalid_correction"
            )
        },
        "example": None,
    }
    plot = destination / "coverage_failure_example.png"
    if invalid.size:
        index = int(invalid[0])
        reason = str(reasons[index])
        summary["example"] = {
            "event_id": int(dataset.event_index[index]),
            "index": index, "reason": reason, "plot": plot.name,
        }
        _plot_example(plot, dataset, index, diagnostics, options, reason)
    else:
        plot.unlink(missing_ok=True)
    atomic_json(destination / "coverage_failure_summary.json", summary)
    return summary
