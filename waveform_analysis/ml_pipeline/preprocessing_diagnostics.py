from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from utils.signal import INVALID_TIME_FS

_FS_PER_NS = 1.0e6


def _channel_numbers(cache: Any, family: str) -> list[int]:
    key = f"{family}_channels_one_based"
    values = cache.manifest.get(key, []) if isinstance(cache.manifest, dict) else []
    if isinstance(values, (list, tuple)) and len(values) >= 2:
        return [int(values[0]), int(values[1])]
    if family == "energy":
        values = cache.manifest.get("energy_channels_one_based", [])
        if isinstance(values, (list, tuple)) and len(values) >= 2:
            return [int(values[0]), int(values[1])]
    return [1, 2]


def _save(fig: plt.Figure, path: Path, dpi: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def _photopeak_plot(cache: Any, summary: dict[str, Any], path: Path, dpi: int) -> None:
    amplitudes = np.asarray(cache.amplitude_mV, dtype=np.float64)
    rows = list(summary.get("photopeak", []) or [])
    channels = _channel_numbers(cache, "energy")
    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.8), squeeze=False)
    for position, axis in enumerate(axes[0]):
        values = amplitudes[:, position]
        finite = values[np.isfinite(values)]
        axis.hist(finite, bins=180, histtype="step", linewidth=1.4)
        row = rows[position] if position < len(rows) else {}
        low = row.get("selection_low_mV")
        high = row.get("selection_high_mV")
        if low is not None and high is not None and np.isfinite(float(low)) and np.isfinite(float(high)):
            axis.axvspan(float(low), float(high), alpha=0.2, label="Selected photopeak")
            axis.axvline(float(low), linestyle="--", linewidth=1.0)
            axis.axvline(float(high), linestyle="--", linewidth=1.0)
            axis.legend(loc="best")
        axis.set_title(f"Energy channel {channels[position]}")
        axis.set_xlabel("Amplitude [mV]")
        axis.set_ylabel("Events")
        axis.grid(alpha=0.25)
    fig.suptitle("Energy photopeak selection")
    _save(fig, path, dpi)


def _baseline_rmse_plot(cache: Any, summary: dict[str, Any], path: Path, dpi: int) -> None:
    rmse = np.asarray(cache.noise_rms_mV, dtype=np.float64)
    rows = list((summary.get("baseline_rmse_filter", {}) or {}).get("channels", []) or [])
    channels = _channel_numbers(cache, "energy")
    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.8), squeeze=False)
    for position, axis in enumerate(axes[0]):
        values = rmse[:, position]
        finite = values[np.isfinite(values)]
        axis.hist(finite, bins=180, histtype="step", linewidth=1.4)
        row = rows[position] if position < len(rows) else {}
        limit = row.get("upper_limit_mV")
        if limit is not None and np.isfinite(float(limit)):
            axis.axvline(float(limit), linestyle="--", linewidth=1.5, label=f"RMSE max = {float(limit):.3g} mV")
            axis.legend(loc="best")
        axis.set_title(f"Energy channel {channels[position]}")
        axis.set_xlabel("Baseline RMSE [mV]")
        axis.set_ylabel("Events")
        axis.grid(alpha=0.25)
    fig.suptitle("Baseline-noise RMSE")
    _save(fig, path, dpi)


def _trigger_rows(summary: dict[str, Any]) -> dict[tuple[str, int], dict[str, Any]]:
    output: dict[tuple[str, int], dict[str, Any]] = {}
    for row in list((summary.get("trigger_time_filter", {}) or {}).get("channels", []) or []):
        try:
            output[(str(row["family"]), int(row["position"]))] = row
        except (KeyError, TypeError, ValueError):
            continue
    return output


def _trigger_time_plot(
    cache: Any,
    summary: dict[str, Any],
    active_families: tuple[str, ...],
    path: Path,
    dpi: int,
) -> None:
    families = [family for family in active_families if family in {"energy", "timing"}]
    if not families:
        families = ["energy"]
    rows = _trigger_rows(summary)
    fig, axes = plt.subplots(len(families), 2, figsize=(12.0, 4.2 * len(families)), squeeze=False)
    for family_index, family in enumerate(families):
        anchors = getattr(cache, f"{family}_window_anchor_time_fs", None)
        channels = _channel_numbers(cache, family)
        for position in range(2):
            axis = axes[family_index, position]
            if anchors is None:
                axis.text(0.5, 0.5, f"No {family} trigger data", transform=axis.transAxes, ha="center", va="center")
                axis.set_axis_off()
                continue
            values_fs = np.asarray(anchors[:, position], dtype=np.int64)
            valid = values_fs != int(INVALID_TIME_FS)
            values_ns = values_fs[valid].astype(np.float64) / _FS_PER_NS
            if values_ns.size:
                axis.hist(values_ns, bins=180, histtype="step", linewidth=1.4)
            row = rows.get((family, position), {})
            low = row.get("lower_ns")
            high = row.get("upper_ns")
            if low is not None and high is not None and np.isfinite(float(low)) and np.isfinite(float(high)):
                axis.axvspan(float(low), float(high), alpha=0.2, label="Accepted interval")
                axis.axvline(float(low), linestyle="--", linewidth=1.0)
                axis.axvline(float(high), linestyle="--", linewidth=1.0)
                axis.legend(loc="best")
            axis.set_title(f"{family.capitalize()} channel {channels[position]}")
            axis.set_xlabel("Absolute trigger time from oscilloscope acquisition origin [ns]")
            axis.set_ylabel("Events")
            axis.grid(alpha=0.25)
    fig.suptitle("Search-trigger absolute-time distributions")
    _save(fig, path, dpi)


def _rejected_noise_example(cache: Any, summary: dict[str, Any], path: Path, dpi: int) -> None:
    rmse = np.asarray(cache.noise_rms_mV, dtype=np.float64)
    windows = cache.windows_mV
    rows = list((summary.get("baseline_rmse_filter", {}) or {}).get("channels", []) or [])
    limits = np.asarray(
        [float(row.get("upper_limit_mV", np.nan)) for row in rows[:2]],
        dtype=np.float64,
    )
    candidate = np.zeros(rmse.shape[0], dtype=bool)
    if limits.size == 2 and np.all(np.isfinite(limits)):
        candidate = np.any(rmse > limits.reshape(1, 2), axis=1)

    # Choose the strongest rejected event first, but inspect waveform rows one at
    # a time so a large memmapped cache is never materialized in RAM.
    event: int | None = None
    event_waveforms: np.ndarray | None = None
    indices = np.flatnonzero(candidate)
    if indices.size:
        score = np.nanmax(
            rmse[indices] / np.maximum(limits.reshape(1, 2), 1e-12),
            axis=1,
        )
        for local in np.argsort(score)[::-1]:
            row_index = int(indices[int(local)])
            block = np.asarray(windows[row_index], dtype=np.float64)
            if block.ndim == 2 and block.shape[0] >= 2 and np.all(np.isfinite(block)):
                event = row_index
                event_waveforms = block
                break

    fig, axes = plt.subplots(2, 1, figsize=(10.5, 6.5), squeeze=False)
    if event is not None and event_waveforms is not None:
        time_ns = np.asarray(cache.relative_time_ps, dtype=np.float64) / 1000.0
        channels = _channel_numbers(cache, "energy")
        for position in range(2):
            axis = axes[position, 0]
            axis.plot(
                time_ns,
                event_waveforms[position],
                linewidth=1.1,
            )
            threshold_text = ""
            if limits.size == 2 and np.isfinite(limits[position]):
                threshold_text = f"; limit={limits[position]:.3g} mV"
            axis.set_title(
                f"Energy channel {channels[position]} | "
                f"baseline RMSE={rmse[event, position]:.3g} mV{threshold_text}"
            )
            axis.set_xlabel("Time relative to native search-trigger sample [ns]")
            axis.set_ylabel("Voltage [mV]")
            axis.grid(alpha=0.25)
        fig.suptitle(
            f"Example event rejected for high baseline noise | cache row {event}"
        )
    else:
        for axis in axes[:, 0]:
            axis.text(
                0.5,
                0.5,
                "No finite waveform was rejected by the configured baseline-RMSE cut",
                transform=axis.transAxes,
                ha="center",
                va="center",
            )
            axis.set_axis_off()
        fig.suptitle("High-noise rejected-event diagnostic")
    _save(fig, path, dpi)


def write_preprocessing_diagnostics(
    cache: Any,
    output_dir: Path,
    *,
    physical_summary: dict[str, Any],
    active_families: tuple[str, ...],
    dpi: int = 180,
    logger: Any | None = None,
) -> dict[str, str]:
    """Write the diagnostics requested for permanent preprocessing."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs = {
        "photopeak_selection": output_dir / "photopeak_selection.png",
        "baseline_rmse_distribution": output_dir / "baseline_rmse_distribution.png",
        "trigger_time_distributions": output_dir / "trigger_time_distributions.png",
        "rejected_high_noise_event": output_dir / "rejected_high_noise_event.png",
    }
    _photopeak_plot(cache, physical_summary, outputs["photopeak_selection"], dpi)
    _baseline_rmse_plot(cache, physical_summary, outputs["baseline_rmse_distribution"], dpi)
    _trigger_time_plot(cache, physical_summary, active_families, outputs["trigger_time_distributions"], dpi)
    _rejected_noise_example(cache, physical_summary, outputs["rejected_high_noise_event"], dpi)
    if logger is not None:
        logger.info("Preprocessing diagnostics written | %s", output_dir)
    return {key: str(value.name) for key, value in outputs.items()}
