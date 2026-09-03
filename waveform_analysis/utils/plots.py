from __future__ import annotations

from pathlib import Path
from typing import Iterable

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

from .photopeak import PhotopeakResult
from .signal import INVALID_TIME_FS


# Shared visual language, adapted from plotting(6).py without mutating global
# Matplotlib state.  All public function signatures remain unchanged so this
# module can replace the current waveform-pipeline plots.py directly.
_PLOT_STYLE = {
    "font.size": 16,
    "axes.titlesize": 17,
    "axes.labelsize": 16,
    "xtick.labelsize": 13,
    "ytick.labelsize": 13,
    "legend.fontsize": 12,
    "figure.titlesize": 19,
    "axes.grid": True,
    "grid.alpha": 0.20,
    "grid.linewidth": 0.8,
    "axes.axisbelow": True,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.facecolor": "white",
}

_DATA_COLOR = "tab:blue"
_FIT_COLOR = "tab:orange"
_RANGE_COLOR = "tab:green"
_SELECTED_COLOR = "tab:orange"
_REJECTED_COLOR = "0.55"
_LIMIT_COLOR = "tab:red"


def _fmt3(value: float | int) -> str:
    value_f = float(value)
    if not np.isfinite(value_f):
        return "nan"
    magnitude = abs(value_f)
    if magnitude >= 1_000_000:
        return f"{value_f / 1_000_000:.3g}M"
    if magnitude >= 1_000:
        return f"{value_f / 1_000:.3g}k"
    return f"{value_f:.3g}"


def _save_figure(figure: plt.Figure, path: str | Path, dpi: int) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=int(dpi), bbox_inches="tight")
    plt.close(figure)


def _annotation_box() -> dict[str, object]:
    return {
        "boxstyle": "round,pad=0.35",
        "facecolor": "white",
        "edgecolor": "0.75",
        "alpha": 0.90,
    }


def _safe_optional_float(obj: object, *names: str) -> float | None:
    """Return the first finite optional numeric attribute without requiring it."""
    for name in names:
        value = getattr(obj, name, None)
        if value is None:
            continue
        try:
            converted = float(value)
        except (TypeError, ValueError):
            continue
        if np.isfinite(converted):
            return converted
    return None


def _histogram_bar(
    axis: plt.Axes,
    edges: np.ndarray,
    counts: np.ndarray,
    *,
    label: str = "Data",
) -> object:
    widths = np.diff(edges)
    return axis.bar(
        edges[:-1],
        counts,
        width=widths,
        align="edge",
        alpha=0.65,
        color=_DATA_COLOR,
        edgecolor=_DATA_COLOR,
        linewidth=0.5,
        label=label,
    )


def plot_energy_photopeaks(
    amplitudes_mV: np.ndarray,
    channels_zero_based: np.ndarray,
    results: list[PhotopeakResult],
    selected: np.ndarray,
    path: Path,
    *,
    dpi: int,
    bins: int,
) -> None:
    with plt.rc_context(_PLOT_STYLE):
        figure, axes = plt.subplots(1, 2, figsize=(12.8, 5.8), sharey=False)
        total = int(amplitudes_mV.shape[0])
        selected_count = int(np.count_nonzero(selected))

        for axis, channel, result in zip(
            axes, channels_zero_based, results, strict=True
        ):
            values = amplitudes_mV[:, int(channel)]
            finite = values[np.isfinite(values)]

            if result.success and result.edges_mV.size >= 2:
                centers = 0.5 * (result.edges_mV[:-1] + result.edges_mV[1:])
                histogram = _histogram_bar(axis, result.edges_mV, result.counts)
                gaussian_line, = axis.plot(
                    centers,
                    result.expected,
                    linewidth=2.2,
                    color=_FIT_COLOR,
                    label="Gaussian fit",
                )
                selection_span = axis.axvspan(
                    result.selection_low_mV,
                    result.selection_high_mV,
                    alpha=0.12,
                    color=_RANGE_COLOR,
                    label="Selected range",
                )
                fit_text = (
                    "Gaussian fit\n"
                    f"μ={_fmt3(result.mean_mV)} mV\n"
                    f"σ={_fmt3(result.sigma_mV)} mV\n"
                    f"cut=[{_fmt3(result.selection_low_mV)}, "
                    f"{_fmt3(result.selection_high_mV)}] mV\n"
                    f"χ²/ndof={_fmt3(result.chi2_ndof)}"
                )

                data_legend = axis.legend(
                    handles=[histogram, selection_span],
                    labels=["Data", "Selected range"],
                    loc="lower left",
                )
                axis.add_artist(data_legend)
                axis.legend(
                    handles=[gaussian_line],
                    labels=[fit_text],
                    loc="upper left",
                )
            else:
                axis.hist(
                    finite,
                    bins=bins,
                    alpha=0.65,
                    color=_DATA_COLOR,
                    label="Data",
                )
                axis.legend(loc="upper left")

            axis.set_title(f"Energy C{int(channel) + 1}")
            axis.set_xlabel("Amplitude [mV]")
            axis.set_ylabel("Events")

        rejected_count = total - selected_count
        selected_fraction = 100.0 * selected_count / total if total else 0.0
        figure.suptitle(
            "Photopeak selection — "
            f"selected={_fmt3(selected_count)} "
            f"({selected_fraction:.1f}%), rejected={_fmt3(rejected_count)}"
        )
        figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.94))
        _save_figure(figure, path, dpi)


def plot_energy_correlation(
    amplitudes_mV: np.ndarray,
    channels_zero_based: np.ndarray,
    selected: np.ndarray,
    path: Path,
    *,
    dpi: int,
) -> None:
    a = amplitudes_mV[:, int(channels_zero_based[0])]
    b = amplitudes_mV[:, int(channels_zero_based[1])]
    finite = np.isfinite(a) & np.isfinite(b)
    accepted = finite & selected
    rejected = finite & ~selected

    with plt.rc_context(_PLOT_STYLE):
        figure, axis = plt.subplots(figsize=(8.0, 6.8))
        axis.scatter(
            a[rejected],
            b[rejected],
            s=8,
            alpha=0.18,
            color=_REJECTED_COLOR,
            label=f"Rejected (N={_fmt3(np.count_nonzero(rejected))})",
            rasterized=True,
        )
        axis.scatter(
            a[accepted],
            b[accepted],
            s=10,
            alpha=0.42,
            color=_DATA_COLOR,
            label=f"Selected (N={_fmt3(np.count_nonzero(accepted))})",
            rasterized=True,
        )

        if np.count_nonzero(accepted) >= 2:
            correlation = float(np.corrcoef(a[accepted], b[accepted])[0, 1])
            axis.text(
                0.98,
                0.04,
                f"Selected Pearson r={correlation:.3f}",
                transform=axis.transAxes,
                ha="right",
                va="bottom",
                bbox=_annotation_box(),
            )

        axis.set_xlabel(f"C{int(channels_zero_based[0]) + 1} amplitude [mV]")
        axis.set_ylabel(f"C{int(channels_zero_based[1]) + 1} amplitude [mV]")
        axis.set_title("Energy-channel correlation")
        axis.legend(loc="upper left")
        figure.tight_layout()
        _save_figure(figure, path, dpi)


def plot_noise_distributions(
    noise_mV: np.ndarray,
    timing_channels_zero_based: np.ndarray,
    selected: np.ndarray,
    path: Path,
    *,
    dpi: int,
    noise_limit_mV: float | None,
) -> None:
    with plt.rc_context(_PLOT_STYLE):
        figure, axes = plt.subplots(1, 2, figsize=(12.4, 5.3), sharey=False)

        for axis, channel in zip(axes, timing_channels_zero_based, strict=True):
            values = noise_mV[:, int(channel)]
            finite = np.isfinite(values)
            selected_finite = selected & finite

            axis.hist(
                values[finite],
                bins=180,
                alpha=0.42,
                color=_DATA_COLOR,
                label=f"All valid (N={_fmt3(np.count_nonzero(finite))})",
            )
            axis.hist(
                values[selected_finite],
                bins=180,
                histtype="step",
                linewidth=2.0,
                color=_SELECTED_COLOR,
                label=f"Selected (N={_fmt3(np.count_nonzero(selected_finite))})",
            )

            if noise_limit_mV is not None:
                axis.axvline(
                    float(noise_limit_mV),
                    color=_LIMIT_COLOR,
                    linestyle="--",
                    linewidth=1.8,
                    label=f"Limit={_fmt3(noise_limit_mV)} mV",
                )

            if np.count_nonzero(selected_finite):
                selected_values = values[selected_finite]
                axis.text(
                    0.98,
                    0.96,
                    f"median={_fmt3(np.median(selected_values))} mV\n"
                    f"std={_fmt3(np.std(selected_values))} mV",
                    transform=axis.transAxes,
                    ha="right",
                    va="top",
                    bbox=_annotation_box(),
                )

            axis.set_title(f"Timing noise C{int(channel) + 1}")
            axis.set_xlabel("RMS [mV]")
            axis.set_ylabel("Events")
            axis.legend(loc="upper left")

        figure.suptitle("Timing-channel baseline noise")
        figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.94))
        _save_figure(figure, path, dpi)


def plot_trigger_toa(
    trigger_time_fs: np.ndarray,
    timing_channels_zero_based: np.ndarray,
    selected: np.ndarray,
    path: Path,
    *,
    dpi: int,
    bins: int,
) -> None:
    with plt.rc_context(_PLOT_STYLE):
        figure, axes = plt.subplots(1, 2, figsize=(12.4, 5.3), sharey=False)

        for axis, channel in zip(axes, timing_channels_zero_based, strict=True):
            times_fs = trigger_time_fs[:, int(channel)]
            valid = selected & (times_fs != INVALID_TIME_FS)
            values_ns = times_fs[valid].astype(np.float64) / 1.0e6

            axis.hist(
                values_ns,
                bins=bins,
                alpha=0.65,
                color=_DATA_COLOR,
                label="Selected events",
            )
            if values_ns.size:
                axis.text(
                    0.98,
                    0.96,
                    f"N={_fmt3(values_ns.size)}\n"
                    f"mean={_fmt3(np.mean(values_ns))} ns\n"
                    f"median={_fmt3(np.median(values_ns))} ns\n"
                    f"std={_fmt3(np.std(values_ns))} ns",
                    transform=axis.transAxes,
                    va="top",
                    ha="left",
                    bbox=_annotation_box(),
                )

            axis.set_title(f"50 mV trigger C{int(channel) + 1}")
            axis.set_xlabel("Time of arrival [ns]")
            axis.set_ylabel("Events")
            axis.legend(loc="lower left")

        figure.suptitle("Fixed-threshold trigger time of arrival")
        figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.94))
        _save_figure(figure, path, dpi)


def plot_scan(
    results: list[FitResult],
    best: FitResult | None,
    xlabel: str,
    path: Path,
    *,
    dpi: int,
    errorbars: bool,
) -> None:
    successful = [
        item for item in results if item.success and np.isfinite(item.ctr_ps)
    ]
    if not successful:
        return

    x = np.asarray([item.parameter for item in successful], dtype=np.float64)
    y = np.asarray([item.ctr_ps for item in successful], dtype=np.float64)
    errors = np.asarray(
        [item.ctr_error_ps for item in successful], dtype=np.float64
    )

    order = np.argsort(x)
    x = x[order]
    y = y[order]
    errors = errors[order]

    with plt.rc_context(_PLOT_STYLE):
        figure, axis = plt.subplots(figsize=(9.0, 5.8))
        finite_errors = np.isfinite(errors) & (errors >= 0.0)

        if errorbars and np.any(finite_errors):
            yerr = np.where(finite_errors, errors, 0.0)
            scan_handle = axis.errorbar(
                x,
                y,
                yerr=yerr,
                marker="o",
                markersize=5,
                linewidth=1.6,
                capsize=3,
                color=_DATA_COLOR,
                label="CTR scan",
            )
        else:
            scan_handle, = axis.plot(
                x,
                y,
                marker="o",
                markersize=5,
                linewidth=1.6,
                color=_DATA_COLOR,
                label="CTR scan",
            )

        handles: list[object] = [scan_handle]
        labels = ["CTR scan"]
        if best is not None and best.success and np.isfinite(best.ctr_ps):
            best_error = _safe_optional_float(best, "ctr_error_ps")
            error_text = (
                f" ± {best_error:.1f} ps"
                if best_error is not None and best_error >= 0.0
                else ""
            )
            best_handle = axis.scatter(
                [best.parameter],
                [best.ctr_ps],
                s=150,
                marker="*",
                color=_FIT_COLOR,
                edgecolor="black",
                linewidth=0.6,
                zorder=5,
            )
            handles.append(best_handle)
            labels.append(
                f"Best: {best.ctr_ps:.1f}{error_text} at {_fmt3(best.parameter)}"
            )

        method_names = {str(item.method) for item in successful}
        if len(method_names) == 1:
            axis.set_title(f"{next(iter(method_names))} parameter scan")
        else:
            axis.set_title("Timing-parameter scan")
        axis.set_xlabel(xlabel)
        axis.set_ylabel("CTR FWHM [ps]")
        axis.legend(handles, labels, loc="best")
        figure.tight_layout()
        _save_figure(figure, path, dpi)

def plot_toa_for_parameter(
    times_a_fs: np.ndarray,
    times_b_fs: np.ndarray,
    selected: np.ndarray,
    index: int,
    channel_numbers: Iterable[int],
    path: Path,
    *,
    title: str,
    dpi: int,
    bins: int,
) -> None:
    channels = list(channel_numbers)

    with plt.rc_context(_PLOT_STYLE):
        figure, axes = plt.subplots(1, 2, figsize=(12.4, 5.3), sharey=False)

        for axis, grid, channel in zip(
            axes, (times_a_fs, times_b_fs), channels, strict=True
        ):
            times_fs = grid[selected, index].astype(np.int64, copy=False)
            times_fs = times_fs[times_fs != INVALID_TIME_FS]
            values_ns = times_fs.astype(np.float64) / 1.0e6

            axis.hist(
                values_ns,
                bins=bins,
                alpha=0.65,
                color=_DATA_COLOR,
                label="Selected events",
            )
            if values_ns.size:
                axis.text(
                    0.98,
                    0.96,
                    f"N={_fmt3(values_ns.size)}\n"
                    f"mean={_fmt3(np.mean(values_ns))} ns\n"
                    f"median={_fmt3(np.median(values_ns))} ns\n"
                    f"std={_fmt3(np.std(values_ns))} ns",
                    transform=axis.transAxes,
                    va="top",
                    ha="right",
                    bbox=_annotation_box(),
                )

            axis.set_title(f"C{channel}")
            axis.set_xlabel("Time of arrival [ns]")
            axis.set_ylabel("Events")
            axis.legend(loc="upper left")

        figure.suptitle(title)
        figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.94))
        _save_figure(figure, path, dpi)

def _normalize_xai_importance(values: np.ndarray) -> np.ndarray:
    values = np.abs(np.asarray(values, dtype=np.float64).reshape(-1))
    peak = float(np.max(values)) if values.size else 0.0
    if not np.isfinite(peak) or peak <= 0.0:
        return np.zeros_like(values)
    return values / peak


def _xai_display_levels(
    values: np.ndarray,
    *,
    gamma: float,
    n_levels: int,
) -> np.ndarray:
    x = np.clip(np.asarray(values, dtype=np.float64).reshape(-1), 0.0, 1.0)
    if x.size == 0:
        return x
    gamma = float(gamma)
    if not np.isfinite(gamma) or gamma <= 0.0:
        raise ValueError("contrast_gamma must be positive")
    levels = int(n_levels)
    if levels < 2:
        raise ValueError("n_levels must be at least 2")
    x = np.power(x, gamma)
    indices = np.minimum((x * levels).astype(np.int64), levels - 1)
    return (indices + 0.5) / levels


def _xai_regional_importance(
    time_ns: np.ndarray,
    importance: np.ndarray,
    *,
    window_ns: float,
) -> list[tuple[float, float, float]]:
    time_ns = np.asarray(time_ns, dtype=np.float64).reshape(-1)
    importance = _normalize_xai_importance(importance)
    if time_ns.size != importance.size or time_ns.size < 2:
        raise ValueError("XAI time and importance arrays must have matching length >= 2")
    if np.any(~np.isfinite(time_ns)):
        raise ValueError("XAI time array contains non-finite values")
    window_ns = float(window_ns)
    if not np.isfinite(window_ns) or window_ns <= 0.0:
        raise ValueError("region_window_ns must be positive")

    dt = np.diff(time_ns)
    positive = dt[np.isfinite(dt) & (dt > 0.0)]
    half_step = 0.5 * float(np.median(positive)) if positive.size else 0.0
    left = float(time_ns[0]) - half_step
    right = float(time_ns[-1]) + half_step
    n_bins = max(1, int(np.ceil((right - left) / window_ns)))
    edges = left + window_ns * np.arange(n_bins + 1, dtype=np.float64)
    if edges[-1] < right:
        edges = np.append(edges, right)
    else:
        edges[-1] = right

    regions: list[tuple[float, float, float]] = []
    for index, (lo, hi) in enumerate(zip(edges[:-1], edges[1:], strict=True)):
        if index == len(edges) - 2:
            mask = (time_ns >= lo) & (time_ns <= hi)
        else:
            mask = (time_ns >= lo) & (time_ns < hi)
        if np.any(mask):
            mean_importance = float(np.mean(importance[mask]))
        else:
            center = 0.5 * (lo + hi)
            nearest = int(np.argmin(np.abs(time_ns - center)))
            mean_importance = float(importance[nearest])
        regions.append((float(lo), float(hi), mean_importance))
    return regions


def plot_xai_waveform_importance(
    path: str | Path,
    *,
    waveform_time_ps: np.ndarray,
    waveforms_mV: np.ndarray,
    xai_time_ps: np.ndarray,
    importance: np.ndarray,
    title: str,
    dpi: int,
    region_window_ns: float = 1.0,
    n_levels: int = 6,
    contrast_gamma: float = 0.55,
    anchor_label: str = "LED anchor",
) -> None:
    """Shared waveform + regional XAI background + normalized importance plot."""
    waveform_time_ps = np.asarray(waveform_time_ps, dtype=np.float64).reshape(-1)
    waveforms = np.asarray(waveforms_mV, dtype=np.float64)
    if (
        waveforms.ndim != 2
        or waveforms.shape[0] != 2
        or waveforms.shape[1] != waveform_time_ps.size
    ):
        raise ValueError("waveforms_mV must have shape (2, len(waveform_time_ps))")
    if (
        waveform_time_ps.size < 2
        or np.any(~np.isfinite(waveform_time_ps))
        or np.any(~np.isfinite(waveforms))
    ):
        raise ValueError("XAI waveform example must contain finite samples")

    xai_time_ps = np.asarray(xai_time_ps, dtype=np.float64).reshape(-1)
    normalized_importance = _normalize_xai_importance(importance)
    if xai_time_ps.size != normalized_importance.size or xai_time_ps.size < 2:
        raise ValueError("xai_time_ps and importance must have matching length >= 2")

    waveform_time_ns = waveform_time_ps / 1000.0
    xai_time_ns = xai_time_ps / 1000.0
    regions = _xai_regional_importance(
        xai_time_ns,
        normalized_importance,
        window_ns=float(region_window_ns),
    )
    region_strength = _xai_display_levels(
        np.asarray([value for _lo, _hi, value in regions], dtype=np.float64),
        gamma=float(contrast_gamma),
        n_levels=int(n_levels),
    )
    boundaries = np.linspace(0.0, 1.0, int(n_levels) + 1)

    with plt.rc_context(_PLOT_STYLE):
        figure, (waveform_axis, importance_axis) = plt.subplots(
            2,
            1,
            figsize=(9.2, 5.4),
            sharex=True,
            gridspec_kw={"height_ratios": [2.8, 1.0], "hspace": 0.08},
            constrained_layout=True,
        )
        cmap = mpl.colormaps["YlOrRd"]
        norm = mpl.colors.BoundaryNorm(boundaries, cmap.N, clip=True)

        for (lo, hi, _mean), strength in zip(regions, region_strength, strict=True):
            waveform_axis.axvspan(
                lo,
                hi,
                color=cmap(norm(float(strength))),
                alpha=0.12 + 0.58 * float(strength),
                linewidth=0.0,
                zorder=0,
            )

        waveform_axis.plot(
            waveform_time_ns,
            waveforms[0],
            linewidth=1.55,
            label="Detector 1",
            zorder=3,
        )
        waveform_axis.plot(
            waveform_time_ns,
            waveforms[1],
            linewidth=1.55,
            label="Detector 2",
            zorder=3,
        )
        waveform_axis.axvline(
            0.0,
            linestyle="--",
            linewidth=0.9,
            color="0.25",
            alpha=0.8,
            label=str(anchor_label),
            zorder=2,
        )
        waveform_axis.set_xlim(
            min(region[0] for region in regions),
            max(region[1] for region in regions),
        )
        waveform_axis.set_ylabel("Voltage [mV]")
        waveform_axis.set_title(str(title))
        waveform_axis.grid(alpha=0.18)
        waveform_axis.legend(frameon=False, loc="best")

        importance_axis.plot(
            xai_time_ns,
            normalized_importance,
            linewidth=1.55,
        )
        importance_axis.fill_between(
            xai_time_ns,
            0.0,
            normalized_importance,
            alpha=0.15,
        )
        importance_axis.set_ylim(0.0, 1.05)
        importance_axis.set_ylabel("Normalized\n|importance|")
        importance_axis.set_xlabel("LED-relative time [ns]")
        importance_axis.grid(alpha=0.18)

        scalar = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
        scalar.set_array([])
        ticks = 0.5 * (boundaries[:-1] + boundaries[1:])
        colorbar = figure.colorbar(
            scalar,
            ax=[waveform_axis, importance_axis],
            location="right",
            fraction=0.055,
            pad=0.02,
            boundaries=boundaries,
            ticks=ticks,
            spacing="proportional",
        )
        colorbar.ax.set_yticklabels([f"{value:.2f}" for value in ticks])
        colorbar.set_label("Regional mean normalized |importance|")
        _save_figure(figure, path, int(dpi))
