from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path

import matplotlib as mpl

SINGLE_COLUMN = (3.45, 2.55)
DOUBLE_COLUMN = (7.0, 3.35)
DOUBLE_COLUMN_TALL = (7.0, 4.35)

MODEL_ORDER = (
    "led",
    "cfd",
    "mlp",
    "onishi_cnn",
)
LABELS = {
    "led": "LED",
    "cfd": "CFD",
    "mlp": "Antisymmetric MLP",
    "onishi_cnn": "Onishi paired CNN",
}

# Okabe-Ito-derived, color-vision-friendly identities. Marker and line style
# remain distinct so figures also survive grayscale printing.
MODEL_STYLES = {
    "led": {"color": "#000000", "marker": "o", "linestyle": "--"},
    "cfd": {"color": "#7F7F7F", "marker": "x", "linestyle": ":"},
    "mlp": {"color": "#E69F00", "marker": "o", "linestyle": "-"},
    "onishi_cnn": {"color": "#009E73", "marker": "s", "linestyle": "-"},
}
_FALLBACK_STYLES = (
    {"color": "#56B4E9", "marker": "P", "linestyle": "-"},
    {"color": "#F0E442", "marker": "X", "linestyle": "--"},
)

DETECTOR_STYLES = (
    {"color": "#0072B2", "linestyle": "-", "linewidth": 1.25},
    {"color": "#D55E00", "linestyle": "--", "linewidth": 1.25},
)

_WINDOW_VARIANTS = (
    {"marker": "o", "linestyle": "-"},
    {"marker": "s", "linestyle": "--"},
    {"marker": "^", "linestyle": "-."},
    {"marker": "D", "linestyle": ":"},
    {"marker": "P", "linestyle": "-"},
    {"marker": "X", "linestyle": "--"},
)


@contextmanager
def paper_context():
    rc = {
        "font.family": "sans-serif",
        "font.size": 8.0,
        "axes.labelsize": 8.0,
        "axes.titlesize": 8.0,
        "legend.fontsize": 7.0,
        "xtick.labelsize": 7.0,
        "ytick.labelsize": 7.0,
        "axes.linewidth": 0.8,
        "lines.linewidth": 1.2,
        "lines.markersize": 4.0,
        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "legend.frameon": False,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "savefig.transparent": False,
    }
    with mpl.rc_context(rc):
        yield


def model_style(name: str, index: int = 0) -> dict:
    style = MODEL_STYLES.get(str(name))
    if style is None:
        style = _FALLBACK_STYLES[index % len(_FALLBACK_STYLES)]
    return dict(style)


def window_style(model: str, index: int) -> dict:
    style = model_style(model)
    style.update(_WINDOW_VARIANTS[index % len(_WINDOW_VARIANTS)])
    return style


def set_voltage_ticks(ax, values) -> None:
    import numpy as np

    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return
    low = int(np.ceil(np.min(finite)))
    high = int(np.floor(np.max(finite)))
    if high < low:
        return
    ticks = np.arange(low, high + 1, dtype=int)
    ax.set_xticks(ticks)
    ax.set_xticklabels([str(int(value)) for value in ticks])


def plot_voltage_series(
    ax,
    voltages,
    values,
    *,
    label: str | None = None,
    style: dict | None = None,
    errors=None,
    capsize: float = 2.5,
) -> bool:
    import numpy as np

    x = np.asarray(voltages, dtype=float)
    y = np.asarray(values, dtype=float)
    if x.shape != y.shape:
        raise ValueError("Voltage and value arrays must have the same shape")
    finite = np.isfinite(x) & np.isfinite(y)
    if not np.any(finite):
        return False

    kwargs = dict(style or {})
    if label is not None:
        kwargs["label"] = label

    if errors is None:
        ax.plot(x[finite], y[finite], **kwargs)
    else:
        err = np.asarray(errors, dtype=float)
        if err.shape != y.shape:
            raise ValueError("Error array must match the value array shape")
        ax.errorbar(
            x[finite],
            y[finite],
            yerr=np.where(np.isfinite(err[finite]), err[finite], 0.0),
            capsize=float(capsize),
            **kwargs,
        )
    return True


def finish_voltage_axis(
    ax,
    voltages,
    *,
    ylabel: str,
    legend: bool = True,
    ncol: int = 2,
    zero_line: bool = False,
) -> None:
    if zero_line:
        ax.axhline(0.0, color="#7F7F7F", linestyle=":", linewidth=0.9)
    set_voltage_ticks(ax, voltages)
    ax.set_xlabel("Voltage [V]")
    ax.set_ylabel(ylabel)
    if legend:
        ax.legend(loc="best", ncol=ncol)
    clean_axis(ax, grid="y")


def clean_axis(ax, *, grid: str | None = "y") -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    if grid == "y":
        ax.grid(axis="y", linewidth=0.5, alpha=0.22)
    elif grid == "both":
        ax.grid(True, linewidth=0.5, alpha=0.18)
    else:
        ax.grid(False)


def panel_label(ax, label: str) -> None:
    ax.text(
        0.02,
        0.98,
        label,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontweight="bold",
    )


def save_figure(fig, path: str | Path) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(target, bbox_inches="tight", pad_inches=0.03)
    return target
