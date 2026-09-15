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
    "linear_svr",
    "mlp",
    "onishi_cnn",
    "minirocket",
    "difference_knn",
)
LABELS = {
    "led": "LED",
    "cfd": "CFD",
    "linear_svr": "Linear SVR",
    "cnn": "CNN",
    "mlp": "Antisymmetric MLP",
    "onishi_cnn": "Onishi paired CNN",
    "minirocket": "MiniRocket",
    "difference_knn": "Difference k-NN",
}

# Okabe-Ito-derived, color-vision-friendly identities. Marker and line style
# remain distinct so figures also survive grayscale printing.
MODEL_STYLES = {
    "led": {"color": "#000000", "marker": "o", "linestyle": "--"},
    "cfd": {"color": "#7F7F7F", "marker": "x", "linestyle": ":"},
    "linear_svr": {"color": "#0072B2", "marker": "^", "linestyle": "-."},
    "cnn": {"color": "#E69F00", "marker": "o", "linestyle": "-"},
    "mlp": {"color": "#E69F00", "marker": "o", "linestyle": "-"},
    "onishi_cnn": {"color": "#009E73", "marker": "s", "linestyle": "-"},
    "minirocket": {"color": "#CC79A7", "marker": "D", "linestyle": "-."},
    "difference_knn": {"color": "#D55E00", "marker": "v", "linestyle": ":"},
}
_FALLBACK_STYLES = (
    {"color": "#56B4E9", "marker": "P", "linestyle": "-"},
    {"color": "#F0E442", "marker": "X", "linestyle": "--"},
)

DETECTOR_STYLES = (
    {"color": "#0072B2", "linestyle": "-", "linewidth": 1.25},
    {"color": "#D55E00", "linestyle": "--", "linewidth": 1.25},
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
