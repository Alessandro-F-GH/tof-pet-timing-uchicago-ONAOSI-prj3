from __future__ import annotations
import csv, json
from contextlib import contextmanager
from pathlib import Path
from typing import Any
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from waveform_analysis.reporting.stats import residual_summary


def load_plot_config(root):
    return json.loads((Path(root).resolve() / "plots.json").read_text(encoding="utf-8"))


@contextmanager
def plot_context(config):
    f = config["font"]
    settings = {
        "font.family": f["family"],
        "font.size": float(f["size"]),
        "axes.titlesize": float(f["title_size"]),
        "axes.labelsize": float(f["label_size"]),
        "xtick.labelsize": float(f["tick_size"]),
        "ytick.labelsize": float(f["tick_size"]),
        "legend.fontsize": float(f["legend_size"]),
        "figure.dpi": float(config["output"]["dpi"]),
        "savefig.dpi": float(config["output"]["dpi"]),
        "lines.linewidth": float(config["line"]["width"]),
    }
    axes = config.get("axes", {})
    ticks = config.get("ticks", {})
    legend = config.get("legend", {})
    settings.update(
        {
            "axes.prop_cycle": mpl.cycler(
                color=config.get("palette", ["#0072B2", "#D55E00", "#009E73"])
            ),
            "axes.linewidth": axes.get("line_width", 0.8),
            "axes.edgecolor": axes.get("edge_color", "#333333"),
            "axes.labelcolor": axes.get("label_color", "#222222"),
            "axes.titleweight": axes.get("title_weight", "normal"),
            "axes.titlepad": axes.get("title_pad", 10),
            "axes.labelpad": axes.get("label_pad", 6),
            "axes.axisbelow": axes.get("axis_below", True),
            "legend.loc": legend.get("location", "best"),
            "legend.frameon": legend.get("frame", False),
            "legend.handlelength": legend.get("handle_length", 2),
            "legend.labelspacing": legend.get("label_spacing", 0.4),
            "legend.borderaxespad": legend.get("border_axes_pad", 0.5),
            "mathtext.fontset": f.get("mathtext_fontset", "dejavuserif"),
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.bbox": config["output"].get("bbox_inches", "tight"),
            "savefig.pad_inches": config["output"].get("pad_inches", 0.12),
        }
    )
    for side in ("top", "right", "bottom", "left"):
        settings[f"axes.spines.{side}"] = axes.get("spines", {}).get(
            side, side in {"bottom", "left"}
        )
    for axis in ("x", "y"):
        for key, default in (
            ("direction", "out"),
            ("color", "#333333"),
            ("major.size", 4),
            ("major.width", 0.8),
            ("minor.size", 2),
            ("minor.width", 0.6),
        ):
            settings[f"{axis}tick.{key}"] = ticks.get(key.replace(".", "_"), default)
    with mpl.rc_context(settings):
        yield


def output_path(directory, stem, config):
    return (
        Path(directory)
        / f"{stem}.{str(config['output'].get('format', 'png')).lstrip('.')}"
    )


def _finish(ax, cfg):
    g = cfg["grid"]
    ax.grid(False)
    if g["enabled"]:
        ax.grid(
            True,
            axis=g.get("axis", "both"),
            alpha=float(g["alpha"]),
            linestyle=g["linestyle"],
            color=g.get("color", "#B0B0B0"),
            linewidth=g.get("line_width", 0.5),
        )


def _save(fig, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return path


def _csv(path):
    with Path(path).open("r", encoding="utf-8", newline="") as s:
        return list(csv.DictReader(s))


def _ordered_cv_candidates(
    run: Path, rows: list[dict[str, str]]
) -> list[tuple[int, dict[str, str]]]:
    """Resolve one-based evaluation order without changing persisted metrics.

    Legacy fixed/grid runs lack order metadata; use their stable table order.
    Missing entries in a partial registry follow the known candidate numbers.
    """
    path = run / "metadata" / "candidates.json"
    registry = json.loads(path.read_text()) if path.is_file() else {}
    numbered = []
    missing = []
    for row in rows:
        candidate = registry.get(row["candidate_id"], {})
        if "candidate_order" in candidate:
            number = int(candidate["candidate_order"])
        elif "trial_number" in candidate:
            number = int(candidate["trial_number"]) + 1
        else:
            missing.append(row)
            continue
        numbered.append((number, row))
    next_number = max((number for number, _ in numbered), default=0) + 1
    numbered.extend((next_number + index, row) for index, row in enumerate(missing))
    return sorted(numbered, key=lambda item: item[0])


def plot_run_cv(run, cfg):
    run = Path(run)
    path = run / "tables" / "cv.csv"
    if not path.is_file():
        return None
    rows = _csv(path)
    manifest = json.loads((run / "metadata" / "manifest.json").read_text())
    if manifest.get("selection_method") == "ridge_cv":
        return None
    metric = manifest["cv"]["metric"]
    mean = f"{metric}_mean_ps"
    std = f"{metric}_std_ps"
    ordered = _ordered_cv_candidates(run, rows)
    x = np.asarray([number for number, _ in ordered], dtype=np.int64)
    rows = [row for _, row in ordered]
    v = np.asarray([float(r[mean]) for r in rows])
    e = np.asarray([float(r[std]) for r in rows])
    pr = np.asarray([str(r.get("pruned", "")).lower() in {"true", "1"} for r in rows])
    with plot_context(cfg):
        fig, ax = plt.subplots(figsize=tuple(cfg["cv"]["figsize"]))
        ax.errorbar(
            x[~pr],
            v[~pr],
            yerr=e[~pr],
            fmt=cfg["markers"]["default"],
            capsize=cfg["cv"].get("error_capsize", 3),
            color=cfg["cv"].get(
                "complete_color", cfg["formulations"]["shared"]["color"]
            ),
            markersize=cfg["markers"]["size"],
            label="Complete",
        )
        if np.any(pr):
            ax.scatter(
                x[pr],
                v[pr],
                marker=cfg["cv"]["pruned_marker"],
                color=cfg["cv"].get(
                    "pruned_color", cfg["formulations"]["direct"]["color"]
                ),
                label="Pruned",
            )
        ax.set_xticks(x)
        ax.set_xticklabels(
            [str(number) for number in x],
            rotation=float(cfg["cv"]["label_rotation"]),
            ha="right",
        )
        ax.set_xlabel("Candidate order")
        ax.set_ylabel(f"Development CV {metric.upper()} [ps]")
        ax.legend(ncols=int(cfg.get("legend", {}).get("columns", 1)))
        _finish(ax, cfg)
        return _save(fig, output_path(run / "plots", "cv", cfg))


def _plot_run_residual_distribution(run, cfg, *, dataset_role):
    run = Path(run)
    filename = "pred.npz" if dataset_role == "blind" else "development_pred.npz"
    stem = "blind" if dataset_role == "blind" else "development"
    path = run / "artifacts" / filename
    if not path.is_file():
        output_path(run / "plots", stem, cfg).unlink(missing_ok=True)
        return None
    with np.load(path) as d:
        if dataset_role == "development":
            manifest_path = run / "metadata" / "manifest.json"
            if manifest_path.is_file():
                manifest = json.loads(manifest_path.read_text())
                fingerprint = manifest.get("stage_fingerprints", {}).get("final_fit")
                if ("final_fit_fingerprint" not in d
                        or str(d["final_fit_fingerprint"].item()) != fingerprint):
                    output_path(run / "plots", stem, cfg).unlink(missing_ok=True)
                    return None
        corrected = np.asarray(d["corrected_ps"])
        led = np.asarray(d["led_residual_ps"])
    # Display a zero-centered LED reference; numerical residuals remain unchanged.
    led = np.asarray(led, dtype=float)
    finite_led = np.isfinite(led)
    if np.any(finite_led):
        led = led - np.mean(led[finite_led])
    summaries = [residual_summary(a) for a in (led, corrected)]
    valid = [s for s in summaries if s["n_finite"]]
    if not valid:
        return None
    lo = min(s["q01_ps"] for s in valid)
    hi = max(s["q99_ps"] for s in valid)
    span = hi - lo
    if span <= 0:
        span = max(abs(lo) * 0.1, 1.0)
        lo -= span / 2
        hi += span / 2
    else:
        lo -= 0.05 * span
        hi += 0.05 * span
    with plot_context(cfg):
        fig, ax = plt.subplots(figsize=tuple(cfg["histogram"]["figsize"]))
        for values, label, color in (
            (led, "LED (mean-centered)", cfg["reference"]["color"]),
            (
                corrected,
                "ML corrected",
                cfg["histogram"].get(
                    "corrected_color", cfg["formulations"]["shared"]["color"]
                ),
            ),
        ):
            values = np.asarray(values, float).ravel()
            ax.hist(
                values[np.isfinite(values)],
                bins=int(cfg["histogram"]["bins"]),
                range=(lo, hi),
                alpha=float(cfg["histogram"]["alpha"]),
                histtype=cfg["histogram"]["histtype"],
                color=color,
                label=label,
            )
        ax.set_xlabel("Blind residual [ps]" if dataset_role == "blind"
                      else "Development residual [ps] (final model)")
        ax.set_ylabel("Events")
        ax.legend(ncols=int(cfg.get("legend", {}).get("columns", 1)))
        _finish(ax, cfg)
        return _save(fig, output_path(run / "plots", stem, cfg))


def plot_run_blind(run, cfg):
    return _plot_run_residual_distribution(run, cfg, dataset_role="blind")


def plot_run_development(run, cfg):
    """Plot final-model training residuals with the blind distribution style."""
    return _plot_run_residual_distribution(run, cfg, dataset_role="development")


def blind_residual_boxplot(
    pairs: list[dict[str, Any]], path: str | Path, cfg: dict[str, Any], *, title: str
) -> Path | None:
    """Compare paired blind residuals without centering or truncating tails.

    Each entry has a label and one-dimensional LED/corrected residual arrays.
    Whiskers use the configured multiple of the interquartile range. Every finite
    paired event is included and all points beyond the whiskers are displayed.
    """
    style = cfg.get("boxplot", {})
    data, labels = [], []
    for pair in pairs:
        led = np.asarray(pair["led"], dtype=float).reshape(-1)
        corrected = np.asarray(pair["corrected"], dtype=float).reshape(-1)
        if led.shape != corrected.shape:
            raise ValueError("LED and corrected boxplot residuals must be paired")
        valid = np.isfinite(led) & np.isfinite(corrected)
        if np.any(valid):
            labels.append(pair["label"])
            data.extend([led[valid], corrected[valid]])
    if not data:
        Path(path).unlink(missing_ok=True)
        return None
    whis = float(style.get("whisker_iqr", 1.5))
    centers = np.arange(len(labels), dtype=float)
    separation = float(style.get("pair_separation", 0.36))
    positions = np.column_stack([centers - separation / 2, centers + separation / 2]).ravel()
    colors = [cfg["reference"]["color"], style.get(
        "corrected_color", cfg["histogram"].get("corrected_color", "#0072B2"))]
    with plot_context(cfg):
        fig, ax = plt.subplots(figsize=(
            max(float(style.get("figure_width_min", 6.4)),
                float(style.get("width_per_model", 1.15)) * len(labels)),
            float(style.get("figure_height", 5.0)),
        ))
        boxes = ax.boxplot(
            data, positions=positions, widths=float(style.get("box_width", 0.28)),
            whis=whis, patch_artist=True, showfliers=True,
            medianprops={"color": style.get("median_color", "#222222")},
            flierprops={"marker": style.get("outlier_marker", "."),
                        "markersize": float(style.get("outlier_size", 2.5)),
                        "alpha": float(style.get("outlier_alpha", 0.3))},
        )
        for index, (box, flier) in enumerate(zip(boxes["boxes"], boxes["fliers"])):
            color = colors[index % 2]
            box.set_facecolor(color)
            box.set_alpha(float(style.get("box_alpha", 0.55)))
            flier.set_markerfacecolor(color)
            flier.set_markeredgecolor(color)
        ax.axhline(0, color=cfg["reference"]["color"], linestyle=":", linewidth=0.8)
        ax.set_xticks(centers, labels)
        ax.set_xlim(-0.6, len(labels) - 0.4)
        ax.set_ylabel("Blind residual [ps]")
        ax.set_xlabel(
            "Boxes: 25th–75th percentiles; line: median\n"
            f"Whiskers: {whis:g} × interquartile range; all outliers shown",
            fontsize=float(cfg["font"]["annotation_size"]),
        )
        ax.set_title(title)
        ax.legend(
            boxes["boxes"][:2], ["LED", "ML corrected"],
            loc=style.get("legend_location", "upper left"),
            bbox_to_anchor=style.get("legend_anchor", [1.02, 1.0]),
        )
        _finish(ax, cfg)
        return _save(fig, path)


def plot_run_blind_boxplot(run, cfg):
    """Render the LED/correction comparison for one saved blind population."""
    run = Path(run)
    output = output_path(run / "plots", "blind_boxplot", cfg)
    path = run / "artifacts" / "pred.npz"
    if not path.is_file():
        output.unlink(missing_ok=True)
        return None
    with np.load(path) as data:
        return blind_residual_boxplot(
            [{"label": "Blind events", "led": data["led_residual_ps"],
              "corrected": data["corrected_ps"]}], output, cfg,
            title="Blind LED and corrected residual distributions",
        )


def _xai_one_ns(time, importance, *, normalize=True):
    # Aggregate per-sample group importance in fixed 1 ns time bins.
    edges = np.arange(np.floor(time.min()), np.ceil(time.max()) + 1, 1.0)
    if edges.size < 2:
        edges = np.array([time.min() - 0.5, time.max() + 0.5])
    centers = (edges[:-1] + edges[1:]) / 2
    values = np.full(centers.size, np.nan)
    indices = np.clip(
        np.searchsorted(edges, time, side="right") - 1, 0, centers.size - 1
    )
    for i in range(centers.size):
        portion = importance[indices == i]
        portion = portion[np.isfinite(portion)]
        if portion.size:
            values[i] = float(np.mean(portion))
    maximum = float(np.nanmax(values)) if np.any(np.isfinite(values)) else 0.0
    normalized = (
        np.nan_to_num(values / maximum, nan=0.0, posinf=0.0, neginf=0.0)
        if maximum > 0
        else np.zeros_like(values)
    )
    return (
        edges,
        centers,
        normalized
        if normalize
        else np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0),
    )


def plot_run_xai(run, cfg):
    run = Path(run)
    path = run / "artifacts" / "xai.npz"
    if not path.is_file():
        return None
    with np.load(path) as d:
        time = np.asarray(d["time_ps"], float).reshape(-1) / 1000.0
        importance = np.asarray(d["importance_ps"], float)
        example = np.asarray(d["example_waveforms_mV"], float)
        formulation = str(np.asarray(d["estimator_formulation"]).item())
    if formulation not in {"shared", "direct"}:
        raise ValueError(f"Unknown XAI formulation {formulation!r} in {path}")
    if time.size < 2 or not np.all(np.isfinite(time)):
        return None
    if example.shape != (2, time.size):
        raise ValueError(f"XAI waveform pair shape mismatch in {path}: {example.shape}")
    expected_shape = (time.size,) if formulation == "shared" else (2, time.size)
    if importance.shape != expected_shape:
        raise ValueError(
            f"XAI importance shape mismatch in {path}: {importance.shape}, expected {expected_shape}"
        )
    style = cfg["xai"]
    width, height = style["figsize"]
    cmap = mpl.colormaps.get_cmap(style.get("cmap", "viridis"))
    norm = mpl.colors.Normalize(vmin=0, vmax=1)
    colors = style.get(
        "detector_colors",
        cfg.get("preprocessing", {}).get("detector_colors", ["#0072B2", "#D55E00"]),
    )
    with plot_context(cfg):
        if formulation == "shared":
            edges, centers, values = _xai_one_ns(time, importance)
            fig, (top, bottom) = plt.subplots(
                2,
                1,
                figsize=(max(float(width), 7.2), max(float(height), 5.4)),
                sharex=True,
                gridspec_kw={"height_ratios": [2.0, 1.0]},
                layout="constrained",
            )
            for index, (color, linestyle) in enumerate(
                zip(colors, style.get("detector_linestyles", ["-", "--"]))
            ):
                top.plot(
                    time,
                    example[index],
                    color=color,
                    linestyle=linestyle,
                    label=f"Detector {index + 1}",
                    zorder=3,
                )
            top.legend(ncols=int(cfg.get("legend", {}).get("columns", 1)))
            top.set_ylabel("Signal [mV]")
            for left, right, value in zip(edges[:-1], edges[1:], values):
                top.axvspan(
                    left,
                    right,
                    color=cmap(norm(value)),
                    alpha=float(style.get("band_alpha", 0.25)),
                    linewidth=0,
                    zorder=0,
                )
            bottom.bar(
                centers,
                values,
                width=np.diff(edges),
                color=[cmap(norm(v)) for v in values],
                edgecolor="white",
                linewidth=0.5,
            )
            bottom.plot(
                centers,
                values,
                color=style.get("importance_color", "#333333"),
                marker=style.get("importance_marker", "o"),
                markersize=style.get("importance_marker_size", 2.5),
            )
            bottom.set_ylim(0, 1.08)
            bottom.set_ylabel("Normalized importance")
            axes = [top, bottom]
        else:
            # Independent interventions for each input channel; normalize to
            # one common maximum so channel values and colors are comparable.
            bins = [
                _xai_one_ns(time, importance[channel], normalize=False)
                for channel in range(2)
            ]
            global_max = max(max(float(np.max(v)), 0.0) for _, _, v in bins)
            fig, axes = plt.subplots(
                2,
                1,
                figsize=(max(float(width), 7.2), max(float(height), 5.8)),
                sharex=True,
                layout="constrained",
            )
            for channel, ax in enumerate(axes):
                edges, _, raw_scores = bins[channel]
                values = (
                    raw_scores / global_max
                    if global_max > 0
                    else np.zeros_like(raw_scores)
                )
                ax.plot(
                    time,
                    example[channel],
                    color=colors[channel],
                    label=f"Detector {channel + 1}",
                    zorder=3,
                )
                for left, right, value in zip(edges[:-1], edges[1:], values):
                    ax.axvspan(
                        left,
                        right,
                        color=cmap(norm(value)),
                        alpha=float(style.get("band_alpha", 0.32)),
                        linewidth=0,
                        zorder=0,
                    )
                ax.set_ylabel(f"Detector {channel + 1} [mV]")
                ax.legend(ncols=int(cfg.get("legend", {}).get("columns", 1)))
        axes[-1].set_xlabel("Time relative to LED [ns]")
        axes[0].set_xlim(time.min(), time.max())
        for ax in axes:
            _finish(ax, cfg)
        colorbar = fig.colorbar(
            mpl.cm.ScalarMappable(norm=norm, cmap=cmap),
            ax=axes,
            location="right",
            fraction=0.035,
            pad=0.045,
        )
        colorbar.set_label("Normalized importance", labelpad=9)
        output = output_path(run / "plots", "xai", cfg)
        output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output)
        plt.close(fig)
        return output


def render_run_plots(run, cfg):
    return {
        "cv": plot_run_cv(run, cfg),
        "blind": plot_run_blind(run, cfg),
        "blind_boxplot": plot_run_blind_boxplot(run, cfg),
        "development": plot_run_development(run, cfg),
        "xai": plot_run_xai(run, cfg),
    }


def heatmap(matrix, labels, path, cfg, *, title, correlation=False, value_format=".1f"):
    matrix = np.asarray(matrix, float)
    style = cfg["heatmap"]
    kwargs = {
        "cmap": style["correlation_cmap"] if correlation else style["difference_cmap"]
    }
    if correlation:
        kwargs.update(
            vmin=float(style["correlation_limits"][0]),
            vmax=float(style["correlation_limits"][1]),
        )
    elif np.any(np.isfinite(matrix)):
        limit = float(np.nanmax(np.abs(matrix)))
        if limit > 0:
            kwargs.update(vmin=-limit, vmax=limit)
    with plot_context(cfg):
        figsize = [
            max(float(minimum), float(per_model) * len(labels))
            for minimum, per_model in zip(style["min_figsize"], style["per_model"])
        ]
        fig, ax = plt.subplots(figsize=tuple(figsize))
        im = ax.imshow(matrix, **kwargs)
        ax.set_xticks(range(len(labels)))
        ax.set_yticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=float(style["label_rotation"]), ha="right")
        ax.set_yticklabels(labels)
        ax.set_title(title)
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                if np.isfinite(matrix[i, j]):
                    ax.text(
                        j,
                        i,
                        format(matrix[i, j], value_format),
                        ha="center",
                        va="center",
                        fontsize=float(style["cell_text_size"]),
                        color=style.get("contrast_text_color", "white")
                        if abs(float(im.norm(matrix[i, j])) - 0.5) * 2
                        > style.get("contrast_threshold", 0.7)
                        else style.get("text_color", "#222222"),
                    )
        fig.colorbar(im, ax=ax)
        return _save(fig, path)


def scatter_with_labels(
    x,
    y,
    labels,
    path,
    cfg,
    *,
    xlabel,
    ylabel,
    title,
    xerr=None,
    yerr=None,
    annotation=None,
):
    with plot_context(cfg):
        fig, ax = plt.subplots(figsize=tuple(cfg["scatter"]["figsize"]))
        x = np.asarray(x, float)
        y = np.asarray(y, float)
        if xerr is not None or yerr is not None:
            ax.errorbar(
                x,
                y,
                xerr=xerr,
                yerr=yerr,
                fmt="none",
                capsize=float(cfg["scatter"]["error_capsize"]),
                ecolor=cfg["scatter"].get("error_color", "#777777"),
            )
        palette = cfg.get("palette", ["#0072B2"])
        colors = [
            cfg["reference"]["color"]
            if label == "LED reference"
            else palette[index % len(palette)]
            for index, label in enumerate(labels)
        ]
        for xi, yi, label, color in zip(x, y, labels, colors):
            ax.scatter(
                xi, yi, label=label,
                s=float(cfg["scatter"]["marker_size"]), color=color,
                edgecolors=cfg["scatter"].get("edge_color", "white"),
                linewidths=cfg["scatter"].get("edge_line_width", 0.5), zorder=3,
            )
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title + (f"\n{annotation}" if annotation else ""))
        ax.legend(
            loc=cfg["scatter"].get("legend_location", "upper left"),
            bbox_to_anchor=cfg["scatter"].get("legend_anchor", [1.02, 1.0]),
        )
        _finish(ax, cfg)
        return _save(fig, path)


def blind_metric_bar(
    rows: list[dict[str, Any]], path: str | Path, cfg: dict[str, Any], *,
    metric: str, led_value: float,
) -> Path:
    """Plot performance-ranked persisted timing metrics with bootstrap errors.

    ``rows`` contains rank-ordered code, display name, formulation, metric and error.
    No fitting, selection or numerical metric calculation is performed here.
    """
    style = cfg["bar"]
    with plot_context(cfg):
        fig, ax = plt.subplots(figsize=(
            max(float(style["figure_width_min"]),
                float(style["width_per_category"]) * len(rows)),
            float(style["figure_height"]),
        ))
        handles = []
        for index, row in enumerate(rows):
            value = float(row[f"{metric}_ps"])
            error = float(row[f"{metric}_std_ps"])
            handles.append(ax.bar(index, value, yerr=error,
                   width=float(style["group_width"]),
                   capsize=float(style["error_capsize"]),
                   color=cfg["formulations"][row["formulation"]]["color"],
                   edgecolor=style["edge_color"],
                   linewidth=float(style["edge_line_width"]),
                   label=f"{row['code']}: {row['display_name']}"))
            if np.isfinite(value):
                ax.annotate(
                    f"{int(round(value))} ps",
                    (index, value + (error if np.isfinite(error) else 0.0)),
                    xytext=(0, float(style["annotation_offset_points"])),
                    textcoords="offset points", ha="center", va="bottom",
                    fontsize=float(cfg["font"]["annotation_size"]),
                )
        handles.append(ax.axhline(
            led_value, color=cfg["reference"]["color"], linestyle="--",
            label=cfg["reference"]["label"]))
        lower, upper = ax.get_ylim()
        ax.set_ylim(lower, upper + (upper - lower) * float(style.get("headroom_fraction", 0.25)))
        ax.set_xticks(range(len(rows)), [row["code"] for row in rows])
        ax.set_ylabel(f"Blind {metric.upper()} [ps]")
        ax.set_xlabel(f"Model (ascending blind {metric.upper()})")
        ax.set_title("Blind coincidence timing resolution" if metric == "ctr"
                     else "Blind root mean squared error")
        ax.legend(
            handles=handles, loc=style.get("legend_location", "upper left"),
            bbox_to_anchor=style.get("legend_anchor", [1.02, 1.0]),
        )
        _finish(ax, cfg)
        return _save(fig, path)


def grouped_bar(labels, series, path, cfg, *, ylabel, title):
    style = cfg["bar"]
    width = max(
        float(style["figure_width_min"]),
        float(style["width_per_category"]) * len(labels),
    )
    with plot_context(cfg):
        fig, ax = plt.subplots(figsize=(width, float(style["figure_height"])))
        x = np.arange(len(labels), dtype=float)
        bar_width = float(style["group_width"]) / max(1, len(series))
        for index, item in enumerate(series):
            off = (index - (len(series) - 1) / 2) * bar_width
            v = np.asarray(item["values"], float)
            e = np.asarray(item.get("errors", np.zeros_like(v)), float)
            color = next(
                (
                    style["color"]
                    for style in [*cfg["formulations"].values(), cfg["reference"]]
                    if style["label"] == item["label"]
                ),
                cfg.get("palette", ["#0072B2"])[
                    index % len(cfg.get("palette", ["#0072B2"]))
                ],
            )
            bars = ax.bar(
                x + off,
                v,
                width=bar_width,
                yerr=e,
                capsize=float(style["error_capsize"]),
                label=item["label"],
                color=color,
                edgecolor=style["edge_color"],
                linewidth=float(style["edge_line_width"]),
            )
            for bar, value, error in zip(bars, v, e):
                if not np.isfinite(value) or not style.get("show_values", True):
                    continue
                text = (
                    f"{int(round(value))} ± {int(round(error))} ps"
                    if np.isfinite(error) and error > 0
                    else f"{int(round(value))} ps"
                )
                ax.annotate(
                    text,
                    (
                        bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + (error if np.isfinite(error) else 0),
                    ),
                    xytext=(0, float(style["annotation_offset_points"])),
                    textcoords="offset points",
                    ha="center",
                    va="bottom",
                    fontsize=float(cfg["font"]["annotation_size"]),
                    rotation=style.get("annotation_rotation", 90),
                )
        lower, upper = ax.get_ylim()
        ax.set_ylim(
            lower, upper + (upper - lower) * float(style.get("headroom_fraction", 0.25))
        )
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend(ncols=int(cfg.get("legend", {}).get("columns", 1)))
        _finish(ax, cfg)
        return _save(fig, path)


# Retain serialized identities while legacy imports resolve to this module.
from waveform_analysis.core.compat import (
    preserve_legacy_identity as _preserve_legacy_identity,
)

_preserve_legacy_identity(__name__, "waveform_analysis.ml_pipeline.plotting")
