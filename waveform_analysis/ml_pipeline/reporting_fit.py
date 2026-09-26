from __future__ import annotations

from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path
from typing import Callable

import numpy as np

from utils_fit import (
    DEFAULT_FIT_CONFIG,
    DEFAULT_HISTOGRAM_BIN_WIDTH_PS,
    fit_ctr_ps,
    fit_direct_fwhm,
    fixed_width_histogram_edges,
    validate_histogram_bin_width_ps,
)

from . import latex_tables as latex_tables_module
from . import reporting as reporting_module
from .common import load_artifact_array
from .splits import semantic_seed


def resolve_histogram_bin_width_ps(value: float | None) -> float:
    """Resolve reporting bin width without consulting the stored run config."""
    return validate_histogram_bin_width_ps(
        DEFAULT_HISTOGRAM_BIN_WIDTH_PS if value is None else value
    )


def _reporting_fit_config(histogram_bin_width_ps: float) -> dict[str, float | int]:
    """Current reporting-only F1 configuration, independent of run history."""
    return {
        "histogram_bin_width_ps": validate_histogram_bin_width_ps(
            histogram_bin_width_ps
        ),
        "bootstrap_samples": int(DEFAULT_FIT_CONFIG["bootstrap_samples"]),
    }


def _reporting_ctr(
    values_ps: np.ndarray,
    fit_config: dict[str, float | int],
    *,
    seed: int,
    bootstrap: bool,
):
    """Direct-F1 CTR for reporting, using the bootstrap median as central value.

    The original-sample fit is retained for geometric diagnostics such as peak
    position and half-maximum crossings. When bootstrap is enabled, the reported
    scalar CTR is the median of successful event-bootstrap F1 estimates and its
    uncertainty is their sample standard deviation.
    """
    values = np.asarray(values_ps, dtype=np.float64).reshape(-1)
    finite = values[np.isfinite(values)]

    base = fit_ctr_ps(
        finite,
        fit_config,
        seed=int(seed),
        bootstrap=False,
    )
    if not bootstrap:
        return base

    requested = int(fit_config["bootstrap_samples"])
    if requested <= 1:
        return replace(
            base,
            ctr_error_ps=float("nan"),
            bootstrap_samples=requested,
            bootstrap_successful=0,
        )

    bin_width = float(fit_config["histogram_bin_width_ps"])
    rng = np.random.default_rng(int(seed))
    bootstrap_ctrs: list[float] = []
    n_valid = int(finite.size)
    for _ in range(requested):
        sample = finite[rng.integers(0, n_valid, size=n_valid)]
        try:
            trial = fit_direct_fwhm(
                sample,
                histogram_bin_width_ps=bin_width,
            )
        except ValueError:
            continue
        if np.isfinite(trial.ctr_ps):
            bootstrap_ctrs.append(float(trial.ctr_ps))

    if not bootstrap_ctrs:
        raise ValueError("No successful direct-F1 bootstrap replicate for reporting")

    bootstrap_values = np.asarray(bootstrap_ctrs, dtype=np.float64)
    central = float(np.median(bootstrap_values))
    error = (
        float(np.std(bootstrap_values, ddof=1))
        if bootstrap_values.size > 1
        else float("nan")
    )
    return replace(
        base,
        ctr_ps=central,
        ctr_error_ps=error,
        bootstrap_samples=requested,
        bootstrap_successful=int(bootstrap_values.size),
    )


def _recompute_rows(
    run_dir: str | Path,
    original_reader: Callable[[str | Path], list[dict]],
    *,
    histogram_bin_width_ps: float,
    cache: dict[Path, list[dict]],
) -> list[dict]:
    """Recompute all reportable CTR rows from persisted residual artifacts.

    Stored CTR values and stored fit configuration are intentionally ignored.
    The persisted results table is used only to discover dataset/method/stage rows
    and retain unrelated metadata.
    """
    run = Path(run_dir).resolve()
    if run in cache:
        return [dict(row) for row in cache[run]]

    rows = original_reader(run)
    fit_config = _reporting_fit_config(histogram_bin_width_ps)
    recomputed: list[dict] = []

    for row in rows:
        updated = dict(row)
        stage = str(row.get("stage", ""))
        method = str(row.get("method", ""))
        dataset = str(row.get("dataset", ""))
        if stage not in {"train", "test"} or not dataset or not method:
            recomputed.append(updated)
            continue

        residual = load_artifact_array(run, dataset, method, stage, "residuals_ps")
        if residual is None:
            raise FileNotFoundError(
                "Reporting CTR recomputation requires residual artifacts for "
                f"{dataset}/{method}/{stage}"
            )
        values = np.asarray(residual, dtype=np.float64).reshape(-1)
        result = _reporting_ctr(
            values,
            fit_config,
            seed=semantic_seed(
                0,
                dataset,
                method,
                stage,
                "reporting_direct_f1_ctr",
            ),
            bootstrap=True,
        )
        updated.update(
            {
                "ctr_ps": float(result.ctr_ps),
                "ctr_uncertainty_ps": float(result.ctr_error_ps),
                "center_ps": float(result.center_ps),
                "peak_height": float(result.peak_height),
                "half_max_events": float(result.half_max_events),
                "left_half_ps": float(result.left_half_ps),
                "right_half_ps": float(result.right_half_ps),
                "histogram_bin_width_ps": float(result.histogram_bin_width_ps),
                "histogram_bins": int(result.histogram_bins),
                "bootstrap_samples": int(result.bootstrap_samples),
                "bootstrap_successful": int(result.bootstrap_successful),
                "ctr_central_value": "bootstrap_median",
                "ctr_uncertainty_method": "bootstrap_standard_deviation",
            }
        )
        recomputed.append(updated)

    cache[run] = [dict(row) for row in recomputed]
    return [dict(row) for row in recomputed]


def _direct_distribution_plot(
    output,
    run,
    rows,
    mode,
    dataset,
    stage,
    paths,
    *,
    histogram_bin_width_ps: float,
) -> None:
    """CTR distribution with direct F1 half-maximum width overlaid."""
    import matplotlib.pyplot as plt

    available = []
    for method in reporting_module._distribution_methods(rows, dataset, stage):
        residual = reporting_module._residual(run, dataset, method, stage)
        if residual is None:
            continue
        residual = np.asarray(residual, dtype=float)
        residual = residual[np.isfinite(residual)]
        if not residual.size:
            continue
        row = next(
            (
                item
                for item in rows
                if item["dataset"] == dataset
                and item["method"] == method
                and item.get("stage") == stage
            ),
            None,
        )
        if row is not None:
            available.append((method, residual, row))

    led = next((item for item in available if item[0] == "led"), None)
    models = [item for item in available if item[0] not in {"led", "cfd"}]
    if led is None or not models:
        return

    for model, model_residual, model_row in models:
        pair = [led, (model, model_residual, model_row)]
        direct_fits = [
            (
                method,
                residual,
                row,
                fit_direct_fwhm(
                    residual,
                    histogram_bin_width_ps=histogram_bin_width_ps,
                ),
            )
            for method, residual, row in pair
        ]

        xlim = reporting_module._robust_display_range(
            [residual for _method, residual, _row, _fit in direct_fits],
            quantiles=(0.005, 0.995),
            margin_fraction=0.06,
        )
        xlim = (
            min(
                float(xlim[0]),
                min(fit.half_max_left_ps for _m, _r, _row, fit in direct_fits) - 20.0,
            ),
            max(
                float(xlim[1]),
                max(fit.half_max_right_ps for _m, _r, _row, fit in direct_fits) + 20.0,
            ),
        )

        fig, ax = plt.subplots(figsize=reporting_module.SINGLE_COLUMN)
        peak = 0.0
        for index, (method, residual, row, fit) in enumerate(direct_fits):
            edges = fixed_width_histogram_edges(
                residual,
                histogram_bin_width_ps,
                low_ps=float(fit.fit_low_ps),
                high_ps=float(fit.fit_high_ps),
            )
            counts, _ = np.histogram(residual, bins=edges)
            if counts.size:
                peak = max(peak, float(np.max(counts)))
            peak = max(peak, float(fit.peak_height))

            style = reporting_module.model_style(method, index)
            color = style["color"]
            label = (
                f"{reporting_module.LABELS.get(method, method)}, CTR "
                f"{reporting_module._measurement_text(reporting_module._float(row.get('ctr_ps')), reporting_module._float(row.get('ctr_uncertainty_ps')))} ps"
            )
            ax.hist(
                residual,
                bins=edges,
                histtype="step",
                color=color,
                linestyle=style["linestyle"],
                linewidth=1.35,
                label=label,
            )

            half_height = 0.5 * float(fit.peak_height)
            ax.plot(
                float(fit.center_ps),
                float(fit.peak_height),
                marker="o",
                markersize=4.5,
                linestyle="none",
                color=color,
                zorder=5,
            )
            ax.hlines(
                half_height,
                float(fit.half_max_left_ps),
                float(fit.half_max_right_ps),
                color=color,
                linestyle=":",
                linewidth=1.35,
                zorder=4,
            )
            ax.vlines(
                [float(fit.half_max_left_ps), float(fit.half_max_right_ps)],
                0.0,
                half_height,
                color=color,
                linestyle=":",
                linewidth=0.9,
                alpha=0.8,
                zorder=3,
            )

        if peak > 0:
            ax.set_ylim(0.0, peak * 1.16)
        ax.set_xlim(*xlim)
        ax.set_xlabel("Residual [ps]")
        ax.set_ylabel("Events [count]")
        ax.legend(loc="best")
        reporting_module.clean_axis(ax, grid="y")
        fig.tight_layout()
        target = reporting_module.save_figure(
            fig,
            output / f"ctr_distribution_{stage}_{dataset}_{model}.pdf",
        )
        plt.close(fig)
        paths.append(target)


@contextmanager
def reporting_fit_options(*, histogram_bin_width_ps: float | None = None):
    """Recompute report CTR values from residual artifacts using current F1 settings."""
    from matplotlib.axes import Axes

    bin_width = resolve_histogram_bin_width_ps(histogram_bin_width_ps)
    fit_config = _reporting_fit_config(bin_width)
    original_read_results = reporting_module.read_results
    original_fit_ctr = reporting_module.fit_ctr_ps
    original_distribution_plot = reporting_module._distribution_plot
    original_legend = Axes.legend
    original_latex_read_csv = latex_tables_module._read_csv
    cache: dict[Path, list[dict]] = {}

    def configured_read_results(run_dir):
        return _recompute_rows(
            run_dir,
            original_read_results,
            histogram_bin_width_ps=bin_width,
            cache=cache,
        )

    def configured_fit_ctr(values_ps, config=None, *, seed=0, bootstrap=True):
        del config
        return _reporting_ctr(
            values_ps,
            fit_config,
            seed=seed,
            bootstrap=bootstrap,
        )

    def configured_distribution_plot(output, run, rows, mode, dataset, stage, paths):
        return _direct_distribution_plot(
            output,
            run,
            rows,
            mode,
            dataset,
            stage,
            paths,
            histogram_bin_width_ps=bin_width,
        )

    def configured_legend(self, *args, **kwargs):
        if self.get_xlabel() == "Residual [ps]" and self.get_ylabel() == "Events [count]":
            kwargs["loc"] = "lower center"
            kwargs["bbox_to_anchor"] = (0.5, 1.02)
            kwargs.setdefault("ncol", 2)
            kwargs.setdefault("borderaxespad", 0.0)
        return original_legend(self, *args, **kwargs)

    def configured_latex_read_csv(path):
        candidate = Path(path)
        if candidate.name == "results.csv" and candidate.parent.name == "csv":
            return configured_read_results(candidate.parent.parent)
        return original_latex_read_csv(path)

    reporting_module.read_results = configured_read_results
    reporting_module.fit_ctr_ps = configured_fit_ctr
    reporting_module._distribution_plot = configured_distribution_plot
    latex_tables_module._read_csv = configured_latex_read_csv
    Axes.legend = configured_legend
    try:
        yield bin_width
    finally:
        reporting_module.read_results = original_read_results
        reporting_module.fit_ctr_ps = original_fit_ctr
        reporting_module._distribution_plot = original_distribution_plot
        latex_tables_module._read_csv = original_latex_read_csv
        Axes.legend = original_legend
