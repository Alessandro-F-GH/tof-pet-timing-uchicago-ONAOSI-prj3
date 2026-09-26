from __future__ import annotations

from contextlib import contextmanager
import json
from pathlib import Path
from typing import Any, Callable

import numpy as np

from utils_fit import (
    CTR_DEFINITIONS,
    DEFAULT_CTR_DEFINITION,
    DEFAULT_HISTOGRAM_BIN_WIDTH_PS,
    fit_ctr_ps,
    fit_nema_fwhm,
    fixed_width_histogram_edges,
    validate_histogram_bin_width_ps,
)

from . import latex_tables as latex_tables_module
from . import reporting as reporting_module
from .common import load_artifact_array
from .splits import semantic_seed


def resolve_ctr_definition(value: str | None) -> str:
    definition = DEFAULT_CTR_DEFINITION if value is None else str(value).strip().lower()
    if definition not in CTR_DEFINITIONS:
        raise ValueError(
            f"Unknown CTR definition {value!r}; expected one of {CTR_DEFINITIONS}"
        )
    return definition


def resolve_histogram_bin_width_ps(value: float | None) -> float:
    return validate_histogram_bin_width_ps(
        DEFAULT_HISTOGRAM_BIN_WIDTH_PS if value is None else value
    )


def _fit_config(manifest: dict[str, Any]) -> dict[str, Any]:
    return dict((manifest.get("config") or {}).get("fit") or {})


def _base_seed(manifest: dict[str, Any]) -> int:
    return int(((manifest.get("config") or {}).get("validation") or {}).get("seed", 0))


def _recompute_rows(
    run_dir: str | Path,
    original_reader: Callable[[str | Path], list[dict[str, Any]]],
    *,
    definition: str,
    histogram_bin_width_ps: float,
    cache: dict[Path, list[dict[str, Any]]],
) -> list[dict[str, Any]]:
    run = Path(run_dir).resolve()
    if run in cache:
        return [dict(row) for row in cache[run]]

    rows = original_reader(run)
    if definition == DEFAULT_CTR_DEFINITION:
        cache[run] = [dict(row) for row in rows]
        return [dict(row) for row in rows]

    manifest = json.loads((run / "manifest.json").read_text(encoding="utf-8"))
    fit_config = _fit_config(manifest)
    base_seed = _base_seed(manifest)
    recomputed: list[dict[str, Any]] = []
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
        result = fit_ctr_ps(
            values,
            fit_config,
            seed=semantic_seed(
                base_seed,
                dataset,
                method,
                stage,
                "reporting_ctr",
                definition,
            ),
            bootstrap=True,
            definition=definition,
            histogram_bin_width_ps=histogram_bin_width_ps,
        )
        updated.update(
            {
                "ctr_ps": float(result.ctr_ps),
                "ctr_uncertainty_ps": float(result.ctr_error_ps),
                "center_ps": float(result.center_ps),
                "coverage_fraction": float(result.coverage_fraction),
                "interval_events": int(result.interval_events),
                "interval_low_ps": float(result.interval_low_ps),
                "interval_high_ps": float(result.interval_high_ps),
                "interval_width_ps": float(result.interval_width_ps),
                "gaussian_equivalent_scale": float(result.gaussian_equivalent_scale),
                "bootstrap_samples": int(result.bootstrap_samples),
                "bootstrap_successful": int(result.bootstrap_successful),
                "ctr_definition": result.definition,
                "histogram_bins": int(result.histogram_bins),
                "histogram_bin_width_ps": float(result.histogram_bin_width_ps),
                "sigma_narrow_ps": float(result.sigma_narrow_ps),
                "sigma_wide_ps": float(result.sigma_wide_ps),
                "narrow_fraction": float(result.narrow_fraction),
            }
        )
        recomputed.append(updated)

    cache[run] = [dict(row) for row in recomputed]
    return [dict(row) for row in recomputed]


def _nema_distribution_plot(
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
    """CTR distribution plot with the NEMA peak and half-maximum width overlaid."""
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
        nema_fits = []
        for method, residual, row in pair:
            fit = fit_nema_fwhm(
                residual,
                histogram_bin_width_ps=histogram_bin_width_ps,
            )
            nema_fits.append((method, residual, row, fit))

        xlim = reporting_module._robust_display_range(
            [residual for _method, residual, _row, _fit in nema_fits],
            quantiles=(0.005, 0.995),
            margin_fraction=0.06,
        )
        half_left = [fit.half_max_left_ps for _m, _r, _row, fit in nema_fits]
        half_right = [fit.half_max_right_ps for _m, _r, _row, fit in nema_fits]
        xlim = (
            min(float(xlim[0]), min(half_left) - 20.0),
            max(float(xlim[1]), max(half_right) + 20.0),
        )

        fig, ax = plt.subplots(figsize=reporting_module.SINGLE_COLUMN)
        peak = 0.0
        for index, (method, residual, row, fit) in enumerate(nema_fits):
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
def reporting_fit_options(
    *,
    ctr_definition: str | None = None,
    histogram_bin_width_ps: float | None = None,
):
    """Apply CTR-definition and fixed histogram width only while rebuilding reports."""
    from matplotlib.axes import Axes

    definition = resolve_ctr_definition(ctr_definition)
    bin_width = resolve_histogram_bin_width_ps(histogram_bin_width_ps)

    original_read_results = reporting_module.read_results
    original_fit_ctr = reporting_module.fit_ctr_ps
    original_edges = reporting_module._median_centered_display_edges
    original_distribution_plot = reporting_module._distribution_plot
    original_legend = Axes.legend
    original_latex_read_csv = latex_tables_module._read_csv
    cache: dict[Path, list[dict[str, Any]]] = {}

    def configured_read_results(run_dir):
        return _recompute_rows(
            run_dir,
            original_read_results,
            definition=definition,
            histogram_bin_width_ps=bin_width,
            cache=cache,
        )

    def configured_fit_ctr(values_ps, config=None, *, seed=0, bootstrap=True):
        return fit_ctr_ps(
            values_ps,
            config,
            seed=seed,
            bootstrap=bootstrap,
            definition=definition,
            histogram_bin_width_ps=bin_width,
        )

    def configured_edges(values, xlim, n_bins=None):
        return fixed_width_histogram_edges(
            values,
            bin_width,
            low_ps=float(xlim[0]),
            high_ps=float(xlim[1]),
        )

    def configured_distribution_plot(output, run, rows, mode, dataset, stage, paths):
        if definition == "nema":
            return _nema_distribution_plot(
                output,
                run,
                rows,
                mode,
                dataset,
                stage,
                paths,
                histogram_bin_width_ps=bin_width,
            )
        return original_distribution_plot(output, run, rows, mode, dataset, stage, paths)

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
    reporting_module._median_centered_display_edges = configured_edges
    reporting_module._distribution_plot = configured_distribution_plot
    latex_tables_module._read_csv = configured_latex_read_csv
    Axes.legend = configured_legend
    try:
        yield definition, bin_width
    finally:
        reporting_module.read_results = original_read_results
        reporting_module.fit_ctr_ps = original_fit_ctr
        reporting_module._median_centered_display_edges = original_edges
        reporting_module._distribution_plot = original_distribution_plot
        latex_tables_module._read_csv = original_latex_read_csv
        Axes.legend = original_legend
