from __future__ import annotations

from contextlib import contextmanager
import json
from pathlib import Path
from typing import Any, Callable

import numpy as np

from utils_fit import (
    CTR_DEFINITIONS,
    DEFAULT_CTR_DEFINITION,
    DEFAULT_HISTOGRAM_BINS,
    fit_ctr_ps,
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


def resolve_histogram_bins(value: int | None) -> int:
    if isinstance(value, bool):
        raise ValueError("histogram_bins must be a positive integer")
    bins = DEFAULT_HISTOGRAM_BINS if value is None else int(value)
    if value is not None and bins != value:
        raise ValueError("histogram_bins must be a positive integer")
    if bins <= 0:
        raise ValueError("histogram_bins must be a positive integer")
    return bins


def _fit_config(manifest: dict[str, Any]) -> dict[str, Any]:
    return dict((manifest.get("config") or {}).get("fit") or {})


def _base_seed(manifest: dict[str, Any]) -> int:
    return int(((manifest.get("config") or {}).get("validation") or {}).get("seed", 0))


def _recompute_rows(
    run_dir: str | Path,
    original_reader: Callable[[str | Path], list[dict[str, Any]]],
    *,
    definition: str,
    histogram_bins: int,
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
            histogram_bins=histogram_bins,
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
                "sigma_narrow_ps": float(result.sigma_narrow_ps),
                "sigma_wide_ps": float(result.sigma_wide_ps),
                "narrow_fraction": float(result.narrow_fraction),
            }
        )
        recomputed.append(updated)

    cache[run] = [dict(row) for row in recomputed]
    return [dict(row) for row in recomputed]


@contextmanager
def reporting_fit_options(
    *,
    ctr_definition: str | None = None,
    histogram_bins: int | None = None,
):
    """Apply CTR-definition and histogram settings only while rebuilding reports."""
    from matplotlib.axes import Axes

    definition = resolve_ctr_definition(ctr_definition)
    bins = resolve_histogram_bins(histogram_bins)
    if definition == "double_gaussian" and bins < 5:
        raise ValueError("double_gaussian reporting requires at least 5 histogram bins")

    original_read_results = reporting_module.read_results
    original_fit_ctr = reporting_module.fit_ctr_ps
    original_edges = reporting_module._median_centered_display_edges
    original_legend = Axes.legend
    original_latex_read_csv = latex_tables_module._read_csv
    cache: dict[Path, list[dict[str, Any]]] = {}

    def configured_read_results(run_dir):
        return _recompute_rows(
            run_dir,
            original_read_results,
            definition=definition,
            histogram_bins=bins,
            cache=cache,
        )

    def configured_fit_ctr(values_ps, config=None, *, seed=0, bootstrap=True):
        return fit_ctr_ps(
            values_ps,
            config,
            seed=seed,
            bootstrap=bootstrap,
            definition=definition,
            histogram_bins=bins,
        )

    def configured_edges(values, xlim, n_bins=DEFAULT_HISTOGRAM_BINS):
        return original_edges(values, xlim, bins)

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
    latex_tables_module._read_csv = configured_latex_read_csv
    Axes.legend = configured_legend
    try:
        yield definition, bins
    finally:
        reporting_module.read_results = original_read_results
        reporting_module.fit_ctr_ps = original_fit_ctr
        reporting_module._median_centered_display_edges = original_edges
        latex_tables_module._read_csv = original_latex_read_csv
        Axes.legend = original_legend
