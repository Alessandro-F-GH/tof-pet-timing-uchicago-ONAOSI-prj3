from __future__ import annotations
from pathlib import Path
from waveform_analysis.reporting.plotting import load_plot_config, render_run_plots
from waveform_analysis.reporting.report import collect_runs, generate_report


def remake_plots(result_root, *, logger=None):
    root = Path(result_root).expanduser().resolve()
    cfg = load_plot_config(root)
    for run in collect_runs(root):
        render_run_plots(run["directory"], cfg)
    return generate_report(root, logger=logger, reuse_numeric=True)


# Retain serialized identities while legacy imports resolve to this module.
from waveform_analysis.core.compat import (
    preserve_legacy_identity as _preserve_legacy_identity,
)

_preserve_legacy_identity(__name__, "waveform_analysis.ml_pipeline.postprocess")
