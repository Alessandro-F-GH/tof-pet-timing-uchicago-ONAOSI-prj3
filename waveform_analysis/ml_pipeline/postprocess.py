from __future__ import annotations
from pathlib import Path
from .plotting import load_plot_config,render_run_plots
from .report import collect_runs,generate_report
def remake_plots(result_root,*,logger=None):
    root=Path(result_root).expanduser().resolve();cfg=load_plot_config(root)
    for run in collect_runs(root):render_run_plots(run["directory"],cfg)
    return generate_report(root,logger=logger,reuse_numeric=True)
