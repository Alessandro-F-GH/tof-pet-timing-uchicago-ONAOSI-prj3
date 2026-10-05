from __future__ import annotations

import json
import logging
from pathlib import Path

from .config import BatchConfig
from .hyperparameter_plot import plot_hyperparameter_validation
from .result_plots import make_study_result_plots
from .storage import RunStore


def remake_study_plots(config, *, logger=None):
    run_dir = Path(config["output_dir"]).resolve()
    if not run_dir.is_dir() or not (run_dir / "results.csv").is_file():
        raise FileNotFoundError(f"No completed study results found in {run_dir}")

    log = logger or logging.getLogger("waveform-postprocess")
    store = RunStore(run_dir, resume=True)
    rows = store.read_results()
    if not rows:
        raise RuntimeError(f"No result rows found in {store.results_path}")

    selection_name = str(config["model_selection"]["metric"])
    selection_field = "ctr_ps" if selection_name == "ctr" else "rmse_ps"
    selection_label = "CTR" if selection_name == "ctr" else "RMSE"

    candidates_path = run_dir / "candidates.json"
    if candidates_path.is_file():
        candidates = json.loads(candidates_path.read_text(encoding="utf-8"))
        if any(row.get("phase") == "hyperparameter_validation" for row in rows):
            plot_hyperparameter_validation(
                rows,
                candidates,
                run_dir / f"hyperparameter_validation_{selection_name}.png",
                log,
                metric=selection_field,
                metric_label=selection_label,
            )

    plots = make_study_result_plots(
        rows,
        run_dir,
        model=config["model"]["name"],
        mode=config["mode"],
        window_ns=config["window_ns"],
    )
    log.info("Plots remade without training | %s", run_dir)
    return plots


def remake_batch_plots(batch, *, logger=None):
    if not isinstance(batch, BatchConfig):
        raise TypeError("remake_batch_plots requires a BatchConfig")
    outputs = []
    total = len(batch.runs)
    for index, config in enumerate(batch.runs, 1):
        if logger:
            logger.info("Remake plots %d/%d | %s", index, total, config["name"])
        outputs.append(
            (Path(config["output_dir"]).resolve(), remake_study_plots(config, logger=logger))
        )
    return outputs
