from __future__ import annotations

import json
import logging
from pathlib import Path

from .hyperparameter_plot import plot_hyperparameter_validation
from .result_plots import make_study_result_plots
from .storage import RunStore


def remake_study_plots(config, *, logger=None):
    """Regenerate plots from saved study outputs without preprocessing or training."""
    run_dir = Path(config["output_dir"]).resolve()
    if not run_dir.is_dir() or not (run_dir / "results.csv").is_file():
        raise FileNotFoundError(f"No completed study results found in {run_dir}")

    log = logger or logging.getLogger("waveform-postprocess")
    store = RunStore(run_dir, resume=True)
    rows = store.read_results()
    if not rows:
        raise RuntimeError(f"No result rows found in {store.results_path}")

    candidates_path = run_dir / "candidates.json"
    if candidates_path.is_file():
        candidates = json.loads(candidates_path.read_text(encoding="utf-8"))
        if len(candidates) > 1 and any(r.get("stage") == "validation" for r in rows):
            plot_hyperparameter_validation(
                rows, candidates, run_dir / "hyperparameter_validation_ctr.png", log,
                metric="ctr_ps", metric_label="CTR",
            )
            plot_hyperparameter_validation(
                rows, candidates, run_dir / "hyperparameter_validation_rmse.png", log,
                metric="rmse_ps", metric_label="RMSE",
            )

    plots = make_study_result_plots(
        rows,
        run_dir,
        model=config["model"]["name"],
        mode=config["mode"],
        window_ns=config["window_ns"],
    )

    if plots.get("paired_ctr_improvement") is None:
        log.warning(
            "Paired CTR bootstrap plot unavailable for %s: no saved paired-bootstrap CTR samples.",
            run_dir,
        )
    if plots.get("paired_rmse_improvement") is None:
        log.warning(
            "Paired RMSE bootstrap plot unavailable for %s: saved paired-bootstrap files predate RMSE persistence. "
            "Blind RMSE distributions/correlation can still be remade from results.csv when rmse_ps is present, "
            "but paired RMSE uncertainty cannot be reconstructed from aggregate rows.",
            run_dir,
        )
    log.info("Plots remade without training | %s", run_dir)
    return plots


def remake_batch_plots(configs, *, logger=None):
    outputs = []
    total = len(configs)
    for i, config in enumerate(configs, 1):
        if logger:
            logger.info("Remake plots %d/%d | %s", i, total, config["name"])
        outputs.append((Path(config["output_dir"]).resolve(), remake_study_plots(config, logger=logger)))
    return outputs
