from __future__ import annotations

import csv
import json
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from .dataset import load_prepared_dataset
from .models import get_model
from .stats import ctr_estimate, rmse_ps
from .storage import RunStore
from .train import load_fitted_model, predict_indices
from .view import model_target


def _pearson(x, y):
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    if x.size < 2 or np.std(x) == 0 or np.std(y) == 0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def _replica_rows(rows):
    return sorted(
        [row for row in rows if row.get("phase") == "replica"],
        key=lambda row: int(row["replica_index"]),
    )


def _replica_split_path(store, replica_index):
    manifest = json.loads((store.root / "manifest.json").read_text(encoding="utf-8"))
    shared = (manifest.get("shared_replicas") or {}).get(str(int(replica_index)))
    if not shared:
        raise FileNotFoundError(f"Missing shared replica artifact for replica {replica_index}")
    path = Path(shared) / "split.npz"
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def _residuals_for_row(store, spec, dataset, config, candidates, row, target, logger):
    seed = int(row["seed"])
    replica_index = int(row["replica_index"])
    candidate_id = row["candidate_id"]
    residual_path = store.blind_residuals_path(seed, candidate_id)
    if residual_path.is_file():
        with np.load(residual_path) as data:
            return np.asarray(data["corrected_ps"], float)

    split_path = _replica_split_path(store, replica_index)
    with np.load(split_path) as split_data:
        test = np.asarray(split_data["test"], np.int64)

    model_dir = (
        store.root
        / "models"
        / f"replica_{replica_index:03d}_seed_{seed}"
        / candidate_id
    )
    if not model_dir.is_dir():
        raise FileNotFoundError(
            f"Missing residuals and saved replica model for replica {replica_index}"
        )
    fitted = load_fitted_model(spec, model_dir, candidates[candidate_id], config)
    prediction = predict_indices(spec, fitted, dataset, config["mode"], test)
    corrected = np.asarray(target[test], float) - np.asarray(prediction, float)
    store.save_blind_residuals(seed, candidate_id, corrected)
    logger.info(
        "Blind residuals reconstructed | replica=%d | seed=%d | events=%d",
        replica_index,
        seed,
        len(test),
    )
    return corrected


def run_ctr_binning_scan(config, bin_widths_ps, *, logger=None):
    log = logger or logging.getLogger("ctr-binning-scan")
    run_dir = Path(config["output_dir"]).resolve()
    store = RunStore(run_dir, resume=True)
    rows = store.read_results()
    replicas = _replica_rows(rows)
    if not replicas:
        raise RuntimeError(f"No replica results found in {store.results_path}")

    candidates_path = run_dir / "candidates.json"
    if not candidates_path.is_file():
        raise FileNotFoundError(candidates_path)
    candidates = json.loads(candidates_path.read_text(encoding="utf-8"))
    spec = get_model(config["model"]["name"])
    widths = sorted({float(value) for value in bin_widths_ps})
    if not widths or any(not np.isfinite(value) or value <= 0 for value in widths):
        raise ValueError("bin widths must be finite positive values")

    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    dataset = load_prepared_dataset(manifest["prepared_dataset"])
    target = model_target(dataset, config["mode"])

    residuals = []
    for row in replicas:
        corrected = _residuals_for_row(
            store, spec, dataset, config, candidates, row, target, log
        )
        residuals.append((
            int(row["replica_index"]),
            int(row["seed"]),
            corrected,
            rmse_ps(corrected),
        ))

    output_rows = []
    for width in widths:
        fit_cfg = dict(config["fit"])
        fit_cfg["histogram_bin_width_ps"] = float(width)
        ctrs = []
        rmses = []
        for replica_index, seed, corrected, rmse in residuals:
            try:
                ctr = float(ctr_estimate(corrected, fit_cfg, seed=seed, bootstrap=False).ctr_ps)
            except ValueError:
                ctr = float("nan")
            output_rows.append({
                "replica_index": replica_index,
                "seed": seed,
                "bin_width_ps": width,
                "rmse_ps": rmse,
                "ctr_ps": ctr,
            })
            if np.isfinite(ctr):
                ctrs.append(ctr)
                rmses.append(rmse)
        correlation = _pearson(rmses, ctrs)
        for item in output_rows:
            if item["bin_width_ps"] == width:
                item["rmse_ctr_pearson_r"] = correlation
        log.info(
            "CTR binning scan | width=%.3f ps | valid=%d/%d | RMSE-vs-CTR r=%.4f",
            width,
            len(ctrs),
            len(residuals),
            correlation,
        )

    csv_path = run_dir / "ctr_binning_scan.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=[
                "replica_index", "seed", "bin_width_ps", "rmse_ps",
                "ctr_ps", "rmse_ctr_pearson_r",
            ],
        )
        writer.writeheader()
        writer.writerows(output_rows)

    correlations = []
    for width in widths:
        values = [row for row in output_rows if row["bin_width_ps"] == width]
        correlations.append(float(values[0]["rmse_ctr_pearson_r"]))

    fig, ax = plt.subplots()
    ax.plot(widths, correlations, marker="o")
    ax.axhline(0.0, linestyle=":")
    ax.set_xlabel("Histogram bin width [ps]")
    ax.set_ylabel("Pearson r (RMSE, CTR)")
    ax.set_title(
        f"CTR binning sensitivity\n{spec.name} | {config['mode']} | "
        f"[{config['window_ns']['start']}, {config['window_ns']['end']}] ns"
    )
    for x, y in zip(widths, correlations):
        if np.isfinite(y):
            ax.annotate(f"{y:.3f}", (x, y), textcoords="offset points", xytext=(0, 6), ha="center")
    fig.tight_layout()
    plot_path = run_dir / "rmse_ctr_correlation_vs_bin_width.png"
    fig.savefig(plot_path)
    plt.close(fig)

    summary_path = run_dir / "ctr_binning_scan_summary.csv"
    with summary_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=["bin_width_ps", "rmse_ctr_pearson_r", "n_replicas"],
        )
        writer.writeheader()
        for width, correlation in zip(widths, correlations):
            writer.writerow({
                "bin_width_ps": width,
                "rmse_ctr_pearson_r": correlation,
                "n_replicas": len(residuals),
            })

    log.info("CTR binning scan complete | %s", run_dir)
    return {"plot": plot_path, "csv": csv_path, "summary": summary_path}
