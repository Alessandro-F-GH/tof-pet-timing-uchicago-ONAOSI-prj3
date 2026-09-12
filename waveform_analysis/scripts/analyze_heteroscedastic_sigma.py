from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from waveform_analysis.ml_pipeline.dataset import load_prepared_dataset
from waveform_analysis.ml_pipeline.models.cnn_heteroscedastic import (
    HeteroscedasticCNNArtifact,
    HeteroscedasticSharedScorerCNN,
    predict_distribution,
)
from waveform_analysis.ml_pipeline.sample_mask import apply_sample_mask
from waveform_analysis.ml_pipeline.view import model_target, waveform_view


MODEL_NAME = "cnn_heteroscedastic"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Inspect heteroscedastic-CNN predicted sigma distributions and compare "
            "predicted uncertainty with empirical blind-test residual spread."
        )
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        required=True,
        help="Completed study directory containing trained cnn_heteroscedastic models",
    )
    parser.add_argument(
        "--dataset",
        nargs="+",
        help="Dataset name(s) to analyze; default: every dataset with a trained heteroscedastic CNN",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Output root (default: <run-dir>/sigma_analysis)",
    )
    parser.add_argument(
        "--calibration-bins",
        type=int,
        default=8,
        help="Number of equal-population blind-test sigma bins (default: 8)",
    )
    return parser


def _read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as stream:
        value = json.load(stream)
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return value


def _torch_checkpoint(path: Path):
    import torch

    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _available_datasets(run: Path) -> list[str]:
    models_root = run / "models"
    if not models_root.is_dir():
        return []
    return sorted(
        child.name
        for child in models_root.iterdir()
        if child.is_dir() and (child / MODEL_NAME / "model.pt").is_file()
    )


def _load_model(run: Path, dataset_name: str, manifest: dict[str, Any]):
    model_dir = run / "models" / dataset_name / MODEL_NAME
    metadata = _read_json(model_dir / "metadata.json")
    checkpoint = _torch_checkpoint(model_dir / "model.pt")

    model_config = (
        ((manifest.get("config") or {}).get("models") or {}).get(MODEL_NAME) or {}
    )
    model = HeteroscedasticSharedScorerCNN(model_config.get("architecture", {}))
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    training_metadata = dict(metadata.get("training") or {})
    sigma_max_ps = float(
        checkpoint.get(
            "sigma_max_ps",
            training_metadata.get("sigma_max_ps", 20.0),
        )
    )
    artifact = HeteroscedasticCNNArtifact(
        model=model,
        device="cpu",
        sigma_max_ps=sigma_max_ps,
        metadata=dict(checkpoint.get("metadata") or training_metadata),
    )

    sample_mask = None
    sample_mask_file = metadata.get("sample_mask_file")
    if sample_mask_file:
        sample_mask = np.asarray(
            np.load(model_dir / str(sample_mask_file)),
            dtype=bool,
        ).reshape(-1)

    return artifact, sample_mask


def _stage_arrays(dataset, mode: str, artifact, sample_mask):
    target_all = np.asarray(model_target(dataset, mode), dtype=np.float64)
    stages = {
        "train": np.asarray(dataset.training, dtype=np.int64),
        "validation": np.asarray(dataset.validation, dtype=np.int64),
        "blind": np.asarray(dataset.test, dtype=np.int64),
    }
    output = {}
    for stage, indices in stages.items():
        pair = waveform_view(dataset, mode, indices).materialize()
        pair = apply_sample_mask(pair, sample_mask)
        mean, sigma = predict_distribution(artifact, pair)
        target = target_all[indices]
        raw_residual = target - mean
        gated_prediction = np.where(sigma <= artifact.sigma_max_ps, mean, 0.0)
        gated_residual = target - gated_prediction
        output[stage] = {
            "indices": indices,
            "target": target,
            "mean": mean,
            "sigma": sigma,
            "raw_residual": raw_residual,
            "gated_prediction": gated_prediction,
            "gated_residual": gated_residual,
        }
    return output


def _write_event_csv(path: Path, dataset, stage_data, sigma_max_ps: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "prepared_index",
        "event_index",
        "target_ps",
        "mu_ps",
        "sigma_pred_ps",
        "raw_residual_ps",
        "abs_raw_residual_ps",
        "gated_prediction_ps",
        "gated_residual_ps",
        "sigma_max_ps",
        "correction_applied",
    ]
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for position, prepared_index in enumerate(stage_data["indices"]):
            sigma = float(stage_data["sigma"][position])
            writer.writerow(
                {
                    "prepared_index": int(prepared_index),
                    "event_index": int(dataset.event_index[prepared_index]),
                    "target_ps": float(stage_data["target"][position]),
                    "mu_ps": float(stage_data["mean"][position]),
                    "sigma_pred_ps": sigma,
                    "raw_residual_ps": float(stage_data["raw_residual"][position]),
                    "abs_raw_residual_ps": abs(
                        float(stage_data["raw_residual"][position])
                    ),
                    "gated_prediction_ps": float(
                        stage_data["gated_prediction"][position]
                    ),
                    "gated_residual_ps": float(
                        stage_data["gated_residual"][position]
                    ),
                    "sigma_max_ps": float(sigma_max_ps),
                    "correction_applied": bool(sigma <= sigma_max_ps),
                }
            )


def _calibration_rows(sigma: np.ndarray, residual: np.ndarray, n_bins: int):
    sigma = np.asarray(sigma, dtype=np.float64)
    residual = np.asarray(residual, dtype=np.float64)
    finite = np.isfinite(sigma) & np.isfinite(residual) & (sigma > 0.0)
    sigma = sigma[finite]
    residual = residual[finite]
    if sigma.size < 2:
        return []

    n_bins = max(1, min(int(n_bins), int(sigma.size)))
    quantiles = np.linspace(0.0, 1.0, n_bins + 1)
    edges = np.quantile(sigma, quantiles)
    edges = np.unique(edges)
    if edges.size < 2:
        edges = np.asarray([float(np.min(sigma)), float(np.max(sigma)) + 1e-12])

    rows = []
    for index, (low, high) in enumerate(zip(edges[:-1], edges[1:])):
        if index == len(edges) - 2:
            mask = (sigma >= low) & (sigma <= high)
        else:
            mask = (sigma >= low) & (sigma < high)
        if not np.any(mask):
            continue
        s = sigma[mask]
        r = residual[mask]
        rows.append(
            {
                "bin": len(rows) + 1,
                "sigma_low_ps": float(low),
                "sigma_high_ps": float(high),
                "n": int(s.size),
                "mean_sigma_pred_ps": float(np.mean(s)),
                "median_sigma_pred_ps": float(np.median(s)),
                "empirical_rmse_ps": float(np.sqrt(np.mean(r**2))),
                "empirical_residual_std_ps": float(np.std(r, ddof=1))
                if r.size > 1
                else 0.0,
                "mean_abs_error_ps": float(np.mean(np.abs(r))),
                "mean_residual_ps": float(np.mean(r)),
                "coverage_1sigma": float(np.mean(np.abs(r) <= s)),
                "coverage_2sigma": float(np.mean(np.abs(r) <= 2.0 * s)),
            }
        )
    return rows


def _write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _distribution_plot(
    path: Path,
    dataset_name: str,
    stages: dict[str, dict[str, np.ndarray]],
    sigma_max_ps: float,
) -> None:
    import matplotlib.pyplot as plt

    pooled = np.concatenate(
        [
            np.asarray(stages[name]["sigma"], dtype=np.float64)
            for name in ("train", "validation", "blind")
        ]
    )
    pooled = pooled[np.isfinite(pooled)]
    if not pooled.size:
        return
    low, high = np.quantile(pooled, [0.005, 0.995])
    if high <= low:
        low = float(np.min(pooled))
        high = float(np.max(pooled)) + 1.0
    bins = np.linspace(float(low), float(high), 41)

    fig, ax = plt.subplots(figsize=(8.6, 5.2))
    for stage in ("train", "validation", "blind"):
        values = np.asarray(stages[stage]["sigma"], dtype=np.float64)
        values = values[np.isfinite(values)]
        ax.hist(
            values,
            bins=bins,
            histtype="step",
            density=True,
            linewidth=1.5,
            label=f"{stage} (n={values.size})",
        )
    ax.axvline(
        sigma_max_ps,
        linestyle="--",
        linewidth=1.3,
        label=f"selected sigma_max = {sigma_max_ps:g} ps",
    )
    ax.set_xlabel("Predicted pair sigma [ps]")
    ax.set_ylabel("Density")
    ax.set_title(f"{dataset_name} · predicted heteroscedastic sigma")
    ax.grid(True, alpha=0.2)
    ax.legend()
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)


def _blind_scatter_plot(path: Path, dataset_name: str, blind) -> None:
    import matplotlib.pyplot as plt

    sigma = np.asarray(blind["sigma"], dtype=np.float64)
    absolute_error = np.abs(np.asarray(blind["raw_residual"], dtype=np.float64))
    finite = np.isfinite(sigma) & np.isfinite(absolute_error)
    sigma = sigma[finite]
    absolute_error = absolute_error[finite]
    if not sigma.size:
        return

    fig, ax = plt.subplots(figsize=(7.2, 5.4))
    ax.scatter(sigma, absolute_error, s=14, alpha=0.45)
    xmax = float(np.quantile(sigma, 0.995))
    ymax = float(np.quantile(absolute_error, 0.995))
    ax.set_xlim(left=0.0, right=max(xmax, 1.0))
    ax.set_ylim(bottom=0.0, top=max(ymax, 1.0))
    ax.set_xlabel("Predicted sigma [ps]")
    ax.set_ylabel("Blind |target - mu| [ps]")
    ax.set_title(f"{dataset_name} · blind event uncertainty vs realized error")
    ax.grid(True, alpha=0.2)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)


def _calibration_plot(
    path: Path,
    dataset_name: str,
    rows: list[dict[str, Any]],
) -> None:
    import matplotlib.pyplot as plt

    if not rows:
        return
    predicted = np.asarray([row["mean_sigma_pred_ps"] for row in rows], dtype=float)
    empirical = np.asarray([row["empirical_rmse_ps"] for row in rows], dtype=float)
    counts = np.asarray([row["n"] for row in rows], dtype=int)

    limit = 1.08 * max(float(np.max(predicted)), float(np.max(empirical)), 1.0)
    fig, ax = plt.subplots(figsize=(6.4, 5.4))
    ax.plot([0.0, limit], [0.0, limit], linestyle="--", linewidth=1.2, label="ideal")
    ax.plot(predicted, empirical, marker="o", linewidth=1.4, label="blind bins")
    for x, y, n in zip(predicted, empirical, counts):
        ax.annotate(f"n={n}", (x, y), xytext=(4, 4), textcoords="offset points", fontsize=8)
    ax.set_xlim(0.0, limit)
    ax.set_ylim(0.0, limit)
    ax.set_xlabel("Mean predicted sigma [ps]")
    ax.set_ylabel("Blind empirical RMSE(target - mu) [ps]")
    ax.set_title(f"{dataset_name} · sigma calibration on blind test")
    ax.grid(True, alpha=0.2)
    ax.legend()
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)


def _standardized_residual_plot(path: Path, dataset_name: str, blind) -> None:
    import matplotlib.pyplot as plt

    sigma = np.asarray(blind["sigma"], dtype=np.float64)
    residual = np.asarray(blind["raw_residual"], dtype=np.float64)
    finite = np.isfinite(sigma) & np.isfinite(residual) & (sigma > 0.0)
    z = residual[finite] / sigma[finite]
    if not z.size:
        return

    display = z[(z >= -5.0) & (z <= 5.0)]
    x = np.linspace(-5.0, 5.0, 500)
    normal = np.exp(-0.5 * x**2) / math.sqrt(2.0 * math.pi)

    fig, ax = plt.subplots(figsize=(7.2, 5.2))
    ax.hist(display, bins=50, density=True, histtype="step", linewidth=1.5, label="blind standardized residual")
    ax.plot(x, normal, linestyle="--", linewidth=1.2, label="N(0,1)")
    ax.set_xlim(-5.0, 5.0)
    ax.set_xlabel("(target - mu) / sigma_pred")
    ax.set_ylabel("Density")
    ax.set_title(f"{dataset_name} · blind standardized residual calibration")
    ax.grid(True, alpha=0.2)
    ax.legend()
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)


def analyze(
    run_dir: Path,
    *,
    requested_datasets: list[str] | None = None,
    output_dir: Path | None = None,
    calibration_bins: int = 8,
) -> list[Path]:
    run = run_dir.resolve()
    manifest = _read_json(run / "manifest.json")
    mode = str(manifest.get("mode") or manifest["config"]["mode"])
    available = _available_datasets(run)
    if requested_datasets:
        missing = sorted(set(requested_datasets) - set(available))
        if missing:
            raise ValueError(
                f"No trained {MODEL_NAME} model found for dataset(s): {missing}"
            )
        datasets = list(requested_datasets)
    else:
        datasets = available
    if not datasets:
        raise ValueError(f"No trained {MODEL_NAME} models found in {run}")

    output = (output_dir or (run / "sigma_analysis")).resolve()
    plot_dir = output / "plots"
    csv_dir = output / "csv"
    generated: list[Path] = []

    for dataset_name in datasets:
        dataset_manifest = manifest["datasets"][dataset_name]
        dataset = load_prepared_dataset(dataset_manifest["prepared_dir"])
        artifact, sample_mask = _load_model(run, dataset_name, manifest)
        stages = _stage_arrays(dataset, mode, artifact, sample_mask)

        for stage in ("train", "validation", "blind"):
            target = csv_dir / dataset_name / f"{stage}_events.csv"
            _write_event_csv(target, dataset, stages[stage], artifact.sigma_max_ps)
            generated.append(target)

        blind = stages["blind"]
        calibration = _calibration_rows(
            blind["sigma"],
            blind["raw_residual"],
            calibration_bins,
        )
        calibration_csv = csv_dir / dataset_name / "blind_sigma_calibration.csv"
        _write_rows(calibration_csv, calibration)
        if calibration:
            generated.append(calibration_csv)

        distribution_pdf = plot_dir / dataset_name / "sigma_distribution.pdf"
        _distribution_plot(
            distribution_pdf,
            dataset_name,
            stages,
            artifact.sigma_max_ps,
        )
        generated.append(distribution_pdf)

        scatter_pdf = plot_dir / dataset_name / "blind_sigma_vs_abs_error.pdf"
        _blind_scatter_plot(scatter_pdf, dataset_name, blind)
        generated.append(scatter_pdf)

        calibration_pdf = plot_dir / dataset_name / "blind_sigma_calibration.pdf"
        _calibration_plot(calibration_pdf, dataset_name, calibration)
        generated.append(calibration_pdf)

        standardized_pdf = plot_dir / dataset_name / "blind_standardized_residual.pdf"
        _standardized_residual_plot(standardized_pdf, dataset_name, blind)
        generated.append(standardized_pdf)

        coverage_1 = float(
            np.mean(np.abs(blind["raw_residual"]) <= blind["sigma"])
        )
        coverage_2 = float(
            np.mean(np.abs(blind["raw_residual"]) <= 2.0 * blind["sigma"])
        )
        print(
            f"{dataset_name}: sigma_max={artifact.sigma_max_ps:g} ps | "
            f"blind mean sigma={np.mean(blind['sigma']):.3f} ps | "
            f"median sigma={np.median(blind['sigma']):.3f} ps | "
            f"coverage |r|<=sigma={100.0*coverage_1:.1f}% | "
            f"|r|<=2sigma={100.0*coverage_2:.1f}%"
        )

    return generated


def main() -> None:
    args = _parser().parse_args()
    for path in analyze(
        args.run_dir,
        requested_datasets=args.dataset,
        output_dir=args.output_dir,
        calibration_bins=args.calibration_bins,
    ):
        print(path)


if __name__ == "__main__":
    main()
