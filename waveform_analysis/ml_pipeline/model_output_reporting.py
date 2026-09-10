from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import numpy as np

from .dataset import load_prepared_dataset
from .view import model_target


def _model_output(run: Path, dataset: str, model: str, stage: str) -> np.ndarray | None:
    path = run / "artifacts" / dataset / f"{model}_{stage}_model_output_ps.npy"
    return np.asarray(np.load(path), dtype=np.float64).reshape(-1) if path.is_file() else None


def _stage_indices(run: Path, dataset: str, stage: str) -> np.ndarray:
    path = run / "splits" / f"{dataset}.npz"
    if not path.is_file():
        raise FileNotFoundError(f"Missing split artifact: {path}")
    key = "training" if stage == "train" else "test"
    with np.load(path) as split:
        return np.asarray(split[key], dtype=np.int64)


def _target(run: Path, manifest: dict[str, Any], dataset: str, mode: str, stage: str) -> np.ndarray:
    prepared = load_prepared_dataset(manifest["datasets"][dataset]["prepared_dir"])
    indices = _stage_indices(run, dataset, stage)
    return np.asarray(model_target(prepared, mode)[indices], dtype=np.float64).reshape(-1)


def _pearson(x: np.ndarray, y: np.ndarray) -> tuple[float, int]:
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    if x.shape != y.shape:
        raise ValueError(f"Correlation arrays have different shapes: {x.shape} != {y.shape}")
    finite = np.isfinite(x) & np.isfinite(y)
    n = int(np.count_nonzero(finite))
    if n < 2:
        return float("nan"), n
    x = x[finite]
    y = y[finite]
    if np.std(x) == 0.0 or np.std(y) == 0.0:
        return float("nan"), n
    return float(np.corrcoef(x, y)[0, 1]), n


def _display_limits(arrays: list[np.ndarray]) -> tuple[float, float]:
    finite = []
    for array in arrays:
        values = np.asarray(array, dtype=np.float64).reshape(-1)
        values = values[np.isfinite(values)]
        if values.size:
            finite.append(values)
    if not finite:
        return -1.0, 1.0
    pooled = np.concatenate(finite)
    low, high = np.quantile(pooled, [0.005, 0.995])
    if not np.isfinite(low) or not np.isfinite(high):
        return -1.0, 1.0
    if high <= low:
        margin = max(abs(float(low)) * 0.05, 1.0)
        return float(low - margin), float(high + margin)
    margin = 0.06 * float(high - low)
    return float(low - margin), float(high + margin)


def plot_prediction_vs_target(
    output: Path,
    run: Path,
    manifest: dict[str, Any],
    mode: str,
    dataset: str,
    model: str,
    label: str,
    paths: list[Path],
) -> None:
    """Scatter model correction versus the exact ML target for train and blind/test."""
    import matplotlib.pyplot as plt

    stages: list[tuple[str, np.ndarray, np.ndarray]] = []
    for stage in ("train", "test"):
        prediction = _model_output(run, dataset, model, stage)
        if prediction is None:
            continue
        target = _target(run, manifest, dataset, mode, stage)
        if prediction.size != target.size:
            raise ValueError(
                f"{dataset}/{model}/{stage}: prediction/target length mismatch "
                f"{prediction.size} != {target.size}"
            )
        stages.append((stage, target, prediction))
    if not stages:
        return

    limits = _display_limits([array for _, target, prediction in stages for array in (target, prediction)])
    fig, axes = plt.subplots(1, len(stages), figsize=(6.1 * len(stages), 5.4), squeeze=False)
    for ax, (stage, target, prediction) in zip(axes[0], stages):
        finite = np.isfinite(target) & np.isfinite(prediction)
        x = target[finite]
        y = prediction[finite]
        correlation, n = _pearson(target, prediction)
        ax.scatter(x, y, s=7, alpha=0.28, rasterized=True)
        ax.plot(limits, limits, ls="--", lw=1.1, label="ideal prediction")
        ax.set_xlim(*limits)
        ax.set_ylim(*limits)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel("ML target [ps]")
        ax.set_ylabel("Model prediction [ps]")
        ax.set_title(f"{stage.capitalize()} · Pearson r={correlation:.4f} · n={n}")
        ax.grid(True, alpha=0.2)
        ax.legend(loc="best")
    fig.suptitle(f"{label} prediction vs target · {mode.replace('_', ' ')} · {dataset}")
    fig.tight_layout()
    target_path = output / f"prediction_vs_target_{dataset}.pdf"
    fig.savefig(target_path, bbox_inches="tight")
    plt.close(fig)
    paths.append(target_path)


def _stage_output_matrix(
    run: Path,
    dataset: str,
    models: list[str],
    stage: str,
) -> tuple[list[str], np.ndarray, np.ndarray]:
    available: list[tuple[str, np.ndarray]] = []
    for model in models:
        values = _model_output(run, dataset, model, stage)
        if values is not None:
            available.append((model, values))
    if len(available) < 2:
        return [], np.empty((0, 0), dtype=np.float64), np.empty((0, 0), dtype=np.int64)

    lengths = {values.size for _, values in available}
    if len(lengths) != 1:
        detail = {model: int(values.size) for model, values in available}
        raise ValueError(f"{dataset}/{stage}: model-output lengths differ: {detail}")

    names = [model for model, _ in available]
    n_models = len(names)
    corr = np.full((n_models, n_models), np.nan, dtype=np.float64)
    counts = np.zeros((n_models, n_models), dtype=np.int64)
    for i, (_, left) in enumerate(available):
        for j, (_, right) in enumerate(available):
            value, n = _pearson(left, right)
            corr[i, j] = value
            counts[i, j] = n
    return names, corr, counts


def _write_matrix_csv(path: Path, names: list[str], matrix: np.ndarray) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["model", *names])
        for name, row in zip(names, matrix):
            writer.writerow([name, *row.tolist()])


def plot_model_output_correlations(
    output: Path,
    run: Path,
    mode: str,
    dataset: str,
    models: list[str],
    labels: dict[str, str],
    paths: list[Path],
) -> None:
    """Plot pairwise Pearson correlations between model correction outputs."""
    import matplotlib.pyplot as plt

    stages = []
    for stage in ("train", "test"):
        names, matrix, counts = _stage_output_matrix(run, dataset, models, stage)
        if names:
            stages.append((stage, names, matrix, counts))
    if not stages:
        return

    fig, axes = plt.subplots(1, len(stages), figsize=(6.3 * len(stages), 5.7), squeeze=False)
    image = None
    for ax, (stage, names, matrix, counts) in zip(axes[0], stages):
        image = ax.imshow(matrix, vmin=-1.0, vmax=1.0, cmap="coolwarm")
        display_names = [labels.get(name, name) for name in names]
        ax.set_xticks(np.arange(len(names)))
        ax.set_yticks(np.arange(len(names)))
        ax.set_xticklabels(display_names, rotation=35, ha="right")
        ax.set_yticklabels(display_names)
        for i in range(len(names)):
            for j in range(len(names)):
                value = matrix[i, j]
                text = "nan" if not np.isfinite(value) else f"{value:.3f}"
                ax.text(j, i, text, ha="center", va="center", fontsize=9)
        ax.set_title(f"{stage.capitalize()} model outputs")
        csv_path = output / f"model_output_correlation_{stage}_{dataset}.csv"
        _write_matrix_csv(csv_path, names, matrix)
        paths.append(csv_path)
        count_path = output / f"model_output_correlation_counts_{stage}_{dataset}.csv"
        _write_matrix_csv(count_path, names, counts)
        paths.append(count_path)
    if image is not None:
        cbar = fig.colorbar(image, ax=axes.ravel().tolist(), fraction=0.035, pad=0.04)
        cbar.set_label("Pearson correlation")
    fig.suptitle(f"Model-output correlation · {mode.replace('_', ' ')} · {dataset}")
    fig.subplots_adjust(bottom=0.22, top=0.88, wspace=0.35)
    target_path = output / f"model_output_correlation_{dataset}.pdf"
    fig.savefig(target_path, bbox_inches="tight")
    plt.close(fig)
    paths.append(target_path)


def make_model_output_reports(
    run_dir: str | Path,
    output_dir: str | Path,
    *,
    labels: dict[str, str] | None = None,
) -> list[Path]:
    """Create prediction-target scatters and multi-model output correlations."""
    run = Path(run_dir).resolve()
    output_root = Path(output_dir).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((run / "manifest.json").read_text(encoding="utf-8"))
    mode = str(manifest.get("mode") or manifest["config"]["mode"])
    labels = dict(labels or {})
    paths: list[Path] = []

    datasets = list((manifest.get("datasets") or {}).keys())
    for dataset in datasets:
        artifact_dir = run / "artifacts" / dataset
        models = sorted(
            {
                path.name[: -len("_test_model_output_ps.npy")]
                for path in artifact_dir.glob("*_test_model_output_ps.npy")
                if path.name.endswith("_test_model_output_ps.npy")
            }
        )
        if not models:
            models = sorted(
                {
                    path.name[: -len("_train_model_output_ps.npy")]
                    for path in artifact_dir.glob("*_train_model_output_ps.npy")
                    if path.name.endswith("_train_model_output_ps.npy")
                }
            )
        for model in models:
            model_dir = output_root / model
            model_dir.mkdir(parents=True, exist_ok=True)
            plot_prediction_vs_target(
                model_dir,
                run,
                manifest,
                mode,
                dataset,
                model,
                labels.get(model, model),
                paths,
            )
        if len(models) > 1:
            correlation_dir = output_root / "correlations"
            correlation_dir.mkdir(parents=True, exist_ok=True)
            plot_model_output_correlations(
                correlation_dir,
                run,
                mode,
                dataset,
                models,
                labels,
                paths,
            )
    return paths
